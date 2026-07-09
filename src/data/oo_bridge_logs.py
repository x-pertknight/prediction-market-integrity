"""
src/data/oo_bridge_logs.py — Decoders for the Polygon->mainnet UMA oracle bridge.

When an OptimisticOracleV2 request is disputed on Polygon, the OracleChildTunnel
emits, in that same transaction, the exact price request it forwards to the
mainnet DVM. That event is the ground truth for linking a dispute to the votes
cast on it, and it resolves two things a Polygon-side dispute log cannot:

  * `time` -- Polymarket's requests are event-based, so OOv2 escalates under the
    PROPOSAL time, not DisputePrice.timestamp. These differ by days.
  * `ancillaryData` -- already compressed to the "ancillaryDataHash:...,
    childChainId:137" blob that the DVM (and every VoteRevealed) carries.

Pure functions over raw log dicts. No network.
"""
from __future__ import annotations

from eth_abi import decode as abi_decode
from eth_utils import keccak

# Polygon OracleChildTunnel (verified in dispute tx 0xaba67ff5...).
ORACLE_CHILD_TUNNEL = "0xac60353a54873c446101216829a6A98cDbbC3f3D"

PRICE_REQUEST_ADDED_SIG = "PriceRequestAdded(bytes32,uint256,bytes,bytes32)"
PRICE_REQUEST_ADDED_TOPIC = "0x" + keccak(text=PRICE_REQUEST_ADDED_SIG).hex()

PRICE_REQUEST_BRIDGED_SIG = "PriceRequestBridged(address,bytes32,uint256,bytes,bytes32,bytes32)"
PRICE_REQUEST_BRIDGED_TOPIC = "0x" + keccak(text=PRICE_REQUEST_BRIDGED_SIG).hex()


def _data_bytes(log: dict) -> bytes:
    data = log["data"]
    return bytes.fromhex(data[2:] if data.startswith("0x") else data)


def decode_price_request_added(log: dict) -> dict:
    """PriceRequestAdded(bytes32 indexed identifier, uint256 time, bytes ancillaryData, bytes32 indexed requestHash)."""
    if log["topics"][0].lower() != PRICE_REQUEST_ADDED_TOPIC:
        raise ValueError(f"not a PriceRequestAdded log: topic0={log['topics'][0]}")
    time_, ancillary_data = abi_decode(["uint256", "bytes"], _data_bytes(log))
    return {
        "identifier": log["topics"][1].lower(),
        "request_hash": log["topics"][2].lower(),
        "dvm_request_time": time_,
        "ancillary_data": ancillary_data,
    }


def decode_price_request_bridged(log: dict) -> dict:
    """PriceRequestBridged(address indexed requester, bytes32 identifier, uint256 time,
    bytes ancillaryData, bytes32 indexed childRequestId, bytes32 indexed parentRequestId)."""
    if log["topics"][0].lower() != PRICE_REQUEST_BRIDGED_TOPIC:
        raise ValueError(f"not a PriceRequestBridged log: topic0={log['topics'][0]}")
    identifier, time_, ancillary_data = abi_decode(["bytes32", "uint256", "bytes"], _data_bytes(log))
    return {
        "requester": "0x" + log["topics"][1][-40:].lower(),
        "child_request_id": log["topics"][2].lower(),
        "parent_request_id": log["topics"][3].lower(),
        "identifier": "0x" + identifier.hex(),
        "dvm_request_time": time_,
        "ancillary_data": ancillary_data,
    }


def dvm_request_from_receipt(receipt: dict) -> dict | None:
    """Pull the DVM price request a dispute transaction forwarded to mainnet.

    Prefers PriceRequestAdded; falls back to PriceRequestBridged, which is emitted
    even when the child tunnel has already seen the (identifier, time, ancillary)
    triple and so skips PriceRequestAdded. Returns None when the receipt carries
    neither, meaning the dispute never escalated.
    """
    tunnel = ORACLE_CHILD_TUNNEL.lower()
    logs = [lg for lg in receipt["logs"] if lg["address"].lower() == tunnel]
    for log in logs:
        if log["topics"][0].lower() == PRICE_REQUEST_ADDED_TOPIC:
            return decode_price_request_added(log)
    for log in logs:
        if log["topics"][0].lower() == PRICE_REQUEST_BRIDGED_TOPIC:
            return decode_price_request_bridged(log)
    return None
