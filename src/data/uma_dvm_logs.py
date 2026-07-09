"""
src/data/uma_dvm_logs.py — Decoders for UMA DVM (VotingV2) VoteRevealed logs.

Pure functions over raw eth_getLogs dicts. No network.

Event, verified against the deployed VotingV2 source
(mainnet 0x004395edb43EFca9885CEdad51EC9fAf93Bd34ac):

    event VoteRevealed(
        address indexed voter,
        address indexed caller,
        uint32          roundId,
        bytes32 indexed identifier,
        uint256         time,
        bytes           ancillaryData,
        int256          price,
        uint128         numTokens
    );

Note `numTokens` is uint128 and `roundId` is uint32 — decoding either as
uint256 silently corrupts every field after it.
"""
from __future__ import annotations

import re

from eth_abi import decode as abi_decode
from eth_utils import keccak

VOTING_V2 = "0x004395edb43EFca9885CEdad51EC9fAf93Bd34ac"

VOTE_REVEALED_SIG = "VoteRevealed(address,address,uint32,bytes32,uint256,bytes,int256,uint128)"
VOTE_REVEALED_TOPIC = "0x" + keccak(text=VOTE_REVEALED_SIG).hex()

# UMIP-107 identifier used by every Polymarket price request.
YES_OR_NO_IDENTIFIER = "0x" + b"YES_OR_NO_QUERY".hex().ljust(64, "0")

_NON_INDEXED = ["uint32", "uint256", "bytes", "int256", "uint128"]

# OracleChildTunnel compresses the stamped ancillary data into this key/value blob.
_ANC_HASH_RE = re.compile(rb"ancillaryDataHash:([0-9a-fA-F]{64})")


def extract_ancillary_hash(ancillary_data: bytes) -> str | None:
    """Pull the `ancillaryDataHash` a Polygon-originated DVM request carries.

    Returns None for mainnet-native requests, whose ancillary data is the raw
    text rather than the compressed form.
    """
    match = _ANC_HASH_RE.search(ancillary_data)
    return match.group(1).decode().lower() if match else None


def _topic_address(topic: str) -> str:
    return "0x" + topic[-40:]


def decode_vote_revealed(log: dict) -> dict:
    """Decode one raw VoteRevealed log into a flat record."""
    topics = log["topics"]
    if topics[0].lower() != VOTE_REVEALED_TOPIC:
        raise ValueError(f"not a VoteRevealed log: topic0={topics[0]}")

    data = log["data"]
    round_id, request_timestamp, ancillary_data, price, num_tokens = abi_decode(
        _NON_INDEXED, bytes.fromhex(data[2:] if data.startswith("0x") else data)
    )
    return {
        "voter": _topic_address(topics[1]).lower(),
        "caller": _topic_address(topics[2]).lower(),
        "identifier": topics[3].lower(),
        "round_id": round_id,
        "request_timestamp": request_timestamp,
        "ancillary_data": ancillary_data,
        "stamped_hash": extract_ancillary_hash(ancillary_data),
        "vote_price": price,
        # numTokens is the voter's staked UMA (1e18), i.e. their voting weight.
        "stake": num_tokens / 1e18,
        "block_number": int(log["blockNumber"], 16),
        "vote_tx_hash": log["transactionHash"],
    }
