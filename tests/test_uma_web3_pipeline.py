"""tests/test_uma_web3_pipeline.py — Decoders and CTF balance reconstruction.

Synthetic logs throughout, zero network access. The one exception to "synthetic"
is a handful of field values copied verbatim from real on-chain events, used as
regression anchors so a decoder change that silently corrupts a field is caught.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from eth_abi import encode as abi_encode
from eth_utils import keccak

from data.ctf_positions import (
    TRANSFER_BATCH_TOPIC, TRANSFER_SINGLE_TOPIC, ZERO_ADDRESS,
    balance_at, condition_id, decode_transfer, position_side, replay_balances, safe_proxy_address,
)
from data.oo_bridge_logs import (
    PRICE_REQUEST_ADDED_TOPIC, decode_price_request_added, dvm_request_from_receipt,
    ORACLE_CHILD_TUNNEL,
)
from data.uma_dvm_logs import (
    VOTE_REVEALED_TOPIC, YES_OR_NO_IDENTIFIER, decode_vote_revealed, extract_ancillary_hash,
)

ALICE = "0x" + "a1" * 20
BOB = "0x" + "b0" * 20
UMA_IGNORE_PRICE = -(2 ** 255)


def _topic(addr: str) -> str:
    return "0x" + "00" * 12 + addr[2:]


# ── VoteRevealed decoding ────────────────────────────────────────────────────

def _vote_log(voter=ALICE, caller=BOB, round_id=10132, request_time=1750780863,
              ancillary=b"ancillaryDataHash:" + b"c2" * 32, price=10 ** 18, num_tokens=552_456_613_972_901_357_245):
    data = abi_encode(["uint32", "uint256", "bytes", "int256", "uint128"],
                      [round_id, request_time, ancillary, price, num_tokens])
    return {
        "topics": [VOTE_REVEALED_TOPIC, _topic(voter), _topic(caller), YES_OR_NO_IDENTIFIER],
        "data": "0x" + data.hex(),
        "blockNumber": hex(22785382),
        "transactionHash": "0x" + "ab" * 32,
    }


class TestVoteRevealedDecoding:
    def test_decodes_all_fields(self):
        rec = decode_vote_revealed(_vote_log())
        assert rec["voter"] == ALICE.lower()
        assert rec["caller"] == BOB.lower()
        assert rec["round_id"] == 10132
        assert rec["request_timestamp"] == 1750780863
        assert rec["vote_price"] == 10 ** 18
        assert rec["stake"] == pytest.approx(552.4566139729013)
        assert rec["block_number"] == 22785382

    def test_num_tokens_is_uint128_not_uint256(self):
        """uint128 numTokens: decoding it as uint256 would shift/corrupt the field.

        Anchored on the real reveal in tx 0x5fa1aa56... (stake 552.4566... UMA).
        """
        rec = decode_vote_revealed(_vote_log(num_tokens=552_456_613_972_901_357_245))
        assert rec["stake"] == pytest.approx(552.456613972901, rel=1e-12)

    def test_ignore_price_survives_int256_roundtrip(self):
        """The sentinel is int256.min; an unsigned decode would wrap it to a huge positive."""
        rec = decode_vote_revealed(_vote_log(price=UMA_IGNORE_PRICE))
        assert rec["vote_price"] == UMA_IGNORE_PRICE

    def test_rejects_foreign_topic(self):
        log = _vote_log()
        log["topics"][0] = "0x" + "ff" * 32
        with pytest.raises(ValueError, match="not a VoteRevealed"):
            decode_vote_revealed(log)

    def test_extracts_ancillary_hash(self):
        h = "c2883106c1be3ad8374f6dfa49d8589aad207582aa0c49dbe8e7be9113ed8c6e"
        blob = f"ancillaryDataHash:{h},childBlockNumber:73169792,childChainId:137".encode()
        assert extract_ancillary_hash(blob) == h

    def test_ancillary_hash_none_for_uncompressed(self):
        """Mainnet-native requests carry raw text, not the compressed blob."""
        assert extract_ancillary_hash(b"q: title: Will X happen?, res_data: p1: 0") is None


# ── OracleChildTunnel bridge decoding ────────────────────────────────────────

def _price_request_added_log(dvm_time=1750780863, anc_hash="c2" * 32):
    blob = f"ancillaryDataHash:{anc_hash},childBlockNumber:73169792,childChainId:137".encode()
    data = abi_encode(["uint256", "bytes"], [dvm_time, blob])
    return {
        "address": ORACLE_CHILD_TUNNEL,
        "topics": [PRICE_REQUEST_ADDED_TOPIC, YES_OR_NO_IDENTIFIER, "0x" + "e2" * 32],
        "data": "0x" + data.hex(),
    }


class TestBridgeDecoding:
    def test_decodes_dvm_request_time(self):
        rec = decode_price_request_added(_price_request_added_log())
        assert rec["dvm_request_time"] == 1750780863
        assert extract_ancillary_hash(rec["ancillary_data"]) == "c2" * 32

    def test_dvm_request_from_receipt_finds_tunnel_log(self):
        receipt = {"logs": [
            {"address": "0x" + "de" * 20, "topics": ["0x" + "11" * 32], "data": "0x"},
            _price_request_added_log(),
        ]}
        assert dvm_request_from_receipt(receipt)["dvm_request_time"] == 1750780863

    def test_dvm_request_from_receipt_none_when_never_escalated(self):
        receipt = {"logs": [{"address": "0x" + "de" * 20, "topics": ["0x" + "11" * 32], "data": "0x"}]}
        assert dvm_request_from_receipt(receipt) is None

    def test_ignores_tunnel_lookalike_from_other_address(self):
        """A PriceRequestAdded from a non-tunnel address must not be trusted."""
        log = _price_request_added_log()
        log["address"] = "0x" + "de" * 20
        assert dvm_request_from_receipt({"logs": [log]}) is None


# ── CTF ERC-1155 balance reconstruction ──────────────────────────────────────

def _single(frm, to, token_id, value, block):
    return {
        "topics": [TRANSFER_SINGLE_TOPIC, _topic(BOB), _topic(frm), _topic(to)],
        "data": "0x" + abi_encode(["uint256", "uint256"], [token_id, value]).hex(),
        "blockNumber": hex(block),
    }


def _batch(frm, to, ids, values, block):
    return {
        "topics": [TRANSFER_BATCH_TOPIC, _topic(BOB), _topic(frm), _topic(to)],
        "data": "0x" + abi_encode(["uint256[]", "uint256[]"], [ids, values]).hex(),
        "blockNumber": hex(block),
    }


def _flatten(logs):
    return [t for log in logs for t in decode_transfer(log)]


class TestTransferDecoding:
    def test_single(self):
        [t] = decode_transfer(_single(ZERO_ADDRESS, ALICE, 7, 100, 10))
        assert (t["from"], t["to"], t["position_id"], t["value"], t["block_number"]) == \
               (ZERO_ADDRESS.lower(), ALICE.lower(), 7, 100, 10)

    def test_batch_flattens_to_one_record_per_id(self):
        ts = decode_transfer(_batch(ZERO_ADDRESS, ALICE, [7, 8], [100, 250], 10))
        assert [(t["position_id"], t["value"]) for t in ts] == [(7, 100), (8, 250)]

    def test_rejects_foreign_topic(self):
        log = _single(ALICE, BOB, 1, 1, 1)
        log["topics"][0] = "0x" + "ff" * 32
        with pytest.raises(ValueError, match="not an ERC-1155 transfer"):
            decode_transfer(log)


class TestReplayBalances:
    def test_mint_then_partial_sale(self):
        transfers = _flatten([
            _single(ZERO_ADDRESS, ALICE, 7, 1000, 10),   # split: mint
            _single(ALICE, BOB, 7, 400, 20),             # sells 400
        ])
        assert replay_balances(transfers, ALICE) == {7: 600}
        assert replay_balances(transfers, BOB) == {7: 400}

    def test_balance_is_point_in_time(self):
        """A dispute-time balance must ignore transfers after the dispute block."""
        transfers = _flatten([
            _single(ZERO_ADDRESS, ALICE, 7, 1000, 10),
            _single(ALICE, BOB, 7, 1000, 30),   # dumps the whole position later
        ])
        assert balance_at(transfers, ALICE, 7, up_to_block=20) == 1000
        assert balance_at(transfers, ALICE, 7, up_to_block=30) == 0

    def test_transfer_on_the_boundary_block_is_included(self):
        transfers = _flatten([_single(ZERO_ADDRESS, ALICE, 7, 500, 25)])
        assert balance_at(transfers, ALICE, 7, up_to_block=24) == 0
        assert balance_at(transfers, ALICE, 7, up_to_block=25) == 500

    def test_burn_to_zero_address_debits(self):
        transfers = _flatten([
            _single(ZERO_ADDRESS, ALICE, 7, 1000, 10),
            _single(ALICE, ZERO_ADDRESS, 7, 1000, 20),   # merge/redeem
        ])
        assert replay_balances(transfers, ALICE) == {}

    def test_self_transfer_nets_to_zero(self):
        transfers = _flatten([
            _single(ZERO_ADDRESS, ALICE, 7, 100, 10),
            _single(ALICE, ALICE, 7, 100, 20),
        ])
        assert replay_balances(transfers, ALICE) == {7: 100}

    def test_batch_and_single_compose(self):
        transfers = _flatten([
            _batch(ZERO_ADDRESS, ALICE, [1, 2], [100, 200], 5),
            _single(ALICE, BOB, 2, 50, 6),
        ])
        assert replay_balances(transfers, ALICE) == {1: 100, 2: 150}

    def test_unrelated_holder_sees_nothing(self):
        transfers = _flatten([_single(ZERO_ADDRESS, ALICE, 7, 1000, 10)])
        assert replay_balances(transfers, BOB) == {}

    def test_holder_case_insensitive(self):
        transfers = _flatten([_single(ZERO_ADDRESS, ALICE.upper(), 7, 10, 1)])
        assert replay_balances(transfers, ALICE.lower()) == {7: 10}


class TestPositionIdentity:
    def test_condition_id_matches_ctf_formula(self):
        oracle, qid = "0x" + "11" * 20, "0x" + "22" * 32
        expected = "0x" + keccak(bytes.fromhex("11" * 20) + bytes.fromhex("22" * 32)
                                 + (2).to_bytes(32, "big")).hex()
        assert condition_id(oracle, qid) == expected

    def test_index_set_1_is_yes(self):
        """Polymarket reports payouts as [YES, NO], so slot 0 is YES."""
        assert position_side(1) == "YES"
        assert position_side(2) == "NO"


class TestSafeProxyAddress:
    @pytest.mark.parametrize("owner,proxy", [
        ("0x8e0be5dc9323b31017206dd736a532c6c23b3bd8", "0x4bcb70a23e54ef47a1a5e71b82cd20824952af88"),
        ("0x9650d2167634f00a902ae23b1a9632e104deb76d", "0x2b56992dc7fddf032f63f1f433ebd13468966498"),
    ])
    def test_reproduces_real_proxy_creation_logs(self, owner, proxy):
        """Anchors on two real ProxyCreation(proxy, owner) events from the Polymarket
        Safe factory. If the pinned initcode hash or the salt rule ever drifts, every
        voter's position silently reads as zero -- so this must fail loudly."""
        assert safe_proxy_address(owner) == proxy

    def test_case_insensitive(self):
        a = safe_proxy_address("0x8E0BE5DC9323B31017206DD736A532C6C23B3BD8".lower())
        b = safe_proxy_address("0x8e0be5dc9323b31017206dd736a532c6c23b3bd8")
        assert a == b

    def test_rejects_malformed_owner(self):
        with pytest.raises(ValueError, match="20-byte address"):
            safe_proxy_address("0xdeadbeef")

    def test_distinct_owners_distinct_proxies(self):
        assert safe_proxy_address("0x" + "11" * 20) != safe_proxy_address("0x" + "22" * 20)
