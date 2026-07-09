"""
src/data/ctf_positions.py — Reconstruct Polymarket CTF (ERC-1155) balances at a
historical block from transfer logs alone.

Why replay instead of `balanceOf`: an eth_call pinned to an old block needs an
archive node, which free/public RPCs do not serve. Event logs, by contrast, are
always retrievable. So a holder's balance at block B is derived by replaying
every TransferSingle/TransferBatch that touches them up to and including B.

ERC-1155 mints appear as transfers from the zero address and burns as transfers
to it, so split/merge/redeem on the ConditionalTokens contract are all captured
without special-casing.

Pure functions over raw log dicts. No network.
"""
from __future__ import annotations

from collections import defaultdict

from eth_abi import decode as abi_decode
from eth_utils import keccak

# Polygon deployments.
CONDITIONAL_TOKENS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
ZERO_ADDRESS = "0x" + "00" * 20

TRANSFER_SINGLE_TOPIC = "0x" + keccak(text="TransferSingle(address,address,address,uint256,uint256)").hex()
TRANSFER_BATCH_TOPIC = "0x" + keccak(text="TransferBatch(address,address,address,uint256[],uint256[])").hex()

# ConditionPreparation(bytes32 indexed conditionId, address indexed oracle,
#                      bytes32 indexed questionId, uint outcomeSlotCount)
CONDITION_PREPARATION_TOPIC = "0x" + keccak(
    text="ConditionPreparation(bytes32,address,bytes32,uint256)").hex()


def _topic_address(topic: str) -> str:
    return "0x" + topic[-40:].lower()


def _data_bytes(log: dict) -> bytes:
    data = log["data"]
    return bytes.fromhex(data[2:] if data.startswith("0x") else data)


def decode_transfer(log: dict) -> list[dict]:
    """Decode a TransferSingle or TransferBatch into one record per (id, value).

    A TransferBatch carries parallel id/value arrays; it is flattened so callers
    handle a single uniform shape.
    """
    topic0 = log["topics"][0].lower()
    frm = _topic_address(log["topics"][2])
    to = _topic_address(log["topics"][3])
    block = int(log["blockNumber"], 16)

    if topic0 == TRANSFER_SINGLE_TOPIC:
        token_id, value = abi_decode(["uint256", "uint256"], _data_bytes(log))
        pairs = [(token_id, value)]
    elif topic0 == TRANSFER_BATCH_TOPIC:
        ids, values = abi_decode(["uint256[]", "uint256[]"], _data_bytes(log))
        if len(ids) != len(values):
            raise ValueError("TransferBatch id/value length mismatch")
        pairs = list(zip(ids, values))
    else:
        raise ValueError(f"not an ERC-1155 transfer log: topic0={topic0}")

    return [{"from": frm, "to": to, "position_id": tid, "value": val, "block_number": block}
            for tid, val in pairs]


def replay_balances(transfers: list[dict], holder: str, up_to_block: int | None = None) -> dict[int, int]:
    """Net ERC-1155 balances for `holder` after replaying `transfers`.

    Credits transfers in, debits transfers out, ignores everything else. A
    self-transfer nets to zero. Only positions with a nonzero final balance are
    returned.
    """
    holder = holder.lower()
    balances: dict[int, int] = defaultdict(int)
    for t in transfers:
        if up_to_block is not None and t["block_number"] > up_to_block:
            continue
        if t["to"] == holder:
            balances[t["position_id"]] += t["value"]
        if t["from"] == holder:
            balances[t["position_id"]] -= t["value"]
    return {pid: bal for pid, bal in balances.items() if bal != 0}


def balance_at(transfers: list[dict], holder: str, position_id: int, up_to_block: int) -> int:
    return replay_balances(transfers, holder, up_to_block).get(position_id, 0)


# ── Position identity ────────────────────────────────────────────────────────
# conditionId = keccak256(abi.encodePacked(oracle, questionId, outcomeSlotCount))
# is plain hashing, but collectionId is an alt_bn128 curve point derived on-chain,
# so getCollectionId/getPositionId must be read from the contract via eth_call
# rather than reimplemented here.

def condition_id(oracle: str, question_id_hex: str, outcome_slot_count: int = 2) -> str:
    """CTF conditionId for a question. `oracle` is the adapter that prepared it."""
    oracle_bytes = bytes.fromhex(oracle[2:] if oracle.startswith("0x") else oracle)
    qid = question_id_hex[2:] if question_id_hex.startswith("0x") else question_id_hex
    packed = oracle_bytes + bytes.fromhex(qid) + outcome_slot_count.to_bytes(32, "big")
    return "0x" + keccak(packed).hex()


def position_side(index_set: int) -> str:
    """Polymarket reports payouts as [YES, NO], so slot 0 (indexSet 1) is YES."""
    return {1: "YES", 2: "NO"}.get(index_set, f"indexSet:{index_set}")


# ── Polymarket Safe proxy ────────────────────────────────────────────────────
# The factory deploys with CREATE2 under salt = keccak(abi.encode(owner)) and
# initcode = proxyCreationCode ++ abi.encode(masterCopy). Both the factory and
# its masterCopy are immutable, so the initcode hash below is a constant, read
# once from `getContractBytecode()` and pinned. Derivation is exact and offline:
# it avoids one eth_call per voter, and is defined even for an owner who never
# deployed a proxy (their balance is then simply zero).
SAFE_PROXY_FACTORY = "0xaacFeEa03eb1561C4e67d661e40682Bd20E3541b"
SAFE_PROXY_INITCODE_HASH = bytes.fromhex(
    "2bce2127ff07fb632d16c8347c4ebf501f4841168bed00d9e6ef715ddb6fcecf")


def safe_proxy_address(owner: str, factory: str = SAFE_PROXY_FACTORY,
                       initcode_hash: bytes = SAFE_PROXY_INITCODE_HASH) -> str:
    """The Polymarket proxy wallet a given EOA owns, whether or not it exists yet."""
    addr = owner[2:] if owner.startswith("0x") else owner
    if len(addr) != 40:
        raise ValueError(f"owner must be a 20-byte address, got {owner!r}")
    salt = keccak(bytes(12) + bytes.fromhex(addr))          # keccak(abi.encode(owner))
    fac = bytes.fromhex(factory[2:] if factory.startswith("0x") else factory)
    return "0x" + keccak(b"\xff" + fac + salt + initcode_hash)[12:].hex()
