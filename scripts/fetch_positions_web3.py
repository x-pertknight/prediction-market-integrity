"""
scripts/fetch_positions_web3.py — Reconstruct each DVM voter's Polymarket
position on the market they voted on, AT THE BLOCK OF THE DISPUTE, from public
RPC only.

This is what makes the WSJ whale-conflict hypothesis testable. Three facts shape
the implementation:

  1. A DVM voter's mainnet EOA is not the address that holds their Polymarket
     position. Positions sit in a per-user Safe proxy on Polygon, deployed with
     CREATE2 under salt = keccak(abi.encode(owner)). The factory's
     ProxyCreation(proxy, owner) event indexes neither field, so owners cannot be
     filtered server-side. But the factory's CREATE2 salt and initcode are fixed,
     so each voter's proxy is derived offline (verified to reproduce real
     ProxyCreation logs exactly) with no RPC call at all.

  2. `data-api.polymarket.com/positions` returns CURRENT positions only. For a
     dispute that settled months ago that is noise, not signal, so it is not used.

  3. An `eth_call` of `balanceOf` pinned to a historical block needs an archive
     node, which free/public RPCs do not serve. Event logs, however, are always
     retrievable, so a holder's balance at the dispute block is replayed from
     the ConditionalTokens TransferSingle/TransferBatch history.

Scope limit, and it is a real one: the CTF position id for a question requires a
conditionId. For the V2 adapter (0x6A9D...) the adapter itself is the CTF oracle,
so conditionId = keccak(adapter, questionID, 2) and the position ids follow from
eth_call to getCollectionId/getPositionId. The V3 adapter (0x2F5e...) was deployed
against the NegRiskOperator, so its conditions are prepared by the NegRiskAdapter
under a different questionId and the derivation above does not hold. Those
disputes are reported as `negrisk_unsupported` rather than silently mislinked.

The full run is ~7,000 eth_getLogs over ~26M Polygon blocks (roughly two hours
on public endpoints). Run it with -u, or a redirected stdout block-buffers and
the progress is indistinguishable from a hang.

Usage:
    python -u scripts/fetch_positions_web3.py                  # all linkable voters
    python -u scripts/fetch_positions_web3.py --max-disputes 5 # smoke test

Optional .env: POLYGON_RPC_URL (comma-separated). Public by default.
Inputs:  data/uma_votes.csv, data/uma_dvm_requests.csv
Outputs: data/uma_positions.csv
"""
from __future__ import annotations

import argparse
import sys
import time

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.ctf_positions import (  # noqa: E402
    CONDITION_PREPARATION_TOPIC, CONDITIONAL_TOKENS, TRANSFER_BATCH_TOPIC, TRANSFER_SINGLE_TOPIC,
    condition_id, decode_transfer, position_side, replay_balances, safe_proxy_address,
)
from data.rpc import JsonRpc, RpcError  # noqa: E402

DATA = ROOT / "data"
VOTES_CSV = DATA / "uma_votes.csv"
REQUESTS_CSV = DATA / "uma_dvm_requests.csv"
POSITIONS_CSV = DATA / "uma_positions.csv"

# Adapters. Only the V2 adapter prepares conditions directly on the CTF.
ADAPTER_V2 = "0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74"
ADAPTER_V3_NEGRISK = "0x2F5e3684cb1F318ec51b00Edba38d79Ac2c0aA9d"

USDC_POLYGON = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

CTF_DEPLOY_BLOCK = 4_023_686  # ConditionalTokens on Polygon

# eth_call selectors (keccak of the canonical signature).
SEL_GET_COLLECTION_ID = "0x856296f7"    # getCollectionId(bytes32,bytes32,uint256)
SEL_GET_POSITION_ID = "0x39dd7530"      # getPositionId(address,bytes32)


def _hex32(x: int) -> str:
    return f"{x:064x}"


def _call(poly: JsonRpc, to: str, data: str) -> str:
    return poly.call("eth_call", [{"to": to, "data": data}, "latest"])


def position_ids_for_condition(poly: JsonRpc, cond_id: str) -> dict[str, int]:
    """(YES, NO) CTF position ids. collectionId is an alt_bn128 point, so it is
    read from the contract rather than reimplemented."""
    out: dict[str, int] = {}
    for index_set in (1, 2):
        collection = _call(poly, CONDITIONAL_TOKENS,
                           SEL_GET_COLLECTION_ID + _hex32(0) + cond_id[2:] + _hex32(index_set))
        position = _call(poly, CONDITIONAL_TOKENS,
                         SEL_GET_POSITION_ID + _hex32(int(USDC_POLYGON, 16)) + collection[2:])
        out[position_side(index_set)] = int(position, 16)
    return out


def resolve_proxies(voters: set[str]) -> dict[str, str]:
    """Map each voter EOA to its deterministic Polymarket Safe proxy address."""
    return {v: safe_proxy_address(v) for v in sorted(voters)}


def condition_prepared_blocks(poly: JsonRpc, condition_ids: list[str], end: int,
                              chunk: int = 100) -> dict[str, int]:
    """Block at which each condition was prepared on the CTF.

    A position in a market cannot predate its condition, so this bounds the
    transfer replay to [prepared, dispute] instead of scanning from the CTF's
    2020 deployment for every dispute. conditionId is indexed, so the set is
    filtered server-side; it is chunked because providers cap OR-list size.
    """
    out: dict[str, int] = {}
    ids = [c.lower() for c in condition_ids]
    for i in range(0, len(ids), chunk):
        for log in poly.get_logs_chunked(
            address=CONDITIONAL_TOKENS,
            topics=[CONDITION_PREPARATION_TOPIC, ids[i:i + chunk]],
            start=CTF_DEPLOY_BLOCK, end=end, window=2_000_000,
        ):
            out[log["topics"][1].lower()] = int(log["blockNumber"], 16)
    return out


def transfers_for_holders(poly: JsonRpc, holders: list[str], start: int, end: int,
                          keep_ids: set[int], chunk: int = 40, progress=None) -> list[dict]:
    """CTF ERC-1155 transfers touching `holders` in [start, end], for `keep_ids` only.

    ERC-1155 does not index the token id, so transfers cannot be filtered by
    position server-side; they are filtered by holder (from/to are indexed) and
    the id is matched after decoding. Holders are chunked because providers cap
    the size of a topic OR-list.

    Scanned once for the whole voter set rather than per dispute: the same DVM
    voters recur across disputes, so per-dispute scans would refetch the same
    logs hundreds of times.
    """
    records: list[dict] = []
    for i in range(0, len(holders), chunk):
        topics = ["0x" + "00" * 12 + h[2:] for h in holders[i:i + chunk]]
        for topic0 in (TRANSFER_SINGLE_TOPIC, TRANSFER_BATCH_TOPIC):
            for position in (2, 3):  # `from` is topic2, `to` is topic3
                filt = [topic0, None, None, None]
                filt[position] = topics
                for log in poly.get_logs_chunked(address=CONDITIONAL_TOKENS, topics=filt,
                                                 start=start, end=end, window=1_000_000):
                    records.extend(t for t in decode_transfer(log) if t["position_id"] in keep_ids)
        if progress:
            progress(min(i + chunk, len(holders)), len(records))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-disputes", type=int, default=0)
    args = parser.parse_args()

    if not VOTES_CSV.exists():
        raise SystemExit(f"{VOTES_CSV} missing -- run scripts/fetch_votes_web3.py first")
    votes = pd.read_csv(VOTES_CSV)
    requests_df = pd.read_csv(REQUESTS_CSV)
    if votes.empty:
        raise SystemExit("uma_votes.csv is empty -- the dispute<->vote join produced nothing")

    blocks = requests_df.set_index("dispute_tx_hash")["dispute_block"].to_dict()
    poly = JsonRpc.polygon()
    t0 = time.time()

    # Only the V2 adapter's conditions are derivable as keccak(adapter, questionID, 2).
    # V3 rides the NegRiskOperator, whose questionId differs; those are reported,
    # not guessed at.
    n_negrisk = int((votes["requester"].str.lower() == ADAPTER_V3_NEGRISK.lower()).sum())
    v2 = votes[votes["requester"].str.lower() == ADAPTER_V2.lower()]
    print(f"[positions] {n_negrisk:,} votes on negRisk (V3) disputes -> unsupported conditionId")
    if v2.empty:
        pd.DataFrame([{"status": "negrisk_unsupported"}]).to_csv(POSITIONS_CSV, index=False)
        raise SystemExit("[positions] no V2-adapter disputes to work with")

    groups = list(v2.groupby("dispute_tx_hash"))
    if args.max_disputes:
        groups = groups[:args.max_disputes]
    print(f"[positions] {len(groups)} V2 disputes in scope")

    voters = {v.lower() for _, g in groups for v in g["voter"].unique()}
    proxies = resolve_proxies(voters)
    print(f"[positions] derived {len(proxies)} proxy addresses (offline CREATE2)")

    conds = {dispute_tx: condition_id(ADAPTER_V2, g["question_id"].iloc[0]) for dispute_tx, g in groups}
    max_block = max(int(blocks[d]) for d, _ in groups)
    prepared = condition_prepared_blocks(poly, sorted(set(conds.values())), max_block)
    print(f"[positions] {len(prepared)}/{len(set(conds.values()))} conditions located on the CTF "
          f"({time.time() - t0:.0f}s)")

    pids_by_cond: dict[str, dict[str, int]] = {}
    for cond in sorted(set(conds.values())):
        if cond.lower() in prepared:
            pids_by_cond[cond] = position_ids_for_condition(poly, cond)
    keep_ids = {pid for pids in pids_by_cond.values() for pid in pids.values()}
    print(f"[positions] {len(keep_ids)} position ids of interest ({time.time() - t0:.0f}s)")

    scan_start = min(prepared.values())
    holders = sorted(set(proxies.values()))
    owner_of = {p: v for v, p in proxies.items()}
    print(f"[positions] scanning CTF transfers for {len(holders)} proxies over "
          f"blocks {scan_start:,}..{max_block:,}")

    def progress(done: int, kept: int) -> None:
        print(f"  [transfers] {done}/{len(holders)} proxies  kept={kept:,}  "
              f"{time.time() - t0:.0f}s", flush=True)

    transfers = transfers_for_holders(poly, holders, scan_start, max_block, keep_ids,
                                      progress=progress)
    print(f"[positions] {len(transfers):,} relevant transfers ({time.time() - t0:.0f}s)")

    by_proxy: dict[str, list[dict]] = {}
    for t in transfers:
        by_proxy.setdefault(t["to"], []).append(t)
        if t["from"] != t["to"]:
            by_proxy.setdefault(t["from"], []).append(t)

    rows: list[dict] = []
    for dispute_tx, g in groups:
        dispute_block = int(blocks[dispute_tx])
        qid, cond = g["question_id"].iloc[0], conds[dispute_tx]
        if cond not in pids_by_cond:
            rows.append({"dispute_tx_hash": dispute_tx, "status": "condition_not_found"})
            continue
        pids = pids_by_cond[cond]
        n_pos = 0
        for voter in g["voter"].str.lower().unique():
            proxy = proxies[voter]
            if proxy not in by_proxy:
                continue
            bal = replay_balances(by_proxy[proxy], proxy, up_to_block=dispute_block)
            yes, no = bal.get(pids["YES"], 0), bal.get(pids["NO"], 0)
            if not yes and not no:
                continue
            n_pos += 1
            rows.append({
                "dispute_tx_hash": dispute_tx, "question_id": qid, "condition_id": cond,
                "voter": voter, "proxy": proxy,
                # CTF shares inherit the collateral's decimals (USDC: 6).
                "yes_shares": yes / 1e6, "no_shares": no / 1e6,
                "dispute_block": dispute_block, "status": "ok",
            })
        if n_pos == 0:
            rows.append({"dispute_tx_hash": dispute_tx, "status": "no_voter_position"})

    frame = pd.DataFrame(rows)
    frame.to_csv(POSITIONS_CSV, index=False)
    ok = frame[frame["status"] == "ok"] if not frame.empty else frame
    print(f"\n[positions] wrote {POSITIONS_CSV} ({len(frame)} rows, {len(ok)} voter positions "
          f"across {ok['dispute_tx_hash'].nunique() if len(ok) else 0} disputes)")
    print(frame["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
