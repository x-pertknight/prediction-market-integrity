"""
scripts/fetch_positions_web3.py — Reconstruct each DVM voter's Polymarket
position on the market they voted on, AT THE BLOCK OF THE DISPUTE, from public
RPC only. No Dune, no API keys.

This is what makes the WSJ whale-conflict hypothesis testable. Three facts shape
the implementation:

  1. A DVM voter's mainnet EOA is not the address that holds their Polymarket
     position. Positions sit in a per-user Safe proxy on Polygon, deployed with
     CREATE2 under salt = keccak(abi.encode(owner)). The factory's
     ProxyCreation(proxy, owner) event indexes neither field, so owners cannot be
     filtered server-side. But the CREATE2 salt and initcode are fixed, so each
     voter's proxy is derived offline (verified to reproduce real ProxyCreation
     logs exactly) with no RPC call at all.

  2. `data-api.polymarket.com/positions` returns CURRENT positions only. For a
     dispute that settled months ago that is noise, not signal, so it is not used.

  3. The balance must be as of the dispute block. Two ways to get it:
     `balanceOfBatch` pinned to that block (needs archive state), or replaying the
     ConditionalTokens TransferSingle/TransferBatch log. Archive turns out to be
     served by tenderly and drpc, and it is far cheaper: one eth_call per dispute
     versus ~17k eth_getLogs for a holder-filtered scan (ERC-1155 does not index
     the token id, so logs can only be filtered by holder, and 1,619 of our 2,586
     voters have a deployed proxy). Archive is the default; `--mode replay` keeps
     the log path for endpoints that prune state.

     Archive is verified, never assumed: a pruned node answers 0 for every
     historical query, which would silently read as "no voter held anything" --
     the same class of silent-wrong that a null receipt caused upstream. The run
     asserts a known-nonzero control balance before trusting any result.

Both adapter families are covered. A CTF positionId needs a conditionId:

  * V2 adapter (0x6A9D...) is itself the CTF oracle, so
    conditionId = keccak(adapter ‖ questionID ‖ 2), collateral USDC.

  * V3 adapter (0x2F5e...) is deployed against the NegRiskOperator, whose
    `prepareCondition` is an explicit no-op. The real condition is prepared by the
    NegRiskAdapter under a DIFFERENT questionId. The two are linked on-chain by
    NegRiskOperator.questionIds[requestId], where requestId is the UMA adapter's
    own questionID (= keccak(ancillaryData)) -- that is what the oracle passes to
    `reportPayouts(_requestId, payouts)`. Then
    conditionId = keccak(NegRiskAdapter ‖ negRiskQuestionId ‖ 2), and the collateral
    is the adapter's WrappedCollateral, NOT USDC.

Every condition is positively controlled with getOutcomeSlotCount == 2 before use.
A wrong derivation returns a well-formed 32-byte id that simply has no holders, so
without that check every voter would silently read as holding nothing.

Usage:
    python -u scripts/fetch_positions_web3.py                  # archive (default)
    python -u scripts/fetch_positions_web3.py --mode replay    # log-replay fallback
    python -u scripts/fetch_positions_web3.py --max-disputes 5 # smoke test

Optional .env: POLYGON_RPC_URL (comma-separated). Public by default.
Inputs:  data/uma_votes.parquet, data/uma_dvm_requests.csv
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
    CONDITION_PREPARATION_TOPIC, CONDITIONAL_TOKENS, NEG_RISK_ADAPTER, NEG_RISK_OPERATOR,
    SEL_NEGRISK_POSITION_ID, SEL_NEGRISK_QUESTION_IDS, SEL_OUTCOME_SLOT_COUNT,
    TRANSFER_BATCH_TOPIC, TRANSFER_SINGLE_TOPIC,
    condition_id, decode_balance_of_batch, decode_transfer, encode_balance_of_batch,
    position_side, replay_balances, safe_proxy_address,
)
from data.rpc import JsonRpc, RpcError  # noqa: E402

DATA = ROOT / "data"
VOTES_PARQUET = DATA / "uma_votes.parquet"
REQUESTS_CSV = DATA / "uma_dvm_requests.csv"
POSITIONS_CSV = DATA / "uma_positions.csv"
CONDITIONS_CSV = DATA / "uma_conditions.csv"

ADAPTER_V2 = "0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74"
ADAPTER_V3_NEGRISK = "0x2F5e3684cb1F318ec51b00Edba38d79Ac2c0aA9d"
USDC_POLYGON = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_DEPLOY_BLOCK = 4_023_686

SEL_GET_COLLECTION_ID = "0x856296f7"    # getCollectionId(bytes32,bytes32,uint256)
SEL_GET_POSITION_ID = "0x39dd7530"      # getPositionId(address,bytes32)

# balanceOfBatch payloads get large; keep each eth_call to this many (owner, id) pairs.
BATCH_PAIRS = 400


def _hex32(x: int) -> str:
    return f"{x:064x}"


def _call(poly: JsonRpc, to: str, data: str, block: str = "latest") -> str:
    return poly.call("eth_call", [{"to": to, "data": data}, block])


def verify_archive_state(poly: JsonRpc) -> None:
    """Prove the endpoint returns real historical balances before we trust zeros.

    Finds a genuine CTF transfer in an old window and checks that balanceOf pinned
    to that block is nonzero for at least one recipient. A pruned node returns 0
    for every historical query, so without this a fully-zero result set would be
    indistinguishable from "no DVM voter ever held a position".
    """
    block = 73_000_000
    logs = poly.call("eth_getLogs", [{
        "address": CONDITIONAL_TOKENS, "fromBlock": hex(block), "toBlock": hex(block + 300),
        "topics": [TRANSFER_SINGLE_TOPIC]}])
    sel_bal = "0x00fdd58e"  # balanceOf(address,uint256)
    for log in logs:
        for t in decode_transfer(log):
            if t["value"] <= 0 or int(t["to"], 16) == 0:
                continue
            data = sel_bal + _hex32(int(t["to"], 16)) + _hex32(t["position_id"])
            got = int(_call(poly, CONDITIONAL_TOKENS, data, hex(t["block_number"])), 16)
            if got > 0:
                print(f"[positions] archive state confirmed: balanceOf({t['to'][:10]}..) "
                      f"@block {t['block_number']} = {got}")
                return
    raise SystemExit("[positions] endpoint does not serve archive state -- rerun with --mode replay")


def load_condition_cache() -> dict[str, tuple[str, dict[str, int]] | None]:
    """question_id -> (conditionId, {YES,NO}) or None for a question with no condition."""
    if not CONDITIONS_CSV.exists():
        return {}
    frame = pd.read_csv(CONDITIONS_CSV, dtype=str)
    out: dict[str, tuple[str, dict[str, int]] | None] = {}
    for _, r in frame.iterrows():
        if pd.isna(r["condition_id"]):
            out[r["question_id"]] = None
        else:
            out[r["question_id"]] = (r["condition_id"], {"YES": int(r["yes_id"]), "NO": int(r["no_id"])})
    return out


def save_condition_cache(ids: dict[str, tuple[str, dict[str, int]] | None]) -> None:
    rows = [{"question_id": q,
             "condition_id": v[0] if v else None,
             "yes_id": str(v[1]["YES"]) if v else None,
             "no_id": str(v[1]["NO"]) if v else None}
            for q, v in ids.items()]
    pd.DataFrame(rows).to_csv(CONDITIONS_CSV, index=False)


def outcome_slot_count(poly: JsonRpc, cond_id: str) -> int:
    return int(_call(poly, CONDITIONAL_TOKENS, SEL_OUTCOME_SLOT_COUNT + cond_id[2:]), 16)


def position_ids_for_condition(poly: JsonRpc, cond_id: str) -> dict[str, int]:
    """(YES, NO) CTF position ids for a V2 (plain USDC) condition.

    collectionId is an alt_bn128 curve point, so it is read from the contract
    rather than reimplemented."""
    out: dict[str, int] = {}
    for index_set in (1, 2):
        collection = _call(poly, CONDITIONAL_TOKENS,
                           SEL_GET_COLLECTION_ID + _hex32(0) + cond_id[2:] + _hex32(index_set))
        position = _call(poly, CONDITIONAL_TOKENS,
                         SEL_GET_POSITION_ID + _hex32(int(USDC_POLYGON, 16)) + collection[2:])
        out[position_side(index_set)] = int(position, 16)
    return out


def negrisk_ids(poly: JsonRpc, uma_question_id: str) -> tuple[str, dict[str, int]] | None:
    """Resolve a negRisk dispute to (conditionId, {YES, NO} positionIds), or None.

    The UMA adapter's questionID is the NegRiskOperator's `requestId`. The position
    ids come from NegRiskAdapter.getPositionId, which prices them in the adapter's
    WrappedCollateral rather than USDC -- deriving them against USDC would produce
    plausible ids that no one ever holds.
    """
    nr_qid = _call(poly, NEG_RISK_OPERATOR, SEL_NEGRISK_QUESTION_IDS + uma_question_id[2:])
    if int(nr_qid, 16) == 0:
        return None
    cond = condition_id(NEG_RISK_ADAPTER, nr_qid)
    # Positive control: a wrong derivation yields a well-formed but unprepared id.
    if outcome_slot_count(poly, cond) != 2:
        return None
    pids = {}
    for side, flag in (("YES", 1), ("NO", 0)):
        pids[side] = int(_call(poly, NEG_RISK_ADAPTER,
                               SEL_NEGRISK_POSITION_ID + nr_qid[2:] + _hex32(flag)), 16)
    return cond, pids


def condition_prepared_blocks(poly: JsonRpc, condition_ids: list[str], end: int,
                              chunk: int = 100) -> dict[str, int]:
    """Block at which each condition was prepared on the CTF (replay mode only)."""
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


def balances_at_block(poly: JsonRpc, pairs: list[tuple[str, int]], block: int,
                      batch: int = BATCH_PAIRS) -> list[int]:
    """balanceOfBatch(owners, ids) pinned to `block`, chunked to bound payload size.

    A big archive call can exceed the provider's compute budget and come back as a
    timeout. Halve the batch and retry rather than dropping the dispute -- a
    dropped dispute is a silently missing conflict, not a neutral one.
    """
    out: list[int] = []
    i = 0
    while i < len(pairs):
        window = pairs[i:i + batch]
        data = encode_balance_of_batch([o for o, _ in window], [pid for _, pid in window])
        try:
            result = _call(poly, CONDITIONAL_TOKENS, data, hex(block))
        except RpcError as exc:
            if batch > 25 and ("timeout" in str(exc).lower() or "exceed" in str(exc).lower()):
                batch //= 2
                continue
            raise
        out.extend(decode_balance_of_batch(result, len(window)))
        i += batch
    return out


def transfers_for_holders(poly: JsonRpc, holders: list[str], start: int, end: int,
                          keep_ids: set[int], chunk: int = 8) -> list[dict]:
    """Replay-mode fallback: CTF transfers touching `holders`, filtered to `keep_ids`.

    Holders are chunked small (providers cap topic OR-list size and time out on
    wide ones) and the block window adapts downward inside get_logs_chunked.
    """
    records: list[dict] = []
    for i in range(0, len(holders), chunk):
        topics = ["0x" + "00" * 12 + h[2:] for h in holders[i:i + chunk]]
        for topic0 in (TRANSFER_SINGLE_TOPIC, TRANSFER_BATCH_TOPIC):
            for position in (2, 3):  # `from` is topic2, `to` is topic3
                filt = [topic0, None, None, None]
                filt[position] = topics
                for log in poly.get_logs_chunked(address=CONDITIONAL_TOKENS, topics=filt,
                                                 start=start, end=end, window=500_000):
                    records.extend(t for t in decode_transfer(log) if t["position_id"] in keep_ids)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-disputes", type=int, default=0)
    parser.add_argument("--mode", choices=("archive", "replay"), default="archive")
    args = parser.parse_args()

    votes = pd.read_parquet(VOTES_PARQUET)
    requests_df = pd.read_csv(REQUESTS_CSV)
    meta = requests_df.set_index("dispute_tx_hash")
    votes = votes.assign(
        requester=votes["dispute_tx_hash"].map(meta["requester"]).str.lower(),
        question_id=votes["dispute_tx_hash"].map(meta["question_id"]),
    )

    poly = JsonRpc.polygon()
    t0 = time.time()

    groups = list(votes.groupby("dispute_tx_hash", observed=True))
    if args.max_disputes:
        groups = groups[:args.max_disputes]
    n_v2 = sum(1 for _, g in groups if g["requester"].iloc[0] == ADAPTER_V2.lower())
    print(f"[positions] {len(groups)} disputes in scope ({n_v2} V2, {len(groups) - n_v2} negRisk)")

    if args.mode == "archive":
        verify_archive_state(poly)

    voters = {v.lower() for _, g in groups for v in g["voter"].unique()}
    proxies = {v: safe_proxy_address(v) for v in voters}
    print(f"[positions] derived {len(proxies)} proxy addresses (offline CREATE2)")

    # Resolve (conditionId, positionIds) per question, deduped: a question disputed
    # more than once reuses the same condition. Cached to disk so a run interrupted
    # partway through ~3,700 eth_calls resumes instead of restarting.
    ids_by_question = load_condition_cache()
    todo = [(tx, g) for tx, g in groups if g["question_id"].iloc[0] not in ids_by_question]
    print(f"[positions] {len(ids_by_question)} conditions cached, {len(todo)} questions to resolve")
    for n, (tx, g) in enumerate(todo, 1):
        qid = g["question_id"].iloc[0]
        if qid in ids_by_question:
            continue
        if g["requester"].iloc[0] == ADAPTER_V2.lower():
            cond = condition_id(ADAPTER_V2, qid)
            ids_by_question[qid] = (cond, position_ids_for_condition(poly, cond)) \
                if outcome_slot_count(poly, cond) == 2 else None
        else:
            ids_by_question[qid] = negrisk_ids(poly, qid)
        if n % 100 == 0:
            save_condition_cache(ids_by_question)
            print(f"  [conditions {n}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    save_condition_cache(ids_by_question)
    resolved = sum(1 for v in ids_by_question.values() if v)
    print(f"[positions] resolved {resolved}/{len(ids_by_question)} questions to a CTF condition "
          f"({len(ids_by_question) - resolved} unresolved) ({time.time() - t0:.0f}s)")

    transfers_by_proxy: dict[str, list[dict]] = {}
    if args.mode == "replay":
        max_block = int(meta.loc[[tx for tx, _ in groups], "dispute_block"].max())
        conds = [c for v in ids_by_question.values() if v for c in (v[0],)]
        prepared = condition_prepared_blocks(poly, sorted(set(conds)), max_block)
        keep = {pid for v in ids_by_question.values() if v for pid in v[1].values()}
        holders = sorted(set(proxies.values()))
        print(f"[positions] replay scan: {len(holders)} proxies, blocks "
              f"{min(prepared.values()):,}..{max_block:,}")
        for t in transfers_for_holders(poly, holders, min(prepared.values()), max_block, keep):
            transfers_by_proxy.setdefault(t["to"], []).append(t)
            if t["from"] != t["to"]:
                transfers_by_proxy.setdefault(t["from"], []).append(t)

    rows: list[dict] = []
    for n, (dispute_tx, g) in enumerate(groups, 1):
        dispute_block = int(meta.loc[dispute_tx, "dispute_block"])
        qid = g["question_id"].iloc[0]
        resolved_ids = ids_by_question[qid]
        if resolved_ids is None:
            rows.append({"dispute_tx_hash": dispute_tx, "status": "condition_unresolved"})
            continue
        cond, pids = resolved_ids
        voters_here = sorted(g["voter"].str.lower().unique())

        if args.mode == "archive":
            pairs = [(proxies[v], pid) for v in voters_here for pid in (pids["YES"], pids["NO"])]
            try:
                bals = balances_at_block(poly, pairs, dispute_block)
            except RpcError as exc:
                print(f"[positions] balanceOfBatch failed for {dispute_tx[:12]}: {exc}")
                rows.append({"dispute_tx_hash": dispute_tx, "status": "balance_call_failed"})
                continue
            held = {v: (bals[2 * i], bals[2 * i + 1]) for i, v in enumerate(voters_here)}
        else:
            held = {}
            for v in voters_here:
                proxy = proxies[v]
                bal = replay_balances(transfers_by_proxy.get(proxy, []), proxy, up_to_block=dispute_block)
                held[v] = (bal.get(pids["YES"], 0), bal.get(pids["NO"], 0))

        n_pos = 0
        for voter, (yes, no) in held.items():
            if not yes and not no:
                continue
            n_pos += 1
            rows.append({
                "dispute_tx_hash": dispute_tx, "question_id": qid, "condition_id": cond,
                "voter": voter, "proxy": proxies[voter],
                # Shares inherit the collateral's decimals; WrappedCollateral mirrors USDC's 6.
                "yes_shares": yes / 1e6, "no_shares": no / 1e6,
                "dispute_block": dispute_block,
                "adapter": "v2" if g["requester"].iloc[0] == ADAPTER_V2.lower() else "negrisk",
                "status": "ok",
            })
        if n_pos == 0:
            rows.append({"dispute_tx_hash": dispute_tx, "status": "no_voter_position"})
        if n % 20 == 0 or n == len(groups):
            print(f"  [positions {n}/{len(groups)}] voters={len(voters_here)} positions={n_pos} "
                  f"{time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(POSITIONS_CSV, index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(POSITIONS_CSV, index=False)
    ok = frame[frame["status"] == "ok"]
    print(f"\n[positions] wrote {POSITIONS_CSV}: {len(ok)} voter positions across "
          f"{ok['dispute_tx_hash'].nunique() if len(ok) else 0} disputes")
    print(frame["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
