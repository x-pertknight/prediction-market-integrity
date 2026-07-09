"""
scripts/fetch_votes_web3.py — Link Polymarket OO disputes to the mainnet DVM
votes cast on them, using only public RPC. No Dune, no API keys.

This is the "join pulito". Two facts make the naive joins fail:

  1. A Polygon dispute reaches the Ethereum DVM with its ancillary data rewritten
     twice: OptimisticOracleV2 appends ",ooRequester:<adapter>", then
     OracleChildTunnel COMPRESSES the result, so the DVM only ever sees
     "ancillaryDataHash:<keccak of the stamped bytes>,...,childChainId:137".
     Matching on the raw text finds nothing (the earlier attempt scored 0/973).

  2. Polymarket's requests are event-based, so on dispute OOv2 escalates under the
     PROPOSAL time, not DisputePrice.timestamp. Those differ by days, so keying on
     the OO request_timestamp also matches nothing.

Both the hash and the true DVM time are emitted by OracleChildTunnel inside the
dispute's own Polygon transaction, so stage 1 reads them from chain rather than
guessing. Stage 2 then scans mainnet VoteRevealed and keeps the reveals whose
(ancillaryDataHash, time) hits that set -- an exact join, correct even for a
question that was disputed, reset, and disputed again under identical ancillary
data.

Usage:
    python scripts/fetch_votes_web3.py                     # full run
    python scripts/fetch_votes_web3.py --max-disputes 25   # smoke test
    python scripts/fetch_votes_web3.py --requests-only     # stage 1 only

Optional .env: ETH_RPC_URL, POLYGON_RPC_URL (comma-separated). Public by default.
Outputs: data/uma_dvm_requests.csv (stage 1 cache), data/uma_votes.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from alpha.uma_dispute import question_id, stamped_ancillary_hash, vote_side  # noqa: E402
from data.oo_bridge_logs import dvm_request_from_receipt  # noqa: E402
from data.rpc import JsonRpc, RpcError  # noqa: E402
from data.uma_dvm_logs import (  # noqa: E402
    VOTE_REVEALED_TOPIC, VOTING_V2, YES_OR_NO_IDENTIFIER,
    decode_vote_revealed, extract_ancillary_hash,
)

DATA = ROOT / "data"
DISPUTES_CSV = DATA / "uma_disputes.csv"
REQUESTS_CSV = DATA / "uma_dvm_requests.csv"
VOTES_CSV = DATA / "uma_votes.csv"
# ~460k votes is ~276 MB as CSV but ~5 MB as zstd parquet, so the parquet is the
# artifact that lives in git; the CSV stays as a local working file.
VOTES_PARQUET = DATA / "uma_votes.parquet"

SEARCH_LO = 21_000_000  # VotingV2 predates the post-MOOV2 window; bound the search

REQUEST_FIELDS = [
    "dispute_tx_hash", "dispute_block", "question_id", "stamped_hash", "onchain_ancillary_hash",
    "dvm_request_time", "request_timestamp", "dispute_time", "settle_time",
    "settled_price", "proposedPrice", "requester",
]
VOTE_FIELDS = [
    "dispute_tx_hash", "question_id", "stamped_hash", "dvm_request_time",
    "dispute_time", "settled_price", "proposedPrice", "requester",
    "voter", "caller", "side", "vote_price", "stake", "round_id",
    "vote_tx_hash", "block_number",
]


def stage1_dvm_requests(disputes: pd.DataFrame, max_disputes: int = 0) -> pd.DataFrame:
    """Read each dispute's forwarded DVM price request off its Polygon receipt.

    Incremental: only disputes absent from the cache are fetched, so a run
    interrupted partway through 973 receipts resumes where it stopped.
    """
    subset = disputes.head(max_disputes) if max_disputes else disputes
    rows: list[dict] = []
    if REQUESTS_CSV.exists():
        cached = pd.read_csv(REQUESTS_CSV, dtype={"settled_price": str, "proposedPrice": str})
        rows = cached.to_dict("records")
    have = {r["dispute_tx_hash"] for r in rows}
    todo = subset[~subset["dispute_tx_hash"].isin(have)]
    print(f"[stage1] {len(have)} cached, {len(todo)} receipts to fetch")

    poly = JsonRpc.polygon()
    missing, failed, t0 = 0, 0, time.time()
    for i, (_, d) in enumerate(todo.iterrows(), 1):
        try:
            # call_nonnull: a pruned node answers with null rather than an error,
            # which would masquerade as "this dispute never escalated".
            receipt = poly.call_nonnull("eth_getTransactionReceipt", [d["dispute_tx_hash"]])
        except RpcError as exc:
            print(f"[stage1] receipt unavailable {d['dispute_tx_hash'][:12]}: {exc}")
            failed += 1
            continue
        req = dvm_request_from_receipt(receipt)
        if req is None:
            missing += 1
            continue

        anc = bytes.fromhex(d["ancillaryData"][2:])
        rows.append({
            "dispute_tx_hash": d["dispute_tx_hash"],
            "dispute_block": int(receipt["blockNumber"], 16),
            "question_id": "0x" + question_id(anc),
            "stamped_hash": stamped_ancillary_hash(anc, d["requester"]),
            "onchain_ancillary_hash": extract_ancillary_hash(req["ancillary_data"]),
            "dvm_request_time": req["dvm_request_time"],
            "request_timestamp": int(d["request_timestamp"]),
            "dispute_time": d["dispute_time"],
            "settle_time": d["settle_time"],
            "settled_price": d["settled_price"],
            "proposedPrice": d["proposedPrice"],
            "requester": d["requester"],
        })
        if i % 50 == 0:
            pd.DataFrame(rows, columns=REQUEST_FIELDS).to_csv(REQUESTS_CSV, index=False)
            print(f"  [stage1] {i}/{len(todo)} receipts  missing={missing}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    frame = pd.DataFrame(rows, columns=REQUEST_FIELDS)
    frame.to_csv(REQUESTS_CSV, index=False)

    # The locally computed hash must equal what the tunnel actually emitted.
    # A mismatch means the stamping rule changed and the join is unsafe.
    both = frame.dropna(subset=["onchain_ancillary_hash"])
    agree = (both["stamped_hash"] == both["onchain_ancillary_hash"]).sum()
    print(f"[stage1] {len(frame)} DVM requests, {missing} disputes never escalated, "
          f"{failed} receipts unavailable")
    print(f"[stage1] locally-derived hash agrees with on-chain: {agree}/{len(both)}")
    if len(both) and agree != len(both):
        raise SystemExit("[stage1] ancillaryDataHash mismatch -- refusing to build a wrong join")
    lag = (frame["dvm_request_time"] - frame["request_timestamp"])
    print(f"[stage1] dvm_request_time - request_timestamp: median {lag.median() / 3600:.1f}h, "
          f"zero for {int((lag == 0).sum())}/{len(frame)}")
    return frame


def stage2_votes(requests: pd.DataFrame, window: int, limit_blocks: int) -> list[dict]:
    """Scan mainnet VoteRevealed, keeping reveals that hit a known DVM request."""
    index = {(r["stamped_hash"], int(r["dvm_request_time"])): r for _, r in requests.iterrows()}
    print(f"[stage2] {len(index)} distinct (stamped_hash, dvm_request_time) keys")

    eth = JsonRpc.ethereum()
    latest = eth.block_number()
    d_min = pd.to_datetime(requests["dispute_time"], format="mixed").min()
    s_max = pd.to_datetime(requests["settle_time"], format="mixed").max()

    # Reveals land between the dispute and the settle; pad both ends for rolled rounds.
    start = eth.block_at_timestamp(int(d_min.timestamp()), SEARCH_LO, latest)
    end = min(eth.block_at_timestamp(int(s_max.timestamp()), start, latest) + 50_000, latest)
    if limit_blocks:
        end = min(end, start + limit_blocks)
    print(f"[stage2] scanning mainnet {start:,}..{end:,} ({end - start:,} blocks) via {eth.url}")

    matched: list[dict] = []
    seen = 0
    t0 = time.time()

    def progress(block: int, _n: int) -> None:
        pct = 100 * (block - start) / max(end - start, 1)
        print(f"  [stage2 {pct:5.1f}%] block {block:,}  reveals={seen:,}  "
              f"matched={len(matched):,}  {time.time() - t0:5.0f}s", flush=True)

    for log in eth.get_logs_chunked(
        address=VOTING_V2,
        topics=[VOTE_REVEALED_TOPIC, None, None, YES_OR_NO_IDENTIFIER],
        start=start, end=end, window=window, progress=progress,
    ):
        seen += 1
        rec = decode_vote_revealed(log)
        if rec["stamped_hash"] is None:
            continue
        row = index.get((rec["stamped_hash"], rec["request_timestamp"]))
        if row is None:
            continue
        matched.append({
            "dispute_tx_hash": row["dispute_tx_hash"],
            "question_id": row["question_id"],
            "stamped_hash": rec["stamped_hash"],
            "dvm_request_time": rec["request_timestamp"],
            "dispute_time": row["dispute_time"],
            "settled_price": row["settled_price"],
            "proposedPrice": row["proposedPrice"],
            "requester": row["requester"],
            "voter": rec["voter"],
            "caller": rec["caller"],
            "side": vote_side(rec["vote_price"]),
            "vote_price": rec["vote_price"],
            "stake": rec["stake"],
            "round_id": rec["round_id"],
            "vote_tx_hash": rec["vote_tx_hash"],
            "block_number": rec["block_number"],
        })

    print(f"\n[stage2] scanned {seen:,} YES_OR_NO reveals in {time.time() - t0:.0f}s")
    return matched


def write_votes_parquet(matched: list[dict]) -> None:
    """Compact, git-friendly copy of the votes.

    Per-dispute attributes (settled_price, requester, ...) are dropped: they are
    one join away in uma_dvm_requests.csv and repeating them 460k times is what
    makes the CSV 50x larger. vote_price stays a string because the ignore-price
    sentinel is int256.min and overflows int64.
    """
    keep = ["dispute_tx_hash", "stamped_hash", "dvm_request_time", "voter", "caller",
            "side", "vote_price", "stake", "round_id", "vote_tx_hash", "block_number"]
    frame = pd.DataFrame(matched, columns=VOTE_FIELDS)[keep]
    frame["vote_price"] = frame["vote_price"].astype(str)
    for col in ("dispute_tx_hash", "stamped_hash", "voter", "caller", "side", "vote_tx_hash"):
        frame[col] = frame[col].astype("category")
    frame.to_parquet(VOTES_PARQUET, compression="zstd", index=False)
    print(f"[votes] wrote {VOTES_PARQUET}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-disputes", type=int, default=0, help="cap stage-1 disputes (smoke test)")
    parser.add_argument("--limit-blocks", type=int, default=0, help="cap stage-2 block span (smoke test)")
    parser.add_argument("--window", type=int, default=20_000, help="initial getLogs block window")
    parser.add_argument("--requests-only", action="store_true", help="run stage 1 and stop")
    args = parser.parse_args()

    disputes = pd.read_csv(DISPUTES_CSV, dtype={"settled_price": str, "proposedPrice": str})
    print(f"[votes] {len(disputes)} disputes, {disputes['ancillaryData'].nunique()} distinct questions")

    requests = stage1_dvm_requests(disputes, args.max_disputes)
    if args.requests_only or requests.empty:
        return

    matched = stage2_votes(requests, args.window, args.limit_blocks)

    VOTES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(VOTES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VOTE_FIELDS)
        writer.writeheader()
        writer.writerows(matched)
    write_votes_parquet(matched)

    n_disputes = len({m["dispute_tx_hash"] for m in matched})
    n_voters = len({m["voter"] for m in matched})
    print(f"[votes] matched {len(matched):,} votes across {n_disputes}/{len(requests)} DVM requests, "
          f"{n_voters} distinct voters")
    print(f"[votes] wrote {VOTES_CSV}")
    if not matched:
        print("[votes] 0 matches -- this is a broken linkage, NOT evidence against the hypothesis.")


if __name__ == "__main__":
    main()
