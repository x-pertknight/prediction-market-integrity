"""
scripts/backfill_uma_disputes.py — Historical backfill for the UMA dispute
whale-conflict hypothesis test (see
docs/experiments/2026-07-uma-dispute-preregistration.md).

Standalone script. Uses the Dune API (api.dune.com) via DUNE_API_KEY in
.env — no interactive session required to re-run this later.

Scope, post-2025-04-01 (post-MOOV2/UMIP-189 regime only):

1. Base-rate metric ("does the DVM confirm the original proposal") — FULLY
   WORKING. Polygon-only join between uma_polygon.optimisticoraclev2_evt_disputeprice
   and uma_polygon.optimisticoraclev2_evt_settle for the two known Polymarket
   requester/adapter addresses. -> data/uma_disputes.csv

2. Voter-level conflict analysis (the actual WSJ hypothesis: does a DVM
   voter with a Polymarket position on the disputed market vote their own
   book) — BLOCKED. Investigated live on 2026-07-08: three join strategies
   between a Polygon dispute and its Ethereum-mainnet DVM VoteRevealed
   event were tested against real, fully-resolved disputes:
     a. direct (identifier, timestamp) equality -> 0/25 matched
     b. same-tx-hash correlation via uma_polygon.oraclechildtunnel_evt_pricerequestadded
        (the Polygon<->mainnet bridge) -> 0/25 matched
     c. content-substring match on the decoded `market_id:` field in
        ancillaryData, tested against the FULL post-2025-04-01 resolved
        Polymarket dispute set (973 disputes) -> 0/973 matched
   This is a structural finding, not noise: post-MOOV2 Polymarket disputes
   do not appear to resolve via classic identifier-keyed VotingV2
   token-holder voting in the way the public UMA docs describe, or they do
   via a mechanism this script does not yet know how to trace (a different
   contract/chain, a non-identifier-keyed request, or an off-chain/committee
   step introduced alongside MOOV2's proposer whitelist). Cracking this
   needs either UMA/Polymarket-side documentation of the exact post-MOOV2
   escalation path, or hand-tracing one publicly documented dispute
   end-to-end.
   -> data/uma_votes.csv is written with a header only (or whatever the
      linkage query finds, if the mechanism is later identified and this
      script's query is updated) — never fabricated.
   -> data/uma_positions.csv is derived FROM uma_votes.csv, so it will
      likewise be empty until the linkage above is solved.

   NOTE on a separate, already-resolved sub-question (2026-07-08): a DVM
   voter's mainnet EOA is not the same address as their Polymarket Polygon
   position holder — Polymarket positions are held by a per-user Safe
   proxy wallet. This IS bridgeable without inventing anything: Polymarket's
   `polymarket_polygon.safeproxyfactory_evt_proxycreation` decoded table
   emits `ProxyCreation(owner, proxy)` on Polygon. Measured match rate:
   2986/4888 (61.1%) of distinct DVM voters since 2025-07-01 resolve
   directly to an owned proxy this way. So the EOA-to-proxy bridge is NOT
   the blocker — the still-unresolved dispute-to-vote linkage above is.
   Once that's solved, join votes.voter -> proxycreation.owner -> proxy,
   then query positions for `proxy` (not the raw voter EOA).

Both Dune queries below were created and validated via Dune's query
builder (query_id pinned below); this script re-executes them by ID so it
never needs to embed ad-hoc SQL through the API, which the base Dune API
tier does not support.

Usage:
    python scripts/backfill_uma_disputes.py
Requires DUNE_API_KEY in .env.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DUNE_API_BASE = "https://api.dune.com/api/v1"
POLYMARKET_DATA_API = "https://data-api.polymarket.com"

# Pinned Dune query IDs (created 2026-07-08, see module docstring for SQL intent).
DISPUTES_QUERY_ID = 7917876   # traptrade_uma_polymarket_disputes_backfill
VOTES_QUERY_ID = 7917907      # traptrade_uma_votes_besteffort_linkage (known: 0 rows today)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DISPUTES_CSV = DATA_DIR / "uma_disputes.csv"
VOTES_CSV = DATA_DIR / "uma_votes.csv"
POSITIONS_CSV = DATA_DIR / "uma_positions.csv"

POLL_INTERVAL_S = 3
POLL_TIMEOUT_S = 420

# UMA's "ignore/void this price request" sentinel (== -2**255, int256 min).
# Discovered 2026-07-08 while sanity-checking the backfill: 67.4% of
# "settled" Polymarket disputes settle to this value, meaning the market
# was voided (e.g. cancelled/ambiguous event), NOT that the DVM ruled
# against the proposer. Counting these as "proposal rejected" would badly
# understate the true confirm-rate. They are excluded from the base-rate
# calc and reported as a separate VOIDED bucket.
UMA_IGNORE_PRICE = -(2 ** 255)


def _dune_headers(api_key: str) -> dict:
    return {"X-Dune-API-Key": api_key}


def dune_execute_and_fetch(query_id: int, api_key: str) -> list[dict]:
    """Execute a pinned Dune query by ID and return all result rows.

    Re-executes rather than reading a cached result, so the backfill always
    reflects on-chain state at run time.
    """
    resp = requests.post(
        f"{DUNE_API_BASE}/query/{query_id}/execute",
        headers=_dune_headers(api_key),
        json={"performance": "medium"},
        timeout=30,
    )
    resp.raise_for_status()
    execution_id = resp.json()["execution_id"]

    deadline = time.monotonic() + POLL_TIMEOUT_S
    state = "QUERY_STATE_PENDING"
    while state not in ("QUERY_STATE_COMPLETED", "QUERY_STATE_FAILED"):
        if time.monotonic() > deadline:
            raise TimeoutError(f"Dune execution {execution_id} did not finish in {POLL_TIMEOUT_S}s")
        time.sleep(POLL_INTERVAL_S)
        status_resp = requests.get(
            f"{DUNE_API_BASE}/execution/{execution_id}/status",
            headers=_dune_headers(api_key),
            timeout=30,
        )
        status_resp.raise_for_status()
        state = status_resp.json()["state"]

    if state == "QUERY_STATE_FAILED":
        raise RuntimeError(f"Dune execution {execution_id} failed")

    rows: list[dict] = []
    offset = 0
    limit = 5000
    while True:
        results_resp = requests.get(
            f"{DUNE_API_BASE}/execution/{execution_id}/results",
            headers=_dune_headers(api_key),
            params={"limit": limit, "offset": offset},
            timeout=60,
        )
        results_resp.raise_for_status()
        batch = results_resp.json().get("result", {}).get("rows", [])
        rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Still write a stable header so downstream code has a known schema.
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["_empty"])
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_polymarket_positions(voters: list[str]) -> list[dict]:
    """Fetch current Polymarket positions for each voter wallet.

    data-api.polymarket.com is public, no auth required (verified 2026-07-08,
    HTTP 200 on /positions?user=0x...). Returns CURRENT positions only —
    not a historical snapshot at dispute time. Best used as a live
    supplementary check once uma_votes.csv has real linked voters.
    """
    out: list[dict] = []
    for voter in voters:
        try:
            resp = requests.get(
                f"{POLYMARKET_DATA_API}/positions",
                params={"user": voter, "limit": 500},
                timeout=15,
            )
            resp.raise_for_status()
            for pos in resp.json():
                pos["_voter"] = voter
                out.append(pos)
        except requests.RequestException as e:
            print(f"[backfill_uma_disputes] position fetch failed for {voter}: {e}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    api_key = os.environ.get("DUNE_API_KEY")
    if not api_key:
        print("DUNE_API_KEY not set in .env — cannot run the backfill.", file=sys.stderr)
        sys.exit(1)

    print("[backfill_uma_disputes] fetching dispute+settle base-rate dataset from Dune...")
    disputes = dune_execute_and_fetch(DISPUTES_QUERY_ID, api_key)
    write_csv(disputes, DISPUTES_CSV)
    print(f"[backfill_uma_disputes] wrote {len(disputes)} rows -> {DISPUTES_CSV}")

    resolved = [d for d in disputes if d.get("settle_time")]
    voided = [d for d in resolved if int(d["settled_price"]) == UMA_IGNORE_PRICE]
    decided = [d for d in resolved if int(d["settled_price"]) != UMA_IGNORE_PRICE]
    confirmed = [d for d in decided if d["dvm_confirmed_proposal"]]
    print(f"[backfill_uma_disputes] {len(resolved)}/{len(disputes)} disputes resolved so far")
    print(f"[backfill_uma_disputes] {len(voided)}/{len(resolved)} resolved as VOIDED "
          f"(settled to UMA's ignore-price sentinel -- market cancelled/ambiguous, "
          f"not a proposal confirm/reject)")
    if decided:
        base_rate = len(confirmed) / len(decided)
        print(f"[backfill_uma_disputes] measured base rate (DVM confirms proposal, "
              f"VOIDED excluded): {base_rate:.3f} ({len(confirmed)}/{len(decided)})")
    else:
        print("[backfill_uma_disputes] zero non-voided resolved disputes -- base rate NOT TESTABLE yet")

    print("[backfill_uma_disputes] attempting voter-linkage query (known limitation, see module docstring)...")
    votes = dune_execute_and_fetch(VOTES_QUERY_ID, api_key)
    write_csv(votes, VOTES_CSV)
    print(f"[backfill_uma_disputes] wrote {len(votes)} rows -> {VOTES_CSV}")
    if not votes:
        print("[backfill_uma_disputes] 0 linked votes -- voter-conflict hypothesis NOT TESTABLE "
              "with the current linkage method (documented, not a bug).")

    voters = sorted({v["voter"] for v in votes if v.get("voter")})
    positions = fetch_polymarket_positions(voters)
    write_csv(positions, POSITIONS_CSV)
    print(f"[backfill_uma_disputes] wrote {len(positions)} rows -> {POSITIONS_CSV} "
          f"({len(voters)} distinct voters queried)")


if __name__ == "__main__":
    main()
