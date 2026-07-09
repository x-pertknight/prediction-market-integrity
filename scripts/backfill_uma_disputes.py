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

2. Voter-level conflict analysis (the actual WSJ hypothesis) — SUPERSEDED,
   2026-07-09. The dispute<->vote linkage this script could not solve is now
   solved, and it needs no Dune at all. It lives in
   scripts/fetch_votes_web3.py, over public RPC.

   The linkage failed here for two compounding reasons, both since verified
   on-chain:
     a. OptimisticOracleV2 appends ",ooRequester:<adapter>" and then
        OracleChildTunnel COMPRESSES the ancillary data, so the DVM only ever
        sees "ancillaryDataHash:<keccak of the stamped bytes>,...". Matching
        the raw text is guaranteed to score 0. Matching the hash scores 939/973.
     b. Polymarket's requests are event-based, so on dispute the OO escalates
        to the DVM under the PROPOSAL time, not `request_timestamp`. Those
        differ by a median of 125 hours, so keying on request_timestamp also
        scores 0.
   Both the hash and the true DVM time are emitted by OracleChildTunnel inside
   the dispute's own Polygon transaction (`PriceRequestAdded`), so they are read
   from chain rather than inferred.

   The old linkage query (Dune 7917907) still returns 0 rows. Do not read that 0
   as evidence against the WSJ hypothesis. Use scripts/fetch_votes_web3.py.

   Historical positions likewise come from scripts/fetch_positions_web3.py:
   a DVM voter's mainnet EOA is not their Polymarket position holder (positions
   sit in a per-user Safe proxy on Polygon, derived deterministically via the
   factory's `computeProxyAddress(owner)`).

The disputes query below was created and validated via Dune's query builder
(query_id pinned); this script re-executes it by ID. Note the base tier CAN
create saved queries over the API — what it runs out of is execution credits
and per-cycle datapoints, which is what made the RPC path the right call.

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

# Pinned Dune query IDs (created 2026-07-08, see module docstring for SQL intent).
DISPUTES_QUERY_ID = 7917876   # traptrade_uma_polymarket_disputes_backfill

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DISPUTES_CSV = DATA_DIR / "uma_disputes.csv"

POLL_INTERVAL_S = 3
POLL_TIMEOUT_S = 420

# UMA's "ignore price" sentinel (== -2**255, int256 min). 67.4% of settled
# Polymarket disputes land here.
#
# CORRECTED 2026-07-09 against the deployed UmaCtfAdapter source: this is a
# RESET, not a void. `_resolve` calls `_reset(...)` on the sentinel, sending a
# fresh price request out; the market is not cancelled and nobody is refunded.
# The only genuine void is 0.5 ether -> payouts [1,1], which is 5/973 disputes.
# Counting resets as "proposal rejected" understates the confirm-rate; calling
# them "voided" misstates what a position holder actually received. They are
# excluded from the base rate and reported as their own RESET bucket.
# Canonical logic: src/alpha/uma_dispute.classify_outcome / expected_payouts.
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
    reset = [d for d in resolved if int(d["settled_price"]) == UMA_IGNORE_PRICE]
    decided = [d for d in resolved if int(d["settled_price"]) != UMA_IGNORE_PRICE]
    confirmed = [d for d in decided if d["dvm_confirmed_proposal"]]
    void_5050 = [d for d in decided if int(d["settled_price"]) == 5 * 10 ** 17]
    print(f"[backfill_uma_disputes] {len(resolved)}/{len(disputes)} disputes resolved so far")
    print(f"[backfill_uma_disputes] {len(reset)}/{len(resolved)} RESET "
          f"(ignore-price sentinel: the question is re-requested, NOT cancelled or "
          f"refunded, and the DVM issued no ruling on the proposal)")
    print(f"[backfill_uma_disputes] {len(void_5050)}/{len(resolved)} VOID 50/50 "
          f"(0.5 ether -> payouts [1,1]; the only genuine void)")
    if decided:
        base_rate = len(confirmed) / len(decided)
        print(f"[backfill_uma_disputes] measured base rate (DVM confirms proposal, "
              f"RESET excluded): {base_rate:.3f} ({len(confirmed)}/{len(decided)})")
    else:
        print("[backfill_uma_disputes] zero non-reset resolved disputes -- base rate NOT TESTABLE yet")

    # uma_votes.csv / uma_positions.csv are owned by the web3 scripts now. This
    # backfill must not touch them: writing an empty header here would silently
    # destroy a real, hard-won dataset.
    print("[backfill_uma_disputes] votes/positions: see scripts/fetch_votes_web3.py "
          "and scripts/fetch_positions_web3.py (public RPC, no Dune).")


if __name__ == "__main__":
    main()
