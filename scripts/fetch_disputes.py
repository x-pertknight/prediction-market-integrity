"""
scripts/fetch_disputes.py: fetch the list of Polymarket disputes that reached
the UMA DVM, from 2025-04-01 on.

Source: a saved Dune query over Polygon OptimisticOracleV2 DisputePrice joined
to Settle, for the two classic Polymarket adapters (UmaCtfAdapter v2
0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74 and v3 negRisk
0x2F5e3684cb1F318ec51b00Edba38d79Ac2c0aA9d). The script re executes the query
by id so the list reflects chain state at run time. Needs DUNE_API_KEY.

Everything downstream (DVM votes, positions) uses public RPC only:
scripts/fetch_votes_web3.py and scripts/fetch_positions_web3.py.

Output: data/raw/uma_disputes.csv (names wallets; never published as is,
see scripts/build_tables.py).

    python3 -I scripts/fetch_disputes.py
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

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DUNE_API_BASE = "https://api.dune.com/api/v1"
# The id of the saved Dune query is not published here. Set
# DUNE_DISPUTES_QUERY_ID to the id of your own query (see the docstring).
DISPUTES_QUERY_ID = os.environ.get("DUNE_DISPUTES_QUERY_ID")
DISPUTES_CSV = ROOT / "data" / "raw" / "uma_disputes.csv"

POLL_INTERVAL_S = 3
POLL_TIMEOUT_S = 420

# UMA's ignore price sentinel (int256 min). It is a RESET, not a void: the
# adapter's `_resolve` calls `_reset(...)` and asks the question again. The
# only genuine void is 0.5 ether, payouts [1,1]. See scripts/uma_dvm.py.
UMA_IGNORE_PRICE = -(2 ** 255)


def _headers(api_key: str) -> dict:
    return {"X-Dune-API-Key": api_key}


def wait_for(execution_id: str, api_key: str) -> None:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    state = "QUERY_STATE_PENDING"
    while state not in ("QUERY_STATE_COMPLETED", "QUERY_STATE_FAILED"):
        if time.monotonic() > deadline:
            raise TimeoutError(f"execution {execution_id} did not finish in {POLL_TIMEOUT_S}s")
        time.sleep(POLL_INTERVAL_S)
        resp = requests.get(f"{DUNE_API_BASE}/execution/{execution_id}/status",
                            headers=_headers(api_key), timeout=30)
        resp.raise_for_status()
        state = resp.json()["state"]
    if state == "QUERY_STATE_FAILED":
        raise RuntimeError(f"execution {execution_id} failed")


def execute_and_fetch(query_id: int, api_key: str) -> list[dict]:
    """Execute a saved query by id and return all result rows."""
    resp = requests.post(f"{DUNE_API_BASE}/query/{query_id}/execute",
                         headers=_headers(api_key), json={"performance": "medium"}, timeout=30)
    resp.raise_for_status()
    execution_id = resp.json()["execution_id"]
    wait_for(execution_id, api_key)
    rows: list[dict] = []
    offset, limit = 0, 5000
    while True:
        r = requests.get(f"{DUNE_API_BASE}/execution/{execution_id}/results",
                         headers=_headers(api_key), params={"limit": limit, "offset": offset},
                         timeout=60)
        r.raise_for_status()
        batch = r.json().get("result", {}).get("rows", [])
        rows.extend(batch)
        if len(batch) < limit:
            return rows
        offset += limit


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["_empty"])
        writer.writeheader()
        writer.writerows(rows)


def summarize(disputes: list[dict]) -> None:
    resolved = [d for d in disputes if d.get("settle_time")]
    reset = [d for d in resolved if int(d["settled_price"]) == UMA_IGNORE_PRICE]
    decided = [d for d in resolved if int(d["settled_price"]) != UMA_IGNORE_PRICE]
    confirmed = [d for d in decided if d["dvm_confirmed_proposal"]]
    print(f"{len(resolved)}/{len(disputes)} disputes resolved")
    print(f"{len(reset)}/{len(resolved)} RESET (too early: the question is asked again)")
    if decided:
        print(f"DVM confirms the proposal, RESET excluded: "
              f"{len(confirmed) / len(decided):.3f} ({len(confirmed)}/{len(decided)})")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    api_key = os.environ.get("DUNE_API_KEY")
    if not api_key:
        sys.exit("DUNE_API_KEY not set (environment or .env)")
    if not DISPUTES_QUERY_ID:
        sys.exit("DUNE_DISPUTES_QUERY_ID not set (environment or .env)")
    disputes = execute_and_fetch(int(DISPUTES_QUERY_ID), api_key)
    write_csv(disputes, DISPUTES_CSV)
    print(f"wrote {len(disputes)} rows to {DISPUTES_CSV}")
    summarize(disputes)


if __name__ == "__main__":
    main()
