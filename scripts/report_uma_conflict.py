"""
scripts/report_uma_conflict.py: score the UMA voter conflict question against
the published aggregate tables.

Question: do DVM voters who hold a Polymarket position on the disputed market
vote their own book? Reported, without editorialising:

  * the dispute outcome breakdown and the DVM base rate (RESET excluded)
  * vote counts by side, from data/dvm_round_tally.csv
  * the own book rate on directional votes against a coin flip, from
    data/own_book_votes.csv (one row per positioned vote, no voter id)

Figures that need voter level rows (distinct voters, coverage of the position
rebuild, the per voter conflict_rate metric) come from data/summary_raw.json,
computed by scripts/build_tables.py from the raw tables, and are labelled.

The predictions quoted below were written down and are scored here, misses
included. The predictions file is not in this repo and no hash was recorded,
so their date cannot be checked from the repo.

    python3 -I scripts/report_uma_conflict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_dvm import summarize_disputes, vote_side  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
MIN_CONTESTED_DISPUTES = 20   # testability floor from the written predictions
CONFLICT_THRESHOLD = 0.8      # threshold from the written predictions (predicted share: 0.18)
RAW_LABEL = "(raw tables, not recomputable from shipped data)"


def print_breakdown() -> None:
    disputes = pd.read_csv(DATA / "uma_disputes.csv", dtype={"settled_price": str, "proposedPrice": str})
    b = summarize_disputes(disputes)
    print("=" * 72)
    print("DISPUTE OUTCOME BREAKDOWN  (N =", b.n_disputes, "resolved disputes)")
    print("=" * 72)
    for label, n in [("RESET (too early, asked again)", b.n_reset), ("NO  [0,1]", b.n_no),
                     ("YES [1,0]", b.n_yes), ("VOID 50/50 [1,1]", b.n_void_5050)]:
        print(f"  {label:34s} {n:5d}  {100 * n / b.n_disputes:5.1f}%")
    print(f"\n  decided (RESET excluded)            {b.n_decided:5d}")
    print(f"  DVM confirmed the proposer          {b.n_confirmed:5d}")
    print(f"\n  BASE RATE  = {b.base_rate:.4f}   (written prediction: 0.85)")
    print(f"  RESET RATE = {b.reset_rate:.4f}   a dispute buys delay, not a reversal")


def print_linkage(tally: pd.DataFrame, raw: dict) -> pd.Series:
    sides = tally.assign(side=tally["price"].map(vote_side)).groupby("side")["n_votes"].sum()
    sides = sides.sort_values(ascending=False)
    total = int(sides.sum())
    print("\n" + "=" * 72)
    print("VOTE LINKAGE")
    print("=" * 72)
    print(f"  votes linked            {total:,}")
    print(f"  disputes with votes     {tally['dispute'].nunique()}")
    print(f"  distinct DVM voters     {raw['distinct_dvm_voters']}  {RAW_LABEL}")
    print("\n  vote sides:")
    for side, n in sides.items():
        print(f"    {side:8s} {n:8,}  {100 * n / total:5.1f}%")
    return sides


def print_coverage(raw: dict, own: pd.DataFrame) -> int:
    print(f"  disputes with a resolved condition   {raw['disputes_with_resolved_condition']}  {RAW_LABEL}")
    mix = ", ".join(f"{k}={v}" for k, v in raw["positions_by_adapter"].items())
    print(f"  positions by adapter                 {mix}  {RAW_LABEL}")
    for status, n in raw["dropped_status"].items():
        print(f"  dropped (NOT counted as no conflict): {status}: {n}")
    contested = own["case"].nunique()
    print(f"  disputes with >=1 conflicted voter   {contested}")
    print(f"  voter dispute pairs with a position  {len(own):,}")
    return contested


def print_own_book(own: pd.DataFrame, sides: pd.Series) -> dict:
    """Directional (YES/NO) votes only: an IGNORE vote can never match a book."""
    directional = own[own["vote_side"].isin(["YES", "NO"])]
    n_dir = len(directional)
    n_own = int((directional["vote_side"] == directional["position_side"]).sum())
    n_ign = int((own["vote_side"] == "IGNORE").sum())
    print(f"\n  position linked votes                {len(own)}")
    print(f"    of which directional (YES/NO)      {n_dir}")
    print(f"    of which IGNORE (a vote to reset)  {n_ign}")
    print(f"\n  >> OWN BOOK RATE on directional votes: {n_own / n_dir:.4f}  ({n_own}/{n_dir})")
    print("     null if voters ignored their book:  0.5000")
    try:
        from scipy.stats import binomtest
        p = binomtest(n_own, n_dir, 0.5, alternative="two-sided").pvalue
        print(f"     binomial test vs 0.50:              p = {p:.3g}")
    except ImportError:
        print("     (install scipy for the binomial p value)")
    # Holders may also prefer to stall rather than vote against themselves.
    free_ign = (sides.get("IGNORE", 0) - n_ign) / (sides.sum() - len(own))
    print(f"\n  IGNORE share | holds a position      {n_ign / len(own):.4f}")
    print(f"  IGNORE share | holds nothing         {free_ign:.4f}")
    return {"own_book": n_own, "directional": n_dir, "p_own_book": n_own / n_dir}


def print_sensitivity(raw: dict) -> None:
    print(f"\n  per voter metric: share of eligible voters with conflict_rate > "
          f"{CONFLICT_THRESHOLD}  (predicted 0.18)  {RAW_LABEL}")
    print("  (a voter with 1 linked vote scores 1.00 or 0.00 by construction)")
    for r in raw["conflict_rate_sensitivity"]:
        print(f"    min_linked>={r['min_linked']}: eligible={r['eligible']:4d}  "
              f"share>{CONFLICT_THRESHOLD}={r['above_0_8'] / r['eligible']:.3f} "
              f"({r['above_0_8']}/{r['eligible']})  pooled_conflict_rate={r['pooled_conflict_rate']:.4f}")


def main() -> dict:
    """Print the report and return the own book figures (empty if not testable)."""
    tally = pd.read_csv(DATA / "dvm_round_tally.csv", dtype={"price": str})
    own = pd.read_csv(DATA / "own_book_votes.csv")
    raw = json.loads((DATA / "summary_raw.json").read_text())
    print_breakdown()
    sides = print_linkage(tally, raw)
    print("\n" + "=" * 72)
    print("CONFLICT OF INTEREST")
    print("=" * 72)
    contested = print_coverage(raw, own)
    if contested < MIN_CONTESTED_DISPUTES:
        print(f"\n  N = {contested} < {MIN_CONTESTED_DISPUTES}: NOT TESTABLE, reporting N only.")
        return {}
    figures = print_own_book(own, sides)
    print_sensitivity(raw)
    return figures


if __name__ == "__main__":
    main()
