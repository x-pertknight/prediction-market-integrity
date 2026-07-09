"""
scripts/report_uma_conflict.py — Score the preregistered UMA whale-conflict
hypothesis against the reconstructed data.

Reads the artifacts of the web3 pipeline and reports, without editorialising:

  * the dispute outcome breakdown and the DVM base rate (RESET excluded)
  * how many voters held a directional Polymarket position at the dispute block
  * conflict_rate per voter, and the share of eligible whales above the
    preregistered 0.8 threshold

Refuses to report a conflict_rate at all when the sample is below the
preregistered testability floor (20 contested disputes). See
docs/experiments/2026-07-uma-dispute-preregistration.md.

Usage:
    python scripts/report_uma_conflict.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from alpha.uma_dispute import (  # noqa: E402
    MIN_HISTORICAL_VOTES, DisputeOutcome, annotate_votes_with_positions,
    classify_outcome, compute_voter_stats, summarize_disputes,
)

DATA = ROOT / "data"
MIN_CONTESTED_DISPUTES = 20   # preregistered testability floor
CONFLICT_THRESHOLD = 0.8      # preregistered: predicted 18% (mine) / <30% (advisor)


def main() -> None:
    disputes = pd.read_csv(DATA / "uma_disputes.csv",
                           dtype={"settled_price": str, "proposedPrice": str})
    votes = pd.read_parquet(DATA / "uma_votes.parquet")
    positions = pd.read_csv(DATA / "uma_positions.csv") if (DATA / "uma_positions.csv").exists() \
        else pd.DataFrame()

    b = summarize_disputes(disputes)
    print("=" * 72)
    print("DISPUTE OUTCOME BREAKDOWN  (N =", b.n_disputes, "resolved disputes)")
    print("=" * 72)
    for label, n in [("RESET (sentinel, re-requested)", b.n_reset), ("NO  [0,1]", b.n_no),
                     ("YES [1,0]", b.n_yes), ("VOID 50/50 [1,1]", b.n_void_5050)]:
        print(f"  {label:34s} {n:5d}  {100 * n / b.n_disputes:5.1f}%")
    print(f"\n  decided (RESET excluded)            {b.n_decided:5d}")
    print(f"  DVM confirmed the proposer          {b.n_confirmed:5d}")
    print(f"\n  BASE RATE  = {b.base_rate:.4f}   (predicted 0.85 mine / 0.80-0.85 advisor)")
    print(f"  RESET RATE = {b.reset_rate:.4f}   a dispute buys delay, not a reversal")

    print("\n" + "=" * 72)
    print("VOTE LINKAGE")
    print("=" * 72)
    print(f"  votes linked            {len(votes):,}")
    print(f"  disputes with votes     {votes['dispute_tx_hash'].nunique()}")
    print(f"  distinct DVM voters     {votes['voter'].nunique()}")
    print("\n  vote sides:")
    for side, n in votes["side"].value_counts().items():
        print(f"    {side:8s} {n:8,}  {100 * n / len(votes):5.1f}%")

    print("\n" + "=" * 72)
    print("CONFLICT OF INTEREST")
    print("=" * 72)
    ok = positions[positions["status"] == "ok"] if "status" in positions else pd.DataFrame()
    if positions.empty or ok.empty:
        print("  0 voter positions reconstructed -> conflict_rate NOT CALCULABLE.")
        print("  This is a coverage gap, NOT evidence against the WSJ hypothesis.")
        if "status" in positions:
            print("\n  status breakdown:")
            print("   ", positions["status"].value_counts().to_string().replace("\n", "\n    "))
        return

    # Coverage: what fraction of the linked disputes could we even look at?
    n_linked = votes["dispute_tx_hash"].nunique()
    covered = positions.loc[positions["status"].isin(["ok", "no_voter_position"]), "dispute_tx_hash"].nunique()
    print(f"  disputes with votes                  {n_linked}")
    print(f"  disputes with a resolved condition   {covered}  ({100 * covered / n_linked:.1f}% of sample)")
    if "adapter" in ok:
        mix = ok["adapter"].value_counts()
        print(f"  positions by adapter                 " +
              ", ".join(f"{k}={v}" for k, v in mix.items()))
    dropped = positions[~positions["status"].isin(["ok", "no_voter_position"])]
    if len(dropped):
        print("  dropped (NOT counted as 'no conflict'):")
        for status, n in dropped["status"].value_counts().items():
            print(f"    {status}: {n}")

    # Annotate the FULL vote record, not just the disputes whose positions we could
    # rebuild. total_votes is a voter's whole DVM history (that's what the ">=3
    # historical votes" whale filter means); votes_with_position is only ever
    # nonzero where a position was actually measured, so restricting the frame
    # would shrink total_votes and wrongly admit small-sample voters as whales.
    annotated = annotate_votes_with_positions(votes, ok)
    contested = annotated.loc[annotated["has_position"], "dispute_tx_hash"].nunique()
    print(f"  disputes covered by position rebuild {ok['dispute_tx_hash'].nunique()}")
    print(f"  disputes with >=1 conflicted voter   {contested}")
    print(f"  voter-dispute pairs with a position  {int(annotated['has_position'].sum()):,}")

    if contested < MIN_CONTESTED_DISPUTES:
        print(f"\n  N = {contested} < {MIN_CONTESTED_DISPUTES} (preregistered floor)")
        print("  VERDICT: NOT TESTABLE. Reporting N only, per the preregistration.")
        return

    # ── The headline test ────────────────────────────────────────────────────
    # conflict_rate's denominator counts every position-linked vote, including
    # IGNORE. An IGNORE is a vote to reset, so it can never match a book, which
    # structurally caps conflict_rate near (1 - IGNORE share) and makes it hard to
    # read. The clean test restricts to DIRECTIONAL (YES/NO) position-linked votes
    # and asks whether they land on the holder's own side more often than a coin.
    linked = annotated[annotated["has_position"]]
    directional = linked[linked["side"].isin(["YES", "NO"])]
    own = int(directional["voted_with_own_position"].sum())
    n_dir = len(directional)
    print(f"\n  position-linked votes                {len(linked)}")
    print(f"    of which directional (YES/NO)      {n_dir}")
    print(f"    of which IGNORE (a vote to reset)  {int((linked['side'] == 'IGNORE').sum())}")

    if n_dir:
        rate = own / n_dir
        print(f"\n  >> OWN-BOOK RATE on directional votes: {rate:.4f}  ({own}/{n_dir})")
        print(f"     null if voters ignored their book:  0.5000")
        try:
            from scipy.stats import binomtest
            p = binomtest(own, n_dir, 0.5, alternative="two-sided").pvalue
            print(f"     binomial test vs 0.50:              p = {p:.3g}")
        except ImportError:
            pass

    # Holders may also prefer to stall rather than vote against themselves.
    ign_held = (linked["side"] == "IGNORE").mean()
    ign_free = (annotated.loc[~annotated["has_position"], "side"] == "IGNORE").mean()
    print(f"\n  IGNORE share | holds a position      {ign_held:.4f}")
    print(f"  IGNORE share | holds nothing         {ign_free:.4f}")

    # ── The preregistered metric, and why it must be read with care ──────────
    stats = compute_voter_stats(annotated)
    print(f"\n  preregistered metric: share of eligible whales with conflict_rate > "
          f"{CONFLICT_THRESHOLD}  (predicted 0.18 mine / <0.30 advisor)")
    print("  sensitivity to the minimum number of POSITION-LINKED votes per whale:")
    print("  (a whale with 1 linked vote scores 1.00 or 0.00 by construction)")
    for min_linked in (1, 2, 3, 5):
        eligible = [s for s in stats.values()
                    if s.total_votes >= MIN_HISTORICAL_VOTES and s.votes_with_position >= min_linked]
        if not eligible:
            print(f"    min_linked>={min_linked}: 0 eligible")
            continue
        above = [s for s in eligible if s.conflict_rate > CONFLICT_THRESHOLD]
        own_book = sum(s.voted_own_book for s in eligible)
        with_pos = sum(s.votes_with_position for s in eligible)
        print(f"    min_linked>={min_linked}: eligible={len(eligible):4d}  "
              f"share>{CONFLICT_THRESHOLD}={len(above) / len(eligible):.3f} "
              f"({len(above)}/{len(eligible)})  pooled_conflict_rate={own_book / with_pos:.4f}")

    eligible = [s for s in stats.values()
                if s.total_votes >= MIN_HISTORICAL_VOTES and s.votes_with_position >= 3]
    above = sorted([s for s in eligible if s.conflict_rate > CONFLICT_THRESHOLD],
                   key=lambda s: -s.avg_stake)
    if above:
        print("\n  whales above threshold with >=3 linked votes (the defensible ones):")
        for s in above[:10]:
            print(f"    {s.voter}  votes={s.total_votes:4d}  linked={s.votes_with_position:3d}  "
                  f"conflict_rate={s.conflict_rate:.2f}  avg_stake={s.avg_stake:,.0f} UMA")


if __name__ == "__main__":
    main()
