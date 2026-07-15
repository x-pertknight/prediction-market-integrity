"""
scripts/flip_test_uma.py — Counterfactual flip test for the positioned-voter
mechanism claim.

The conflict analysis (report_uma_conflict.py) established that positioned
voters vote their own book (63.2% vs 50%, p=1.5e-5) but that outcomes do not
move (corr(tilt, outcome) ~ 0). The README asserts the mechanism — "their
stake is negligible against the total DVM" — without quantifying it. This
script closes that gap with two measurements:

  1. STAKE SHARE — per dispute, the share of final-round DVM stake held by
     voters with a directional Polymarket position on that market.
  2. FLIP TEST — the counterfactual: reassign EVERY positioned voter's stake
     to their own book side and recompute the stake-weighted winner. Count
     how many of the reproduced outcomes change.

Sanity gate: before flipping anything, the stake-weighted plurality of the
final voting round must reproduce the settled price. Disputes that fail the
gate are excluded and reported, never silently flipped.

Pure replay on committed artifacts — no network. Usage:
    python scripts/flip_test_uma.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from alpha.uma_dispute import PRICE_NO, PRICE_YES  # noqa: E402

DATA = ROOT / "data"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    disputes = pd.read_csv(DATA / "uma_disputes.csv",
                           dtype={"settled_price": str, "proposedPrice": str})
    votes = pd.read_parquet(DATA / "uma_votes.parquet")
    positions = pd.read_csv(DATA / "uma_positions.csv")
    return disputes, votes, positions


def final_round(votes: pd.DataFrame) -> pd.DataFrame:
    """Keep only each dispute's final voting round; dedupe voters within it
    (last revealed vote by block wins)."""
    last = votes.groupby("dispute_tx_hash")["round_id"].transform("max")
    fr = votes[votes["round_id"] == last].copy()
    fr = (fr.sort_values("block_number")
            .drop_duplicates(subset=["dispute_tx_hash", "voter"], keep="last"))
    return fr


def winner(group: pd.DataFrame) -> int:
    """Stake-weighted plurality price of one dispute's final round."""
    tally = group.groupby("price_int")["stake"].sum()
    return int(tally.idxmax())


def main() -> None:
    disputes, votes, positions = load()

    fr = final_round(votes)
    fr["price_int"] = fr["vote_price"].apply(int)

    # book side per (dispute, voter): net directional position
    ok = positions[positions["status"] == "ok"].copy()
    ok["net"] = ok["yes_shares"].fillna(0) - ok["no_shares"].fillna(0)
    ok = ok[ok["net"] != 0]
    ok["book_price"] = ok["net"].apply(lambda n: PRICE_YES if n > 0 else PRICE_NO)
    book = ok.set_index(["dispute_tx_hash", "voter"])["book_price"]

    settled = disputes.set_index("dispute_tx_hash")["settled_price"].apply(int)

    n_repro = n_gate_fail = n_flipped = 0
    shares, flipped_disputes, gate_failures = [], [], []

    for tx, g in fr.groupby("dispute_tx_hash"):
        if tx not in settled.index:
            continue
        actual = winner(g)

        # sanity gate: replay must reproduce the chain
        if actual != settled.loc[tx]:
            n_gate_fail += 1
            gate_failures.append(tx)
            continue
        n_repro += 1

        keys = list(zip([tx] * len(g), g["voter"]))
        mask = [k in book.index for k in keys]
        pos_stake = g.loc[mask, "stake"].sum()
        total_stake = g["stake"].sum()
        shares.append(pos_stake / total_stake if total_stake else 0.0)

        if not any(mask):
            continue

        # counterfactual: every positioned voter votes their book
        cf = g.copy()
        cf.loc[mask, "price_int"] = [book.loc[k] for k, m in zip(keys, mask) if m]
        if winner(cf) != actual:
            n_flipped += 1
            flipped_disputes.append(tx)

    s = pd.Series(shares)
    print("=" * 72)
    print("POSITIONED-VOTER STAKE SHARE  (final round, reproduced disputes)")
    print("=" * 72)
    print(f"  disputes replayed & reproduced      {n_repro}")
    print(f"  replay gate failures (excluded)     {n_gate_fail}")
    print(f"  stake share of positioned voters:")
    print(f"    median                            {s.median():.4%}")
    print(f"    mean                              {s.mean():.4%}")
    print(f"    p95                               {s.quantile(0.95):.4%}")
    print(f"    max                               {s.max():.4%}")
    print(f"    disputes with any positioned stake {int((s > 0).sum())}")

    print()
    print("=" * 72)
    print("COUNTERFACTUAL FLIP TEST")
    print("=" * 72)
    print("  (all positioned voters' stake reassigned to their own book side)")
    print(f"  outcomes changed                    {n_flipped} / {n_repro}")
    for tx in flipped_disputes:
        print(f"    FLIPPED: {tx}")
    if gate_failures:
        print(f"\n  gate failures (not reproduced, hence not flipped):")
        for tx in gate_failures[:10]:
            print(f"    {tx}")

    print()
    verdict = ("MECHANISM CONFIRMED: positioned voters vote their book, but their "
               "stake cannot move outcomes." if n_flipped == 0 else
               f"MECHANISM PARTIAL: {n_flipped} outcome(s) WOULD flip — "
               "the 'too little stake' claim needs qualification.")
    print("  VERDICT:", verdict)


if __name__ == "__main__":
    main()
