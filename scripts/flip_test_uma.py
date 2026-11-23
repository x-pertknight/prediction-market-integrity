"""
scripts/flip_test_uma.py: can the stake of voters holding a position move a
DVM outcome?

Published data holds no voter rows, so the test is run as a bound:

  1. GATE: the stake weighted plurality of each dispute's final round
     (data/dvm_round_tally.csv) must reproduce the settled price
     (data/dvm_disputes.csv). Disputes that fail are excluded and reported.
  2. BOUND: moving stake s from one price to another narrows the gap between
     the winner and any other price by at most 2s. If the winning margin is
     larger than twice the positioned share bound, no reassignment of the
     positioned voters' stake, including "every positioned voter votes their
     own book", can change the outcome.

The exact voter level replay (same result) and the stake share statistics
are computed in scripts/build_tables.py from the raw tables and shipped in
data/summary_raw.json.

    python3 -I scripts/flip_test_uma.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def final_round_shares(tally: pd.DataFrame) -> pd.DataFrame:
    """Per dispute: winning price, margin over the runner up, as stake shares."""
    last = tally.groupby("dispute")["round"].transform("max")
    fr = tally[tally["round"] == last]
    rows = []
    for k, g in fr.groupby("dispute"):
        share = (g.set_index("price")["stake"] / g["stake"].sum()).sort_values(ascending=False)
        runner = share.iloc[1] if len(share) > 1 else 0.0
        rows.append({"dispute": k, "winner": int(share.index[0]), "margin": share.iloc[0] - runner})
    return pd.DataFrame(rows).set_index("dispute")


def main() -> dict:
    tally = pd.read_csv(DATA / "dvm_round_tally.csv", dtype={"price": str})
    disputes = pd.read_csv(DATA / "dvm_disputes.csv", dtype={"settled_price": str}).set_index("dispute")
    raw = json.loads((DATA / "summary_raw.json").read_text())["exact_flip_test"]
    fr = final_round_shares(tally).join(disputes)
    gate_ok = fr["winner"] == fr["settled_price"].apply(int)
    ok = fr[gate_ok]
    can_flip = ok[ok["margin"] <= 2 * ok["positioned_share_max"]]
    s = raw["positioned_stake_share"]

    print("=" * 72)
    print("POSITIONED VOTER STAKE SHARE  (final round, reproduced disputes)")
    print("=" * 72)
    print(f"  disputes replayed & reproduced      {len(ok)}")
    print(f"  replay gate failures (excluded)     {int((~gate_ok).sum())}")
    print("  stake share of positioned voters (from the raw tables, see summary_raw.json):")
    print(f"    median                            {s['median']:.4%}")
    print(f"    mean                              {s['mean']:.4%}")
    print(f"    p95                               {s['p95']:.4%}")
    print(f"    max                               {s['max']:.4%}")
    print(f"    disputes with any positioned stake {int((ok['positioned_share_max'] > 0).sum())}")
    print()
    print("=" * 72)
    print("COUNTERFACTUAL FLIP TEST")
    print("=" * 72)
    print(f"  smallest winning margin             {ok['margin'].min():.4f} of final round stake")
    print(f"  positioned share bound              {ok['positioned_share_max'].max():.2f}")
    print(f"  outcomes changed                    {len(can_flip)} / {len(ok)}")
    print(f"  exact voter level replay (raw)      {raw['outcomes_changed']} / {raw['disputes_reproduced']}")
    print()
    verdict = ("positioned voters vote their book, but their stake cannot move outcomes."
               if len(can_flip) == 0 else f"{len(can_flip)} outcome(s) not excluded by the bound.")
    print("  VERDICT:", verdict)
    return {"margin_min": float(ok["margin"].min()), "stake_max": s["max"],
            "reproduced": len(ok), "linked": len(fr), "outcomes_unchanged": len(ok) - len(can_flip),
            "stake_bound": float(ok["positioned_share_max"].max()),
            "raw_changed": raw["outcomes_changed"], "raw_reproduced": raw["disputes_reproduced"]}


if __name__ == "__main__":
    main()
