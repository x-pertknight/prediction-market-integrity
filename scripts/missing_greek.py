"""
scripts/missing_greek.py: recompute the headline numbers of
articles/settlement-risk-article.md from the shipped data. No network.

    python3 -I scripts/missing_greek.py

Inputs:
    data/uma_disputes.csv       973 DVM disputes on the classic UMA adapters
    data/too_early_outcomes.csv final on chain outcome of each too early question
    data/dispute_rate_full.csv  monthly resolutions and DVM disputes
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_dvm import PRICE_VOID, UMA_IGNORE_PRICE  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"


def classify(row) -> str:
    """too_early (reset), confirmed (DVM kept the proposal), half (0.5 tie),
    flip (DVM ruled the opposite outcome)."""
    settled, proposed = int(row["settled_price"]), int(row["proposedPrice"])
    if settled == UMA_IGNORE_PRICE:
        return "too_early"
    if settled == proposed:
        return "confirmed"
    if settled == PRICE_VOID:
        return "half"
    return "flip"


def dvm_decomposition(disputes: pd.DataFrame) -> dict:
    cls = disputes.apply(classify, axis=1).value_counts()
    n = len(disputes)
    overturned = n - cls["confirmed"]
    print("DVM DISPUTES ON THE CLASSIC ADAPTERS")
    print(f"  disputes                               {n}")
    print(f"  disputer 'won' (DVM did not confirm)   {overturned} / {n} = {overturned / n:.1%}  (earlier headline, retracted)")
    print(f"    too early (reset, asked again)       {cls['too_early']}")
    print(f"    direct flips to the other outcome    {cls['flip']}")
    print(f"    settled at 0.5                       {cls['half']}")
    print(f"  DVM confirmed the proposal             {cls['confirmed']}")
    lat = disputes["latency_days"]
    print(f"  dispute to DVM settle, days            median {lat.median():.2f}, p95 {np.percentile(lat, 95):.1f}, max {lat.max():.1f}")
    return {"n": n, "too_early": int(cls["too_early"]), "flip": int(cls["flip"]), "overturned": int(overturned)}


def too_early_join(n_too_early: int) -> dict:
    """Flipped is recomputed from the prices, not read from the flag column.
    A question whose own proposal was 0.5 is ambiguous and counted apart."""
    t = pd.read_csv(DATA / "too_early_outcomes.csv", dtype={"proposedPrice": str, "final_price": str})
    half_prop = t["proposedPrice"].astype(int) == PRICE_VOID
    changed = t["final_price"].astype(int) != t["proposedPrice"].astype(int)
    same, flipped = int((~changed & ~half_prop).sum()), int((changed & ~half_prop).sum())
    print("\nTOO EARLY DISPUTES JOINED TO THE FINAL RESOLUTION")
    print(f"  too early disputes                     {n_too_early}")
    print(f"  joined to a final resolution           {len(t)}  (pending {n_too_early - len(t)})")
    print(f"  final outcome = original proposal      {same}")
    print(f"  final outcome flipped                  {flipped}")
    print(f"  proposal was itself 0.5 (ambiguous)    {int(half_prop.sum())}")
    hours = t["days_to_final"].median() * 24
    wait = t.loc[changed & ~half_prop, "days_to_final"]
    print(f"  wait when the outcome flipped, days    median {wait.median():.2f}, p95 {np.percentile(wait, 95):.1f}, max {wait.max():.1f}")
    print(f"  median time to final outcome           {hours:.1f} hours; within 7 days {(t['days_to_final'] <= 7).mean():.0%}")
    return {"flipped": flipped, "ambiguous_flips": int((changed & half_prop).sum())}


def regime_rates() -> dict:
    """DVM dispute rate on classic adapter (v2/v3) resolutions, by regime."""
    m = pd.read_csv(DATA / "dispute_rate_full.csv")
    m = m[m["month"] != "pre"]
    pre = m[(m["month"] >= "2025-04") & (m["month"] <= "2025-08")]
    post = m[(m["month"] >= "2025-09") & (m["month"] <= "2026-06")]
    out = {}
    print("\nCLASSIC BOOK DVM DISPUTE RATE")
    windows = (("pre", "pre migration  2025-04..2025-08", pre),
               ("post", "post migration 2025-09..2026-06", post),
               ("full", "full window    2025-04..2026-06", pd.concat([pre, post])))
    for key, label, w in windows:
        d, r = int(w["dvm"].sum()), int(w["v23_resolved"].sum())
        out[key] = d / r
        out[f"{key}_disputes"], out[f"{key}_resolutions"] = d, r
        print(f"  {label}   {d:4d} / {r:7,} = {d / r:.2%}")
    return out


def main() -> dict:
    """Print the figures and return them for the README check in reproduce.py."""
    disputes = pd.read_csv(DATA / "uma_disputes.csv", dtype={"settled_price": str, "proposedPrice": str})
    dvm = dvm_decomposition(disputes)
    te = too_early_join(dvm["too_early"])
    changed = dvm["flip"] + te["flipped"]
    p_change = changed / dvm["n"]
    print("\nHEADLINE")
    print(f"  final outcome differs from the proposal {changed} / {dvm['n']} = {p_change:.1%}"
          f"  ({changed + te['ambiguous_flips']} counting the ambiguous 0.5 proposal)")
    rates = regime_rates()
    print("\nREPRICING COMPONENT OF THE SETTLEMENT ADJUSTMENT (dispute rate x P(change))")
    print(f"  current regime   {rates['post'] * p_change * 1e4:.1f} bps of notional")
    print(f"  pre migration    {rates['pre'] * p_change * 1e4:.1f} bps of notional")
    return {"n": dvm["n"], "changed": changed, "p_change": p_change, "overturned": dvm["overturned"],
            "p_overturned": dvm["overturned"] / dvm["n"], "too_early": dvm["too_early"],
            "bps_post": rates["post"] * p_change * 1e4, "bps_pre": rates["pre"] * p_change * 1e4,
            **{f"{k}_{c}": rates[f"{k}_{c}"] for k in ("pre", "post") for c in ("disputes", "resolutions")},
            "rate_pre": rates["pre"], "rate_post": rates["post"]}


if __name__ == "__main__":
    main()
