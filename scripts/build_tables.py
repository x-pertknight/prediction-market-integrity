"""
scripts/build_tables.py: turn the raw fetch outputs (data/raw/, not committed)
into the published tables in data/.

Why: the voter study is published in aggregate only. A voter row with a round,
an exact stake or a vote price can be matched to the public VoteRevealed event
that carries the voter's address, so no published table holds voter rows with
any of those. What is published:

  data/uma_disputes.csv     one row per dispute: adapter, dispute and settle
                            DATE, latency in days, proposed and settled price.
                            Shuffled, no key.
  data/dvm_round_tally.csv  per dispute, round and price: vote count and total
                            stake. No voter information at all.
  data/dvm_disputes.csv     per dispute (same key as the tally): settled price
                            and an upper bound on the final round stake share of
                            voters with a position (0, or 0.02 when any).
  data/own_book_votes.csv   one row per vote cast by a voter holding a position:
                            vote side (YES, NO, IGNORE, OTHER) and position side only. Its case key is a
                            separate random key that does not join to the tally.
  data/summary_raw.json     the few figures that need voter level rows,
                            computed here from the raw tables.

Keys and row order are drawn from the OS random source, so they carry no order
from the chain.

    python3 -I scripts/build_tables.py --raw data/raw
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_dvm import (PRICE_NO, PRICE_YES, annotate_votes_with_positions,
                     compute_voter_stats, vote_side)  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"
ADAPTER_V2 = "0x6a9d222616c90fca5754cd1333cfd9b7fb6a4f74"
POSITIONED_SHARE_BOUND = 0.02   # published bound; checked against the raw maximum
MIN_VOTER_HISTORY = 3           # voters need >= 3 DVM votes for the conflict_rate metric
RNG = secrets.SystemRandom()


def random_keys(values) -> dict:
    """Map each distinct value to a random integer key 1..n."""
    distinct = sorted(set(values))
    keys = list(range(1, len(distinct) + 1))
    RNG.shuffle(keys)
    return dict(zip(distinct, keys))


def shuffled(frame: pd.DataFrame) -> pd.DataFrame:
    order = list(range(len(frame)))
    RNG.shuffle(order)
    return frame.iloc[order].reset_index(drop=True)


def load_raw(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(raw / "uma_disputes.csv", dtype=str)
    v = pd.read_parquet(raw / "uma_votes.parquet")
    for col in ("dispute_tx_hash", "voter", "side", "vote_price"):
        v[col] = v[col].astype(str)
    v["voter"] = v["voter"].str.lower()
    v = v.rename(columns={"dispute_tx_hash": "dispute_id"})
    p = pd.read_csv(raw / "uma_positions.csv", dtype={"voter": str})
    p = p.rename(columns={"dispute_tx_hash": "dispute_id"})
    p["voter"] = p["voter"].str.lower()
    return d, v, p


def publish_disputes(d: pd.DataFrame) -> None:
    """Times are cut to the date; the latency keeps the precision the medians need."""
    t0 = pd.to_datetime(d["dispute_time"], format="mixed")
    t1 = pd.to_datetime(d["settle_time"], format="mixed")
    out = pd.DataFrame({
        "adapter": d["requester"].str.lower().map(lambda r: "v2" if r == ADAPTER_V2 else "negrisk"),
        "dispute_date": t0.dt.strftime("%Y-%m-%d"),
        "settle_date": t1.dt.strftime("%Y-%m-%d"),
        "latency_days": ((t1 - t0).dt.total_seconds() / 86400).round(2),
        "proposedPrice": d["proposedPrice"],
        "settled_price": d["settled_price"],
    })
    shuffled(out).to_csv(OUT / "uma_disputes.csv", index=False)


def publish_too_early(raw: Path) -> None:
    """Same date cut for the too early table, if it is in data/raw."""
    src = raw / "too_early_outcomes.csv"
    if not src.exists():
        return
    t = pd.read_csv(src, dtype=str)
    t["dispute_time"] = pd.to_datetime(t["dispute_time"], format="mixed").dt.strftime("%Y-%m-%d")
    t = t.rename(columns={"dispute_time": "dispute_date"})
    shuffled(t).to_csv(OUT / "too_early_outcomes.csv", index=False)


def final_round(v: pd.DataFrame) -> pd.DataFrame:
    """Each dispute's last round; a voter's last reveal in it wins."""
    last = v.groupby("dispute_id")["round_id"].transform("max")
    fr = v[v["round_id"] == last].sort_values("block_number")
    return fr.drop_duplicates(subset=["dispute_id", "voter"], keep="last")


def book_sides(p: pd.DataFrame) -> pd.Series:
    """(dispute_id, voter) -> price that pays the voter's net position."""
    ok = p[p["status"] == "ok"].copy()
    ok["net"] = ok["yes_shares"].fillna(0) - ok["no_shares"].fillna(0)
    ok = ok[ok["net"] != 0]
    ok["book_price"] = np.where(ok["net"] > 0, PRICE_YES, PRICE_NO)
    return ok.set_index(["dispute_id", "voter"])["book_price"]


def exact_flip_test(fr: pd.DataFrame, settled: pd.Series, book: pd.Series) -> dict:
    """The voter level flip test, run here on the raw rows."""
    shares, flipped, gate_fail = [], 0, 0
    for tx, g in fr.groupby("dispute_id"):
        g = g.assign(price_int=g["vote_price"].apply(int))
        win = int(g.groupby("price_int")["stake"].sum().idxmax())
        if win != int(settled.loc[tx]):
            gate_fail += 1
            continue
        keys = list(zip(g["dispute_id"], g["voter"]))
        mask = np.array([k in book.index for k in keys])
        shares.append(g.loc[mask, "stake"].sum() / g["stake"].sum())
        if mask.any():
            cf = g.copy()
            cf.loc[mask, "price_int"] = [book.loc[k] for k, m in zip(keys, mask) if m]
            flipped += int(cf.groupby("price_int")["stake"].sum().idxmax() != win)
    s = pd.Series(shares)
    return {"disputes_reproduced": len(shares), "gate_failures": gate_fail,
            "outcomes_changed": flipped,
            "positioned_stake_share": {"median": float(s.median()), "mean": float(s.mean()),
                                       "p95": float(np.percentile(s, 95)), "max": float(s.max())}}


def positioned_shares(fr: pd.DataFrame, book: pd.Series) -> pd.Series:
    """Final round stake share of positioned voters, per dispute."""
    pos = pd.Series([k in book.index for k in zip(fr["dispute_id"], fr["voter"])], index=fr.index)
    total = fr.groupby("dispute_id")["stake"].sum()
    return fr[pos].groupby("dispute_id")["stake"].sum().reindex(total.index, fill_value=0) / total


def publish_tally(v: pd.DataFrame, d: pd.DataFrame, shares: pd.Series) -> None:
    """Round tally and per dispute table, under one random dispute key."""
    if shares.max() >= POSITIONED_SHARE_BOUND:
        raise SystemExit("positioned share exceeds the published bound; raise POSITIONED_SHARE_BOUND")
    key = random_keys(v["dispute_id"])
    v = v.assign(dispute=v["dispute_id"].map(key),
                 round=v.groupby("dispute_id")["round_id"].rank(method="dense").astype(int))
    tally = (v.groupby(["dispute", "round", "vote_price"])
               .agg(n_votes=("stake", "size"), stake=("stake", "sum")).reset_index()
               .rename(columns={"vote_price": "price"}))
    tally["stake"] = tally["stake"].round(2)
    shuffled(tally).to_csv(OUT / "dvm_round_tally.csv", index=False)
    settled = d.set_index("dispute_tx_hash")["settled_price"]
    disp = pd.DataFrame({"dispute": [key[x] for x in shares.index],
                         "settled_price": [settled.loc[x] for x in shares.index],
                         "positioned_share_max": np.where(shares > 0, POSITIONED_SHARE_BOUND, 0.0)})
    shuffled(disp).to_csv(OUT / "dvm_disputes.csv", index=False)


def publish_own_book(annotated: pd.DataFrame) -> None:
    linked = annotated[annotated["has_position"]].copy()
    net = linked["yes_shares"] - linked["no_shares"]
    case = random_keys(linked["dispute_id"])
    out = pd.DataFrame({"case": linked["dispute_id"].map(case).values,
                        # VOID and OTHER are rare; merged so a row cannot point at a rare vote
                        "vote_side": linked["side"].replace({"VOID": "OTHER"}).values,
                        "position_side": np.where(net > 0, "YES", "NO")})
    shuffled(out).to_csv(OUT / "own_book_votes.csv", index=False)


def conflict_rate_sensitivity(annotated: pd.DataFrame) -> list[dict]:
    """Share of eligible voters with conflict_rate > 0.8, by minimum linked votes."""
    stats = compute_voter_stats(annotated)
    rows = []
    for min_linked in (1, 2, 3, 5):
        el = [s for s in stats.values()
              if s.total_votes >= MIN_VOTER_HISTORY and s.votes_with_position >= min_linked]
        above = [s for s in el if s.conflict_rate > 0.8]
        rows.append({"min_linked": min_linked, "eligible": len(el), "above_0_8": len(above),
                     "pooled_conflict_rate": sum(s.voted_own_book for s in el)
                     / sum(s.votes_with_position for s in el)})
    return rows


def coverage(v: pd.DataFrame, p: pd.DataFrame) -> dict:
    ok = p[p["status"] == "ok"]
    covered = p.loc[p["status"].isin(["ok", "no_voter_position"]), "dispute_id"].nunique()
    return {"disputes_with_resolved_condition": int(covered),
            "positions_by_adapter": {k: int(n) for k, n in ok["adapter"].value_counts().items()},
            "dropped_status": {k: int(n) for k, n in
                               p.loc[~p["status"].isin(["ok", "no_voter_position"]), "status"]
                               .value_counts().items()},
            "distinct_dvm_voters": int(v["voter"].nunique())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    raw = Path(parser.parse_args().raw)
    d, v, p = load_raw(raw)
    v["side"] = v["vote_price"].map(vote_side)
    publish_disputes(d)
    publish_too_early(raw)
    fr, book = final_round(v), book_sides(p)
    shares = positioned_shares(fr, book)
    publish_tally(v, d, shares)
    annotated = annotate_votes_with_positions(v, p[p["status"] == "ok"])
    publish_own_book(annotated)
    flip = exact_flip_test(fr, d.set_index("dispute_tx_hash")["settled_price"], book)
    summary = {"note": "computed from the raw tables before aggregation, "
                       "not recomputable from the shipped data",
               **coverage(v, p), "exact_flip_test": flip,
               "conflict_rate_sensitivity": conflict_rate_sensitivity(annotated)}
    (OUT / "summary_raw.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote published tables to {OUT}")


if __name__ == "__main__":
    main()
