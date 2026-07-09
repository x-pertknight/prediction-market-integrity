"""
scripts/backtest_uma_whale_edge.py — Backtest the preregistered whale-conflict edge.

Rule and accept/reject gates were fixed in
docs/experiments/2026-07-uma-whale-edge-backtest-preregistration.md and committed
BEFORE this script existed (commit fb66f4a). Nothing here may be tuned to the
result; post-hoc variants must be labelled as such.

Prices come from Polymarket's public data-api trade tape, VWAP'd over the 24h
before the dispute. Fees are the real per-category Polymarket taker schedule,
charged on entry only (redemption of a resolved position pays no taker fee).

Usage:
    python -u scripts/backtest_uma_whale_edge.py
    python -u scripts/backtest_uma_whale_edge.py --refresh-prices

Outputs: data/uma_whale_trades.csv, data/raw/uma_trade_tape.parquet (cache)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from alpha.uma_dispute import annotate_votes_with_positions, compute_voter_stats  # noqa: E402
from alpha.uma_whale_edge import (  # noqa: E402
    CONCENTRATION_FLAG, MIN_TRADABLE_DISPUTES, MIN_TRADES_IN_WINDOW, NOTIONAL,
    PRICE_WINDOW_HOURS, concentration, dispute_tilt, forecast_from_tilt,
    simulate_trade, split_halves, terminal_outcomes, vwap, whale_voters,
)
from data.polymarket_feature_adapter import infer_category  # noqa: E402
from risk.fees import polymarket_taker_fee  # noqa: E402
from risk.taleb import gain_to_pain, hill_estimator  # noqa: E402

DATA = ROOT / "data"
TAPE = DATA / "raw" / "uma_trade_tape.parquet"
TRADES_CSV = DATA / "uma_whale_trades.csv"
DATA_API = "https://data-api.polymarket.com/trades"
PAGE = 500
MAX_PAGES = 60


def fetch_tape(condition_ids: list[str], refresh: bool) -> pd.DataFrame:
    """Full trade tape for the conditions we might trade, cached to parquet."""
    if TAPE.exists() and not refresh:
        cached = pd.read_parquet(TAPE)
        missing = sorted(set(condition_ids) - set(cached["conditionId"].unique()))
        if not missing:
            print(f"[tape] reusing {TAPE.name} ({len(cached):,} trades)")
            return cached
    else:
        cached = pd.DataFrame()
        missing = condition_ids

    rows: list[dict] = []
    t0 = time.time()
    for i, cond in enumerate(missing, 1):
        for page in range(MAX_PAGES):
            try:
                resp = requests.get(DATA_API, params={"market": cond, "limit": PAGE,
                                                      "offset": page * PAGE}, timeout=30)
                resp.raise_for_status()
                batch = resp.json()
            except (requests.RequestException, ValueError) as exc:
                print(f"[tape] {cond[:12]} page {page} failed: {exc}")
                break
            rows.extend(batch)
            if len(batch) < PAGE:
                break
        if i % 25 == 0:
            print(f"  [tape] {i}/{len(missing)} markets  {len(rows):,} trades  "
                  f"{time.time() - t0:.0f}s", flush=True)
    fresh = pd.DataFrame(rows)
    tape = pd.concat([cached, fresh], ignore_index=True) if len(cached) else fresh
    if tape.empty:
        return tape
    tape = tape[["conditionId", "asset", "price", "size", "timestamp", "outcome", "title", "slug"]]
    TAPE.parent.mkdir(parents=True, exist_ok=True)
    tape.to_parquet(TAPE, compression="zstd", index=False)
    print(f"[tape] {len(tape):,} trades over {tape['conditionId'].nunique()} markets -> {TAPE}")
    return tape


def yes_price_window(tape_c: pd.DataFrame, t_end: int) -> tuple[float, int, str, str] | None:
    """VWAP of the YES side over the 24h before the dispute.

    A trade on the NO token prices YES at 1 - p; both sides are used, so a market
    that only printed NO in the window is not silently dropped.
    """
    t_start = t_end - PRICE_WINDOW_HOURS * 3600
    w = tape_c[(tape_c["timestamp"] >= t_start) & (tape_c["timestamp"] < t_end)]
    if len(w) < MIN_TRADES_IN_WINDOW:
        return None
    yes = np.where(w["outcome"].str.lower() == "yes", w["price"], 1.0 - w["price"])
    price = vwap(yes.astype(float), w["size"].to_numpy(dtype=float))
    if not 0.0 < price < 1.0:
        return None
    return price, len(w), w["title"].iloc[0], w["slug"].iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-prices", action="store_true")
    args = parser.parse_args()

    votes = pd.read_parquet(DATA / "uma_votes.parquet")
    positions = pd.read_csv(DATA / "uma_positions.csv")
    requests_df = pd.read_csv(DATA / "uma_dvm_requests.csv", dtype={"settled_price": str})
    ok = positions[positions["status"] == "ok"]

    # 1. Whales: the defensible set (>=3 position-linked votes).
    annotated = annotate_votes_with_positions(votes, ok)
    stats = compute_voter_stats(annotated)
    whales = whale_voters(stats)
    print(f"[backtest] {len(whales)} whale voters (>= {3} position-linked votes)")

    # 2. Terminal resolution per question. RESET is not an outcome.
    outcomes = terminal_outcomes(requests_df)
    print(f"[backtest] {len(outcomes)} questions reach a decided resolution")

    meta = requests_df.set_index("dispute_tx_hash")
    cond_by_q = ok.drop_duplicates("question_id").set_index("question_id")["condition_id"].to_dict()

    # 3. Candidate disputes: >=1 whale with net directional exposure.
    stake_by = annotated.set_index(["dispute_tx_hash", "voter"])["stake"].to_dict()
    candidates: list[dict] = []
    for dispute_tx, g in ok.groupby("dispute_tx_hash"):
        qid = g["question_id"].iloc[0]
        if qid not in outcomes:
            continue
        nets, stakes = {}, {}
        for _, r in g.iterrows():
            v = r["voter"]
            if v not in whales:
                continue
            nets[v] = r["yes_shares"] - r["no_shares"]
            stakes[v] = stake_by.get((dispute_tx, v), 0.0)
        if not nets:
            continue
        tilt = dispute_tilt(nets, stakes)
        if tilt is None:
            continue
        candidates.append({
            "dispute_tx_hash": dispute_tx, "question_id": qid,
            "condition_id": cond_by_q.get(qid, g["condition_id"].iloc[0]),
            "tilt": tilt, "n_whales": len(nets),
            "yes_payout": outcomes[qid],
            "dispute_time": pd.to_datetime(meta.loc[dispute_tx, "dispute_time"], format="mixed"),
        })
    print(f"[backtest] {len(candidates)} candidate disputes (whale signal + decided outcome)")
    if not candidates:
        raise SystemExit("[backtest] no candidates -> NOT TESTABLE")

    # 4. Prices.
    tape = fetch_tape(sorted({c["condition_id"] for c in candidates}), args.refresh_prices)
    if tape.empty:
        raise SystemExit("[backtest] empty trade tape -> NOT TESTABLE")
    by_cond = {c: g for c, g in tape.groupby("conditionId")}

    trades, no_price, no_edge = [], 0, 0
    for c in candidates:
        tc = by_cond.get(c["condition_id"])
        if tc is None:
            no_price += 1
            continue
        got = yes_price_window(tc, int(c["dispute_time"].timestamp()))
        if got is None:
            no_price += 1
            continue
        price, n_tr, title, slug = got
        category = infer_category({"slug": slug, "question": title})
        forecast = forecast_from_tilt(c["tilt"])
        trade = simulate_trade(
            c["dispute_tx_hash"], c["dispute_time"], forecast, price, c["yes_payout"],
            fee_per_share=lambda p, cat=category: polymarket_taker_fee(p, cat),
        )
        if trade is None:
            no_edge += 1
            continue
        trades.append((trade, c, n_tr, category, title))

    print(f"[backtest] {no_price} dropped for no price, {no_edge} below entry threshold")
    print(f"[backtest] {len(trades)} TRADABLE disputes")

    if not trades:
        raise SystemExit("[backtest] 0 tradable -> NOT TESTABLE")

    frame = pd.DataFrame([{
        "dispute_tx_hash": t.dispute_tx_hash, "dispute_time": t.dispute_time,
        "title": title[:70], "category": cat, "n_trades_window": n_tr,
        "n_whales": c["n_whales"], "tilt": c["tilt"], "forecast": t.forecast,
        "yes_price": t.price, "side": t.side, "entry_price": t.entry_price,
        "yes_payout": c["yes_payout"], "payout_per_share": t.payout_per_share,
        "fee": t.fee, "gross_pnl": t.gross_pnl, "net_pnl": t.net_pnl,
        "net_edge": t.net_edge,
    } for t, c, n_tr, cat, title in trades]).sort_values("dispute_time")
    frame.to_csv(TRADES_CSV, index=False)

    tl = [t for t, *_ in trades]
    report(frame, tl)


def report(frame: pd.DataFrame, trades: list) -> None:
    n = len(trades)
    pnl = frame["net_pnl"].to_numpy()
    edges = frame["net_edge"].to_numpy()

    print("\n" + "=" * 72)
    print("PREREGISTERED GATES")
    print("=" * 72)

    print(f"\n1. N tradable = {n}  (floor {MIN_TRADABLE_DISPUTES})")
    if n < MIN_TRADABLE_DISPUTES:
        print("   -> VERDICT: NOT TESTABLE. Reporting N only, per the preregistration.")
        print(f"      (for the record, and NOT to be read as a result: "
              f"net PnL ${pnl.sum():,.0f} on ${n * NOTIONAL:,.0f} notional)")
        return
    print("   -> PASS")

    hit = float((np.sign(frame["forecast"] - frame["yes_price"]) ==
                 np.sign(frame["yes_payout"] - frame["yes_price"])).mean())
    print(f"\n   sign accuracy (forecast direction vs realized): {hit:.4f}  (null 0.50)")
    print(f"   avg net edge per trade: {edges.mean():+.4f}  (${pnl.mean():+,.2f} on ${NOTIONAL:.0f})")
    print(f"   total net PnL: ${pnl.sum():+,.2f}   win rate: {(pnl > 0).mean():.3f}")
    print(f"   total fees paid: ${frame['fee'].sum():,.2f}")

    first, second = split_halves(trades)
    e1 = np.mean([t.net_edge for t in first])
    e2 = np.mean([t.net_edge for t in second])
    print(f"\n2. Temporal halves (split at median dispute_time)")
    print(f"   first  half: n={len(first):3d}  avg net edge {e1:+.4f}")
    print(f"   second half: n={len(second):3d}  avg net edge {e2:+.4f}")
    both = e1 > 0 and e2 > 0
    print(f"   -> {'PASS' if both else 'FAIL'} (needs > 0 in BOTH)")

    conc = concentration(pnl)
    print(f"\n3. Concentration: top-3 trades = "
          f"{'n/a (total net PnL <= 0)' if np.isinf(conc) else f'{conc:.1%}'} of net PnL")
    top = frame.nlargest(3, "net_pnl")[["dispute_time", "title", "net_pnl"]]
    for _, r in top.iterrows():
        print(f"     {str(r['dispute_time'])[:10]}  ${r['net_pnl']:+8,.0f}  {r['title'][:48]}")
    if not np.isinf(conc) and conc >= CONCENTRATION_FLAG:
        print(f"   -> FLAG: >= {CONCENTRATION_FLAG:.0%} of PnL is 3 disputes. Tail-driven.")

    alpha = hill_estimator(pnl)
    print(f"\n4. Hill alpha on net PnL: {alpha}")
    if not np.isfinite(alpha):
        # hill_estimator returns inf on INSUFFICIENT TAIL OBSERVATIONS, which is
        # not the same as thin tails. Never read that as licence to use Sharpe.
        print("   alpha not estimable (too few tail observations at q=0.95).")
        print("   Cannot certify alpha >= 2 -> Sharpe withheld, per project rule.")
        print(f"   gain-to-pain ratio: {gain_to_pain(pnl):.4f}")
    elif alpha < 2:
        print("   alpha < 2 -> fat tails; Sharpe is not meaningful.")
        print(f"   gain-to-pain ratio: {gain_to_pain(pnl):.4f}")
    else:
        print(f"   alpha >= 2 -> Sharpe admissible: {pnl.mean() / pnl.std(ddof=1):.4f}")

    print("\n" + "=" * 72)
    verdict = "TRADABLE (all gates pass)" if (both and n >= MIN_TRADABLE_DISPUTES and edges.mean() > 0) \
        else "NOT TRADABLE"
    print(f"VERDICT: {verdict}")
    print("=" * 72)

    diagnose_specification(frame)


def diagnose_specification(frame: pd.DataFrame) -> None:
    """POST-HOC. Not preregistered. Reported because the preregistered rule turned
    out not to test the hypothesis it was written for, and saying only "it lost
    money" would hide that."""
    from alpha.uma_whale_edge import BASE_YES

    print("\n" + "=" * 72)
    print("POST-HOC DIAGNOSTICS (not preregistered)")
    print("=" * 72)

    side_with = np.where(frame["forecast"] > frame["yes_price"], "YES", "NO")
    side_null = np.where(BASE_YES > frame["yes_price"], "YES", "NO")
    flips = int((side_with != side_null).sum())
    print(f"\n  The forecast is base_yes +- K*tilt, so it can only disagree with a price")
    print(f"  inside [{BASE_YES - 0.25:.2f}, {BASE_YES + 0.25:.2f}]. Disputed markets sit at extremes:")
    print(f"    yes_price  p25={frame['yes_price'].quantile(.25):.3f}  "
          f"median={frame['yes_price'].median():.3f}  p75={frame['yes_price'].quantile(.75):.3f}")
    print(f"  -> the whale tilt flipped the trade side in {flips}/{len(frame)} trades.")
    print(f"     The rule as preregistered was, in effect, 'fade every disputed market")
    print(f"     back toward {BASE_YES:.2f}'. It barely tested the whale signal at all.")

    decided = frame[frame["yes_payout"].isin([0.0, 1.0])]
    print(f"\n  Is the market simply right? corr(yes_price, outcome) = "
          f"{np.corrcoef(decided['yes_price'], decided['yes_payout'])[0, 1]:+.4f}")
    print(f"    mean price | resolved YES: {decided.loc[decided['yes_payout'] == 1, 'yes_price'].mean():.3f}")
    print(f"    mean price | resolved NO : {decided.loc[decided['yes_payout'] == 0, 'yes_price'].mean():.3f}")

    print(f"\n  THE ACTUAL HYPOTHESIS, rule-independent: does whale tilt predict the outcome?")
    print(f"    corr(tilt, outcome) = {np.corrcoef(decided['tilt'], decided['yes_payout'])[0, 1]:+.4f}")
    for lo, hi, label in [(-1.01, -1e-9, "whales net NO  (tilt<0)"),
                          (1e-9, 1.01, "whales net YES (tilt>0)")]:
        s = decided[(decided["tilt"] > lo) & (decided["tilt"] <= hi)]
        if len(s):
            print(f"    {label:26s} n={len(s):3d}  P(YES resolution) = {s['yes_payout'].mean():.3f}")
    print(f"    {'unconditional':26s} n={len(decided):3d}  P(YES resolution) = "
          f"{decided['yes_payout'].mean():.3f}")


if __name__ == "__main__":
    main()
