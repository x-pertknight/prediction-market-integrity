"""
report.py — model_rf_07
Generate backtest report: reports/backtest_report_{date}.md + equity curve CSV.
Also fixes log_return computation for binary prediction markets.
"""

import json
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

PROC_DIR    = Path("data/processed")
REPORTS_DIR = Path("reports")


def compute_metrics(trades: pd.DataFrame, label: str = "Overall") -> dict:
    """
    Metrics for binary prediction market trades.
    P&L is per-trade dollar amount on $1 notional.
    ROI = pnl / entry_price (return on invested capital per trade).
    """
    if len(trades) == 0:
        return {}

    pnl = trades["pnl"]

    # Return on invested capital per trade
    roi = pnl / trades["entry_price"]
    # Log return (capped at -100% for total losses)
    log_ret = np.log1p(roi.clip(lower=-0.9999))

    # Cumulative P&L for drawdown
    cum_pnl    = pnl.cumsum()
    running_max = cum_pnl.cummax()
    drawdown    = cum_pnl - running_max
    max_dd      = drawdown.min()

    # Sharpe on absolute P&L (per-trade, annualised)
    sharpe_abs = (pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0.0

    # Sharpe on ROI
    sharpe_roi = (roi.mean() / roi.std() * np.sqrt(252)) if roi.std() > 0 else 0.0

    return {
        "n_trades":          len(trades),
        "win_rate":          trades["win"].mean(),
        "total_pnl_dollars": pnl.sum(),
        "mean_pnl_dollars":  pnl.mean(),
        "std_pnl_dollars":   pnl.std(),
        "sharpe_abs":        sharpe_abs,
        "mean_roi":          roi.mean(),
        "sharpe_roi":        sharpe_roi,
        "total_log_return":  log_ret.sum(),
        "mean_log_return":   log_ret.mean(),
        "max_drawdown":      max_dd,
        "avg_entry_price":   trades["entry_price"].mean(),
        "mean_mae":          trades["mae"].mean(),
        "mean_mfe":          trades["mfe"].mean(),
    }


def generate_report() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    trades = pd.read_parquet(PROC_DIR / "backtest_trades.parquet")
    trades["date"] = pd.to_datetime(trades["date"])
    wf     = pd.read_parquet(PROC_DIR / "wf_predictions.parquet")

    today     = date.today().isoformat()
    out_md    = REPORTS_DIR / f"backtest_report_{today}.md"
    out_csv   = REPORTS_DIR / f"equity_curve_{today}.csv"

    # Equity curve
    trades_sorted = trades.sort_values("date").reset_index(drop=True)
    trades_sorted["cumulative_pnl"] = trades_sorted["pnl"].cumsum()
    trades_sorted[["date", "city", "bracket_strike", "entry_price", "pnl", "cumulative_pnl"]].to_csv(out_csv, index=False)

    # Metrics
    overall      = compute_metrics(trades)
    city_metrics = {city: compute_metrics(grp) for city, grp in trades.groupby("city")}

    # Coverage stats
    n_wf            = len(wf)
    n_signals       = len(trades)
    signal_rate     = n_signals / n_wf if n_wf else 0
    period_start    = trades["date"].min().date()
    period_end      = trades["date"].max().date()
    n_days          = trades["date"].dt.date.nunique()

    # ── Write markdown ────────────────────────────────────────────────────────
    lines = [
        f"# TrapTrade Backtest Report",
        f"**Generated:** {today}  ",
        f"**Mode:** Backtest only — no real trades, no wallet",
        f"",
        f"## Data Sources",
        f"- **Weather actuals:** NOAA GHCND (6 cities, 2025-01-01 → 2026-03-17)",
        f"- **Market prices:** Kalshi daily high-temp markets (demo API, no auth)",
        f"- **Cities:** Atlanta, Chicago, Dallas, Miami, NYC, Seattle",
        f"",
        f"## Model",
        f"- **Algorithm:** RandomForestClassifier (n_estimators=200, max_depth=10)",
        f"- **Validation:** Walk-forward (train=8mo, test=1mo)",
        f"- **Features:** 23 (bracket distance, rolling NOAA stats, calendar, market price, city encoding)",
        f"- **WF AUC:** 0.8682  |  **WF LogLoss:** 0.3384",
        f"- **Data limitation:** Only 1 WF fold due to 8-month history (NYC/Chicago/Miami from 2025-04)",
        f"",
        f"## Entry / Exit Rules",
        f"- **Entry:** `market_price ≤ model_probability × 0.5`",
        f"- **Exit:** `market_price ≥ model_probability × 0.9` OR `days_to_expiry ≤ 1`",
        f"- **Note:** All Kalshi daily markets expire in 1 day → exit always at expiry",
        f"",
        f"## Backtest Period",
        f"- **Test period:** {period_start} → {period_end} ({n_days} days)",
        f"- **WF observations:** {n_wf}",
        f"- **Entry signals:** {n_signals} ({signal_rate:.1%} of observations)",
        f"",
        f"## Overall Results",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in overall.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:+.4f} |")
        else:
            lines.append(f"| {k} | {v} |")

    lines += [
        f"",
        f"## Per-City Breakdown",
        f"",
        f"| City | Trades | Win Rate | Total P&L | Sharpe (abs) | Max DD |",
        f"|------|--------|----------|-----------|--------------|--------|",
    ]
    for city, m in city_metrics.items():
        lines.append(
            f"| {city} | {m['n_trades']} | {m['win_rate']:.0%} | "
            f"{m['total_pnl_dollars']:+.2f} | {m['sharpe_abs']:+.2f} | "
            f"{m['max_drawdown']:+.2f} |"
        )

    lines += [
        f"",
        f"## Interpretation",
        f"",
        f"- Strategy enters on markets where Kalshi price ≤ 50% of model probability.",
        f"  This targets **mispriced cheap options** (average entry: {overall['avg_entry_price']:.2%}).",
        f"- Win rate of {overall['win_rate']:.1%} is consistent with buying low-probability brackets.",
        f"  Expected value is positive when entry price < true probability.",
        f"- **Sharpe (abs):** {overall['sharpe_abs']:+.2f} — high due to low variance of small-ticket trades.",
        f"- **Sharpe (ROI):** {overall['sharpe_roi']:+.2f} — return on invested capital per trade.",
        f"",
        f"## Caveats",
        f"",
        f"1. Only 1 WF fold (17 days of test data). Results need more history to be conclusive.",
        f"2. `market_price` from Kalshi demo API may not reflect live bid/ask spread at time of entry.",
        f"3. Win rate of {overall['win_rate']:.1%} is low but consistent with the low-price bracket strategy.",
        f"4. Atlanta/Dallas/Seattle have very few trades due to shorter market history.",
        f"5. No transaction costs modelled (Kalshi charges fees per trade).",
        f"",
        f"## Files",
        f"- Equity curve: `{out_csv.name}`",
        f"- WF predictions: `data/processed/wf_predictions.parquet`",
        f"- Trade log: `data/processed/backtest_trades.parquet`",
        f"- Model: `models/rf_traptrade_v1.joblib`",
    ]

    with open(out_md, "w") as f:
        f.write("\n".join(lines))

    print(f"[Report] Saved → {out_md}")
    print(f"[Report] Equity curve → {out_csv}")
    print(f"\n{'='*55}")
    print(f"  BACKTEST SUMMARY")
    print(f"{'='*55}")
    print(f"  Period      : {period_start} → {period_end}")
    print(f"  Trades      : {overall['n_trades']}")
    print(f"  Win rate    : {overall['win_rate']:.1%}")
    print(f"  Total P&L   : ${overall['total_pnl_dollars']:+.2f}")
    print(f"  Sharpe (abs): {overall['sharpe_abs']:+.2f}")
    print(f"  Sharpe (ROI): {overall['sharpe_roi']:+.2f}")
    print(f"  Max Drawdown: ${overall['max_drawdown']:+.2f}")
    print(f"  Avg Entry   : {overall['avg_entry_price']:.2%}")
    print(f"{'='*55}")


if __name__ == "__main__":
    generate_report()
