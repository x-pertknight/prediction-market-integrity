"""
backtest.py — model_rf_05 + model_rf_06
Apply entry/exit rules to WF predictions, compute backtest metrics.
Entry:  market_price <= model_prob * 0.5
Exit:   market_price >= model_prob * 0.9  OR  days_to_expiry <= 1
Since Kalshi markets are daily (days_to_expiry=1), most exits trigger at expiry.
P&L: binary outcome — if outcome=1, P&L = (1 - entry_price); else P&L = -entry_price.
Output: data/processed/backtest_trades.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROC_DIR = Path("data/processed")


def run_backtest(wf: pd.DataFrame) -> pd.DataFrame:
    """
    Apply entry/exit rules to walk-forward predictions.
    Returns trade log with P&L per trade.
    """
    df = wf.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Entry signal ──────────────────────────────────────────────────────────
    # Buy YES if market underprices the model's probability by at least 50%
    df["entry_signal"] = df["market_price"] <= df["pred_prob"] * 0.5

    # ── Exit signal ───────────────────────────────────────────────────────────
    # For daily Kalshi markets, days_to_expiry=1 → always exit at expiry
    df["exit_signal"] = (
        (df["market_price"] >= df["pred_prob"] * 0.9)
        | (df["days_to_expiry"] <= 1)
    )

    # ── Build trade log ───────────────────────────────────────────────────────
    trades = df[df["entry_signal"]].copy()
    trades["entry_price"] = trades["market_price"]

    # For daily markets: exit price is 1.0 if outcome=yes, 0.0 if outcome=no
    trades["exit_price"] = trades["outcome"].astype(float)

    # P&L per $1 notional
    trades["pnl"]       = trades["exit_price"] - trades["entry_price"]
    trades["log_return"] = np.log(
        (trades["exit_price"] + 1e-9) / (trades["entry_price"] + 1e-9)
    )
    trades["win"] = (trades["pnl"] > 0).astype(int)

    # MAE / MFE (for daily binary markets: simplified)
    # Adverse excursion = max loss if outcome is no
    trades["mae"] = np.where(trades["outcome"] == 0, trades["entry_price"], 0)
    # Favorable excursion = max gain if outcome is yes
    trades["mfe"] = np.where(trades["outcome"] == 1, 1 - trades["entry_price"], 0)

    trades = trades.reset_index(drop=True)
    return trades


def compute_metrics(trades: pd.DataFrame) -> dict:
    """Compute Sharpe Ratio, win rate, log returns, drawdown."""
    if len(trades) == 0:
        return {"error": "no trades"}

    pnl  = trades["pnl"]
    logs = trades["log_return"]

    # Cumulative P&L
    trades["cumulative_pnl"] = pnl.cumsum()

    # Drawdown
    running_max   = trades["cumulative_pnl"].cummax()
    drawdown      = trades["cumulative_pnl"] - running_max
    max_drawdown  = drawdown.min()

    # Sharpe (annualised assuming ~250 trading days, daily trades)
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0

    metrics = {
        "n_trades":       len(trades),
        "win_rate":       trades["win"].mean(),
        "total_pnl":      pnl.sum(),
        "mean_pnl":       pnl.mean(),
        "std_pnl":        pnl.std(),
        "sharpe_ratio":   sharpe,
        "total_log_return": logs.sum(),
        "mean_log_return":  logs.mean(),
        "max_drawdown":   max_drawdown,
        "mean_mae":       trades["mae"].mean(),
        "mean_mfe":       trades["mfe"].mean(),
        "avg_entry_price": trades["entry_price"].mean(),
    }
    return metrics


def run() -> tuple[pd.DataFrame, dict, dict]:
    wf = pd.read_parquet(PROC_DIR / "wf_predictions.parquet")
    print(f"[Backtest] WF predictions: {len(wf)} rows")
    print(f"           Period: {wf['date'].min().date()} → {wf['date'].max().date()}")
    print(f"           Cities: {sorted(wf['city'].unique())}")

    trades = run_backtest(wf)
    print(f"\n[Backtest] Entry signals triggered: {len(trades)} trades")

    if len(trades) == 0:
        print("[Backtest] No trades generated — entry threshold too strict.")
        return trades, {}, {}

    # ── Overall metrics ───────────────────────────────────────────────────────
    overall = compute_metrics(trades)
    print("\n[Backtest] ── Overall ──────────────────────────────")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:+.4f}")
        else:
            print(f"  {k:<22} {v}")

    # ── Per-city metrics ──────────────────────────────────────────────────────
    city_metrics = {}
    print("\n[Backtest] ── Per City ─────────────────────────────")
    for city, grp in trades.groupby("city"):
        m = compute_metrics(grp)
        city_metrics[city] = m
        print(f"  {city:<10} n={m['n_trades']:>3} | win={m['win_rate']:.0%} | "
              f"pnl={m['total_pnl']:+.2f} | sharpe={m['sharpe_ratio']:+.2f} | "
              f"mdd={m['max_drawdown']:+.2f}")

    # Save trades
    out = PROC_DIR / "backtest_trades.parquet"
    trades.to_parquet(out, index=False)
    print(f"\n[Backtest] Saved {len(trades)} trades → {out}")

    return trades, overall, city_metrics


if __name__ == "__main__":
    run()
