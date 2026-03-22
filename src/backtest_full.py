"""
backtest_full.py
Run entry/exit backtest on WF predictions from both models.
Entry: market_price <= pred_prob * 0.5
Exit:  market_price >= pred_prob * 0.9 OR days_to_expiry <= 1
P&L: binary binary → (outcome - entry_price) per $1 notional
Output: reports/backtest_full_{date}.md + data/processed/trades_{name}.parquet
"""

import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

PROC_DIR    = Path("data/processed")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_backtest(wf: pd.DataFrame, label: str) -> pd.DataFrame:
    df = wf.copy()
    df["_date"]        = pd.to_datetime(df["close_date"])
    days_in_market_col = "days_in_market" if "days_in_market" in df.columns else None

    df["entry_signal"] = df["market_price"] <= df["pred_prob"] * 0.5
    df["exit_signal"]  = (
        (df["market_price"] >= df["pred_prob"] * 0.9)
        | (True if days_in_market_col is None else df[days_in_market_col] <= 1)
    )

    trades = df[df["entry_signal"]].copy()
    if trades.empty:
        return trades

    trades["entry_price"]    = trades["market_price"]
    trades["exit_price"]     = trades["outcome"].astype(float)
    trades["pnl"]            = trades["exit_price"] - trades["entry_price"]
    trades["roi"]            = trades["pnl"] / trades["entry_price"]
    trades["log_return"]     = np.log1p(trades["roi"].clip(lower=-0.9999))
    trades["win"]            = (trades["pnl"] > 0).astype(int)
    trades["mae"]            = np.where(trades["outcome"] == 0, trades["entry_price"], 0)
    trades["mfe"]            = np.where(trades["outcome"] == 1, 1 - trades["entry_price"], 0)
    trades["cumulative_pnl"] = trades.sort_values("_date")["pnl"].cumsum()
    return trades


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    pnl = trades["pnl"]
    roi = trades["roi"]
    cum = trades.sort_values("_date")["pnl"].cumsum()
    mdd = (cum - cum.cummax()).min()
    return {
        "n_trades":          len(trades),
        "win_rate":          trades["win"].mean(),
        "total_pnl":         pnl.sum(),
        "mean_pnl":          pnl.mean(),
        "std_pnl":           pnl.std(),
        "sharpe_abs":        pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0,
        "mean_roi":          roi.mean(),
        "sharpe_roi":        roi.mean() / roi.std() * np.sqrt(252) if roi.std() > 0 else 0,
        "total_log_return":  trades["log_return"].sum(),
        "max_drawdown":      mdd,
        "avg_entry_price":   trades["entry_price"].mean(),
        "mean_mae":          trades["mae"].mean(),
        "mean_mfe":          trades["mfe"].mean(),
    }


def format_table(data: dict) -> str:
    lines = ["| Metric | Value |", "|--------|-------|"]
    for k, v in data.items():
        val = f"{v:+.4f}" if isinstance(v, float) else str(v)
        lines.append(f"| {k} | {val} |")
    return "\n".join(lines)


def backtest_and_report(wf_path: str, label: str, category_col: str | None = None) -> dict:
    wf_file = PROC_DIR / wf_path
    if not wf_file.exists():
        print(f"  [Backtest] {wf_path} not found, skipping.")
        return {}

    wf     = pd.read_parquet(wf_file)
    trades = run_backtest(wf, label)
    if trades.empty:
        print(f"  [{label}] No trades generated.")
        return {}

    out_trades = PROC_DIR / f"trades_{label.replace(' ','_').lower()}.parquet"
    trades.to_parquet(out_trades, index=False)

    overall = metrics(trades)
    print(f"\n[{label}] ── Results ──────────────────────────────────")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:+.4f}")
        else:
            print(f"  {k:<22} {v}")

    # Per-category / per-city breakdown
    breakdown = {}
    group_col = category_col or ("city" if "city" in trades.columns else "series_ticker")
    if group_col in trades.columns:
        print(f"\n[{label}] ── Per {group_col} ────────────────────────────────")
        for grp_name, grp in trades.groupby(group_col):
            m = metrics(grp)
            breakdown[str(grp_name)] = m
            print(f"  {str(grp_name):<18} n={m['n_trades']:>4} | "
                  f"win={m['win_rate']:.0%} | pnl={m['total_pnl']:+.2f} | "
                  f"sharpe={m['sharpe_abs']:+.2f} | mdd={m['max_drawdown']:+.2f}")

    return {"overall": overall, "breakdown": breakdown, "trades": trades}


def generate_combined_report(results: dict) -> None:
    today = date.today().isoformat()
    out   = REPORTS_DIR / f"backtest_full_{today}.md"

    lines = [
        f"# TrapTrade Full Backtest Report",
        f"**Generated:** {today}  ",
        f"**Mode:** Backtest only — hypothetical P&L, no real trades",
        f"",
    ]

    for label, res in results.items():
        if not res or "overall" not in res:
            continue
        ov = res["overall"]
        lines += [
            f"---",
            f"## {label}",
            f"",
            format_table(ov),
            f"",
        ]
        if res.get("breakdown"):
            bk = res["breakdown"]
            lines += [
                f"### Breakdown",
                f"| Group | Trades | Win Rate | P&L | Sharpe | MaxDD |",
                f"|-------|--------|----------|-----|--------|-------|",
            ]
            for grp, m in bk.items():
                if not m:
                    continue
                lines.append(
                    f"| {grp} | {m['n_trades']} | {m['win_rate']:.0%} | "
                    f"{m['total_pnl']:+.2f} | {m['sharpe_abs']:+.2f} | "
                    f"{m['max_drawdown']:+.2f} |"
                )
            lines.append("")

    lines += [
        f"---",
        f"## Caveats",
        f"- All P&L is hypothetical. No real money involved.",
        f"- Kalshi demo API prices may differ from live bid/ask at entry time.",
        f"- No fee model (Kalshi charges ~$0.07/contract round-trip).",
        f"- Generic model trained on all categories: predictive power varies by category.",
    ]

    out.write_text("\n".join(lines))
    print(f"\n[Report] Full report → {out}")


def run() -> None:
    results = {}

    results["Weather Model (NOAA + Kalshi)"] = backtest_and_report(
        wf_path      = "wf_weather_v2.parquet",
        label        = "Weather",
        category_col = "city",
    )

    results["Generic Model (All Markets)"] = backtest_and_report(
        wf_path      = "wf_generic_v1.parquet",
        label        = "Generic",
        category_col = "category",
    )

    generate_combined_report(results)


if __name__ == "__main__":
    run()
