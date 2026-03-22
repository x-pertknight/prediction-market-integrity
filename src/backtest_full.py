"""
backtest_full.py
Full backtest on all WF predictions: weather_v2, general_v1, ensemble.
Entry: market_price <= pred_prob * 0.5
Exit:  binary resolution at outcome
P&L:  (outcome - entry_price) per $1 notional

Outputs:
  reports/backtest_full_{date}.md
  data/processed/trades_{label}.parquet
"""

import json
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

PROC_DIR    = Path("data/processed")
MODEL_DIR   = Path("models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_backtest(wf: pd.DataFrame, prob_col: str = "pred_prob") -> pd.DataFrame:
    """Filter to entry signals and compute P&L per trade."""
    df = wf.copy()
    df["_date"] = pd.to_datetime(df["close_date"])

    # Entry: model says YES prob is at least 2× the current market price (mispriced edge)
    if "market_price" not in df.columns:
        print("  [Backtest] No market_price column — cannot compute entry signal.")
        return pd.DataFrame()

    df["entry_signal"] = (df["market_price"] > 0) & \
                         (df["market_price"] <= df[prob_col] * 0.5)

    trades = df[df["entry_signal"]].copy()
    if trades.empty:
        return trades

    trades["entry_price"]    = trades["market_price"]
    trades["exit_price"]     = trades["outcome"].astype(float)
    trades["pnl"]            = trades["exit_price"] - trades["entry_price"]
    trades["roi"]            = trades["pnl"] / trades["entry_price"]
    trades["log_return"]     = np.log1p(trades["roi"].clip(lower=-0.9999))
    trades["win"]            = (trades["pnl"] > 0).astype(int)
    # MAE: max adverse excursion = entry_price when we lose (bought and got 0)
    trades["mae"]            = np.where(trades["outcome"] == 0, trades["entry_price"], 0)
    # MFE: max favorable excursion = (1 - entry_price) when we win (bought and got 1)
    trades["mfe"]            = np.where(trades["outcome"] == 1, 1 - trades["entry_price"], 0)

    trades = trades.sort_values("_date").reset_index(drop=True)
    trades["cumulative_pnl"] = trades["pnl"].cumsum()
    return trades


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    pnl = trades["pnl"]
    roi = trades["roi"]
    cum = trades["cumulative_pnl"] if "cumulative_pnl" in trades.columns \
          else trades.sort_values("_date")["pnl"].cumsum()
    mdd = (cum - cum.cummax()).min()

    return {
        "n_trades":       len(trades),
        "win_rate":       round(trades["win"].mean(), 4),
        "total_pnl":      round(pnl.sum(), 4),
        "mean_pnl":       round(pnl.mean(), 4),
        "sharpe_abs":     round(pnl.mean() / pnl.std() * np.sqrt(252), 4) if pnl.std() > 0 else 0,
        "mean_roi":       round(roi.mean(), 4),
        "sharpe_roi":     round(roi.mean() / roi.std() * np.sqrt(252), 4) if roi.std() > 0 else 0,
        "max_drawdown":   round(mdd, 4),
        "avg_entry":      round(trades["entry_price"].mean(), 4),
        "mean_mae":       round(trades["mae"].mean(), 4),
        "mean_mfe":       round(trades["mfe"].mean(), 4),
    }


def backtest_model(wf_path: str, label: str, prob_col: str = "pred_prob",
                   group_col: str | None = None) -> dict:
    """Run backtest on one WF parquet, return metrics dict."""
    f = PROC_DIR / wf_path
    if not f.exists():
        print(f"  [Backtest] {wf_path} not found, skipping.")
        return {}

    wf = pd.read_parquet(f)

    # Deduplicate by market for ensemble (one row per market, use ensemble prob)
    if prob_col == "pred_prob_ensemble":
        key_cols = ["close_date", "series_ticker", "outcome"]
        # Keep first occurrence after taking the ensemble prob from any row
        wf = wf.drop_duplicates(subset=key_cols, keep="first")

    trades = run_backtest(wf, prob_col)
    if trades.empty:
        print(f"  [{label}] No entry signals generated.")
        return {}

    # Save trades
    out = PROC_DIR / f"trades_{label.replace(' ', '_').lower()}.parquet"
    trades.to_parquet(out, index=False)

    ov = metrics(trades)
    print(f"\n[{label}] ── Overall ─────────────────────────────")
    print(f"  Trades: {ov['n_trades']} | Win: {ov['win_rate']:.1%} | "
          f"P&L: {ov['total_pnl']:+.2f} | Sharpe: {ov['sharpe_abs']:+.2f} | "
          f"MDD: {ov['max_drawdown']:+.2f}")

    # Per-group breakdown
    breakdown = {}
    gcol = group_col or ("category" if "category" in trades.columns
                         else "city" if "city" in trades.columns
                         else None)
    if gcol and gcol in trades.columns:
        print(f"\n[{label}] ── Per {gcol} ────────────────────────────")
        for grp_name, grp in sorted(trades.groupby(gcol)):
            grp = grp.copy()
            grp["cumulative_pnl"] = grp.sort_values("_date")["pnl"].cumsum()
            m = metrics(grp)
            breakdown[str(grp_name)] = m
            print(f"  {str(grp_name):<22} n={m['n_trades']:>4} | "
                  f"win={m['win_rate']:.0%} | pnl={m['total_pnl']:+.2f} | "
                  f"sharpe={m['sharpe_abs']:+.2f} | mdd={m['max_drawdown']:+.2f}")

    return {"label": label, "overall": ov, "breakdown": breakdown}


def load_shap_features() -> dict[str, dict]:
    """Load SHAP importance dicts for all trained models."""
    shap_data = {}
    for f in MODEL_DIR.glob("shap_*.json"):
        name = f.stem.replace("shap_", "")
        try:
            shap_data[name] = json.loads(f.read_text())
        except Exception:
            pass
    return shap_data


def md_table(headers: list[str], rows: list[list]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line    = "|" + "|".join(["---"] * len(headers)) + "|"
    body        = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([header_line, sep_line, body])


def generate_report(results: dict, shap_data: dict) -> None:
    today = date.today().isoformat()
    out   = REPORTS_DIR / f"backtest_full_{today}.md"

    lines = [
        "# TrapTrade — Full Backtest Report",
        f"**Generated:** {today}",
        "**Mode:** Backtest only — hypothetical P&L, zero real trades",
        "",
    ]

    # Overall summary table
    summary_rows = []
    for label, res in results.items():
        if not res or "overall" not in res:
            continue
        ov = res["overall"]
        summary_rows.append([
            label,
            ov.get("n_trades", 0),
            f"{ov.get('win_rate', 0):.1%}",
            f"{ov.get('total_pnl', 0):+.2f}",
            f"{ov.get('sharpe_abs', 0):+.2f}",
            f"{ov.get('max_drawdown', 0):+.2f}",
            f"{ov.get('mean_mae', 0):.4f}",
            f"{ov.get('mean_mfe', 0):.4f}",
        ])

    if summary_rows:
        lines += [
            "## Summary",
            "",
            md_table(
                ["Model", "Trades", "Win Rate", "Total P&L", "Sharpe", "MaxDD", "Avg MAE", "Avg MFE"],
                summary_rows
            ),
            "",
        ]

    # Per-model details
    for label, res in results.items():
        if not res or "overall" not in res:
            continue
        ov = res["overall"]
        bk = res.get("breakdown", {})

        lines += [
            "---",
            f"## {label}",
            "",
            "### Overall Metrics",
            "",
            md_table(["Metric", "Value"], [[k, v] for k, v in ov.items()]),
            "",
        ]

        if bk:
            bk_rows = [
                [grp,
                 m.get("n_trades", 0),
                 f"{m.get('win_rate', 0):.0%}",
                 f"{m.get('total_pnl', 0):+.2f}",
                 f"{m.get('sharpe_abs', 0):+.2f}",
                 f"{m.get('max_drawdown', 0):+.2f}"]
                for grp, m in sorted(bk.items())
                if m.get("n_trades", 0) > 0
            ]
            if bk_rows:
                lines += [
                    "### Breakdown by Category",
                    "",
                    md_table(["Group", "Trades", "Win Rate", "P&L", "Sharpe", "MaxDD"], bk_rows),
                    "",
                ]

    # SHAP top features
    if shap_data:
        lines += ["---", "## Top SHAP Features (by model)", ""]
        for model_name, shap_dict in sorted(shap_data.items()):
            top = list(shap_dict.items())[:20]
            lines += [
                f"### {model_name}",
                "",
                md_table(["Feature", "Mean |SHAP|"], [[f, f"{v:.4f}"] for f, v in top]),
                "",
            ]

    lines += [
        "---",
        "## Caveats",
        "- All P&L is hypothetical. No real money involved.",
        "- Kalshi demo API prices may differ from live bid/ask at entry time.",
        "- No fee model (Kalshi charges ~$0.07/contract round-trip).",
        "- Walk-forward has limited folds due to data starting 2025-04.",
        "- Generic model trained on all categories — predictive power varies by category.",
    ]

    out.write_text("\n".join(lines))
    print(f"\n[Report] → {out}")


def run() -> None:
    results = {}

    # Weather ensemble model
    results["Weather v2"] = backtest_model(
        wf_path  = "wf_weather_v2.parquet",
        label    = "Weather v2",
        group_col = "city",
    )

    # Crypto specialist
    results["Crypto v1"] = backtest_model(
        wf_path  = "wf_crypto_v1.parquet",
        label    = "Crypto v1",
        group_col = "category",
    )

    # General model (model_dual saves wf_generic_v1, model_ensemble saves wf_general_v1)
    gen_path = "wf_general_v1.parquet"
    if not (PROC_DIR / gen_path).exists():
        gen_path = "wf_generic_v1.parquet"   # fallback from model_dual
    results["General v1"] = backtest_model(
        wf_path  = gen_path,
        label    = "General v1",
        group_col = "category",
    )

    # Full ensemble
    results["Ensemble"] = backtest_model(
        wf_path  = "wf_ensemble.parquet",
        label    = "Ensemble",
        prob_col = "pred_prob_ensemble",
        group_col = "category",
    )

    shap_data = load_shap_features()
    generate_report(results, shap_data)


if __name__ == "__main__":
    run()
