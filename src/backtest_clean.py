"""
backtest_clean.py — Honest backtest without lookahead bias
==========================================================
Fixes applied vs the original backtest_full.py:

  Fix 1 — Entry price
    Uses price_lag1 (previous market's settlement price in the same series)
    as the entry price proxy. This is observable when the current market opens:
    you can look up what the previous contract in the series traded for.
    NOT used: last_price_dollars (settlement-time price of the CURRENT market).

  Fix 2 — Entry signal
    Signal: price_lag1 <= pred_prob_clean * EDGE_MULT
    pred_prob_clean comes from model_clean.py (no market_price in features).

  Fix 3 — Macro features on open_date
    Features used by the clean model were already joined on open_date
    (done in features_clean.py), so this fix is upstream.

  Remaining caveat:
    price_lag1 is still derived from last_price_dollars of the PREVIOUS
    market. For very long-duration markets (Politics, Elections >30 days),
    the previous market's final price may have been set weeks before the
    current market opens. This is documented but not fully resolved without
    the production OHLCV API.
    See kalshi_market_history.py for infrastructure to fix this with auth.

Outputs:
  reports/backtest_clean_{date}.md
  data/processed/trades_clean.parquet
  data/processed/trades_clean_weather.parquet
"""

import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

PROC_DIR    = Path("data/processed")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EDGE_MULT = 0.5   # enter when market_price <= pred_prob * 0.5 (2× edge)

# ── Point 5: Fee models ───────────────────────────────────────────────────────
# PnL is in [0, 1] space (entry_price and exit_price are fractions).
# Kalshi: $0.07 per contract per side → 0.07 total round-trip per $1 nominal.
# Polymarket: 2% of each transaction value.
FEE_MODELS: dict[str, dict] = {
    "none":       {"taker": 0.00, "type": "flat"},
    "kalshi":     {"taker": 0.07, "type": "flat"},   # $0.07/contract round-trip
    "polymarket": {"taker": 0.02, "type": "pct"},    # 2% per leg
}

# Long-duration categories where price_lag1 is a weaker proxy
LONG_DURATION_CATS = {"Politics", "Elections", "Companies", "Science and Technology",
                      "Health", "Education", "Social", "World", "Exotics"}


def apply_fees(pnl: pd.Series, entry: pd.Series, exit_: pd.Series,
               fee_model: str = "none") -> pd.Series:
    """Subtract round-trip fees from PnL series."""
    fm = FEE_MODELS.get(fee_model, FEE_MODELS["none"])
    if fm["type"] == "flat":
        return pnl - fm["taker"]  # flat cost per contract (Kalshi: $0.07)
    else:  # pct
        return pnl - entry * fm["taker"] - exit_ * fm["taker"]


# ── Point 6: Bootstrap CI ─────────────────────────────────────────────────────

def bootstrap_sharpe_ci(returns: pd.Series, n: int = 1000, ci: float = 0.95) -> dict:
    """Bootstrap confidence interval on annualised Sharpe (√252 scaling)."""
    r = np.array(returns.dropna())
    if len(r) < 10 or r.std() == 0:
        s = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
        return {"sharpe": s, "ci_lower": s, "ci_upper": s, "n_bootstrap": 0}

    rng     = np.random.default_rng(42)
    sharpes = []
    for _ in range(n):
        sample = rng.choice(r, size=len(r), replace=True)
        if sample.std() > 0:
            sharpes.append(sample.mean() / sample.std() * np.sqrt(252))

    sharpes = np.array(sharpes)
    alpha   = (1 - ci) / 2
    return {
        "sharpe":      float(np.median(sharpes)),
        "ci_lower":    float(np.percentile(sharpes, alpha * 100)),
        "ci_upper":    float(np.percentile(sharpes, (1 - alpha) * 100)),
        "n_bootstrap": n,
    }


# ── Point 7: Robustness tests ─────────────────────────────────────────────────

def run_robustness_tests(trades: pd.DataFrame, wf: pd.DataFrame | None = None,
                         prob_col: str = "pred_prob_clean") -> dict:
    """
    4 robustness checks:
      1. First vs second half of time series
      2. Exclude best-Sharpe category
      3. Threshold perturbation ±20%
      4. 50% random data removal
    Returns dict of {test_name: metrics_dict}.
    """
    if trades.empty:
        return {}

    results: dict[str, dict] = {}

    # 1. First/second half split
    mid = len(trades) // 2
    first  = trades.iloc[:mid].copy()
    second = trades.iloc[mid:].copy()
    for half, label in [(first, "first_half"), (second, "second_half")]:
        half["cumulative_pnl"] = half["pnl"].cumsum()
        results[label] = metrics(half, label)

    # 2. Exclude best category
    if "category" in trades.columns:
        cat_sharpes = {}
        for cat, grp in trades.groupby("category"):
            if len(grp) >= 5:
                pnl = grp["pnl"]
                if pnl.std() > 0:
                    cat_sharpes[cat] = pnl.mean() / pnl.std() * np.sqrt(252)
        if cat_sharpes:
            best_cat = max(cat_sharpes, key=cat_sharpes.__getitem__)
            wo_best  = trades[trades["category"] != best_cat].copy()
            wo_best["cumulative_pnl"] = wo_best["pnl"].cumsum()
            results[f"excl_best_cat({best_cat[:15]})"] = metrics(wo_best, f"excl_{best_cat}")

    # 3. Threshold perturbation ±20% — re-run signal on wf if available
    if wf is not None and prob_col in wf.columns and "price_lag1" in wf.columns:
        for delta_pct, tag in [(-0.2, "threshold_minus20"), (0.2, "threshold_plus20")]:
            mult = EDGE_MULT * (1 + delta_pct)
            df_t = wf.copy()
            df_t["_date"] = pd.to_datetime(df_t["close_date"])
            sig  = (df_t["price_lag1"] > 0) & \
                   (df_t["price_lag1"] <= df_t[prob_col] * mult)
            t_df = df_t[sig].copy()
            if not t_df.empty:
                t_df["entry_price"] = t_df["price_lag1"]
                t_df["exit_price"]  = t_df["outcome"].astype(float)
                t_df["pnl"]         = t_df["exit_price"] - t_df["entry_price"]
                t_df["roi"]         = t_df["pnl"] / t_df["entry_price"].clip(lower=1e-6)
                t_df["win"]         = (t_df["pnl"] > 0).astype(int)
                t_df["mae"]         = np.where(t_df["outcome"] == 0, t_df["entry_price"], 0)
                t_df["mfe"]         = np.where(t_df["outcome"] == 1, 1 - t_df["entry_price"], 0)
                t_df = t_df.sort_values("_date").reset_index(drop=True)
                t_df["cumulative_pnl"] = t_df["pnl"].cumsum()
                results[tag] = metrics(t_df, tag)

    # 4. 50% random removal
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(trades), size=len(trades) // 2, replace=False)
    half = trades.iloc[sorted(idx)].copy()
    half["cumulative_pnl"] = half["pnl"].cumsum()
    results["random_50pct"] = metrics(half, "random_50pct")

    return results


def run_backtest_clean(wf: pd.DataFrame, prob_col: str = "pred_prob_clean",
                       label: str = "clean", fee_model: str = "none") -> pd.DataFrame:
    """
    Clean backtest: entry at price_lag1, signal on pred_prob_clean.
    Returns trades DataFrame.
    """
    df = wf.copy()
    df["_date"] = pd.to_datetime(df["close_date"])

    if "price_lag1" not in df.columns:
        print(f"  [{label}] No price_lag1 column — cannot run clean backtest.")
        return pd.DataFrame()
    if prob_col not in df.columns:
        print(f"  [{label}] No {prob_col} column — skipping.")
        return pd.DataFrame()

    # Entry signal: price_lag1 (observable entry proxy) <= pred_prob_clean * 0.5
    df["entry_signal_clean"] = (
        (df["price_lag1"] > 0) &
        (df["price_lag1"] <= df[prob_col] * EDGE_MULT)
    )

    # Also compute original signal (for comparison)
    if "market_price" in df.columns:
        df["entry_signal_orig"] = (
            (df["market_price"] > 0) &
            (df["market_price"] <= df[prob_col] * EDGE_MULT)
        )

    trades = df[df["entry_signal_clean"]].copy()
    if trades.empty:
        print(f"  [{label}] No entry signals (clean).")
        return pd.DataFrame()

    trades["entry_price"]    = trades["price_lag1"]
    trades["exit_price"]     = trades["outcome"].astype(float)
    trades["pnl_raw"]        = trades["exit_price"] - trades["entry_price"]
    trades["pnl"]            = apply_fees(
                                   trades["pnl_raw"], trades["entry_price"],
                                   trades["exit_price"], fee_model
                               )
    trades["fee_model"]      = fee_model
    trades["roi"]            = trades["pnl"] / trades["entry_price"].clip(lower=1e-6)
    trades["win"]            = (trades["pnl"] > 0).astype(int)
    trades["mae"]            = np.where(trades["outcome"] == 0, trades["entry_price"], 0)
    trades["mfe"]            = np.where(trades["outcome"] == 1, 1 - trades["entry_price"], 0)
    trades["long_duration"]  = trades.get("category", pd.Series(dtype=str)).isin(
                                   LONG_DURATION_CATS).astype(int)

    trades = trades.sort_values("_date").reset_index(drop=True)
    trades["cumulative_pnl"] = trades["pnl"].cumsum()
    return trades


def metrics(trades: pd.DataFrame, label: str = "", bootstrap: bool = False) -> dict:
    if trades.empty:
        return {"n_trades": 0, "label": label}
    pnl = trades["pnl"]
    roi = trades["roi"]
    cum = trades["cumulative_pnl"] if "cumulative_pnl" in trades.columns \
          else trades.sort_values("_date")["pnl"].cumsum()
    mdd    = (cum - cum.cummax()).min()
    t_stat = pnl.mean() / pnl.std() * np.sqrt(len(pnl)) if pnl.std() > 0 else 0
    sharpe = round(pnl.mean() / pnl.std() * np.sqrt(252), 4) if pnl.std() > 0 else 0

    out = {
        "label":        label,
        "n_trades":     len(trades),
        "win_rate":     round(trades["win"].mean(), 4),
        "total_pnl":    round(pnl.sum(), 4),
        "mean_pnl":     round(pnl.mean(), 4),
        "sharpe":       sharpe,
        "sharpe_roi":   round(roi.mean() / roi.std() * np.sqrt(252), 4) if roi.std() > 0 else 0,
        "t_stat":       round(t_stat, 2),
        "max_drawdown": round(mdd, 4),
        "avg_entry":    round(trades["entry_price"].mean(), 4),
        "mean_mae":     round(trades["mae"].mean(), 4),
        "mean_mfe":     round(trades["mfe"].mean(), 4),
    }

    # Point 6: Bootstrap CI (only on primary metrics to avoid slowdown)
    if bootstrap and len(pnl) >= 30:
        ci = bootstrap_sharpe_ci(pnl)
        out["sharpe_ci_lower"] = round(ci["ci_lower"], 4)
        out["sharpe_ci_upper"] = round(ci["ci_upper"], 4)

    return out


def comparison_metrics(wf: pd.DataFrame, prob_col: str) -> dict:
    """Run both clean (price_lag1) and original (market_price) backtests for comparison."""
    result = {}

    # Clean backtest
    trades_clean = run_backtest_clean(wf, prob_col)
    result["clean"] = metrics(trades_clean, "clean (price_lag1 entry)")

    # Original backtest (last_price_dollars entry — lookahead)
    if "market_price" in wf.columns:
        df_orig = wf.copy()
        df_orig["_date"] = pd.to_datetime(df_orig["close_date"])
        df_orig["entry_signal"] = (
            (df_orig["market_price"] > 0) &
            (df_orig["market_price"] <= df_orig[prob_col] * EDGE_MULT)
        )
        trades_orig = df_orig[df_orig["entry_signal"]].copy()
        if not trades_orig.empty:
            trades_orig["entry_price"]    = trades_orig["market_price"]
            trades_orig["exit_price"]     = trades_orig["outcome"].astype(float)
            trades_orig["pnl"]            = trades_orig["exit_price"] - trades_orig["entry_price"]
            trades_orig["roi"]            = trades_orig["pnl"] / trades_orig["entry_price"].clip(lower=1e-6)
            trades_orig["win"]            = (trades_orig["pnl"] > 0).astype(int)
            trades_orig["mae"]            = np.where(trades_orig["outcome"] == 0, trades_orig["entry_price"], 0)
            trades_orig["mfe"]            = np.where(trades_orig["outcome"] == 1, 1 - trades_orig["entry_price"], 0)
            trades_orig = trades_orig.sort_values("_date").reset_index(drop=True)
            trades_orig["cumulative_pnl"] = trades_orig["pnl"].cumsum()
            result["original"] = metrics(trades_orig, "original (market_price — lookahead)")

    return result, trades_clean


def breakdown_by_group(trades: pd.DataFrame, gcol: str) -> dict:
    if gcol not in trades.columns:
        return {}
    result = {}
    for grp, g in sorted(trades.groupby(gcol)):
        g = g.copy()
        g["cumulative_pnl"] = g.sort_values("_date")["pnl"].cumsum()
        m = metrics(g, str(grp))
        result[str(grp)] = m
        if m["n_trades"] >= 5:
            print(f"  {str(grp):<25} n={m['n_trades']:>4} | win={m['win_rate']:.0%} | "
                  f"pnl={m['total_pnl']:+.2f} | sharpe={m['sharpe']:+.2f} | "
                  f"entry={m['avg_entry']:.4f}")
    return result


def md_table(headers: list[str], rows: list[list]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line    = "|" + "|".join(["---"] * len(headers)) + "|"
    body        = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([header_line, sep_line, body])


def generate_report(results: dict, trades_dict: dict,
                    robustness_dict: dict | None = None) -> None:
    today = date.today().isoformat()
    out   = REPORTS_DIR / f"backtest_clean_{today}.md"

    lines = [
        "# TrapTrade — Clean Backtest Report (Lookahead-Fixed)",
        f"**Generated:** {today}",
        "",
        "## What Was Fixed",
        "",
        "| Issue | Old approach | New approach |",
        "|---|---|---|",
        "| Entry price | `last_price_dollars` (settlement-time price) | `price_lag1` (previous market's final price — observable at open) |",
        "| ML features | Included `market_price`, `price_lag1` (lookahead) | Removed both from feature set |",
        "| Macro join date | `close_date` (future resolution date) | `open_date` (date market opened) |",
        "",
        "## Remaining Caveat",
        "",
        "`price_lag1` is the previous market's `last_price_dollars`. For short-duration",
        "markets (<2 days: Crypto, Entertainment, Financials) this is a clean proxy.",
        "For long-duration markets (Politics, Elections, Companies >30 days) the",
        "previous market's final price may lag by months. True fix requires",
        "production API key for intramarket OHLCV (`kalshi_market_history.py`).",
        "",
    ]

    # Comparison table
    for model_label, (cmp, _) in results.items():
        clean_m = cmp.get("clean", {})
        orig_m  = cmp.get("original", {})
        if not clean_m:
            continue

        lines += [
            f"---",
            f"## {model_label}",
            "",
            "### Clean vs Original Backtest Comparison",
            "",
            md_table(
                ["Metric", "Clean (price_lag1)", "Original (last_price — lookahead)", "Δ"],
                [
                    ["Trades",     clean_m.get("n_trades",0),    orig_m.get("n_trades",0), ""],
                    ["Win Rate",   f"{clean_m.get('win_rate',0):.1%}", f"{orig_m.get('win_rate',0):.1%}", ""],
                    ["Total P&L",  f"{clean_m.get('total_pnl',0):+.2f}", f"{orig_m.get('total_pnl',0):+.2f}",
                     f"{clean_m.get('total_pnl',0)-orig_m.get('total_pnl',0):+.2f}"],
                    ["Sharpe",     f"{clean_m.get('sharpe',0):+.2f}",   f"{orig_m.get('sharpe',0):+.2f}",
                     f"{clean_m.get('sharpe',0)-orig_m.get('sharpe',0):+.2f}"],
                    ["Avg Entry",  f"{clean_m.get('avg_entry',0):.4f}", f"{orig_m.get('avg_entry',0):.4f}", ""],
                    ["Max DD",     f"{clean_m.get('max_drawdown',0):+.2f}", f"{orig_m.get('max_drawdown',0):+.2f}", ""],
                    ["t-stat",     f"{clean_m.get('t_stat',0):.1f}",   f"{orig_m.get('t_stat',0):.1f}", ""],
                ]
            ),
            "",
        ]

        # Per-category breakdown for clean trades
        trades_clean = trades_dict.get(model_label)
        if trades_clean is not None and not trades_clean.empty:
            gcol = "category" if "category" in trades_clean.columns \
                   else "city" if "city" in trades_clean.columns else None
            if gcol:
                print(f"\n[{model_label}] ── Per {gcol} (CLEAN) ─────")
                bk = breakdown_by_group(trades_clean, gcol)
                if bk:
                    bk_rows = [
                        [grp, m["n_trades"], f"{m['win_rate']:.0%}",
                         f"{m['total_pnl']:+.2f}", f"{m['sharpe']:+.2f}",
                         f"{m['avg_entry']:.4f}"]
                        for grp, m in sorted(bk.items())
                        if m["n_trades"] > 0
                    ]
                    lines += [
                        f"### Per {gcol} Breakdown (Clean Entry)",
                        "",
                        md_table(["Group","Trades","Win Rate","P&L","Sharpe","Avg Entry"], bk_rows),
                        "",
                    ]

            # Duration analysis: short vs long duration categories
            if "long_duration" in trades_clean.columns:
                short = trades_clean[trades_clean["long_duration"] == 0]
                long  = trades_clean[trades_clean["long_duration"] == 1]
                if len(short) > 0 and len(long) > 0:
                    ms = metrics(short, "short_duration")
                    ml = metrics(long,  "long_duration")
                    lines += [
                        "### Duration Split (price_lag1 reliability)",
                        "",
                        "Short-duration categories: Crypto, Entertainment, Financials, "
                        "Sports, Weather, Mentions, Economics (price_lag1 more reliable).",
                        "",
                        "Long-duration categories: Politics, Elections, Companies, etc. "
                        "(price_lag1 potentially stale by months).",
                        "",
                        md_table(
                            ["Split", "Trades", "Win Rate", "P&L", "Sharpe", "Avg Entry"],
                            [
                                ["Short duration",
                                 ms["n_trades"], f"{ms['win_rate']:.1%}",
                                 f"{ms['total_pnl']:+.2f}", f"{ms['sharpe']:+.2f}",
                                 f"{ms['avg_entry']:.4f}"],
                                ["Long duration (weaker proxy)",
                                 ml["n_trades"], f"{ml['win_rate']:.1%}",
                                 f"{ml['total_pnl']:+.2f}", f"{ml['sharpe']:+.2f}",
                                 f"{ml['avg_entry']:.4f}"],
                            ]
                        ),
                        "",
                    ]

    # ── Bootstrap CI section ──────────────────────────────────────────────────
    lines += [
        "---",
        "## Sharpe Bootstrap CI (1000 resamples, 95%)",
        "",
    ]
    ci_rows = []
    for model_label, (cmp, _) in results.items():
        clean_m = cmp.get("clean", {})
        if clean_m.get("n_trades", 0) == 0:
            continue
        ci_lower = clean_m.get("sharpe_ci_lower", "—")
        ci_upper = clean_m.get("sharpe_ci_upper", "—")
        sharpe   = clean_m.get("sharpe", 0)
        ci_rows.append([
            model_label,
            f"{sharpe:+.2f}",
            f"{ci_lower:+.2f}" if isinstance(ci_lower, float) else ci_lower,
            f"{ci_upper:+.2f}" if isinstance(ci_upper, float) else ci_upper,
        ])
    if ci_rows:
        lines += [
            md_table(["Model", "Sharpe", "CI 95% lower", "CI 95% upper"], ci_rows),
            "",
        ]

    # ── Robustness tests section ──────────────────────────────────────────────
    if robustness_dict:
        lines += [
            "---",
            "## Robustness Tests",
            "",
            "| Test | Trades | Win Rate | Sharpe | P&L |",
            "|---|---|---|---|---|",
        ]
        for test_name, m in robustness_dict.items():
            if m.get("n_trades", 0) == 0:
                continue
            lines.append(
                f"| {test_name} | {m['n_trades']} | {m['win_rate']:.1%} | "
                f"{m['sharpe']:+.2f} | {m['total_pnl']:+.2f} |"
            )
        lines.append("")

    lines += [
        "---",
        "## How to Get True Intramarket Prices",
        "",
        "Set `KALSHI_API_KEY` and run:",
        "```bash",
        "python3 src/kalshi_market_history.py <ticker>",
        "```",
        "This uses `GET /trade-api/v2/markets/{ticker}/history` (prod API, requires auth).",
        "Once OHLCV data is available, re-run this backtest — entry_source will show 'ohlcv'.",
        "",
        "---",
        "## Caveats",
        "- All P&L is hypothetical. Zero real trades.",
        "- price_lag1 proxy is best available without production API auth.",
        "- Model (pred_prob_clean) trained on features that exclude market_price.",
        "- AUC reduction vs original model is expected (market_price was a strong feature).",
    ]

    out.write_text("\n".join(lines))
    print(f"\n[Report] → {out}")


def run() -> None:
    results   = {}
    trades_dict = {}

    # ── General clean model ───────────────────────────────────────────────────
    gen_path = PROC_DIR / "wf_clean_general.parquet"
    if gen_path.exists():
        wf_gen = pd.read_parquet(gen_path)
        print(f"\n[CleanBacktest] General WF: {len(wf_gen):,} rows")
        print(f"  Columns: {list(wf_gen.columns)}")

        cmp, trades_clean = comparison_metrics(wf_gen, "pred_prob_clean")

        # Bootstrap CI on clean returns
        if not trades_clean.empty:
            ci = bootstrap_sharpe_ci(trades_clean["pnl"])
            cmp["clean"]["sharpe_ci_lower"] = round(ci["ci_lower"], 4)
            cmp["clean"]["sharpe_ci_upper"] = round(ci["ci_upper"], 4)

        # Print comparison
        clean_m = cmp.get("clean", {})
        orig_m  = cmp.get("original", {})
        print(f"\n[CleanBacktest] ── COMPARISON ────────────────────────────────────")
        print(f"{'Metric':<20} {'CLEAN (price_lag1)':>22} {'ORIGINAL (lookahead)':>22}")
        print(f"{'Trades':<20} {clean_m.get('n_trades',0):>22} {orig_m.get('n_trades',0):>22}")
        print(f"{'Win Rate':<20} {clean_m.get('win_rate',0):>22.1%} {orig_m.get('win_rate',0):>22.1%}")
        print(f"{'Total P&L':<20} {clean_m.get('total_pnl',0):>+22.2f} {orig_m.get('total_pnl',0):>+22.2f}")
        print(f"{'Sharpe':<20} {clean_m.get('sharpe',0):>+22.2f} {orig_m.get('sharpe',0):>+22.2f}")
        print(f"{'Avg Entry':<20} {clean_m.get('avg_entry',0):>22.4f} {orig_m.get('avg_entry',0):>22.4f}")
        print(f"{'t-stat':<20} {clean_m.get('t_stat',0):>22.1f} {orig_m.get('t_stat',0):>22.1f}")
        print(f"{'Max DD':<20} {clean_m.get('max_drawdown',0):>+22.2f} {orig_m.get('max_drawdown',0):>+22.2f}")
        ci_lo = clean_m.get("sharpe_ci_lower", "—")
        ci_hi = clean_m.get("sharpe_ci_upper", "—")
        print(f"{'Sharpe CI 95%':<20} [{ci_lo}, {ci_hi}]")

        results["General (Clean Model)"] = (cmp, clean_m)
        if not trades_clean.empty:
            trades_clean.to_parquet(PROC_DIR / "trades_clean.parquet", index=False)
            trades_dict["General (Clean Model)"] = trades_clean

    # ── Weather clean model ───────────────────────────────────────────────────
    wx_path = PROC_DIR / "wf_clean_weather.parquet"
    if wx_path.exists():
        wf_wx = pd.read_parquet(wx_path)
        print(f"\n[CleanBacktest] Weather WF: {len(wf_wx):,} rows")

        cmp_wx, trades_wx = comparison_metrics(wf_wx, "pred_prob_clean")
        clean_wx = cmp_wx.get("clean", {})
        orig_wx  = cmp_wx.get("original", {})
        print(f"\n[CleanBacktest] Weather CLEAN — Sharpe={clean_wx.get('sharpe',0):+.2f} | "
              f"ORIGINAL — Sharpe={orig_wx.get('sharpe',0):+.2f}")

        results["Weather (Clean Model)"] = (cmp_wx, clean_wx)
        if not trades_wx.empty:
            trades_wx.to_parquet(PROC_DIR / "trades_clean_weather.parquet", index=False)
            trades_dict["Weather (Clean Model)"] = trades_wx

    if not results:
        print("[CleanBacktest] No WF parquets found. Run features_clean.py + model_clean.py first.")
        return

    # Point 7: Robustness tests on general clean trades
    robustness: dict = {}
    gen_trades = trades_dict.get("General (Clean Model)")
    gen_wf     = pd.read_parquet(PROC_DIR / "wf_clean_general.parquet") \
                 if (PROC_DIR / "wf_clean_general.parquet").exists() else None
    if gen_trades is not None and not gen_trades.empty:
        print("\n[CleanBacktest] Running robustness tests...")
        robustness = run_robustness_tests(gen_trades, gen_wf)
        for test, m in robustness.items():
            if m.get("n_trades", 0) >= 5:
                print(f"  {test:<35} n={m['n_trades']:>5} | sharpe={m['sharpe']:+.2f} | "
                      f"win={m['win_rate']:.1%} | pnl={m['total_pnl']:+.2f}")

    generate_report(results, trades_dict, robustness_dict=robustness if robustness else None)


if __name__ == "__main__":
    run()
