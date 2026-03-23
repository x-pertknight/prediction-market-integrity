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

EDGE_MULT = 0.70  # A6 Simons volume philosophy: enter when price <= pred_prob * 0.70
                  # was 0.50 (only 2× edge); 0.70 targets 1.43× edge = more trades, smaller edge per trade

# ── Point 5: Fee models (accurate parabolic) ────────────────────────────────
# Kalshi: fee = 0.07 * P * (1-P) per contract, entry only (settlement free).
# Polymarket: most markets FREE; crypto/NCAAB/some sports: P*(1-P)*0.02
# See src/risk/fees.py for full implementation.
try:
    from risk.fees import compute_trade_fees
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from risk.fees import compute_trade_fees

FEE_MODEL_DEFAULT = "kalshi"  # backtest uses Kalshi data → Kalshi fees by default

# Long-duration categories where price_lag1 is a weaker proxy
LONG_DURATION_CATS = {"Politics", "Elections", "Companies", "Science and Technology",
                      "Health", "Education", "Social", "World", "Exotics"}


def apply_fees(pnl: pd.Series, entry: pd.Series, exit_: pd.Series,
               fee_model: str = "none",
               categories: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
    """
    Subtract accurate fees from PnL series.
    Returns (pnl_net, fee_total) tuple.
    """
    if fee_model == "none":
        return pnl, pd.Series(0.0, index=pnl.index)
    fees_df = compute_trade_fees(entry, exit_, fee_model=fee_model,
                                 categories=categories)
    return pnl - fees_df["fee_total"], fees_df["fee_total"]


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
                       label: str = "clean", fee_model: str = "none",
                       ohlcv_dict: dict | None = None) -> pd.DataFrame:
    """
    Clean backtest: entry at price_lag1, signal on pred_prob_clean.
    Returns trades DataFrame.

    ohlcv_dict: optional dict {ticker: DataFrame(ts, open_price, close_price, ...)}
                fetched via kalshi_market_history.batch_fetch_ohlcv().
                When provided, overrides price_lag1 with true OHLCV open price.
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

    trades["entry_price"]  = trades["price_lag1"]
    trades["entry_source"] = "price_lag1"

    # Upgrade entry_price with OHLCV data when available
    if ohlcv_dict and "ticker" in trades.columns:
        ohlcv_overrides = 0
        for idx, row in trades.iterrows():
            ticker = row.get("ticker")
            if not ticker or ticker not in ohlcv_dict:
                continue
            hist = ohlcv_dict[ticker]
            if hist.empty:
                continue
            # Use first-day open price (cleanest: price at market open)
            if "open_price" in hist.columns:
                first_price = hist.iloc[0]["open_price"]
            elif "close_price" in hist.columns:
                first_price = hist.iloc[0]["close_price"]
            else:
                continue
            if pd.notna(first_price) and first_price > 0:
                trades.at[idx, "entry_price"]  = first_price
                trades.at[idx, "entry_source"] = "ohlcv"
                ohlcv_overrides += 1
        if ohlcv_overrides > 0:
            print(f"  [{label}] OHLCV entry override: {ohlcv_overrides}/{len(trades)} trades")

    trades["exit_price"]     = trades["outcome"].astype(float)
    trades["pnl_gross"]      = trades["exit_price"] - trades["entry_price"]
    cats = trades["category"] if "category" in trades.columns else None
    trades["pnl"], trades["fee_paid"] = apply_fees(
                                   trades["pnl_gross"], trades["entry_price"],
                                   trades["exit_price"], fee_model,
                                   categories=cats,
                               )
    trades["pnl_raw"]        = trades["pnl_gross"]  # backward compat
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

    # Gross metrics (before fees)
    pnl_gross = trades["pnl_gross"] if "pnl_gross" in trades.columns else pnl
    sharpe_gross = round(pnl_gross.mean() / pnl_gross.std() * np.sqrt(252), 4) \
                   if pnl_gross.std() > 0 else 0
    total_fees = trades["fee_paid"].sum() if "fee_paid" in trades.columns else 0.0

    out = {
        "label":          label,
        "n_trades":       len(trades),
        "win_rate":       round(trades["win"].mean(), 4),
        "total_pnl":      round(pnl.sum(), 4),
        "total_pnl_gross": round(pnl_gross.sum(), 4),
        "total_fees":     round(total_fees, 4),
        "mean_pnl":       round(pnl.mean(), 4),
        "sharpe":         sharpe,
        "sharpe_gross":   sharpe_gross,
        "sharpe_roi":     round(roi.mean() / roi.std() * np.sqrt(252), 4) if roi.std() > 0 else 0,
        "t_stat":         round(t_stat, 2),
        "max_drawdown":   round(mdd, 4),
        "avg_entry":      round(trades["entry_price"].mean(), 4),
        "mean_mae":       round(trades["mae"].mean(), 4),
        "mean_mfe":       round(trades["mfe"].mean(), 4),
        "avg_fee":        round(total_fees / max(len(trades), 1), 6),
        "fee_pct_of_pnl": round(total_fees / max(abs(pnl_gross.sum()), 1e-6) * 100, 2),
    }

    # Point 6: Bootstrap CI (only on primary metrics to avoid slowdown)
    if bootstrap and len(pnl) >= 30:
        ci = bootstrap_sharpe_ci(pnl)
        out["sharpe_ci_lower"] = round(ci["ci_lower"], 4)
        out["sharpe_ci_upper"] = round(ci["ci_upper"], 4)

    return out


def comparison_metrics(wf: pd.DataFrame, prob_col: str,
                        ohlcv_dict: dict | None = None,
                        fee_model: str = "kalshi") -> dict:
    """Run both clean (price_lag1/ohlcv) and original (market_price) backtests for comparison."""
    result = {}

    # Clean backtest (with optional OHLCV upgrade)
    trades_clean = run_backtest_clean(wf, prob_col, fee_model=fee_model,
                                      ohlcv_dict=ohlcv_dict)
    label = "clean (ohlcv entry)" if ohlcv_dict else "clean (price_lag1 entry)"
    result["clean"] = metrics(trades_clean, label)

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
            trades_orig["pnl_gross"]      = trades_orig["exit_price"] - trades_orig["entry_price"]
            cats_orig = trades_orig["category"] if "category" in trades_orig.columns else None
            trades_orig["pnl"], trades_orig["fee_paid"] = apply_fees(
                trades_orig["pnl_gross"], trades_orig["entry_price"],
                trades_orig["exit_price"], fee_model, categories=cats_orig)
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
                    robustness_dict: dict | None = None,
                    v3_data: dict | None = None) -> None:
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
                ["Metric", "Clean (net)", "Original (net)", "Δ"],
                [
                    ["Trades",       clean_m.get("n_trades",0),    orig_m.get("n_trades",0), ""],
                    ["Win Rate",     f"{clean_m.get('win_rate',0):.1%}", f"{orig_m.get('win_rate',0):.1%}", ""],
                    ["P&L Gross",    f"{clean_m.get('total_pnl_gross',0):+.2f}", f"{orig_m.get('total_pnl_gross',0):+.2f}", ""],
                    ["Total Fees",   f"{clean_m.get('total_fees',0):.2f}", f"{orig_m.get('total_fees',0):.2f}", ""],
                    ["**P&L Net**",  f"**{clean_m.get('total_pnl',0):+.2f}**", f"**{orig_m.get('total_pnl',0):+.2f}**",
                     f"{clean_m.get('total_pnl',0)-orig_m.get('total_pnl',0):+.2f}"],
                    ["Sharpe Gross", f"{clean_m.get('sharpe_gross',0):+.2f}",   f"{orig_m.get('sharpe_gross',0):+.2f}", ""],
                    ["**Sharpe Net**", f"**{clean_m.get('sharpe',0):+.2f}**",   f"**{orig_m.get('sharpe',0):+.2f}**",
                     f"{clean_m.get('sharpe',0)-orig_m.get('sharpe',0):+.2f}"],
                    ["Avg Entry",    f"{clean_m.get('avg_entry',0):.4f}", f"{orig_m.get('avg_entry',0):.4f}", ""],
                    ["Avg Fee/Trade", f"${clean_m.get('avg_fee',0):.4f}", f"${orig_m.get('avg_fee',0):.4f}", ""],
                    ["Fee % of P&L", f"{clean_m.get('fee_pct_of_pnl',0):.1f}%", f"{orig_m.get('fee_pct_of_pnl',0):.1f}%", ""],
                    ["Max DD",       f"{clean_m.get('max_drawdown',0):+.2f}", f"{orig_m.get('max_drawdown',0):+.2f}", ""],
                    ["t-stat",       f"{clean_m.get('t_stat',0):.1f}",   f"{orig_m.get('t_stat',0):.1f}", ""],
                ]
            ),
            "",
        ]

        # Sharpe < 1.5 warning
        _net_sharpe = clean_m.get("sharpe", 0)
        if _net_sharpe < 1.5:
            lines += [
                "",
                f"> **WARNING**: Net Sharpe {_net_sharpe:+.2f} < 1.5 — "
                "fees erode most alpha. Consider Polymarket (zero-fee markets) "
                "or maker orders to reduce cost.",
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

    # ── v3 Simons + Academic Alpha sections ──────────────────────────────────
    if v3_data:
        lines += ["---", "## v3 Simons + Academic Alpha Analysis", ""]

        # HMM Regime
        hmm_stats = v3_data.get("hmm_regime_stats")
        if hmm_stats:
            lines += ["### HMM Regime Performance (3-state Gaussian HMM)", ""]
            lines.append(md_table(
                ["Regime", "Trades", "Win Rate", "Sharpe", "P&L", "Kelly Mult"],
                [[r, d["n_trades"], f"{d['win_rate']:.1%}", f"{d['sharpe']:+.2f}",
                  f"{d['total_pnl']:+.2f}", f"{d.get('kelly_mult', 1.0):.2f}×"]
                 for r, d in hmm_stats.items()]
            ))
            lines.append("")

        # Gamma
        gamma_sharpe = v3_data.get("gamma_sharpe")
        if gamma_sharpe is not None:
            lines += [
                "### Gamma Near-Expiry Sizing (Bawa 2025)",
                "",
                f"| Metric | Value |",
                f"|---|---|",
                f"| Gamma-adjusted Sharpe | {gamma_sharpe:+.4f} |",
                f"| Average size multiplier | {v3_data.get('gamma_avg_mult', 1.0):.4f} |",
            ]
            for bucket, n in (v3_data.get("gamma_buckets") or {}).items():
                lines.append(f"| Bucket: {bucket} | {n} trades |")
            lines.append("")

        # Transaction Cost
        tc_sharpe = v3_data.get("tc_sharpe")
        if tc_sharpe is not None:
            lines += [
                "### Transaction Cost Impact (Almgren-Chriss model)",
                "",
                f"| Metric | Value |",
                f"|---|---|",
                f"| TC-adjusted Sharpe | {tc_sharpe:+.4f} |",
                f"| Average TC per trade | {v3_data.get('tc_avg_cost', 0):.5f} |",
                f"| Viable trades (edge > cost) | {v3_data.get('tc_viable_pct', 1.0):.1%} |",
                "",
            ]

        # Calibration
        cal = v3_data.get("calibration")
        if cal and cal.category_brier:
            lines += [
                "### Calibration (Brier Score per category)",
                "",
                f"**Global Brier Score: {cal.global_brier_score:.5f}** "
                "(random=0.25, perfect=0.00)",
                "",
                md_table(
                    ["Category", "Brier Score", "Sizing Mult", "Status"],
                    [[cat, f"{bs:.5f}", f"{cal.sizing_multiplier(cat):.1f}×",
                      "⚠️ Miscalibrated" if bs > 0.20 else "✓ OK"]
                     for cat, bs in sorted(cal.category_brier.items())]
                ),
                "",
            ]

        # Favorite-Longshot Bias
        flb_tbl = v3_data.get("flb_table")
        if flb_tbl is not None and not flb_tbl.empty:
            lines += [
                "### Favorite-Longshot Bias (UCD Working Paper WP2025_19)",
                "",
                "bias = actual_win_rate - midpoint_of_price_bucket",
                "Positive bias: market overprices (avoid long) | Negative bias: underpriced (buy)",
                "",
                md_table(
                    ["Price Bucket", "Win Rate", "Bias", "N", "Signal"],
                    [[row["bucket"], f"{row['win_rate']:.3f}",
                      f"{row['bias']:+.3f}", row["n"], row["signal"]]
                     for _, row in flb_tbl.iterrows()]
                ),
                "",
            ]

        # Wash Filter
        wash_pct = v3_data.get("wash_pct")
        if wash_pct is not None:
            lines += [
                "### Wash Trading Filter (Columbia Nov 2025)",
                "",
                f"| Metric | Value |",
                f"|---|---|",
                f"| Wash-flagged trades | {v3_data.get('wash_flagged', 0):,} "
                f"({wash_pct:.1%}) |",
                f"| Wash-clean Sharpe | {v3_data.get('wash_clean_sharpe', 'N/A')} |",
                "",
            ]

        # Performance vs Bawa Benchmarks (B7)
        lines += [
            "---",
            "## v3 vs Bawa (2025) Realistic Performance Benchmarks",
            "",
            "Ref: Navnoor Bawa 'Mathematical Execution Behind Prediction Market Alpha' (Dec 2025)",
            "",
            md_table(
                ["Metric", "Bawa Target (Live)", "TrapTrade v3 (Backtest)"],
                [
                    ["Annual Return",      "15–25%",   "~see Sharpe"],
                    ["Sharpe Ratio",       "2.0–2.8",  f"{results.get('General (Clean Model)', ({}, {}))[0].get('clean', {}).get('sharpe', '—')}"],
                    ["Win Rate",           "52–58%",   f"{results.get('General (Clean Model)', ({}, {}))[0].get('clean', {}).get('win_rate', '—')}"],
                    ["Avg Edge/Trade",     "2–4%",     "see avg_entry"],
                    ["Max Drawdown",       "12–18%",   f"{results.get('General (Clean Model)', ({}, {}))[0].get('clean', {}).get('max_drawdown', '—')}"],
                    ["Capital Util.",      "40–60%",   "TBD (live trading)"],
                    ["Entry Threshold",    "v2: ×0.5", f"v3: ×{EDGE_MULT}"],
                ],
            ),
            "",
            "⚠️ **Sanity check**: TrapTrade backtest Sharpe >2.8 likely reflects:",
            "  1. price_lag1 proxy still overfits short-duration series",
            "  2. Limited history (Kalshi launched Apr 2025 → ~11 months data)",
            "  3. No live execution costs, slippage, or capital constraints",
            "  Strategy needs live paper trading to validate real-world performance.",
            "",
        ]

    lines += [
        "---",
        "## How to Enable True Intramarket OHLCV Entry Prices",
        "",
        "The backtest supports true intramarket entry prices via Kalshi prod API.",
        "When OHLCV is available, `entry_source='ohlcv'` replaces `price_lag1`.",
        "",
        "**Setup:**",
        "```bash",
        "# 1. KALSHI_API_KEY is already set in .env",
        "# 2. Add your RSA private key path:",
        "echo 'KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi_private_key.pem' >> .env",
        "```",
        "The PEM file was generated when you created the API key in Kalshi's dashboard.",
        "It is your local RSA private key — never transmitted to Kalshi.",
        "",
        "Once configured, re-running `python3 src/backtest_clean.py` will automatically:",
        "  1. Fetch OHLCV for all signal tickers via `GET /trade-api/v2/markets/{ticker}/history`",
        "  2. Override price_lag1 with first-day open price where available",
        "  3. Report entry_source breakdown (ohlcv vs price_lag1)",
        "",
        "---",
        "## Caveats",
        "- All P&L is hypothetical. Zero real trades.",
        "- price_lag1 proxy is best available without RSA private key configured.",
        "- Model (pred_prob_clean) trained on features that exclude market_price.",
        "- AUC reduction vs original model is expected (market_price was a strong feature).",
    ]

    out.write_text("\n".join(lines))
    print(f"\n[Report] → {out}")


def _try_fetch_ohlcv(wf: pd.DataFrame) -> dict | None:
    """
    Attempt to fetch OHLCV for all signal tickers using the prod API.
    Returns ohlcv_dict or None if auth is not configured.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from kalshi_market_history import batch_fetch_ohlcv, _has_auth
    except ImportError:
        return None

    if not _has_auth():
        print("[OHLCV] RSA private key not configured — using price_lag1 as entry proxy.")
        print("  Add KALSHI_PRIVATE_KEY_PATH to .env to enable true intramarket entry prices.")
        return None

    # Get unique tickers that have entry signals
    if "price_lag1" not in wf.columns or "pred_prob_clean" not in wf.columns:
        return None
    signal_rows = wf[
        (wf["price_lag1"] > 0) &
        (wf["price_lag1"] <= wf["pred_prob_clean"] * EDGE_MULT)
    ]
    if "ticker" not in signal_rows.columns:
        return None
    tickers = signal_rows["ticker"].dropna().unique().tolist()
    print(f"[OHLCV] Fetching intramarket prices for {len(tickers):,} signal tickers...")
    ohlcv_dict = batch_fetch_ohlcv(tickers, period_minutes=1440)
    n_fetched = sum(1 for df in ohlcv_dict.values() if not df.empty)
    print(f"[OHLCV] Fetched: {n_fetched}/{len(tickers)} tickers with data")
    return ohlcv_dict if n_fetched > 0 else None


def _load_simons_modules():
    """Lazy-load Simons + Academic Alpha modules. Returns dict of modules (None if unavailable)."""
    import sys
    src = Path(__file__).parent
    sys.path.insert(0, str(src))
    mods = {}
    try:
        from models.hmm_regime import CategoryHMMRegistry
        mods["hmm"] = CategoryHMMRegistry
    except Exception as e:
        print(f"[v3] HMM not loaded: {e}")
    try:
        from alpha.favorite_longshot import FavoriteLongshotBias
        mods["flb"] = FavoriteLongshotBias
    except Exception as e:
        print(f"[v3] FLB not loaded: {e}")
    try:
        from models.calibration import CalibrationAnalyzer
        mods["cal"] = CalibrationAnalyzer
    except Exception as e:
        print(f"[v3] Calibration not loaded: {e}")
    try:
        from risk.expiry_gamma import ExpiryGamma
        mods["gamma"] = ExpiryGamma
    except Exception as e:
        print(f"[v3] ExpiryGamma not loaded: {e}")
    try:
        from risk.transaction_cost import TransactionCostModel
        mods["tc"] = TransactionCostModel
    except Exception as e:
        print(f"[v3] TransactionCost not loaded: {e}")
    try:
        from data.wash_filter import WashFilter
        mods["wash"] = WashFilter
    except Exception as e:
        print(f"[v3] WashFilter not loaded: {e}")
    return mods


def run_v3_analysis(trades: pd.DataFrame, wf: pd.DataFrame,
                    mods: dict) -> dict:
    """
    Run all Simons + Academic Alpha analysis on completed trades.
    Returns v3_data dict with all module results.
    """
    v3 = {}

    # ── HMM Regime analysis ────────────────────────────────────────────────────
    if "hmm" in mods and not trades.empty:
        print("[v3] Fitting HMM regimes...")
        try:
            reg = mods["hmm"]()
            reg.fit_all(wf)
            # Annotate trades with HMM regime
            trades_hmm = reg.annotate_wf(trades)
            regime_stats = {}
            if "hmm_regime" in trades_hmm.columns:
                for regime, g in trades_hmm.groupby("hmm_regime"):
                    if len(g) >= 5:
                        pnl = g["pnl"]
                        s = round(pnl.mean() / pnl.std() * np.sqrt(252), 4) \
                            if pnl.std() > 0 else 0.0
                        regime_stats[str(regime)] = {
                            "n_trades": len(g),
                            "sharpe":   s,
                            "win_rate": round(g["win"].mean(), 4),
                            "total_pnl": round(pnl.sum(), 4),
                            "kelly_mult": g["hmm_kelly_mult"].mean() if "hmm_kelly_mult" in g.columns else 1.0,
                        }
                        print(f"  HMM {regime:<18} n={len(g):>6} | sharpe={s:+.2f} | "
                              f"win={g['win'].mean():.1%}")
            v3["hmm_regime_stats"] = regime_stats
            v3["hmm_registry"]     = reg
        except Exception as e:
            print(f"  [v3] HMM analysis failed: {e}")

    # ── Favorite-Longshot Bias ─────────────────────────────────────────────────
    if "flb" in mods and not trades.empty:
        print("[v3] Computing Favorite-Longshot Bias...")
        try:
            flb = mods["flb"]()
            flb.fit(trades)
            tbl = flb.summary_table()
            v3["flb_table"] = tbl
            v3["flb"] = flb
            # Compute how much alpha is from FL bias
            if not tbl.empty:
                underpriced = tbl[tbl["bias"] < -0.05]
                overpriced  = tbl[tbl["bias"] > 0.05]
                print(f"  Underpriced buckets (opportunity): {len(underpriced)}")
                print(f"  Overpriced buckets (avoid long):   {len(overpriced)}")
        except Exception as e:
            print(f"  [v3] FLB analysis failed: {e}")

    # ── Calibration ────────────────────────────────────────────────────────────
    if "cal" in mods and not wf.empty:
        print("[v3] Computing calibration metrics...")
        try:
            # Use cached if available, else fit
            cal_path = Path("data/processed/calibration_cache.json")
            if cal_path.exists():
                ca = mods["cal"].load(cal_path)
                # Re-fit if cache seems stale (no category_brier)
                if not ca.category_brier:
                    ca.fit(wf)
            else:
                ca = mods["cal"]()
                ca.fit(wf)
            v3["calibration"] = ca
            print(f"  Global Brier Score: {ca.global_brier_score:.5f}")
            miscal_cats = [c for c, bs in ca.category_brier.items()
                           if bs > 0.20]
            if miscal_cats:
                print(f"  Miscalibrated categories (BS>0.20): {miscal_cats}")
        except Exception as e:
            print(f"  [v3] Calibration analysis failed: {e}")

    # ── Gamma Near Expiry ──────────────────────────────────────────────────────
    if "gamma" in mods and not trades.empty:
        print("[v3] Computing gamma/expiry sizing...")
        try:
            eg = mods["gamma"]()
            trades_gamma = eg.annotate(trades)
            # V3 gamma-adjusted P&L
            if "gamma_mult" in trades_gamma.columns:
                gamma_pnl = trades_gamma["pnl"] * trades_gamma["gamma_mult"]
                gamma_sharpe = (gamma_pnl.mean() / gamma_pnl.std() * np.sqrt(252)
                                if gamma_pnl.std() > 0 else 0.0)
                v3["gamma_sharpe"] = round(gamma_sharpe, 4)
                v3["gamma_avg_mult"] = round(trades_gamma["gamma_mult"].mean(), 4)
                bucket_counts = trades_gamma["gamma_bucket"].value_counts().to_dict()
                v3["gamma_buckets"] = bucket_counts
                print(f"  Gamma avg mult: {v3['gamma_avg_mult']:.4f} | "
                      f"Gamma-adj Sharpe: {gamma_sharpe:.4f}")
                print(f"  Buckets: {bucket_counts}")
        except Exception as e:
            print(f"  [v3] Gamma analysis failed: {e}")

    # ── Transaction Cost Impact ────────────────────────────────────────────────
    if "tc" in mods and not trades.empty:
        print("[v3] Computing transaction cost impact...")
        try:
            tc = mods["tc"]()
            trades_tc = tc.annotate_trades(trades)
            if "tc_total" in trades_tc.columns:
                tc_pnl = trades_tc["pnl"] - trades_tc["tc_total"]
                tc_sharpe = (tc_pnl.mean() / tc_pnl.std() * np.sqrt(252)
                             if tc_pnl.std() > 0 else 0.0)
                viable_pct = trades_tc["tc_viable"].mean()
                v3["tc_sharpe"]     = round(tc_sharpe, 4)
                v3["tc_viable_pct"] = round(viable_pct, 4)
                v3["tc_avg_cost"]   = round(trades_tc["tc_total"].mean(), 5)
                print(f"  TC-adj Sharpe: {tc_sharpe:.4f} | Viable: {viable_pct:.1%} | "
                      f"Avg cost: {v3['tc_avg_cost']:.5f}")
        except Exception as e:
            print(f"  [v3] Transaction cost analysis failed: {e}")

    # ── Wash Filter Impact ─────────────────────────────────────────────────────
    if "wash" in mods and not trades.empty:
        print("[v3] Applying wash trading filter...")
        try:
            wf_filter = mods["wash"]()
            scored = wf_filter.heuristic_wash_score(trades)
            n_flagged = scored["wash_flag"].sum()
            v3["wash_flagged"] = int(n_flagged)
            v3["wash_pct"]     = round(n_flagged / max(len(trades), 1), 4)
            clean_trades = scored[~scored["wash_flag"]]
            if len(clean_trades) >= 20:
                pnl = clean_trades["pnl"]
                wash_sharpe = (pnl.mean() / pnl.std() * np.sqrt(252)
                               if pnl.std() > 0 else 0.0)
                v3["wash_clean_sharpe"] = round(wash_sharpe, 4)
                print(f"  Wash-flagged: {n_flagged:,} ({v3['wash_pct']:.1%}) | "
                      f"Clean Sharpe: {wash_sharpe:.4f}")
        except Exception as e:
            print(f"  [v3] Wash filter failed: {e}")

    return v3


def run() -> None:
    results   = {}
    trades_dict = {}
    v3_data: dict = {}

    # Load Simons modules once
    simons_mods = _load_simons_modules()
    print(f"[v3] Modules loaded: {list(simons_mods.keys())}")

    # ── General clean model ───────────────────────────────────────────────────
    gen_path = PROC_DIR / "wf_clean_general.parquet"
    if gen_path.exists():
        wf_gen = pd.read_parquet(gen_path)
        print(f"\n[CleanBacktest] General WF: {len(wf_gen):,} rows")
        print(f"  Columns: {list(wf_gen.columns)}")

        # Attempt OHLCV fetch for true intramarket entry prices
        ohlcv_dict = _try_fetch_ohlcv(wf_gen)

        cmp, trades_clean = comparison_metrics(wf_gen, "pred_prob_clean",
                                               ohlcv_dict=ohlcv_dict)

        # Bootstrap CI on clean returns
        if not trades_clean.empty:
            ci = bootstrap_sharpe_ci(trades_clean["pnl"])
            cmp["clean"]["sharpe_ci_lower"] = round(ci["ci_lower"], 4)
            cmp["clean"]["sharpe_ci_upper"] = round(ci["ci_upper"], 4)

        # Print comparison
        clean_m = cmp.get("clean", {})
        orig_m  = cmp.get("original", {})
        print(f"\n[CleanBacktest] ── COMPARISON (fee_model=kalshi) ────────────────────")
        print(f"{'Metric':<20} {'CLEAN (net)':>22} {'ORIGINAL (net)':>22}")
        print(f"{'-'*64}")
        print(f"{'Trades':<20} {clean_m.get('n_trades',0):>22,} {orig_m.get('n_trades',0):>22,}")
        print(f"{'Win Rate':<20} {clean_m.get('win_rate',0):>22.1%} {orig_m.get('win_rate',0):>22.1%}")
        print(f"{'P&L Gross':<20} {clean_m.get('total_pnl_gross',0):>+22.2f} {orig_m.get('total_pnl_gross',0):>+22.2f}")
        print(f"{'Total Fees':<20} {clean_m.get('total_fees',0):>22.2f} {orig_m.get('total_fees',0):>22.2f}")
        print(f"{'P&L Net':<20} {clean_m.get('total_pnl',0):>+22.2f} {orig_m.get('total_pnl',0):>+22.2f}")
        print(f"{'Sharpe Gross':<20} {clean_m.get('sharpe_gross',0):>+22.2f} {orig_m.get('sharpe_gross',0):>+22.2f}")
        print(f"{'Sharpe Net':<20} {clean_m.get('sharpe',0):>+22.2f} {orig_m.get('sharpe',0):>+22.2f}")
        print(f"{'Avg Entry':<20} {clean_m.get('avg_entry',0):>22.4f} {orig_m.get('avg_entry',0):>22.4f}")
        print(f"{'Avg Fee/Trade':<20} {clean_m.get('avg_fee',0):>22.6f} {orig_m.get('avg_fee',0):>22.6f}")
        print(f"{'Fee % of Gross':<20} {clean_m.get('fee_pct_of_pnl',0):>21.2f}% {orig_m.get('fee_pct_of_pnl',0):>21.2f}%")
        print(f"{'Max DD':<20} {clean_m.get('max_drawdown',0):>+22.2f} {orig_m.get('max_drawdown',0):>+22.2f}")
        ci_lo = clean_m.get("sharpe_ci_lower", "—")
        ci_hi = clean_m.get("sharpe_ci_upper", "—")
        print(f"{'Sharpe CI 95%':<20} [{ci_lo}, {ci_hi}]")

        # Warning if net Sharpe < 1.5
        net_sharpe = clean_m.get("sharpe", 0)
        if net_sharpe < 1.5:
            print(f"\n  ⚠️  WARNING: Net Sharpe {net_sharpe:+.2f} < 1.5 — fees erode most alpha!")

        results["General (Clean Model)"] = (cmp, clean_m)
        if not trades_clean.empty:
            trades_clean.to_parquet(PROC_DIR / "trades_clean.parquet", index=False)
            trades_dict["General (Clean Model)"] = trades_clean

    # ── Weather clean model ───────────────────────────────────────────────────
    wx_path = PROC_DIR / "wf_clean_weather.parquet"
    if wx_path.exists():
        wf_wx = pd.read_parquet(wx_path)
        print(f"\n[CleanBacktest] Weather WF: {len(wf_wx):,} rows")

        ohlcv_wx = _try_fetch_ohlcv(wf_wx)
        cmp_wx, trades_wx = comparison_metrics(wf_wx, "pred_prob_clean",
                                               ohlcv_dict=ohlcv_wx)
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

    # ── v3 Simons + Academic Alpha Analysis ───────────────────────────────────
    if gen_trades is not None and not gen_trades.empty and simons_mods:
        print("\n[v3] ═══ SIMONS + ACADEMIC ALPHA ANALYSIS ═══════════════════════")
        _v3_wf  = gen_wf if gen_wf is not None else wf_gen
        v3_data = run_v3_analysis(gen_trades, _v3_wf, simons_mods)

    generate_report(results, trades_dict, robustness_dict=robustness if robustness else None,
                    v3_data=v3_data)


if __name__ == "__main__":
    run()
