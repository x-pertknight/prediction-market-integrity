"""
run_backtest_signal_only.py — Run backtest using cached OHLCV only.
Fetches remaining signal tickers in background.
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROC_DIR  = Path("data/processed")
CACHE_DIR = Path("data/raw/market_history")
EDGE_MULT = 0.70


def load_cached_ohlcv(tickers: list[str], period: int = 1440) -> dict[str, pd.DataFrame]:
    """Load OHLCV only from local cache — no API calls."""
    result = {}
    for t in tickers:
        p = CACHE_DIR / f"{t}_d{period}.parquet"
        if p.exists():
            result[t] = pd.read_parquet(p)
    return result


def main():
    # Load walk-forward data
    wf_path = PROC_DIR / "wf_clean_general.parquet"
    if not wf_path.exists():
        print("[ERROR] wf_clean_general.parquet not found. Run model_clean.py first.")
        return

    wf = pd.read_parquet(wf_path)
    print(f"[WF] Loaded {len(wf):,} rows")

    # Identify signal rows
    signal_mask = (wf["price_lag1"] > 0) & (wf["price_lag1"] <= wf["pred_prob_clean"] * EDGE_MULT)
    n_signals = signal_mask.sum()
    signal_tickers = wf.loc[signal_mask, "ticker"].dropna().unique().tolist()
    print(f"[Signal] {n_signals:,} entry signals across {len(signal_tickers):,} tickers")

    # Load OHLCV from cache only (no API calls)
    ohlcv_dict = load_cached_ohlcv(signal_tickers)
    n_cached = len(ohlcv_dict)
    n_need_fetch = len(signal_tickers) - n_cached
    print(f"[OHLCV] Cache hits: {n_cached:,} | Need fetch: {n_need_fetch:,}")

    # Run the full backtest with cached OHLCV
    from backtest_clean import run_backtest_clean, metrics, bootstrap_sharpe_ci, \
        run_robustness_tests, comparison_metrics, generate_report, \
        _load_simons_modules, run_v3_analysis

    print(f"\n{'='*70}")
    print(f"RUNNING BACKTEST (cache-only OHLCV: {n_cached:,}/{len(signal_tickers):,} tickers)")
    print(f"{'='*70}\n")

    # Run comparison (clean vs original) with Kalshi parabolic fees
    cmp, trades_clean = comparison_metrics(wf, "pred_prob_clean", ohlcv_dict=ohlcv_dict,
                                            fee_model="kalshi")

    if trades_clean.empty:
        print("[ERROR] No trades generated.")
        return

    # Bootstrap CI
    ci = bootstrap_sharpe_ci(trades_clean["pnl"])
    cmp["clean"]["sharpe_ci_lower"] = round(ci["ci_lower"], 4)
    cmp["clean"]["sharpe_ci_upper"] = round(ci["ci_upper"], 4)

    # Print results
    clean_m = cmp.get("clean", {})
    orig_m = cmp.get("original", {})

    print(f"\n{'Metric':<20} {'CLEAN (net)':>22} {'ORIGINAL (net)':>22}")
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
    ci_lo = clean_m.get("sharpe_ci_lower", "N/A")
    ci_hi = clean_m.get("sharpe_ci_upper", "N/A")
    print(f"{'Sharpe CI 95%':<20} [{ci_lo}, {ci_hi}]")
    net_sharpe = clean_m.get("sharpe", 0)
    if net_sharpe < 1.5:
        print(f"\n  WARNING: Net Sharpe {net_sharpe:+.2f} < 1.5 — fees erode most alpha!")

    # OHLCV entry source breakdown
    if "entry_source" in trades_clean.columns:
        src_counts = trades_clean["entry_source"].value_counts()
        print(f"\n[Entry Source]")
        for src, cnt in src_counts.items():
            print(f"  {src}: {cnt:,} ({cnt/len(trades_clean):.1%})")

    # Save trades
    trades_clean.to_parquet(PROC_DIR / "trades_clean.parquet", index=False)
    print(f"\n[Saved] trades_clean.parquet ({len(trades_clean):,} trades)")

    # Robustness
    print("\n[Robustness] Running 4 tests...")
    robustness = run_robustness_tests(trades_clean, wf)
    for test, m in robustness.items():
        if m.get("n_trades", 0) >= 5:
            print(f"  {test:<35} n={m['n_trades']:>5} | sharpe={m['sharpe']:+.2f} | "
                  f"win={m['win_rate']:.1%}")

    # v3 Simons analysis
    simons_mods = _load_simons_modules()
    print(f"\n[v3] Modules: {list(simons_mods.keys())}")
    v3_data = {}
    if simons_mods:
        v3_data = run_v3_analysis(trades_clean, wf, simons_mods)

    # Generate report
    results = {"General (Clean Model)": (cmp, clean_m)}
    trades_dict = {"General (Clean Model)": trades_clean}
    generate_report(results, trades_dict, robustness_dict=robustness, v3_data=v3_data)

    # Weather subset
    wx_path = PROC_DIR / "wf_clean_weather.parquet"
    if wx_path.exists():
        wf_wx = pd.read_parquet(wx_path)
        wx_tickers = wf_wx.loc[
            (wf_wx["price_lag1"] > 0) &
            (wf_wx["price_lag1"] <= wf_wx["pred_prob_clean"] * EDGE_MULT),
            "ticker"
        ].dropna().unique().tolist()
        ohlcv_wx = load_cached_ohlcv(wx_tickers)
        # Weather on Polymarket = FREE (zero fee); on Kalshi = parabolic
        cmp_wx, trades_wx = comparison_metrics(wf_wx, "pred_prob_clean", ohlcv_dict=ohlcv_wx,
                                                fee_model="kalshi")
        clean_wx = cmp_wx.get("clean", {})
        print(f"\n[Weather] Sharpe Net={clean_wx.get('sharpe',0):+.2f} | "
              f"Sharpe Gross={clean_wx.get('sharpe_gross',0):+.2f} | "
              f"Trades={clean_wx.get('n_trades',0):,} | "
              f"Fees=${clean_wx.get('total_fees',0):.2f}")
        if not trades_wx.empty:
            trades_wx.to_parquet(PROC_DIR / "trades_clean_weather.parquet", index=False)

    print("\n[DONE] Backtest complete.")
    print(f"[NEXT] Background fetch running for {n_need_fetch:,} remaining tickers at 50ms delay.")


if __name__ == "__main__":
    main()
