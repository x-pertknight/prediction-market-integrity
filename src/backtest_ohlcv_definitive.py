"""
backtest_ohlcv_definitive.py — Definitive backtest with real OHLCV entry prices.

Reports:
  1. Sharpe Gross (raw P&L, no costs)
  2. Sharpe Net (post Kalshi taker fee)
  3. Sharpe Net (post fee + Almgren-Chriss market impact)
  4. Comparison: OHLCV entry vs price_lag1 proxy
"""

import sys
from pathlib import Path

# Ensure src/ is on path for risk module imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from risk.fees import compute_trade_fees
from risk.market_impact import compute_trade_impact

PROC_DIR  = Path("data/processed")
CACHE_DIR = Path("data/raw/market_history")
EDGE_MULT = 0.70


def load_ohlcv_cache(tickers: list[str]) -> dict:
    """Load cached OHLCV parquets for given tickers."""
    ohlcv = {}
    loaded = 0
    empty = 0
    for t in tickers:
        p = CACHE_DIR / f"{t}_d1440.parquet"
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if not df.empty:
                    ohlcv[t] = df
                    loaded += 1
                else:
                    empty += 1
            except Exception:
                empty += 1
    print(f"[OHLCV Cache] loaded={loaded:,} empty={empty:,} missing={len(tickers)-loaded-empty:,}")
    return ohlcv


def build_trades(wf: pd.DataFrame, ohlcv_dict: dict | None = None,
                 use_ohlcv: bool = True) -> pd.DataFrame:
    """
    Build trades from WF data.
    Entry signal always based on price_lag1 (observable).
    Entry PRICE upgraded to OHLCV open when available if use_ohlcv=True.
    """
    df = wf.copy()
    df["_date"] = pd.to_datetime(df["close_date"])

    # Signal: price_lag1 observable at market open
    sig = (df["price_lag1"] > 0) & (df["price_lag1"] <= df["pred_prob_clean"] * EDGE_MULT)
    trades = df[sig].copy()

    if trades.empty:
        return pd.DataFrame()

    trades["entry_price"] = trades["price_lag1"]
    trades["entry_source"] = "price_lag1"

    # Upgrade entry with OHLCV open price
    if use_ohlcv and ohlcv_dict and "ticker" in trades.columns:
        overrides = 0
        for idx, row in trades.iterrows():
            ticker = row.get("ticker")
            if not ticker or ticker not in ohlcv_dict:
                continue
            hist = ohlcv_dict[ticker]
            if hist.empty:
                continue
            # First-day open price = cleanest entry
            if "open_price" in hist.columns:
                p = hist.iloc[0]["open_price"]
            elif "close_price" in hist.columns:
                p = hist.iloc[0]["close_price"]
            else:
                continue
            if pd.notna(p) and p > 0:
                trades.at[idx, "entry_price"] = p
                trades.at[idx, "entry_source"] = "ohlcv"
                overrides += 1
        print(f"  OHLCV overrides: {overrides:,}/{len(trades):,} trades "
              f"({overrides/len(trades)*100:.1f}%)")

    trades["exit_price"] = trades["outcome"].astype(float)
    trades = trades.sort_values("_date").reset_index(drop=True)
    return trades


def compute_all_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute gross, net-fee, net-fee-impact P&L columns."""
    t = trades.copy()

    # 1. Gross P&L
    t["pnl_gross"] = t["exit_price"] - t["entry_price"]

    # 2. Kalshi taker fees (entry only, settlement free)
    cats = t["category"] if "category" in t.columns else None
    fees_df = compute_trade_fees(t["entry_price"], t["exit_price"],
                                  fee_model="kalshi", categories=cats)
    t["fee_paid"] = fees_df["fee_total"]
    t["pnl_post_fee"] = t["pnl_gross"] - t["fee_paid"]

    # 3. Market impact (Almgren-Chriss)
    vols = t["volume_fp"] if "volume_fp" in t.columns else pd.Series(np.nan, index=t.index)
    impact_df = compute_trade_impact(t["entry_price"], vols, categories=cats)
    t["impact_cost"] = impact_df["impact_cost"]
    t["liquidity_tier"] = impact_df["liquidity_tier"]
    t["trade_allowed"] = impact_df["trade_allowed"]
    t["pnl_net"] = t["pnl_post_fee"] - t["impact_cost"]

    # Derived
    t["roi_gross"] = t["pnl_gross"] / t["entry_price"].clip(lower=1e-6)
    t["roi_net"] = t["pnl_net"] / t["entry_price"].clip(lower=1e-6)
    t["win_gross"] = (t["pnl_gross"] > 0).astype(int)
    t["win_net"] = (t["pnl_net"] > 0).astype(int)

    # Cumulative
    t["cum_pnl_gross"] = t["pnl_gross"].cumsum()
    t["cum_pnl_post_fee"] = t["pnl_post_fee"].cumsum()
    t["cum_pnl_net"] = t["pnl_net"].cumsum()

    return t


def sharpe(pnl: pd.Series) -> float:
    if len(pnl) < 5 or pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def bootstrap_ci(pnl: pd.Series, n: int = 2000) -> tuple[float, float, float]:
    r = pnl.dropna().values
    if len(r) < 30 or r.std() == 0:
        s = sharpe(pnl)
        return s, s, s
    rng = np.random.default_rng(42)
    sharpes = []
    for _ in range(n):
        sample = rng.choice(r, size=len(r), replace=True)
        if sample.std() > 0:
            sharpes.append(sample.mean() / sample.std() * np.sqrt(252))
    arr = np.array(sharpes)
    return float(np.median(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def max_drawdown(cum_pnl: pd.Series) -> float:
    return float((cum_pnl - cum_pnl.cummax()).min())


def report_metrics(t: pd.DataFrame, label: str) -> dict:
    """Full metrics for a set of trades."""
    n = len(t)
    m = {"label": label, "n_trades": n}
    if n == 0:
        return m

    # Gross
    m["sharpe_gross"] = round(sharpe(t["pnl_gross"]), 4)
    m["pnl_gross"] = round(t["pnl_gross"].sum(), 2)
    m["win_rate_gross"] = round(t["win_gross"].mean(), 4)

    # Post-fee
    m["sharpe_post_fee"] = round(sharpe(t["pnl_post_fee"]), 4)
    m["pnl_post_fee"] = round(t["pnl_post_fee"].sum(), 2)

    # Net (fee + impact)
    m["sharpe_net"] = round(sharpe(t["pnl_net"]), 4)
    m["pnl_net"] = round(t["pnl_net"].sum(), 2)
    m["win_rate_net"] = round(t["win_net"].mean(), 4)

    # Costs
    m["total_fees"] = round(t["fee_paid"].sum(), 2)
    m["total_impact"] = round(t["impact_cost"].sum(), 2)
    m["total_costs"] = round(m["total_fees"] + m["total_impact"], 2)
    m["avg_fee"] = round(t["fee_paid"].mean(), 5)
    m["avg_impact"] = round(t["impact_cost"].mean(), 5)

    # Other
    m["avg_entry"] = round(t["entry_price"].mean(), 4)
    m["max_dd_gross"] = round(max_drawdown(t["cum_pnl_gross"]), 2)
    m["max_dd_net"] = round(max_drawdown(t["cum_pnl_net"]), 2)
    m["t_stat"] = round(t["pnl_net"].mean() / t["pnl_net"].std() * np.sqrt(n), 2) \
                  if t["pnl_net"].std() > 0 else 0.0

    # Bootstrap CI on net Sharpe
    med, lo, hi = bootstrap_ci(t["pnl_net"])
    m["sharpe_net_ci"] = f"[{lo:+.2f}, {hi:+.2f}]"
    m["sharpe_net_median"] = round(med, 4)

    return m


def print_comparison(m_ohlcv: dict, m_lag1: dict):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("  DEFINITIVE BACKTEST — OHLCV REAL vs price_lag1 PROXY")
    print("=" * 80)

    rows = [
        ("Trades",           "n_trades",       "d"),
        ("Avg Entry Price",  "avg_entry",      ".4f"),
        ("Win Rate (gross)", "win_rate_gross",  ".1%"),
        ("Win Rate (net)",   "win_rate_net",    ".1%"),
        ("",                 None,              ""),
        ("P&L Gross",        "pnl_gross",       "+.2f"),
        ("P&L Post-Fee",     "pnl_post_fee",    "+.2f"),
        ("P&L Net (all-in)", "pnl_net",         "+.2f"),
        ("",                 None,              ""),
        ("Total Fees",       "total_fees",      ".2f"),
        ("Total Impact",     "total_impact",    ".2f"),
        ("Total Costs",      "total_costs",     ".2f"),
        ("Avg Fee/Trade",    "avg_fee",         ".5f"),
        ("Avg Impact/Trade", "avg_impact",      ".5f"),
        ("",                 None,              ""),
        ("SHARPE GROSS",     "sharpe_gross",    "+.4f"),
        ("SHARPE POST-FEE",  "sharpe_post_fee", "+.4f"),
        ("SHARPE NET",       "sharpe_net",      "+.4f"),
        ("Sharpe CI 95%",    "sharpe_net_ci",   "s"),
        ("",                 None,              ""),
        ("Max DD (gross)",   "max_dd_gross",    "+.2f"),
        ("Max DD (net)",     "max_dd_net",      "+.2f"),
        ("t-stat (net)",     "t_stat",          ".2f"),
    ]

    print(f"\n{'Metric':<22} {'OHLCV (real)':>22} {'price_lag1 (proxy)':>22} {'Delta':>12}")
    print("-" * 80)

    for label, key, fmt in rows:
        if key is None:
            print()
            continue
        v1 = m_ohlcv.get(key, 0)
        v2 = m_lag1.get(key, 0)
        if fmt == "s":
            s1 = str(v1)
            s2 = str(v2)
            delta = ""
        elif fmt == "d":
            s1 = f"{v1:>22,}"
            s2 = f"{v2:>22,}"
            delta = f"{v1-v2:>+12,}" if isinstance(v1, int) else ""
        else:
            s1 = f"{v1:{fmt}}"
            s2 = f"{v2:{fmt}}"
            try:
                d = v1 - v2
                delta = f"{d:+.4f}"
            except (TypeError, ValueError):
                delta = ""
        if fmt not in ("s", "d"):
            print(f"{label:<22} {s1:>22} {s2:>22} {delta:>12}")
        else:
            print(f"{label:<22} {s1:>22} {s2:>22} {delta:>12}")


def print_sharpe_waterfall(m: dict, label: str):
    """Print the 3-tier Sharpe waterfall."""
    print(f"\n{'─' * 60}")
    print(f"  SHARPE WATERFALL — {label}")
    print(f"{'─' * 60}")
    sg = m.get("sharpe_gross", 0)
    sf = m.get("sharpe_post_fee", 0)
    sn = m.get("sharpe_net", 0)
    print(f"  Sharpe Gross          {sg:+.4f}")
    print(f"  - Fee drag            {sf - sg:+.4f}  (fees = ${m.get('total_fees',0):,.2f})")
    print(f"  = Sharpe Post-Fee     {sf:+.4f}")
    print(f"  - Impact drag         {sn - sf:+.4f}  (impact = ${m.get('total_impact',0):,.2f})")
    print(f"  = Sharpe Net (all-in) {sn:+.4f}  CI 95% {m.get('sharpe_net_ci', '?')}")
    pct_erosion = (1 - sn / sg) * 100 if sg != 0 else 0
    print(f"\n  Total cost erosion: {pct_erosion:.1f}% of gross Sharpe")


def category_breakdown(trades: pd.DataFrame, pnl_col: str = "pnl_net"):
    """Print per-category breakdown."""
    if "category" not in trades.columns:
        return
    print(f"\n{'─' * 70}")
    print(f"  PER-CATEGORY BREAKDOWN ({pnl_col})")
    print(f"{'─' * 70}")
    print(f"{'Category':<25} {'N':>6} {'Win%':>7} {'P&L':>10} {'Sharpe':>8} {'AvgEntry':>9}")
    print("-" * 70)
    for cat, g in sorted(trades.groupby("category"), key=lambda x: -x[1][pnl_col].sum()):
        if len(g) < 5:
            continue
        pnl = g[pnl_col]
        s = sharpe(pnl)
        wr = (pnl > 0).mean()
        print(f"{cat:<25} {len(g):>6,} {wr:>7.1%} {pnl.sum():>+10.2f} {s:>+8.2f} "
              f"{g['entry_price'].mean():>9.4f}")


def entry_source_breakdown(trades: pd.DataFrame):
    """Show how many trades use OHLCV vs price_lag1 entry."""
    if "entry_source" not in trades.columns:
        return
    print(f"\n{'─' * 50}")
    print("  ENTRY SOURCE BREAKDOWN")
    print(f"{'─' * 50}")
    for src, g in trades.groupby("entry_source"):
        pnl = g["pnl_net"]
        s = sharpe(pnl)
        print(f"  {src:<15} n={len(g):>6,} | pnl={pnl.sum():>+10.2f} | sharpe={s:>+.4f}")


def main():
    print("=" * 80)
    print("  TRAPTRADE — DEFINITIVE BACKTEST WITH REAL OHLCV")
    print("  31,746 tickers with OHLCV data | Kalshi fees | Almgren-Chriss impact")
    print("=" * 80)

    # Load WF
    wf = pd.read_parquet(PROC_DIR / "wf_clean_general.parquet")
    print(f"\n[Data] WF: {len(wf):,} rows, {wf['ticker'].nunique():,} tickers")

    # Get signal tickers
    sig = (wf["price_lag1"] > 0) & (wf["price_lag1"] <= wf["pred_prob_clean"] * EDGE_MULT)
    sig_tickers = wf.loc[sig, "ticker"].dropna().unique().tolist()
    print(f"[Data] Signal trades: {sig.sum():,} | Signal tickers: {len(sig_tickers):,}")

    # Load OHLCV cache
    print(f"\n[OHLCV] Loading cache from {CACHE_DIR}...")
    ohlcv_dict = load_ohlcv_cache(sig_tickers)

    # ═══════════════════════════════════════════════════════════════════════════
    # RUN 1: OHLCV real entry (upgrade price_lag1 where OHLCV available)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[Backtest 1] Building trades with OHLCV entry...")
    trades_ohlcv = build_trades(wf, ohlcv_dict, use_ohlcv=True)
    trades_ohlcv = compute_all_pnl(trades_ohlcv)
    m_ohlcv = report_metrics(trades_ohlcv, "OHLCV Real Entry")

    # ═══════════════════════════════════════════════════════════════════════════
    # RUN 2: price_lag1 only (proxy baseline)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[Backtest 2] Building trades with price_lag1 only...")
    trades_lag1 = build_trades(wf, ohlcv_dict=None, use_ohlcv=False)
    trades_lag1 = compute_all_pnl(trades_lag1)
    m_lag1 = report_metrics(trades_lag1, "price_lag1 Proxy")

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    print_comparison(m_ohlcv, m_lag1)

    print_sharpe_waterfall(m_ohlcv, "OHLCV REAL ENTRY")
    print_sharpe_waterfall(m_lag1, "price_lag1 PROXY")

    entry_source_breakdown(trades_ohlcv)
    category_breakdown(trades_ohlcv)

    # Save trades
    out_path = PROC_DIR / "trades_ohlcv_definitive.parquet"
    trades_ohlcv.to_parquet(out_path, index=False)
    print(f"\n[Saved] {out_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # OHLCV-only subset analysis (trades where we actually have OHLCV)
    # ═══════════════════════════════════════════════════════════════════════════
    ohlcv_only = trades_ohlcv[trades_ohlcv["entry_source"] == "ohlcv"]
    lag1_only = trades_ohlcv[trades_ohlcv["entry_source"] == "price_lag1"]
    if len(ohlcv_only) >= 30:
        m_oo = report_metrics(ohlcv_only, "OHLCV-only subset")
        print_sharpe_waterfall(m_oo, "OHLCV-ONLY TRADES (true entry price)")
    if len(lag1_only) >= 30:
        m_lo = report_metrics(lag1_only, "price_lag1-only subset")
        print_sharpe_waterfall(m_lo, "price_lag1-ONLY TRADES (no OHLCV available)")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
