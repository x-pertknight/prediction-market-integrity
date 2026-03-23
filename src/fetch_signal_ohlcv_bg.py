"""
fetch_signal_ohlcv_bg.py — Background OHLCV fetch for signal-only tickers.
Skips already-cached tickers. REQ_DELAY=50ms set in kalshi_market_history.py.
"""

import sys
import time
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROC_DIR  = Path("data/processed")
CACHE_DIR = Path("data/raw/market_history")
EDGE_MULT = 0.70


def main():
    wf = pd.read_parquet(PROC_DIR / "wf_clean_general.parquet")
    signal_mask = (wf["price_lag1"] > 0) & (wf["price_lag1"] <= wf["pred_prob_clean"] * EDGE_MULT)
    signal_tickers = wf.loc[signal_mask, "ticker"].dropna().unique().tolist()

    # Filter out already cached
    cached = set()
    for f in CACHE_DIR.glob("*_d1440.parquet"):
        cached.add(f.stem.replace("_d1440", ""))

    need_fetch = [t for t in signal_tickers if t not in cached]
    print(f"[BG-Fetch] {len(need_fetch):,} tickers to fetch (50ms delay)")
    print(f"[BG-Fetch] Estimated time: ~{len(need_fetch) * 0.3 / 60:.0f} min")

    if not need_fetch:
        print("[BG-Fetch] All signal tickers already cached!")
        return

    from kalshi_market_history import batch_fetch_ohlcv
    t0 = time.time()
    results = batch_fetch_ohlcv(need_fetch, period_minutes=1440, max_errors=10)
    elapsed = time.time() - t0

    n_ok = sum(1 for df in results.values() if not df.empty)
    print(f"\n[BG-Fetch] Done in {elapsed/60:.1f} min")
    print(f"[BG-Fetch] Fetched: {n_ok:,}/{len(need_fetch):,} with data")
    print(f"[BG-Fetch] Total cached now: {len(cached) + n_ok:,}")


if __name__ == "__main__":
    main()
