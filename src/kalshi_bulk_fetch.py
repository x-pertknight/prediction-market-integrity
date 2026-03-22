"""
kalshi_bulk_fetch.py
Pull ALL settled markets from ALL 9247 Kalshi series.
Concurrent (10 workers), exponential backoff on 429.
Saves incrementally to data/raw/kalshi_bulk_{N}.parquet every 500 series.
Tracks progress in data/raw/kalshi_bulk_progress.json (resumable).
"""

import json
import time
import threading
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

DEMO       = "https://demo-api.kalshi.co/trade-api/v2"
RAW_DIR    = Path("data/raw")
PROGRESS_F = RAW_DIR / "kalshi_bulk_progress.json"
BATCH_SIZE = 500
MAX_WORKERS = 10
REQ_DELAY  = 0.05   # 50ms between requests per worker → ~200 req/sec aggregate

MARKET_FIELDS = [
    "ticker", "event_ticker", "series_ticker",
    "close_time", "open_time",
    "floor_strike", "result",
    "yes_bid_dollars", "yes_ask_dollars", "last_price_dollars",
    "volume_fp", "open_interest_fp",
    "status", "market_type", "strike_type",
    "title",
]

_lock   = threading.Lock()
_stats  = {"requests": 0, "errors": 0, "markets": 0}
_t0     = time.time()


def _get(url: str, params: dict, retries: int = 6) -> dict | None:
    delay = 1.0
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            with _lock:
                _stats["requests"] += 1
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 60)
            elif r.status_code == 400:
                return None   # bad params, skip
            else:
                time.sleep(delay)
                delay = min(delay * 2, 30)
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 30)
    with _lock:
        _stats["errors"] += 1
    return None


def fetch_series_markets(series_ticker: str) -> list[dict]:
    """Paginate all settled markets for one series."""
    markets = []
    cursor  = ""
    while True:
        params = {"series_ticker": series_ticker, "limit": 100, "status": "settled"}
        if cursor:
            params["cursor"] = cursor
        data = _get(f"{DEMO}/markets", params)
        if not data:
            break
        batch  = data.get("markets", [])
        cursor = data.get("cursor", "")
        markets.extend(batch)
        time.sleep(REQ_DELAY)
        if not cursor or not batch:
            break
    return markets


def markets_to_rows(markets: list[dict], series_ticker: str, category: str) -> list[dict]:
    rows = []
    for m in markets:
        row = {f: m.get(f) for f in MARKET_FIELDS}
        row["series_ticker"] = row.get("series_ticker") or series_ticker
        row["category"]      = category
        rows.append(row)
    return rows


def load_progress() -> set[str]:
    if PROGRESS_F.exists():
        return set(json.loads(PROGRESS_F.read_text()).get("done", []))
    return set()


def save_progress(done: set[str]) -> None:
    PROGRESS_F.write_text(json.dumps({"done": sorted(done), "updated": datetime.utcnow().isoformat()}))


def save_batch(rows: list[dict], batch_num: int) -> None:
    if not rows:
        return
    df  = pd.DataFrame(rows)
    out = RAW_DIR / f"kalshi_bulk_{batch_num:04d}.parquet"
    df.to_parquet(out, index=False)
    with _lock:
        _stats["markets"] += len(rows)
    elapsed = time.time() - _t0
    rps = _stats["requests"] / max(elapsed, 1)
    print(f"  [batch {batch_num:04d}] {len(rows):>7} rows → {out.name} | "
          f"total_markets={_stats['markets']:>8} | req/s={rps:.1f} | "
          f"errors={_stats['errors']} | elapsed={elapsed/60:.1f}min")


def run() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load all series
    print("[BulkFetch] Loading series list...")
    r = requests.get(f"{DEMO}/series", timeout=30)
    all_series = r.json().get("series", [])
    print(f"[BulkFetch] {len(all_series)} series found.")

    done     = load_progress()
    pending  = [(s["ticker"], s.get("category", "")) for s in all_series if s["ticker"] not in done]
    print(f"[BulkFetch] Pending: {len(pending)} (already done: {len(done)})")

    batch_num   = max(
        [int(p.stem.split("_")[-1]) for p in RAW_DIR.glob("kalshi_bulk_*.parquet")] + [-1]
    ) + 1

    total = len(pending)
    chunk_rows: list[dict] = []
    chunk_done: set[str]   = set()
    processed = 0

    def worker(ticker_cat: tuple[str, str]) -> tuple[str, list[dict]]:
        ticker, cat = ticker_cat
        markets = fetch_series_markets(ticker)
        rows    = markets_to_rows(markets, ticker, cat)
        return ticker, rows

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(worker, tc): tc for tc in pending}
        for fut in as_completed(futures):
            ticker, rows = fut.result()
            chunk_rows.extend(rows)
            chunk_done.add(ticker)
            done.add(ticker)
            processed += 1

            # Progress log
            if processed % 100 == 0:
                elapsed = time.time() - _t0
                eta_min = (elapsed / processed) * (total - processed) / 60
                print(f"  [{processed:>5}/{total}] ticker={ticker:<30} "
                      f"elapsed={elapsed/60:.1f}min eta={eta_min:.0f}min "
                      f"markets_so_far={_stats['markets']:>8}")

            # Save batch every BATCH_SIZE series
            if len(chunk_done) >= BATCH_SIZE:
                save_batch(chunk_rows, batch_num)
                save_progress(done)
                batch_num += 1
                chunk_rows = []
                chunk_done = set()

    # Final batch
    if chunk_rows:
        save_batch(chunk_rows, batch_num)
    save_progress(done)

    elapsed = time.time() - _t0
    print(f"\n[BulkFetch] DONE. {_stats['markets']:,} markets | "
          f"{_stats['requests']:,} requests | "
          f"{_stats['errors']} errors | "
          f"{elapsed/60:.1f} min total")


if __name__ == "__main__":
    run()
