"""
fetch_signal_ohlcv_bg.py — Concurrent OHLCV fetch for signal-only tickers.

5 async workers with a global rate limiter (50ms between requests total).
Each ticker needs ~2 API calls (market details + candlesticks), so effective
throughput is ~10 tickers/sec vs ~1.5/sec sequential.
"""

import asyncio
import sys
import os
import time
import base64
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROC_DIR  = Path("data/processed")
CACHE_DIR = Path("data/raw/market_history")
EDGE_MULT = 0.70
BASE_URL  = "https://api.elections.kalshi.com"
REQ_DELAY = 0.05  # 50ms global rate limit
N_WORKERS = 5
PERIOD    = 1440
MAX_CONSECUTIVE_ERRORS = 15

# ── Auth (reuse from kalshi_market_history) ──────────────────────────────────

def _load_private_key():
    from cryptography.hazmat.primitives import serialization
    pem_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
    pem_text = os.getenv("KALSHI_PRIVATE_KEY", "")
    if pem_path and Path(pem_path).exists():
        pem_bytes = Path(pem_path).read_bytes()
    elif pem_text:
        pem_bytes = pem_text.encode("utf-8")
    else:
        return None
    return serialization.load_pem_private_key(pem_bytes, password=None)


_PRIVATE_KEY = None
_KEY_ID = os.getenv("KALSHI_API_KEY", "")


def _sign(method: str, path: str) -> dict:
    global _PRIVATE_KEY
    if _PRIVATE_KEY is None:
        _PRIVATE_KEY = _load_private_key()
    if not _PRIVATE_KEY or not _KEY_ID:
        raise EnvironmentError("Kalshi auth not configured")

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    ts_ms = int(time.time() * 1000)
    message = f"{ts_ms}{method.upper()}{path}".encode("utf-8")
    sig = _PRIVATE_KEY.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                     salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": _KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
    }


def _parse_candles(rows: list) -> pd.DataFrame:
    """Parse Kalshi candlestick response into DataFrame."""
    parsed = []
    for c in rows:
        ts = c.get("end_period_ts", 0)
        price = c.get("price") or {}
        yes_bid = c.get("yes_bid") or {}
        def _p(d, key):
            return d.get(f"{key}_dollars") or d.get(key)
        parsed.append({
            "ts": ts,
            "date": pd.Timestamp(ts, unit="s", tz="UTC").date() if ts else None,
            "open_price": _p(price, "open") or _p(yes_bid, "open"),
            "high_price": _p(price, "high") or _p(yes_bid, "high"),
            "low_price": _p(price, "low") or _p(yes_bid, "low"),
            "close_price": _p(price, "close") or _p(yes_bid, "close"),
            "mean_price": _p(price, "mean"),
            "volume": float(c["volume_fp"]) if c.get("volume_fp") else None,
            "open_interest": float(c["open_interest_fp"]) if c.get("open_interest_fp") else None,
        })
    df = pd.DataFrame(parsed)
    for col in ["open_price", "high_price", "low_price", "close_price", "mean_price", "volume", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)


# ── Async HTTP ───────────────────────────────────────────────────────────────

async def _async_get(session, url, params=None, headers=None, retries=3):
    """Single async GET with retry + backoff."""
    delay = 1.0
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, headers=headers, timeout=20) as r:
                if r.status == 200:
                    return await r.json()
                elif r.status == 429:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
                elif r.status in (401, 403):
                    raise PermissionError(f"Auth error {r.status}")
                elif r.status == 404:
                    return None
                else:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 10)
        except PermissionError:
            raise
        except asyncio.TimeoutError:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10)
        except Exception:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10)
    return None


async def _fetch_one_ticker(session, ticker, rate_limiter, stats):
    """Fetch OHLCV for a single ticker. Returns (ticker, success)."""
    cache_path = CACHE_DIR / f"{ticker}_d{PERIOD}.parquet"
    if cache_path.exists():
        return ticker, True

    start_ts = int(time.time()) - 2 * 365 * 86400
    end_ts = int(time.time())

    try:
        # Step 1: Get series_ticker
        path1 = f"/trade-api/v2/markets/{ticker}"
        async with rate_limiter:
            headers1 = _sign("GET", path1)
            data1 = await _async_get(session, f"{BASE_URL}{path1}", headers=headers1)
        if data1 is None:
            stats["empty"] += 1
            # Save empty parquet so we don't retry
            pd.DataFrame().to_parquet(cache_path, index=False)
            return ticker, False

        market = data1.get("market", data1)
        series_ticker = market.get("series_ticker") or market.get("event_ticker")

        # Step 2: Fetch candlesticks (try active, then historical)
        endpoints = []
        if series_ticker:
            endpoints.append(f"/trade-api/v2/series/{series_ticker}/markets/{ticker}/candlesticks")
        endpoints.append(f"/trade-api/v2/historical/markets/{ticker}/candlesticks")

        params = {
            "period_interval": PERIOD,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "limit": 1000,
        }

        data = None
        used_ep = None
        for ep in endpoints:
            async with rate_limiter:
                hdrs = _sign("GET", ep)
                resp = await _async_get(session, f"{BASE_URL}{ep}", params=params, headers=hdrs)
            if resp is not None:
                data = resp
                used_ep = ep
                break

        if data is None:
            stats["empty"] += 1
            pd.DataFrame().to_parquet(cache_path, index=False)
            return ticker, False

        # Collect all pages
        all_rows = []
        while True:
            batch = data.get("candlesticks", data.get("candles", data.get("history", [])))
            cursor = data.get("cursor", "")
            all_rows.extend(batch)
            if not cursor or not batch:
                break
            async with rate_limiter:
                hdrs = _sign("GET", used_ep)
                params_page = {**params, "cursor": cursor}
                data = await _async_get(session, f"{BASE_URL}{used_ep}",
                                        params=params_page, headers=hdrs)
            if data is None:
                break

        if not all_rows:
            stats["empty"] += 1
            pd.DataFrame().to_parquet(cache_path, index=False)
            return ticker, False

        df = _parse_candles(all_rows)
        df.to_parquet(cache_path, index=False)
        stats["ok"] += 1
        return ticker, True

    except PermissionError:
        stats["auth_errors"] += 1
        raise
    except Exception as e:
        stats["errors"] += 1
        return ticker, False


class TokenBucketLimiter:
    """Global rate limiter: allows 1 request per `interval` seconds across all workers."""
    def __init__(self, interval: float):
        self.interval = interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def __aenter__(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._last + self.interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()

    async def __aexit__(self, *args):
        pass


async def worker(name, queue, session, rate_limiter, stats, stop_event):
    """Worker coroutine: pulls tickers from queue and fetches them."""
    consecutive_auth = 0
    while not stop_event.is_set():
        try:
            ticker = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        try:
            _, success = await _fetch_one_ticker(session, ticker, rate_limiter, stats)
            consecutive_auth = 0
        except PermissionError:
            consecutive_auth += 1
            if consecutive_auth >= 3:
                print(f"  [{name}] Too many auth errors, stopping worker")
                stop_event.set()
                break
        except Exception as e:
            pass
        queue.task_done()


async def run_async(tickers: list[str]):
    import aiohttp

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"ok": 0, "empty": 0, "errors": 0, "auth_errors": 0}
    rate_limiter = TokenBucketLimiter(REQ_DELAY)
    stop_event = asyncio.Event()

    queue = asyncio.Queue()
    for t in tickers:
        queue.put_nowait(t)

    total = len(tickers)
    t0 = time.time()

    connector = aiohttp.TCPConnector(limit=N_WORKERS + 2, limit_per_host=N_WORKERS + 2)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        workers = [
            asyncio.create_task(worker(f"W{i}", queue, session, rate_limiter, stats, stop_event))
            for i in range(N_WORKERS)
        ]

        # Progress reporter
        while not queue.empty() and not stop_event.is_set():
            await asyncio.sleep(10)
            done = stats["ok"] + stats["empty"] + stats["errors"]
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate / 60 if rate > 0 else 0
            print(f"  [{done:,}/{total:,}] ok={stats['ok']:,} empty={stats['empty']:,} "
                  f"err={stats['errors']} auth_err={stats['auth_errors']} "
                  f"rate={rate:.1f}/s ETA={remaining:.0f}min")

        await asyncio.gather(*workers)

    elapsed = time.time() - t0
    return stats, elapsed


def main():
    wf = pd.read_parquet(PROC_DIR / "wf_clean_general.parquet")
    signal_mask = (wf["price_lag1"] > 0) & (wf["price_lag1"] <= wf["pred_prob_clean"] * EDGE_MULT)
    signal_tickers = wf.loc[signal_mask, "ticker"].dropna().unique().tolist()

    # Filter out already cached
    cached = set()
    for f in CACHE_DIR.glob("*_d1440.parquet"):
        cached.add(f.stem.replace("_d1440", ""))

    need_fetch = [t for t in signal_tickers if t not in cached]

    # Estimate: ~2 API calls per ticker, 50ms global rate = 20 req/s = 10 tickers/s
    est_min = len(need_fetch) / 10 / 60
    print(f"[AsyncFetch] {len(need_fetch):,} tickers to fetch")
    print(f"[AsyncFetch] {N_WORKERS} workers, {REQ_DELAY*1000:.0f}ms global rate limit")
    print(f"[AsyncFetch] Estimated: ~{est_min:.0f} min (~{est_min/60:.1f} hours)")

    if not need_fetch:
        print("[AsyncFetch] All signal tickers already cached!")
        return

    stats, elapsed = asyncio.run(run_async(need_fetch))

    print(f"\n[AsyncFetch] DONE in {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"[AsyncFetch] OK: {stats['ok']:,} | Empty: {stats['empty']:,} | "
          f"Errors: {stats['errors']:,} | Auth errors: {stats['auth_errors']:,}")
    print(f"[AsyncFetch] Total cached: {len(cached) + stats['ok'] + stats['empty']:,}")


if __name__ == "__main__":
    main()
