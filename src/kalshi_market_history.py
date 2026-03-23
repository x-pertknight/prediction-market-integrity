"""
kalshi_market_history.py
Fetch intramarket price history for Kalshi markets.

Two modes:
  PROD  — GET /trade-api/v2/markets/{ticker}/history (requires auth token)
          Returns daily OHLCV candlesticks.
  DEMO  — GET /trade-api/v2/markets/{ticker}/orderbook + snapshot polling
          No auth, no history endpoint; returns current live ask only.

Usage (prod, when KALSHI_API_KEY is set):
  rows = fetch_market_ohlcv("KXHIGHNY-26JUL16-T85", period_minutes=1440)

Usage (demo, for live open markets):
  asks = fetch_live_asks(["KXHIGHNY-26JUL16-T85", ...])

Output format (OHLCV):
  date, open_price, high_price, low_price, close_price, volume, open_interest
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path

DEMO_BASE  = "https://demo-api.kalshi.co/trade-api/v2"
PROD_BASE  = "https://api.kalshi.com/trade-api/v2"
RAW_DIR    = Path("data/raw")
CACHE_DIR  = RAW_DIR / "market_history"
REQ_DELAY  = 0.1   # 100ms between requests


def _get(url: str, params: dict | None = None, headers: dict | None = None,
         retries: int = 5) -> dict | None:
    delay = 1.0
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 60)
            elif r.status_code in (401, 403):
                raise PermissionError(f"Auth required: {r.text[:100]}")
            else:
                time.sleep(delay)
                delay = min(delay * 2, 30)
        except PermissionError:
            raise
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 30)
    return None


# ── Production API (auth required) ────────────────────────────────────────────

def fetch_market_ohlcv(ticker: str, period_minutes: int = 1440,
                       api_key: str | None = None) -> pd.DataFrame:
    """
    Fetch daily (or sub-daily) OHLCV candlesticks for a market.
    Requires production API key: pass api_key= or set KALSHI_API_KEY env var.

    period_minutes:
      1    = 1-minute bars
      60   = hourly bars
      1440 = daily bars (default)

    Returns DataFrame with columns:
      date, ts, open_price, high_price, low_price, close_price, volume, open_interest
    """
    key = api_key or os.getenv("KALSHI_API_KEY")
    if not key:
        raise EnvironmentError(
            "KALSHI_API_KEY not set. "
            "The production API requires authentication for market history. "
            "Set KALSHI_API_KEY in your environment and retry."
        )

    # Check cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}_d{period_minutes}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    headers = {"Authorization": f"Bearer {key}"}
    rows = []
    cursor = ""

    while True:
        params: dict = {"period_interval": period_minutes, "limit": 1000}
        if cursor:
            params["cursor"] = cursor

        data = _get(f"{PROD_BASE}/markets/{ticker}/history",
                    params=params, headers=headers)
        if not data:
            break

        batch  = data.get("history", [])
        cursor = data.get("cursor", "")
        rows.extend(batch)
        time.sleep(REQ_DELAY)

        if not cursor or not batch:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Normalise field names (Kalshi may return ts, open_price, etc.)
    if "ts" in df.columns:
        df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.date
    for col in ["open_price", "high_price", "low_price", "close_price", "volume", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("ts" if "ts" in df.columns else df.columns[0]).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    return df


def batch_fetch_ohlcv(tickers: list[str], period_minutes: int = 1440,
                      api_key: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for a list of tickers. Returns dict ticker → DataFrame.
    Caches each market to CACHE_DIR to allow resumable fetching.
    """
    results = {}
    for i, ticker in enumerate(tickers):
        cache_path = CACHE_DIR / f"{ticker}_d{period_minutes}.parquet"
        if cache_path.exists():
            results[ticker] = pd.read_parquet(cache_path)
            continue
        try:
            df = fetch_market_ohlcv(ticker, period_minutes, api_key)
            results[ticker] = df
        except PermissionError:
            print(f"[OHLCV] Auth error on {ticker}. Set KALSHI_API_KEY and retry.")
            break
        except Exception as e:
            print(f"[OHLCV] Failed {ticker}: {e}")
            results[ticker] = pd.DataFrame()
        if i % 50 == 0:
            print(f"  [OHLCV] {i}/{len(tickers)} fetched")
        time.sleep(REQ_DELAY)
    return results


# ── Demo API — live market snapshots (no auth) ─────────────────────────────────

def fetch_live_asks(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch current yes_ask_dollars for a list of market tickers.
    Works on the demo API (no auth). Only returns data for OPEN markets.
    Returns DataFrame: ticker, yes_ask_dollars, yes_bid_dollars, last_price_dollars
    """
    rows = []
    for ticker in tickers:
        data = _get(f"{DEMO_BASE}/markets/{ticker}")
        if not data:
            continue
        m = data.get("market", {})
        if m.get("status") not in ("open", "active"):
            continue  # market already settled — no live price
        rows.append({
            "ticker":              ticker,
            "yes_ask_dollars":     pd.to_numeric(m.get("yes_ask_dollars"), errors="coerce"),
            "yes_bid_dollars":     pd.to_numeric(m.get("yes_bid_dollars"), errors="coerce"),
            "last_price_dollars":  pd.to_numeric(m.get("last_price_dollars"), errors="coerce"),
            "status":              m.get("status"),
        })
        time.sleep(REQ_DELAY)
    return pd.DataFrame(rows)


def fetch_open_markets_for_series(series_ticker: str) -> pd.DataFrame:
    """
    Fetch all currently OPEN markets for a series.
    Returns DataFrame with live bid/ask prices — clean entry prices.
    """
    rows = []
    cursor = ""
    while True:
        params: dict = {"series_ticker": series_ticker, "limit": 100, "status": "open"}
        if cursor:
            params["cursor"] = cursor
        data = _get(f"{DEMO_BASE}/markets", params=params)
        if not data:
            break
        batch  = data.get("markets", [])
        cursor = data.get("cursor", "")
        rows.extend(batch)
        time.sleep(REQ_DELAY)
        if not cursor or not batch:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ["yes_ask_dollars", "yes_bid_dollars", "last_price_dollars"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Find entry prices for clean backtest signals ───────────────────────────────

def estimate_entry_prices(signal_df: pd.DataFrame,
                          api_key: str | None = None) -> pd.DataFrame:
    """
    For each signal (row with ticker + close_date), attempt to get the
    true entry price from OHLCV history (prod API) or fallback to
    price_lag1 (previous market settlement price, best clean proxy
    available without auth).

    Returns signal_df with new column:
      entry_price_clean  — best available entry price
      entry_source       — 'ohlcv' | 'price_lag1' | 'market_price'
    """
    key = api_key or os.getenv("KALSHI_API_KEY")
    out = signal_df.copy()
    out["entry_price_clean"] = out.get("price_lag1", out.get("market_price"))
    out["entry_source"]      = "price_lag1"

    if not key:
        print("[EntryPrice] No KALSHI_API_KEY — using price_lag1 as entry proxy.")
        print("  Set KALSHI_API_KEY to enable true intramarket entry prices.")
        return out

    # With API key: fetch OHLCV and find first day where price ≤ pred_prob * 0.5
    if "ticker" not in out.columns or "pred_prob" not in out.columns:
        print("[EntryPrice] Need 'ticker' and 'pred_prob' columns in signal_df.")
        return out

    tickers = out["ticker"].dropna().unique().tolist()
    print(f"[EntryPrice] Fetching OHLCV for {len(tickers)} markets (prod API)...")

    ohlcv_dict = batch_fetch_ohlcv(tickers, period_minutes=1440, api_key=key)

    for idx, row in out.iterrows():
        ticker = row.get("ticker")
        if not ticker or ticker not in ohlcv_dict:
            continue
        hist = ohlcv_dict[ticker]
        if hist.empty or "close_price" not in hist.columns:
            continue

        # Find first day where close_price ≤ pred_prob * 0.5 (entry signal fires)
        prob = row.get("pred_prob", 0)
        threshold = prob * 0.5
        entry_rows = hist[hist["close_price"] <= threshold]
        if entry_rows.empty:
            continue

        # Use the close price on the first signal day
        entry_price = entry_rows.iloc[0]["close_price"]
        out.at[idx, "entry_price_clean"] = entry_price
        out.at[idx, "entry_source"]      = "ohlcv"

    ohlcv_count = (out["entry_source"] == "ohlcv").sum()
    print(f"[EntryPrice] OHLCV: {ohlcv_count} | price_lag1 fallback: {len(out)-ohlcv_count}")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        try:
            df = fetch_market_ohlcv(ticker)
            print(df)
        except EnvironmentError as e:
            print(e)
    else:
        print("Usage: python kalshi_market_history.py <ticker>")
        print("Set KALSHI_API_KEY for production API access.")
