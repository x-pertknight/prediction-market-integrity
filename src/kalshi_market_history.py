"""
kalshi_market_history.py
Fetch intramarket price history for Kalshi markets.

Two modes:
  PROD  — GET /trade-api/v2/markets/{ticker}/history (RSA-PSS auth required)
          Returns daily OHLCV candlesticks.
  DEMO  — GET /demo-api.kalshi.co/trade-api/v2/markets/{ticker}
          No auth, no history endpoint; returns current live ask only.

Auth (prod):
  Kalshi uses RSA-PSS signing with SHA-256.
  Message = "{timestamp_ms}{METHOD}{path}" (no query params, no body).
  Headers required:
    KALSHI-ACCESS-KEY        — API key UUID (KALSHI_API_KEY env var)
    KALSHI-ACCESS-TIMESTAMP  — Unix ms as string
    KALSHI-ACCESS-SIGNATURE  — base64(RSA-PSS-SHA256(message))
  Private key path: KALSHI_PRIVATE_KEY_PATH env var (PEM file).

Usage (prod):
  rows = fetch_market_ohlcv("KXHIGHNY-26JUL16-T85", period_minutes=1440)

Usage (demo, for live open markets):
  asks = fetch_live_asks(["KXHIGHNY-26JUL16-T85", ...])
"""

import os
import time
import base64
import requests
import pandas as pd
from pathlib import Path

DEMO_BASE  = "https://demo-api.kalshi.co/trade-api/v2"
PROD_BASE  = "https://api.elections.kalshi.com/trade-api/v2"
RAW_DIR    = Path("data/raw")
CACHE_DIR  = RAW_DIR / "market_history"
REQ_DELAY  = 0.1   # 100ms between requests


# ── RSA-PSS Auth ───────────────────────────────────────────────────────────────

def _load_private_key():
    """Load RSA private key from KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY env var."""
    try:
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        raise ImportError("cryptography library not installed. Run: pip install cryptography")

    pem_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
    pem_text = os.getenv("KALSHI_PRIVATE_KEY", "")

    if pem_path and Path(pem_path).exists():
        pem_bytes = Path(pem_path).read_bytes()
    elif pem_text:
        pem_bytes = pem_text.encode("utf-8")
    else:
        return None  # no private key available

    return serialization.load_pem_private_key(pem_bytes, password=None)


def _sign_request(method: str, path: str) -> dict | None:
    """
    Build Kalshi RSA-PSS auth headers.
    Returns dict of headers or None if private key is not configured.
    """
    key_id = os.getenv("KALSHI_API_KEY", "")
    if not key_id:
        return None

    private_key = _load_private_key()
    if private_key is None:
        return None

    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError:
        raise ImportError("cryptography library not installed. Run: pip install cryptography")

    ts_ms   = int(time.time() * 1000)
    message = f"{ts_ms}{method.upper()}{path}".encode("utf-8")

    sig = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )

    return {
        "KALSHI-ACCESS-KEY":       key_id,
        "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
    }


def _has_auth() -> bool:
    """True if both KALSHI_API_KEY and a private key source are configured."""
    if not os.getenv("KALSHI_API_KEY"):
        return False
    pem_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
    pem_text = os.getenv("KALSHI_PRIVATE_KEY", "")
    return bool((pem_path and Path(pem_path).exists()) or pem_text)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

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
                raise PermissionError(f"Auth error {r.status_code}: {r.text[:200]}")
            elif r.status_code == 404:
                return None  # market not found
            else:
                time.sleep(delay)
                delay = min(delay * 2, 30)
        except PermissionError:
            raise
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 30)
    return None


# ── Production API (RSA-PSS auth required) ─────────────────────────────────────

def fetch_market_ohlcv(ticker: str, period_minutes: int = 1440) -> pd.DataFrame:
    """
    Fetch daily (or sub-daily) OHLCV candlesticks for a market via prod API.

    Requires:
      KALSHI_API_KEY         — UUID key ID (env var)
      KALSHI_PRIVATE_KEY_PATH — path to RSA private key PEM file (env var)
        OR
      KALSHI_PRIVATE_KEY     — PEM content as string (env var)

    period_minutes:
      1    = 1-minute bars
      60   = hourly bars
      1440 = daily bars (default)

    Returns DataFrame: ts, date, open_price, high_price, low_price, close_price, volume
    """
    # Check cache first
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}_d{period_minutes}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    if not _has_auth():
        raise EnvironmentError(
            "Kalshi prod API requires RSA-PSS auth.\n"
            "  1. Set KALSHI_API_KEY to your key UUID.\n"
            "  2. Set KALSHI_PRIVATE_KEY_PATH to the path of your RSA private key PEM file.\n"
            "     (Generated when you created the API key in Kalshi's dashboard.)\n"
            "  Example: KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi_private_key.pem"
        )

    path = f"/trade-api/v2/markets/{ticker}/history"
    rows   = []
    cursor = ""

    while True:
        auth_headers = _sign_request("GET", path)
        if not auth_headers:
            raise PermissionError("Failed to build auth headers — check private key.")

        params: dict = {"period_interval": period_minutes, "limit": 1000}
        if cursor:
            params["cursor"] = cursor

        data = _get(f"{PROD_BASE}/markets/{ticker}/history",
                    params=params, headers=auth_headers)
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
    # Normalise field names
    if "ts" in df.columns:
        df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.date
    for col in ["open_price", "high_price", "low_price", "close_price", "volume", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_col = "ts" if "ts" in df.columns else df.columns[0]
    df = df.sort_values(sort_col).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    return df


def batch_fetch_ohlcv(tickers: list[str], period_minutes: int = 1440,
                      max_errors: int = 5) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for a list of tickers. Returns dict ticker → DataFrame.
    Caches each market to CACHE_DIR for resumable fetching.
    Stops after max_errors consecutive auth errors.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    errors  = 0

    for i, ticker in enumerate(tickers):
        cache_path = CACHE_DIR / f"{ticker}_d{period_minutes}.parquet"
        if cache_path.exists():
            results[ticker] = pd.read_parquet(cache_path)
            continue
        try:
            df = fetch_market_ohlcv(ticker, period_minutes)
            results[ticker] = df
            errors = 0
        except EnvironmentError as e:
            print(f"[OHLCV] Auth not configured: {e}")
            break
        except PermissionError as e:
            print(f"[OHLCV] Auth error on {ticker}: {e}")
            errors += 1
            if errors >= max_errors:
                print(f"[OHLCV] Stopping after {max_errors} auth errors.")
                break
        except Exception as e:
            print(f"[OHLCV] Failed {ticker}: {e}")
            results[ticker] = pd.DataFrame()
        if i % 100 == 0 and i > 0:
            print(f"  [OHLCV] {i}/{len(tickers)} fetched")
        time.sleep(REQ_DELAY)

    return results


# ── Demo API — live market snapshots (no auth) ─────────────────────────────────

def fetch_live_asks(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch current yes_ask_dollars for a list of market tickers via demo API.
    Only returns data for OPEN markets.
    Returns DataFrame: ticker, yes_ask_dollars, yes_bid_dollars, last_price_dollars
    """
    rows = []
    for ticker in tickers:
        data = _get(f"{DEMO_BASE}/markets/{ticker}")
        if not data:
            continue
        m = data.get("market", {})
        if m.get("status") not in ("open", "active"):
            continue
        rows.append({
            "ticker":              ticker,
            "yes_ask_dollars":     pd.to_numeric(m.get("yes_ask_dollars"),    errors="coerce"),
            "yes_bid_dollars":     pd.to_numeric(m.get("yes_bid_dollars"),    errors="coerce"),
            "last_price_dollars":  pd.to_numeric(m.get("last_price_dollars"), errors="coerce"),
            "status":              m.get("status"),
        })
        time.sleep(REQ_DELAY)
    return pd.DataFrame(rows)


def fetch_open_markets_for_series(series_ticker: str) -> pd.DataFrame:
    """Fetch all currently OPEN markets for a series with live bid/ask prices."""
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


# ── OHLCV-based entry price estimation ────────────────────────────────────────

def estimate_entry_prices(signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each signal row (ticker + pred_prob), attempt to get the true open-day
    entry price from OHLCV history (prod API). Falls back to price_lag1 when
    OHLCV is unavailable.

    Returns signal_df with columns added:
      entry_price_clean  — best available entry price
      entry_source       — 'ohlcv' | 'price_lag1'
    """
    out = signal_df.copy()
    out["entry_price_clean"] = out.get("price_lag1", out.get("market_price"))
    out["entry_source"]      = "price_lag1"

    if not _has_auth():
        print(
            "[EntryPrice] Kalshi prod API not configured — using price_lag1 as entry proxy.\n"
            "  To enable true OHLCV entry prices:\n"
            "    1. Set KALSHI_API_KEY in .env (already set)\n"
            "    2. Set KALSHI_PRIVATE_KEY_PATH=/path/to/your/kalshi_private_key.pem in .env\n"
            "       (The PEM file was generated when you created the API key in Kalshi's dashboard)"
        )
        return out

    if "ticker" not in out.columns or "pred_prob_clean" not in out.columns:
        print("[EntryPrice] Need 'ticker' and 'pred_prob_clean' columns.")
        return out

    tickers = out["ticker"].dropna().unique().tolist()
    print(f"[EntryPrice] Fetching OHLCV for {len(tickers)} markets (prod API)...")

    ohlcv_dict = batch_fetch_ohlcv(tickers, period_minutes=1440)

    for idx, row in out.iterrows():
        ticker = row.get("ticker")
        if not ticker or ticker not in ohlcv_dict:
            continue
        hist = ohlcv_dict[ticker]
        if hist.empty or "close_price" not in hist.columns:
            continue

        # First day's open price = cleanest entry (market just opened)
        if "open_price" in hist.columns:
            first_open = hist.iloc[0]["open_price"]
            if pd.notna(first_open) and first_open > 0:
                out.at[idx, "entry_price_clean"] = first_open
                out.at[idx, "entry_source"]      = "ohlcv"
                continue

        # Fallback: first close_price
        first_close = hist.iloc[0]["close_price"]
        if pd.notna(first_close) and first_close > 0:
            out.at[idx, "entry_price_clean"] = first_close
            out.at[idx, "entry_source"]      = "ohlcv"

    ohlcv_n   = (out["entry_source"] == "ohlcv").sum()
    lag1_n    = len(out) - ohlcv_n
    print(f"[EntryPrice] OHLCV: {ohlcv_n} | price_lag1 fallback: {lag1_n}")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        try:
            df = fetch_market_ohlcv(ticker)
            print(df.head(20))
        except EnvironmentError as e:
            print(e)
    else:
        print("Usage: python kalshi_market_history.py <ticker>")
        print("\nAuth status:")
        print(f"  KALSHI_API_KEY:          {'SET' if os.getenv('KALSHI_API_KEY') else 'NOT SET'}")
        print(f"  KALSHI_PRIVATE_KEY_PATH: {'SET' if os.getenv('KALSHI_PRIVATE_KEY_PATH') else 'NOT SET'}")
        print(f"  KALSHI_PRIVATE_KEY:      {'SET' if os.getenv('KALSHI_PRIVATE_KEY') else 'NOT SET'}")
        print(f"  Auth ready:              {_has_auth()}")
