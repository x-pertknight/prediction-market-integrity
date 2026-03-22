"""
kalshi_fetch.py — data_noaa_06 + data_noaa_07
Fetch all settled Kalshi daily-high-temp markets for 6 cities.
API: Kalshi demo (https://demo-api.kalshi.co/trade-api/v2, no auth required).
Output: data/raw/kalshi_{city_lower}.csv per city.
"""

import time
import requests
import pandas as pd
from pathlib import Path

KALSHI_BASE = "https://demo-api.kalshi.co/trade-api/v2"

# Series tickers per city
CITY_SERIES = {
    "NYC":     "KXHIGHNY",
    "Chicago": "KXHIGHCHI",
    "Miami":   "KXHIGHMIA",
    "Seattle": "KXHIGHTSEA",
    "Atlanta": "KXHIGHTATL",
    "Dallas":  "KXHIGHTDAL",
}

RAW_DIR = Path("data/raw")

# Fields we extract from each market
MARKET_FIELDS = [
    "ticker",
    "event_ticker",
    "close_time",
    "floor_strike",      # temperature threshold in °F
    "result",            # 'yes' | 'no' | ''
    "yes_bid_dollars",   # last bid price (0-1 scale as string)
    "yes_ask_dollars",
    "last_price_dollars",
    "volume_fp",         # total volume
    "status",
    "open_time",
]


def fetch_kalshi_series(city: str, series_ticker: str) -> pd.DataFrame:
    """Paginate all settled markets for a series."""
    all_markets = []
    cursor = ""
    page = 0

    while True:
        params = {
            "series_ticker": series_ticker,
            "limit":         100,
            "status":        "settled",
        }
        if cursor:
            params["cursor"] = cursor

        for attempt in range(3):
            try:
                r = requests.get(
                    f"{KALSHI_BASE}/markets",
                    params=params,
                    timeout=20,
                )
                if r.status_code == 200:
                    data      = r.json()
                    batch     = data.get("markets", [])
                    cursor    = data.get("cursor", "")
                    all_markets.extend(batch)
                    page += 1
                    break
                elif r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"  [{city}] HTTP {r.status_code}: {r.text[:80]}")
                    cursor = ""
                    break
            except requests.Timeout:
                print(f"  [{city}] Timeout attempt {attempt+1}")
                time.sleep(3)

        if not cursor:
            break
        time.sleep(0.2)

    if not all_markets:
        return pd.DataFrame()

    rows = []
    for m in all_markets:
        row = {field: m.get(field) for field in MARKET_FIELDS}
        row["city"] = city
        rows.append(row)

    df = pd.DataFrame(rows)
    df["close_date"] = pd.to_datetime(df["close_time"]).dt.date

    # Convert price strings to float
    for col in ["yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume_fp"] = pd.to_numeric(df["volume_fp"], errors="coerce")
    df["floor_strike"] = pd.to_numeric(df["floor_strike"], errors="coerce")

    return df


def fetch_all_cities() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for city, series in CITY_SERIES.items():
        print(f"[Kalshi] Fetching {city} ({series})...")
        df = fetch_kalshi_series(city, series)

        if df.empty:
            print(f"  [{city}] WARNING: no data returned")
            continue

        out_path = RAW_DIR / f"kalshi_{city.lower()}.csv"
        df.to_csv(out_path, index=False)

        settled = df[df["result"].isin(["yes", "no"])]
        print(f"  → {len(df)} markets, {len(settled)} settled "
              f"| {df['close_date'].min()} → {df['close_date'].max()} "
              f"| brackets: {sorted(df['floor_strike'].dropna().unique().astype(int).tolist())[:8]}...")


if __name__ == "__main__":
    fetch_all_cities()
