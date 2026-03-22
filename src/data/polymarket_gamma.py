"""
src/data/polymarket_gamma.py
Polymarket Gamma API — active markets, prices, volumes.
No auth required.
https://gamma-api.polymarket.com
"""
import time
import requests
import pandas as pd
from pathlib import Path

BASE    = "https://gamma-api.polymarket.com"
RAW_DIR = Path("data/raw")


def fetch_active_markets(limit: int = 500) -> pd.DataFrame:
    """Fetch currently active Polymarket markets with prices."""
    all_markets = []
    offset = 0
    while True:
        try:
            r = requests.get(
                f"{BASE}/markets",
                params={"active": "true", "limit": min(limit, 100), "offset": offset},
                timeout=15,
            )
            if r.status_code != 200:
                break
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break
            all_markets.extend(batch)
            if len(batch) < 100:
                break
            offset += len(batch)
            time.sleep(0.2)
            if len(all_markets) >= limit:
                break
        except Exception as e:
            print(f"  [PolyGamma] {e}")
            break

    if not all_markets:
        return pd.DataFrame()

    rows = []
    for m in all_markets:
        rows.append({
            "market_id":   m.get("id",""),
            "question":    m.get("question",""),
            "slug":        m.get("slug",""),
            "end_date":    m.get("endDate",""),
            "liquidity":   float(m.get("liquidity", 0) or 0),
            "volume":      float(m.get("volume", 0) or 0),
            "category":    m.get("category",""),
            "active":      m.get("active", False),
            "closed":      m.get("closed", False),
        })
    return pd.DataFrame(rows)


def fetch_market_prices(condition_id: str) -> dict:
    """Fetch current bid/ask for a Polymarket market."""
    try:
        r = requests.get(f"{BASE}/markets/{condition_id}", timeout=10)
        if r.status_code == 200:
            d = r.json()
            return {
                "condition_id": condition_id,
                "best_bid":     float(d.get("bestBid", 0) or 0),
                "best_ask":     float(d.get("bestAsk", 1) or 1),
                "spread":       float(d.get("spread", 0) or 0),
            }
    except Exception:
        pass
    return {}


def fetch_and_save() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("[PolyGamma] Fetching active markets...")
    df = fetch_active_markets(500)
    if df.empty:
        print("  [PolyGamma] No data returned")
        return df
    df.to_csv(RAW_DIR / "polymarket_active.csv", index=False)
    print(f"  [PolyGamma] {len(df)} active markets | top categories: {df['category'].value_counts().head(5).to_dict()}")
    return df


def validate(df: pd.DataFrame) -> None:
    assert "market_id" in df.columns


if __name__ == "__main__":
    df = fetch_and_save()
    validate(df)
