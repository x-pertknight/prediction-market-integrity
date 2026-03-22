"""
src/data/coinalyze.py
Coinalyze API — Open Interest, Funding Rates, CVD for BTC/ETH.
Free tier available at https://coinalyze.net/api/
API key required (free registration). Placeholder: set COINALYZE_KEY env var.
"""
import os
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

API_KEY = os.environ.get("COINALYZE_KEY", "")
BASE    = "https://api.coinalyze.net/v1"
RAW_DIR = Path("data/raw")

SYMBOLS = ["BTCUSDT_PERP.A", "ETHUSDT_PERP.A"]   # Coinalyze aggregated symbols


def _get(endpoint: str, params: dict) -> list:
    if not API_KEY:
        return []
    headers = {"api_key": API_KEY}
    r = requests.get(f"{BASE}/{endpoint}", headers=headers, params=params, timeout=15)
    if r.status_code == 200:
        return r.json()
    elif r.status_code == 401:
        print(f"  [Coinalyze] 401 Unauthorized — set COINALYZE_KEY env var")
    return []


def fetch_funding_rates(symbol: str, days: int = 365) -> pd.DataFrame:
    end   = int(pd.Timestamp.utcnow().timestamp())
    start = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp())
    data  = _get("funding-rate-history", {"symbols": symbol, "from": start, "to": end, "resolution": "daily"})
    rows  = []
    for item in (data if isinstance(data, list) else []):
        for entry in item.get("history", []):
            rows.append({
                "date":         pd.to_datetime(entry["t"], unit="s").date(),
                "symbol":       symbol,
                "funding_rate": float(entry.get("c", 0)),
            })
    return pd.DataFrame(rows)


def fetch_open_interest(symbol: str, days: int = 365) -> pd.DataFrame:
    end   = int(pd.Timestamp.utcnow().timestamp())
    start = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp())
    data  = _get("open-interest-history", {"symbols": symbol, "from": start, "to": end, "resolution": "daily"})
    rows  = []
    for item in (data if isinstance(data, list) else []):
        for entry in item.get("history", []):
            rows.append({
                "date":          pd.to_datetime(entry["t"], unit="s").date(),
                "symbol":        symbol,
                "open_interest": float(entry.get("c", 0)),
            })
    return pd.DataFrame(rows)


def fetch_and_save() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not API_KEY:
        print("[Coinalyze] COINALYZE_KEY not set — creating empty placeholder")
        df = pd.DataFrame(columns=["date","symbol","funding_rate","open_interest"])
        df.to_csv(RAW_DIR / "coinalyze.csv", index=False)
        return df

    frames = []
    for sym in SYMBOLS:
        fr = fetch_funding_rates(sym)
        oi = fetch_open_interest(sym)
        if not fr.empty and not oi.empty:
            merged = fr.merge(oi, on=["date","symbol"], how="outer")
            frames.append(merged)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RAW_DIR / "coinalyze.csv", index=False)
    print(f"[Coinalyze] {len(df)} rows")
    return df


def validate(df: pd.DataFrame) -> None:
    if df.empty:
        return
    assert "funding_rate" in df.columns


if __name__ == "__main__":
    fetch_and_save()
