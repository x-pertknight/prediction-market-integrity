"""
src/data/fred_macro.py
FRED (Federal Reserve Economic Data) — macro indicators.
Free API key at https://fred.stlouisfed.org/docs/api/api_key.html
Set FRED_API_KEY env var.
Series: DXY proxy (DTWEXBGS), Fed Funds (FEDFUNDS), CPI (CPIAUCSL), 10Y yield (DGS10)
"""
import os
import requests
import pandas as pd
from pathlib import Path

FRED_KEY = os.environ.get("FRED_API_KEY", "")
BASE     = "https://api.stlouisfed.org/fred/series/observations"
RAW_DIR  = Path("data/raw")

SERIES = {
    "DTWEXBGS": "dxy_proxy",      # Trade-weighted USD index
    "FEDFUNDS": "fed_funds_rate",  # Federal Funds Rate
    "CPIAUCSL": "cpi",             # CPI All Urban Consumers
    "DGS10":    "yield_10y",       # 10-Year Treasury Yield
}


def fetch_series(series_id: str, start: str = "2020-01-01") -> pd.DataFrame:
    if not FRED_KEY:
        return pd.DataFrame(columns=["date", "value", "series_id"])
    params = {
        "series_id":         series_id,
        "api_key":           FRED_KEY,
        "file_type":         "json",
        "observation_start": start,
        # no frequency override — use native series frequency, then forward-fill
    }
    try:
        r = requests.get(BASE, params=params, timeout=15)
        if r.status_code == 200:
            obs = r.json().get("observations", [])
            df  = pd.DataFrame(obs)[["date","value"]]
            df["value"]     = pd.to_numeric(df["value"], errors="coerce")
            df["series_id"] = series_id
            df["date"]      = pd.to_datetime(df["date"]).dt.date
            return df.dropna(subset=["value"])
    except Exception as e:
        print(f"  [FRED] {series_id}: {e}")
    return pd.DataFrame(columns=["date","value","series_id"])


def fetch_and_save() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not FRED_KEY:
        print("[FRED] FRED_API_KEY not set — creating empty placeholder")
        df = pd.DataFrame(columns=["date"] + list(SERIES.values()))
        df.to_csv(RAW_DIR / "fred_macro.csv", index=False)
        return df

    frames = []
    for sid, name in SERIES.items():
        print(f"  [FRED] Fetching {sid} ({name})...")
        df = fetch_series(sid)
        if not df.empty:
            df = df.rename(columns={"value": name})[["date", name]]
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    # Forward-fill gaps (FRED data has weekends/holidays missing)
    for col in list(SERIES.values()):
        if col in merged.columns:
            merged[col] = merged[col].ffill()

    merged.to_csv(RAW_DIR / "fred_macro.csv", index=False)
    print(f"[FRED] {len(merged)} rows | {merged['date'].min()} → {merged['date'].max()}")
    return merged


def validate(df: pd.DataFrame) -> None:
    if df.empty:
        return
    assert "date" in df.columns


if __name__ == "__main__":
    fetch_and_save()
