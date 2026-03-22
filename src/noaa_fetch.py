"""
noaa_fetch.py — data_noaa_01 + data_noaa_02
Fetch daily TMAX/TMIN from NOAA GHCND for 6 cities.
Date range: 2025-01-01 to yesterday (aligned to Kalshi data window).
Output: data/raw/noaa_{city}.csv per city.
"""

import os
import time
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

NOAA_TOKEN = os.environ.get("NOAA_TOKEN", "")
NOAA_BASE  = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

# data_noaa_02: city → GHCND station ID
CITY_STATIONS = {
    "NYC":     "GHCND:USW00094728",  # NY CITY CENTRAL PARK
    "Chicago": "GHCND:USW00094846",  # CHICAGO OHARE
    "Seattle": "GHCND:USW00024233",  # SEATTLE TACOMA AIRPORT
    "Atlanta": "GHCND:USW00013874",  # ATLANTA HARTSFIELD
    "Dallas":  "GHCND:USW00003927",  # DAL FTW WSCMO
    "Miami":   "GHCND:USW00012839",  # MIAMI INTERNATIONAL
}

START_DATE = "2025-01-01"
# NOAA lags ~3-5 days; use 5 days ago to avoid empty tail
END_DATE   = (date.today() - timedelta(days=5)).isoformat()

RAW_DIR    = Path("data/raw")


def fetch_noaa_city(city: str, station_id: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch TMAX and TMIN for one city.
    NOAA API max 1000 records per request; chunk by year to stay safe.
    Returns DataFrame with columns: date, city, tmax, tmin.
    """
    headers = {"token": NOAA_TOKEN}
    all_records = []

    # Chunk by year (NOAA CDO limits date range per call)
    start_dt = date.fromisoformat(start)
    end_dt   = date.fromisoformat(end)

    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(date(chunk_start.year, 12, 31), end_dt)

        params = {
            "datasetid":  "GHCND",
            "stationid":  station_id,
            "datatypeid": "TMAX,TMIN",
            "startdate":  chunk_start.isoformat(),
            "enddate":    chunk_end.isoformat(),
            "units":      "standard",   # returns °F directly
            "limit":      1000,
        }

        for attempt in range(3):
            try:
                r = requests.get(
                    f"{NOAA_BASE}/data",
                    headers=headers,
                    params=params,
                    timeout=30,
                )
                if r.status_code == 200:
                    results = r.json().get("results", [])
                    all_records.extend(results)
                    break
                elif r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"  [{city}] HTTP {r.status_code} for {chunk_start}→{chunk_end}: {r.text[:100]}")
                    break
            except requests.Timeout:
                print(f"  [{city}] Timeout on attempt {attempt+1}, retrying...")
                time.sleep(3)

        chunk_start = date(chunk_start.year + 1, 1, 1)
        time.sleep(0.3)  # respect NOAA rate limit (5 req/sec)

    if not all_records:
        return pd.DataFrame(columns=["date", "city", "tmax", "tmin"])

    # Pivot: one row per date with tmax + tmin columns
    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["city"] = city

    pivot = (
        df.pivot_table(index=["date", "city"], columns="datatype", values="value", aggfunc="first")
        .reset_index()
    )
    pivot.columns.name = None
    pivot = pivot.rename(columns={"TMAX": "tmax", "TMIN": "tmin"})

    for col in ["tmax", "tmin"]:
        if col not in pivot.columns:
            pivot[col] = float("nan")

    return pivot[["date", "city", "tmax", "tmin"]]


def fetch_all_cities() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for city, station in CITY_STATIONS.items():
        print(f"[NOAA] Fetching {city} ({station}) {START_DATE} → {END_DATE}...")
        df = fetch_noaa_city(city, station, START_DATE, END_DATE)

        out_path = RAW_DIR / f"noaa_{city.lower()}.csv"
        df.to_csv(out_path, index=False)

        summary[city] = {
            "rows": len(df),
            "date_min": str(df["date"].min()) if len(df) else "n/a",
            "date_max": str(df["date"].max()) if len(df) else "n/a",
            "missing_tmax": int(df["tmax"].isna().sum()),
            "missing_tmin": int(df["tmin"].isna().sum()),
        }
        print(f"  → {len(df)} rows saved to {out_path}")

    print("\n[NOAA] Summary:")
    for city, s in summary.items():
        print(f"  {city:<10} rows={s['rows']:>4} | {s['date_min']} → {s['date_max']} | "
              f"missing tmax={s['missing_tmax']} tmin={s['missing_tmin']}")


if __name__ == "__main__":
    fetch_all_cities()
