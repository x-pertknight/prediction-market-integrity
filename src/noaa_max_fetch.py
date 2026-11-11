"""
noaa_max_fetch.py
Fetch NOAA GHCND TMAX/TMIN for 6 cities from 2020-01-01 to present.
Provides rich rolling feature history for the weather model.
Output: data/raw/noaa_{city}_max.csv  (replaces noaa_{city}.csv in model)
        data/processed/noaa_all_cities_max.parquet
"""

import os
import time
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

NOAA_TOKEN = os.environ.get("NOAA_TOKEN", "")  # set NOAA_TOKEN in .env, see .env.example
NOAA_BASE  = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
HDR        = {"token": NOAA_TOKEN}

CITY_STATIONS = {
    "NYC":     "GHCND:USW00094728",
    "Chicago": "GHCND:USW00094846",
    "Seattle": "GHCND:USW00024233",
    "Atlanta": "GHCND:USW00013874",
    "Dallas":  "GHCND:USW00003927",
    "Miami":   "GHCND:USW00012839",
}

START_DATE = "2020-01-01"
END_DATE   = (date.today() - timedelta(days=5)).isoformat()

RAW_DIR    = Path("data/raw")
PROC_DIR   = Path("data/processed")


def fetch_city(city: str, station_id: str) -> pd.DataFrame:
    all_records = []
    start_dt    = date.fromisoformat(START_DATE)
    end_dt      = date.fromisoformat(END_DATE)
    chunk_start = start_dt

    while chunk_start <= end_dt:
        chunk_end = min(date(chunk_start.year, 12, 31), end_dt)
        params = {
            "datasetid":  "GHCND",
            "stationid":  station_id,
            "datatypeid": "TMAX,TMIN",
            "startdate":  chunk_start.isoformat(),
            "enddate":    chunk_end.isoformat(),
            "units":      "standard",
            "limit":      1000,
        }
        for attempt in range(4):
            try:
                r = requests.get(f"{NOAA_BASE}/data", headers=HDR, params=params, timeout=30)
                if r.status_code == 200:
                    all_records.extend(r.json().get("results", []))
                    break
                elif r.status_code == 429:
                    time.sleep(10)
                else:
                    print(f"  [{city}] HTTP {r.status_code} {chunk_start}→{chunk_end}")
                    break
            except Exception as e:
                time.sleep(4 * (attempt + 1))
        chunk_start = date(chunk_start.year + 1, 1, 1)
        time.sleep(0.4)

    if not all_records:
        return pd.DataFrame(columns=["date", "city", "tmax", "tmin", "tavg"])

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
    pivot["tavg"] = (pivot["tmax"] + pivot["tmin"]) / 2.0
    pivot = pivot.drop_duplicates(subset=["date", "city"])
    return pivot[["date", "city", "tmax", "tmin", "tavg"]]


def run() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for city, station in CITY_STATIONS.items():
        print(f"[NOAA-max] {city} {START_DATE}→{END_DATE}...")
        df = fetch_city(city, station)
        out = RAW_DIR / f"noaa_{city.lower()}_max.csv"
        df.to_csv(out, index=False)
        frames.append(df)
        print(f"  → {len(df)} rows | missing tmax={df['tmax'].isna().sum()} | {df['date'].min()}→{df['date'].max()}")

    combined = pd.concat(frames, ignore_index=True).sort_values(["city", "date"]).reset_index(drop=True)
    out_pq = PROC_DIR / "noaa_all_cities_max.parquet"
    combined.to_parquet(out_pq, index=False)
    print(f"\n[NOAA-max] Saved {len(combined)} rows → {out_pq}")


if __name__ == "__main__":
    run()
