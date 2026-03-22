"""
noaa_validate.py — data_noaa_03 + data_noaa_04
Validate schema, check date continuity, compute TAVG, save processed.
Output: data/processed/noaa_all_cities.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")
CITIES   = ["nyc", "chicago", "seattle", "atlanta", "dallas", "miami"]


def validate_and_clean() -> pd.DataFrame:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for city_lower in CITIES:
        path = RAW_DIR / f"noaa_{city_lower}.csv"
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = df["date"].dt.date

        # ── data_noaa_03: schema validation ───────────────────────────────────
        assert "date" in df.columns, f"{city_lower}: missing 'date'"
        assert "tmax" in df.columns, f"{city_lower}: missing 'tmax'"
        assert "tmin" in df.columns, f"{city_lower}: missing 'tmin'"

        missing_tmax = df["tmax"].isna().sum()
        missing_tmin = df["tmin"].isna().sum()
        if missing_tmax > 0:
            print(f"  [{city_lower}] WARN: {missing_tmax} missing tmax rows")
        if missing_tmin > 0:
            print(f"  [{city_lower}] WARN: {missing_tmin} missing tmin rows")

        # Date continuity check
        dates = pd.to_datetime(df["date"]).sort_values()
        diffs = dates.diff().dropna()
        gaps = diffs[diffs > pd.Timedelta(days=1)]
        if len(gaps) > 0:
            print(f"  [{city_lower}] WARN: {len(gaps)} date gaps found: {gaps.values[:3]}")
        else:
            print(f"  [{city_lower}] date continuity: OK ({len(df)} rows)")

        # ── data_noaa_04: cleaning ─────────────────────────────────────────────
        # Compute TAVG (NOAA does not return it for these stations)
        df["tavg"] = (df["tmax"] + df["tmin"]) / 2.0

        # Deduplicate on (date, city)
        before = len(df)
        df = df.drop_duplicates(subset=["date", "city"])
        if len(df) < before:
            print(f"  [{city_lower}] dropped {before - len(df)} duplicates")

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["city", "date"]).reset_index(drop=True)

    # Save
    out_path = PROC_DIR / "noaa_all_cities.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\n[NOAA] Saved {len(combined)} rows → {out_path}")
    print(f"       Cities: {combined['city'].unique().tolist()}")
    print(f"       Date range: {combined['date'].min()} → {combined['date'].max()}")
    print(f"       Columns: {combined.columns.tolist()}")

    return combined


if __name__ == "__main__":
    validate_and_clean()
