"""
noaa_kalshi_join.py — data_noaa_08
Join NOAA actuals with Kalshi settled markets on city + date.
Output: data/processed/joined_dataset.parquet
Schema: city, date, tmax, tmin, tavg, bracket_strike, market_price, result, volume
"""

import pandas as pd
from pathlib import Path

from kalshi_validate import validate_all

PROC_DIR  = Path("data/processed")

# Canonical city name mapping (Kalshi lower → NOAA canonical)
CITY_NORM = {
    "nyc":     "NYC",
    "chicago": "Chicago",
    "miami":   "Miami",
    "seattle": "Seattle",
    "atlanta": "Atlanta",
    "dallas":  "Dallas",
}


def build_joined_dataset() -> pd.DataFrame:
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Load NOAA processed
    noaa = pd.read_parquet(PROC_DIR / "noaa_all_cities.parquet")
    noaa["date"] = pd.to_datetime(noaa["date"]).dt.date
    print(f"[Join] NOAA rows: {len(noaa)} | cities: {sorted(noaa['city'].unique())}")

    # Load + validate Kalshi
    kalshi_frames = validate_all()

    kalshi_all = []
    for city_lower, df in kalshi_frames.items():
        canonical = CITY_NORM[city_lower]
        df = df.copy()
        df["city"]   = canonical
        df["date"]   = pd.to_datetime(df["close_date"]).dt.date
        # Use last_price_dollars as market_price (best available mid)
        df["market_price"] = df["last_price_dollars"].fillna(
            (df["yes_bid_dollars"] + df["yes_ask_dollars"]) / 2
        )
        df = df.rename(columns={
            "floor_strike": "bracket_strike",
            "volume_fp":    "volume",
        })
        kalshi_all.append(df[["city", "date", "bracket_strike", "market_price", "result", "volume", "ticker"]])

    kalshi = pd.concat(kalshi_all, ignore_index=True)
    print(f"[Join] Kalshi rows: {len(kalshi)} | cities: {sorted(kalshi['city'].unique())}")

    # Join on city + date
    joined = kalshi.merge(
        noaa[["city", "date", "tmax", "tmin", "tavg"]],
        on=["city", "date"],
        how="inner",
    )

    print(f"[Join] After inner join: {len(joined)} rows")

    # Drop rows with missing bracket_strike or market_price
    before = len(joined)
    joined = joined.dropna(subset=["bracket_strike", "market_price", "tmax"])
    print(f"[Join] After dropping nulls: {len(joined)} rows (dropped {before - len(joined)})")

    # Add binary label for model
    joined["outcome"] = (joined["result"] == "yes").astype(int)

    # Sort
    joined = joined.sort_values(["city", "date", "bracket_strike"]).reset_index(drop=True)

    out_path = PROC_DIR / "joined_dataset.parquet"
    joined.to_parquet(out_path, index=False)
    print(f"\n[Join] Saved {len(joined)} rows → {out_path}")
    print(f"       Columns: {joined.columns.tolist()}")

    # Per-city summary
    print("\n  City breakdown:")
    for city, grp in joined.groupby("city"):
        win_rate = grp["outcome"].mean()
        print(f"    {city:<10} rows={len(grp):>4} | dates={grp['date'].nunique():>3} | "
              f"brackets={grp['bracket_strike'].nunique():>2} | win_rate={win_rate:.1%}")

    return joined


if __name__ == "__main__":
    build_joined_dataset()
