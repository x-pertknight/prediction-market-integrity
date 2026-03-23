"""
features_generic.py
Build two feature matrices from bulk Kalshi data:
  1. data/processed/feature_matrix_weather.parquet  — weather markets + NOAA features
  2. data/processed/feature_matrix_all.parquet       — all markets, generic features only

Generic features (work for any market):
  market_price, volume_fp, days_in_market, day_of_week, month,
  floor_strike_pct, hist_win_rate_series, market_type_enc, category_enc,
  price_lag1, volume_rank, doy_sin, doy_cos, result_binary

NOAA features (weather-only, joined on city+date):
  tmax_roll3/7/14, tmax_std7/14, tmax_lag1, tmax_zscore14,
  tmax_trend3, bracket_dist_lag1/roll7, bracket_pct_roll7
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")

WEATHER_SERIES = {
    "KXHIGHNY":   "NYC",
    "KXHIGHCHI":  "Chicago",
    "KXHIGHMIA":  "Miami",
    "KXHIGHTSEA": "Seattle",
    "KXHIGHTATL": "Atlanta",
    "KXHIGHTDAL": "Dallas",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_bulk() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("kalshi_bulk_*.parquet"))
    if not files:
        raise FileNotFoundError("No kalshi_bulk_*.parquet found. Run kalshi_bulk_fetch.py first.")
    print(f"[Features] Loading {len(files)} bulk parquet files...")
    parts = [pd.read_parquet(f) for f in files]
    df    = pd.concat(parts, ignore_index=True)
    print(f"[Features] Total rows: {len(df):,}")
    return df


def clean_bulk(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise types, filter to settled markets with result yes/no."""
    df = df.copy()
    df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce")
    df["open_time"]  = pd.to_datetime(df["open_time"],  errors="coerce")
    df = df[df["result"].isin(["yes", "no"])].copy()
    df = df[df["close_time"].notna()].copy()   # drop rows with invalid/missing close_time
    df["close_date"] = df["close_time"].dt.date
    df["outcome"]    = (df["result"] == "yes").astype(int)
    for col in ["yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume_fp"]       = pd.to_numeric(df["volume_fp"],       errors="coerce").fillna(0)
    df["floor_strike"]    = pd.to_numeric(df["floor_strike"],    errors="coerce")
    df["market_price"]    = df["last_price_dollars"].fillna(
        (df["yes_bid_dollars"].fillna(0) + df["yes_ask_dollars"].fillna(0)) / 2
    )
    df = df[df["market_price"] > 0].copy()
    return df


# ── Generic features ──────────────────────────────────────────────────────────

CATEGORY_MAP: dict[str, int] = {}
MARKET_TYPE_MAP: dict[str, int] = {}


def build_generic_features(df: pd.DataFrame) -> pd.DataFrame:
    global CATEGORY_MAP, MARKET_TYPE_MAP
    df = df.copy()
    df = df.sort_values(["series_ticker", "close_date"]).reset_index(drop=True)

    # days_in_market
    df["days_in_market"] = (
        (df["close_time"] - df["open_time"]).dt.total_seconds() / 86400
    ).clip(0, 365).fillna(1)

    # calendar
    close_dt          = pd.to_datetime(df["close_date"])
    df["day_of_week"] = close_dt.dt.dayofweek
    df["month"]       = close_dt.dt.month
    df["day_of_year"] = close_dt.dt.dayofyear
    df["doy_sin"]     = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]     = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # floor_strike_pct: strike / rolling mean price per series (rough normalisation)
    df["floor_strike_pct"] = (
        df["floor_strike"]
        / df.groupby("series_ticker")["floor_strike"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).clip(0, 5).fillna(1.0)

    # volume_rank per series (expanding percentile, shift to avoid leakage)
    df["volume_rank"] = (
        df.groupby("series_ticker")["volume_fp"]
        .transform(lambda x: x.shift(1).expanding().rank(pct=True))
    ).fillna(0.5)

    # price_lag1 per series
    df["price_lag1"] = (
        df.groupby("series_ticker")["market_price"]
        .transform(lambda x: x.shift(1))
    ).fillna(df["market_price"])

    # hist_win_rate per series (expanding, no leakage)
    df["hist_win_rate_series"] = (
        df.groupby("series_ticker")["outcome"]
        .transform(lambda x: x.shift(1).expanding().mean())
    ).fillna(0.5)

    # category encoding
    if not CATEGORY_MAP:
        cats = sorted(df["category"].dropna().unique())
        CATEGORY_MAP = {c: i for i, c in enumerate(cats)}
    df["category_enc"] = df["category"].map(CATEGORY_MAP).fillna(-1).astype(int)

    # market_type encoding
    if not MARKET_TYPE_MAP:
        mts = sorted(df["market_type"].dropna().unique())
        MARKET_TYPE_MAP = {m: i for i, m in enumerate(mts)}
    df["market_type_enc"] = df["market_type"].map(MARKET_TYPE_MAP).fillna(0).astype(int)

    return df


GENERIC_FEATURES = [
    "market_price", "price_lag1",
    "volume_fp", "volume_rank",
    "days_in_market",
    "day_of_week", "month", "doy_sin", "doy_cos",
    "floor_strike_pct",
    "hist_win_rate_series",
    "category_enc",
    "market_type_enc",
]

# ── NOAA / weather features ───────────────────────────────────────────────────

def build_weather_features(df_weather: pd.DataFrame, noaa: pd.DataFrame) -> pd.DataFrame:
    """Add NOAA rolling features to weather markets."""
    df = df_weather.copy()

    # Normalise city column
    df["city"] = df["series_ticker"].map(WEATHER_SERIES)
    df["date"] = pd.to_datetime(df["close_date"])

    noaa = noaa.copy()
    noaa["date"] = pd.to_datetime(noaa["date"])
    noaa = noaa.sort_values(["city", "date"])

    for window in [3, 7, 14]:
        noaa[f"tmax_roll{window}"] = (
            noaa.groupby("city")["tmax"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        noaa[f"tmax_std{window}"] = (
            noaa.groupby("city")["tmax"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )
    noaa["tmax_lag1"]     = noaa.groupby("city")["tmax"].transform(lambda x: x.shift(1))
    noaa["tmax_trend3"]   = (
        noaa.groupby("city")["tmax"].transform(lambda x: x.shift(1))
        - noaa.groupby("city")["tmax"].transform(lambda x: x.shift(3))
    )
    noaa["tmax_zscore14"] = (
        (noaa["tmax_lag1"] - noaa["tmax_roll14"])
        / noaa["tmax_std14"].replace(0, np.nan)
    )

    # Parse bracket_strike from ticker if floor_strike is NaN
    def parse_strike(ticker: str) -> float | None:
        m = re.search(r"-[TB]([\d.]+)$", str(ticker))
        return float(m.group(1)) if m else None

    if df["floor_strike"].isna().any():
        df.loc[df["floor_strike"].isna(), "floor_strike"] = (
            df.loc[df["floor_strike"].isna(), "ticker"].apply(parse_strike)
        )

    df = df.merge(
        noaa[["city", "date",
              "tmax_roll3", "tmax_roll7", "tmax_roll14",
              "tmax_std7", "tmax_std14",
              "tmax_lag1", "tmax_trend3", "tmax_zscore14"]],
        on=["city", "date"], how="left",
    )

    df["bracket_dist_lag1"]  = df["floor_strike"] - df["tmax_lag1"]
    df["bracket_dist_roll7"] = df["floor_strike"] - df["tmax_roll7"]
    df["bracket_pct_roll7"]  = (df["floor_strike"] / df["tmax_roll7"].replace(0, np.nan)).clip(0, 5)
    df["bracket_strike"]     = df["floor_strike"]

    return df


WEATHER_EXTRA_FEATURES = [
    "tmax_roll3", "tmax_roll7", "tmax_roll14",
    "tmax_std7", "tmax_std14",
    "tmax_lag1", "tmax_trend3", "tmax_zscore14",
    "bracket_dist_lag1", "bracket_dist_roll7",
    "bracket_pct_roll7", "bracket_strike",
]

WEATHER_FEATURES = GENERIC_FEATURES + WEATHER_EXTRA_FEATURES
TARGET_COL       = "outcome"


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    df_raw  = load_bulk()
    df      = clean_bulk(df_raw)
    print(f"[Features] After clean (result yes/no, price>0): {len(df):,} rows")
    print(f"           Date range: {df['close_date'].min()} → {df['close_date'].max()}")
    print(f"           Categories: {df['category'].value_counts().head(10).to_dict()}")

    df = build_generic_features(df)

    # ── All-market feature matrix ─────────────────────────────────────────────
    drop_early = df.dropna(subset=["price_lag1", "hist_win_rate_series"])
    all_mat    = drop_early[
        ["series_ticker", "category", "close_date", "outcome"] + GENERIC_FEATURES
    ].copy()
    out_all = PROC_DIR / "feature_matrix_all.parquet"
    all_mat.to_parquet(out_all, index=False)
    print(f"\n[Features] All-market matrix: {len(all_mat):,} rows × {len(GENERIC_FEATURES)} features → {out_all}")

    # ── Weather feature matrix ────────────────────────────────────────────────
    weather_raw = df[df["series_ticker"].isin(WEATHER_SERIES)].copy()
    print(f"\n[Features] Weather markets (raw): {len(weather_raw):,} rows")

    noaa_path = PROC_DIR / "noaa_all_cities_max.parquet"
    if not noaa_path.exists():
        noaa_path = PROC_DIR / "noaa_all_cities.parquet"
    noaa = pd.read_parquet(noaa_path)
    print(f"[Features] NOAA rows: {len(noaa):,} | {noaa['date'].min()} → {noaa['date'].max()}")

    weather = build_weather_features(weather_raw, noaa)
    weather_mat = weather.dropna(subset=["tmax_lag1", "bracket_dist_lag1"])
    out_wx = PROC_DIR / "feature_matrix_weather.parquet"
    weather_mat.to_parquet(out_wx, index=False)
    print(f"[Features] Weather matrix: {len(weather_mat):,} rows × {len(WEATHER_FEATURES)} features → {out_wx}")

    # Per-city summary
    for city, grp in weather_mat.groupby("city"):
        dates = pd.to_datetime(grp["close_date"])
        print(f"  {city:<10} rows={len(grp):>4} | {dates.min().date()} → {dates.max().date()} | win={grp['outcome'].mean():.1%}")


if __name__ == "__main__":
    run()
