"""
features_clean.py — Lookahead-free feature engineering
=======================================================
Fixes the two primary lookahead sources identified in backtest audit:

  1. market_price / price_lag1 removed from ML features.
     Both are derived from last_price_dollars which is the final traded
     price of a SETTLED market — not observable at entry time.
     price_lag1 is retained as a METADATA column for use as the entry
     price proxy in backtest_clean.py (it's the previous market's
     settlement price, observable when the current market opens).

  2. Macro features (BTC, fear/greed) joined on open_date, not close_date.
     At entry time you know today's macro, not the macro on the future
     resolution date.

Clean feature set (CLEAN_FEATURES):
  days_in_market, day_of_week, month, day_of_year, doy_sin, doy_cos,
  floor_strike_pct, volume_rank, hist_win_rate_series,
  category_enc, market_type_enc,
  fear_greed_value, btc_ret_1d, btc_ret_7d, btc_vol_20d, btc_zscore_30,
  smart_money_index, whale_cluster_detected

Outputs:
  data/processed/feature_matrix_clean_all.parquet    — all markets
  data/processed/feature_matrix_clean_weather.parquet — weather markets
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features_generic import (
    load_bulk, clean_bulk, build_generic_features, build_weather_features,
    WEATHER_FEATURES, WEATHER_EXTRA_FEATURES, WEATHER_SERIES, TARGET_COL,
)
from features_v2 import build_macro_features, build_wiki_features, join_whale_features

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")

# Features that carry lookahead — excluded from model training
LOOKAHEAD_FEATURES = {"market_price", "price_lag1"}

# ── Clean generic feature list ────────────────────────────────────────────────
CLEAN_FEATURES = [
    # Market structure (static / observable at open)
    "days_in_market",
    "floor_strike_pct",
    "market_type_enc",
    "category_enc",
    # Calendar (from open_date — no lookahead)
    "day_of_week",
    "month",
    "day_of_year",
    "doy_sin",
    "doy_cos",
    # Historical series statistics (lagged, no lookahead)
    "hist_win_rate_series",
    "volume_rank",
    # Macro on open_date (observable at entry, not future close_date)
    "fear_greed_value",
    "btc_ret_1d",
    "btc_ret_7d",
    "btc_vol_20d",
    "btc_zscore_30",
    # Whale / sentiment (approximate — no strict lookahead since static)
    "smart_money_index",
    "whale_cluster_detected",
]

# Weather features (NOAA rolling/lagged — relative to close_date, which is the
# target date; these use lagged temperature data, not the actual close_date temp)
CLEAN_WEATHER_EXTRA = [f for f in WEATHER_EXTRA_FEATURES
                       if f not in LOOKAHEAD_FEATURES]


def add_open_date(df: pd.DataFrame) -> pd.DataFrame:
    """Add open_date column from open_time."""
    df = df.copy()
    if "open_time" in df.columns:
        df["open_date"] = pd.to_datetime(df["open_time"]).dt.date
    else:
        df["open_date"] = df["close_date"]  # fallback
    return df


def join_macro_on_open_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join macro features (BTC, fear/greed) on open_date.
    This is the critical fix: we know today's macro when deciding to enter,
    not the macro on the future close/resolution date.
    """
    macro = build_macro_features()
    macro["date"] = pd.to_datetime(macro["date"]).dt.date

    df["open_date"] = pd.to_datetime(df["open_date"]).dt.date

    df = df.merge(macro, left_on="open_date", right_on="date",
                  how="left", suffixes=("", "_macro"))
    df = df.drop(columns=["date_macro"], errors="ignore")

    # Fill macro NaNs
    for col in ["fear_greed_value"]:
        if col in df.columns:
            df[col] = df[col].fillna(50)
    for col in ["btc_ret_1d", "btc_ret_7d", "btc_vol_20d", "btc_zscore_30"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def run() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print("[FeatClean] Loading and cleaning bulk data...")
    df_raw = load_bulk()
    df     = clean_bulk(df_raw)

    # Add open_date (needed for macro join)
    df = add_open_date(df)
    print(f"[FeatClean] {len(df):,} rows after clean")

    # Generic features (these include market_price, price_lag1, volume_rank, etc.)
    print("[FeatClean] Building generic features...")
    df = build_generic_features(df)

    # Whale features (approximate, no strict lookahead)
    df = join_whale_features(df)

    # Join macro on OPEN_DATE (fix: was close_date)
    print("[FeatClean] Joining macro on open_date (not close_date)...")
    df = join_macro_on_open_date(df)

    # ── All-market clean matrix ───────────────────────────────────────────────
    avail_clean = [c for c in CLEAN_FEATURES if c in df.columns]
    missing = [c for c in CLEAN_FEATURES if c not in df.columns]
    if missing:
        print(f"[FeatClean] Missing features (skipped): {missing}")

    # Keep metadata columns alongside features for backtest use
    meta_only = ["ticker", "series_ticker", "category", "open_date", "close_date",
                 "outcome", "market_price", "price_lag1", "volume_fp"]
    meta_only = [c for c in meta_only if c in df.columns]
    # Combine, dedup while preserving order
    all_cols = list(dict.fromkeys(meta_only + avail_clean))

    df_all = df.dropna(subset=["hist_win_rate_series"]).copy()
    out_all = PROC_DIR / "feature_matrix_clean_all.parquet"
    df_all[all_cols].to_parquet(out_all, index=False)
    print(f"[FeatClean] All-market clean: {len(df_all):,} rows × {len(avail_clean)} features → {out_all}")
    # meta_cols for weather reuse
    meta_cols = meta_only
    print(f"  Features: {avail_clean}")
    print(f"  Removed (lookahead): {sorted(LOOKAHEAD_FEATURES)}")

    # ── Weather clean matrix ──────────────────────────────────────────────────
    weather_raw = df[df["series_ticker"].isin(WEATHER_SERIES)].copy()
    print(f"\n[FeatClean] Weather markets: {len(weather_raw):,} rows")

    noaa_path = PROC_DIR / "noaa_all_cities_max.parquet"
    if not noaa_path.exists():
        noaa_path = PROC_DIR / "noaa_all_cities.parquet"

    if noaa_path.exists():
        noaa = pd.read_parquet(noaa_path)
        weather_raw["close_date"] = pd.to_datetime(weather_raw["close_date"]).dt.date
        weather = build_weather_features(weather_raw, noaa)
        weather = add_open_date(weather) if "open_date" not in weather.columns else weather
        weather = join_macro_on_open_date(weather)

        avail_wx = list(dict.fromkeys(
            c for c in WEATHER_FEATURES + CLEAN_WEATHER_EXTRA
            if c in weather.columns and c not in LOOKAHEAD_FEATURES
        ))
        wx_meta  = [c for c in meta_cols if c in weather.columns]
        wx_cols  = list(dict.fromkeys(wx_meta + avail_wx))  # dedup, preserve order

        weather_mat = weather.dropna(subset=["tmax_lag1", "bracket_dist_lag1"])
        out_wx = PROC_DIR / "feature_matrix_clean_weather.parquet"
        weather_mat[wx_cols].to_parquet(out_wx, index=False)
        print(f"[FeatClean] Weather clean: {len(weather_mat):,} rows × {len(avail_wx)} features → {out_wx}")
    else:
        print(f"[FeatClean] NOAA parquet not found at {noaa_path}, skipping weather matrix.")

    print(f"\n[FeatClean] Done. Clean features: {len(avail_clean)}, no lookahead.")


if __name__ == "__main__":
    run()
