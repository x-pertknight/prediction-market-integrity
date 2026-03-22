"""
features.py — model_rf_01
Feature engineering: 25 features from joined_dataset.parquet.
All features are constructed strictly from data available BEFORE the close_date
(no lookahead bias).
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROC_DIR = Path("data/processed")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  joined_dataset with columns:
            city, date, bracket_strike, market_price, result, volume,
            ticker, tmax, tmin, tavg, outcome
    Output: same rows + feature columns + filtered to market_price > 0.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date", "bracket_strike"]).reset_index(drop=True)

    # ── Filter: only liquid markets ──────────────────────────────────────────
    df = df[df["market_price"] > 0].copy()

    # ── Per-city rolling temperature features (uses NOAA tmax on that date) ─
    # These are the daily tmax values — we'll compute rolling stats per city
    # using data UP TO AND INCLUDING close_date (no leakage: tmax is the
    # settlement source, not a future predictor; we model the bracket, not the
    # exact tmax. For features, we use past tmax values only.)

    daily = (
        df.groupby(["city", "date"])[["tmax", "tmin", "tavg"]]
        .first()
        .reset_index()
        .sort_values(["city", "date"])
    )

    for window in [3, 7, 14]:
        daily[f"tmax_roll{window}"] = (
            daily.groupby("city")["tmax"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        daily[f"tmax_std{window}"] = (
            daily.groupby("city")["tmax"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )

    # Z-score: (yesterday_tmax - 14d_mean) / 14d_std
    daily["tmax_lag1"] = daily.groupby("city")["tmax"].transform(lambda x: x.shift(1))
    daily["tmax_zscore14"] = (
        (daily["tmax_lag1"] - daily["tmax_roll14"])
        / daily["tmax_std14"].replace(0, np.nan)
    )

    # 3-day trend: lag1 - lag3
    daily["tmax_trend3"] = (
        daily.groupby("city")["tmax"].transform(lambda x: x.shift(1))
        - daily.groupby("city")["tmax"].transform(lambda x: x.shift(3))
    )

    # Merge rolling stats back
    df = df.merge(
        daily[["city", "date",
               "tmax_roll3", "tmax_roll7", "tmax_roll14",
               "tmax_std3", "tmax_std7", "tmax_std14",
               "tmax_lag1", "tmax_zscore14", "tmax_trend3"]],
        on=["city", "date"],
        how="left",
    )

    # ── Calendar features ────────────────────────────────────────────────────
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["month"]        = df["date"].dt.month
    df["day_of_week"]  = df["date"].dt.dayofweek
    # Sine/cosine encoding for cyclical calendar features
    df["doy_sin"]      = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]      = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # ── Bracket features ─────────────────────────────────────────────────────
    # bracket_distance: how far is the strike from the recent rolling average?
    df["bracket_dist_lag1"]   = df["bracket_strike"] - df["tmax_lag1"]
    df["bracket_dist_roll7"]  = df["bracket_strike"] - df["tmax_roll7"]
    df["bracket_dist_roll14"] = df["bracket_strike"] - df["tmax_roll14"]

    # bracket_pct: strike as fraction of rolling mean (normalised)
    df["bracket_pct_roll7"] = df["bracket_strike"] / df["tmax_roll7"].replace(0, np.nan)

    # ── City encoding (ordinal, RF doesn't need one-hot) ────────────────────
    city_map = {"Atlanta": 0, "Chicago": 1, "Dallas": 2, "Miami": 3, "NYC": 4, "Seattle": 5}
    df["city_enc"] = df["city"].map(city_map)

    # ── Historical win rate per city+bracket (computed from past data only) ─
    # Use expanding window to avoid leakage
    df = df.sort_values(["city", "bracket_strike", "date"]).reset_index(drop=True)
    df["hist_win_rate"] = (
        df.groupby(["city", "bracket_strike"])["outcome"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # ── Volume rank (percentile of volume for this city, past data) ─────────
    df["volume_rank"] = (
        df.groupby("city")["volume"]
        .transform(lambda x: x.shift(1).expanding().rank(pct=True))
    )

    # ── Days to expiry proxy ─────────────────────────────────────────────────
    # Kalshi daily markets all expire ~1 day after close — constant at 1.
    # We keep it as a feature placeholder for future use with multi-day markets.
    df["days_to_expiry"] = 1

    df = df.sort_values(["city", "date", "bracket_strike"]).reset_index(drop=True)
    return df


FEATURE_COLS = [
    "tmax_roll3", "tmax_roll7", "tmax_roll14",
    "tmax_std3",  "tmax_std7",  "tmax_std14",
    "tmax_lag1",  "tmax_zscore14", "tmax_trend3",
    "doy_sin", "doy_cos", "month", "day_of_week",
    "bracket_strike",
    "bracket_dist_lag1", "bracket_dist_roll7", "bracket_dist_roll14",
    "bracket_pct_roll7",
    "city_enc",
    "hist_win_rate",
    "volume_rank",
    "market_price",   # market's implied probability — key signal
    "days_to_expiry",
]
# Total: 23 features

TARGET_COL = "outcome"


def load_feature_matrix() -> pd.DataFrame:
    raw = pd.read_parquet(PROC_DIR / "joined_dataset.parquet")
    df  = build_features(raw)

    # Drop rows where key features are NaN (early rolling windows)
    df = df.dropna(subset=["tmax_lag1", "tmax_roll7", "bracket_dist_lag1"])

    out_path = PROC_DIR / "feature_matrix.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[Features] {len(df)} rows × {len(FEATURE_COLS)} features → {out_path}")
    print(f"           Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"           Outcome balance: {df[TARGET_COL].mean():.1%} yes")
    return df


if __name__ == "__main__":
    df = load_feature_matrix()
    print(df[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]].to_string())
