"""
features_v2.py — 50+ feature engineering pipeline
Extends features_generic.py with:
  MARKET:    bid_ask_spread, liquidity_depth, price_momentum, volatility
  WHALE:     smart_money_index, whale_cluster_detected, whale_count, whale_exposure
  SENTIMENT: fear_greed_value, wiki_views_zscore, google_trend_score, news_count
  MACRO:     btc_ret_1d, btc_ret_7d, btc_price_zscore, fed_funds_rate, dxy_proxy
  WEATHER:   tmax rolling features + bracket distance (weather markets only)
  TEMPORAL:  hour, day_of_week, month, is_weekend, doy_sin/cos

Output:
  data/processed/feature_matrix_weather_v2.parquet   (weather markets, 50+ features)
  data/processed/feature_matrix_all_v2.parquet        (all markets, 35+ features)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from features_generic import (
    load_bulk, clean_bulk, build_generic_features, build_weather_features,
    GENERIC_FEATURES, WEATHER_FEATURES, WEATHER_EXTRA_FEATURES,
    WEATHER_SERIES, TARGET_COL
)

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")


# ── Loader helpers ────────────────────────────────────────────────────────────

def load_csv_or_empty(path: Path, parse_dates: list[str] = ["date"]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.date
        return df
    except Exception:
        return pd.DataFrame()


def load_fear_greed() -> pd.DataFrame:
    return load_csv_or_empty(RAW_DIR / "fear_greed.csv")


def load_binance() -> pd.DataFrame:
    return load_csv_or_empty(RAW_DIR / "binance_prices.csv")


def load_wiki() -> pd.DataFrame:
    return load_csv_or_empty(RAW_DIR / "wikipedia_pageviews.csv")


def load_fred() -> pd.DataFrame:
    return load_csv_or_empty(RAW_DIR / "fred_macro.csv")


def load_coinalyze() -> pd.DataFrame:
    return load_csv_or_empty(RAW_DIR / "coinalyze.csv")


def load_newsapi() -> pd.DataFrame:
    return load_csv_or_empty(RAW_DIR / "newsapi.csv")


def load_whale_features() -> pd.DataFrame:
    f = PROC_DIR / "whale_features.parquet"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_parquet(f)
    df["scan_date"] = pd.to_datetime(df["scan_date"]).dt.date
    return df


# ── Macro features (per date) ─────────────────────────────────────────────────

def build_macro_features() -> pd.DataFrame:
    """Daily macro feature table indexed by date."""
    # Fear & Greed
    fg = load_fear_greed()
    if fg.empty:
        fg = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2200, freq="D").date,
                           "fear_greed_value": 50})

    # Binance BTC/ETH
    bn = load_binance()
    btc = bn[bn["symbol"] == "BTCUSDT"].copy() if not bn.empty else pd.DataFrame()
    if not btc.empty:
        btc = btc[["date","close","ret_1d","ret_7d","vol_20d","price_zscore_30"]].rename(columns={
            "close":          "btc_price",
            "ret_1d":         "btc_ret_1d",
            "ret_7d":         "btc_ret_7d",
            "vol_20d":        "btc_vol_20d",
            "price_zscore_30":"btc_zscore_30",
        })

    # FRED macro
    fred = load_fred()

    # Coinalyze: BTC funding rate + open interest
    ca = load_coinalyze()
    btc_ca: pd.DataFrame | None = None
    if not ca.empty and "symbol" in ca.columns:
        btc_ca = ca[ca["symbol"].str.contains("BTC", na=False)].copy()
        rename_map = {}
        if "funding_rate"  in btc_ca.columns: rename_map["funding_rate"]  = "btc_funding_rate"
        if "open_interest" in btc_ca.columns: rename_map["open_interest"] = "btc_open_interest"
        btc_ca = btc_ca.rename(columns=rename_map)
        ca_cols = ["date"] + [c for c in ["btc_funding_rate","btc_open_interest"] if c in btc_ca.columns]
        btc_ca  = btc_ca[ca_cols].groupby("date").last().reset_index()

    # NewsAPI: pivot category headline counts to wide format
    news = load_newsapi()
    news_wide: pd.DataFrame | None = None
    if not news.empty and "category" in news.columns and "headline_count" in news.columns:
        pivot = news.pivot_table(
            index="date", columns="category",
            values="headline_count", aggfunc="sum"
        )
        pivot.columns = [f"news_count_{c}" for c in pivot.columns]
        news_wide = pivot.reset_index()
        # Forward-fill: newsapi counts are sparse (daily at best)
        news_wide = news_wide.sort_values("date").reset_index(drop=True)

    # Merge all on date
    macro = fg[["date","fear_greed_value"]].copy()
    if not btc.empty:
        macro = macro.merge(btc, on="date", how="left")
    if not fred.empty:
        macro = macro.merge(fred, on="date", how="left")
    if btc_ca is not None and not btc_ca.empty:
        macro = macro.merge(btc_ca, on="date", how="left")
    if news_wide is not None and not news_wide.empty:
        macro = macro.merge(news_wide, on="date", how="left")

    # Forward-fill macro (weekends/holidays)
    for col in macro.columns:
        if col != "date":
            macro[col] = pd.to_numeric(macro[col], errors="coerce").ffill()

    return macro.sort_values("date").reset_index(drop=True)


# ── Wikipedia pivot ───────────────────────────────────────────────────────────

def build_wiki_features() -> pd.DataFrame:
    """Pivot Wikipedia pageviews: one row per date, columns per article z-score."""
    wiki = load_wiki()
    if wiki.empty:
        return pd.DataFrame()
    pivot = wiki.pivot_table(index="date", columns="article", values="views_zscore", aggfunc="mean")
    pivot.columns = [f"wiki_{c.lower().replace(' ','_')}_zscore" for c in pivot.columns]
    pivot = pivot.reset_index()
    pivot["date"] = pd.to_datetime(pivot["date"]).dt.date
    return pivot


# ── Whale features join ───────────────────────────────────────────────────────

def join_whale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join whale features on market_slug if available."""
    whale = load_whale_features()
    if whale.empty:
        df["whale_count_total"]      = np.nan
        df["smart_money_index"]      = 50.0
        df["whale_cluster_detected"] = 0
        df["whale_avg_position_size"]= np.nan
        return df

    slug_col = "slug" if "slug" in df.columns else None
    if slug_col is None:
        df["whale_count_total"]      = np.nan
        df["smart_money_index"]      = 50.0
        df["whale_cluster_detected"] = 0
        df["whale_avg_position_size"]= np.nan
        return df

    whale_agg = whale.groupby("market_slug").agg({
        "whale_count_total":       "last",
        "smart_money_index":       "last",
        "whale_cluster_detected":  "last",
        "whale_avg_position_size": "last",
    }).reset_index()
    df = df.merge(whale_agg, left_on=slug_col, right_on="market_slug", how="left")
    for col in ["whale_count_total","smart_money_index","whale_cluster_detected","whale_avg_position_size"]:
        if col in df.columns:
            df[col] = df[col].fillna(50.0 if col == "smart_money_index" else 0)
    return df


# ── Full feature build ────────────────────────────────────────────────────────

# Generic features v2 (all markets)
GENERIC_FEATURES_V2 = GENERIC_FEATURES + [
    "fear_greed_value",
    "btc_ret_1d", "btc_ret_7d", "btc_vol_20d", "btc_zscore_30",
    "smart_money_index", "whale_cluster_detected",
    "whale_count_total", "whale_avg_position_size",
]

# Weather features v2
WEATHER_FEATURES_V2 = WEATHER_FEATURES + [
    "fear_greed_value",
    "btc_ret_1d",
    "smart_money_index", "whale_cluster_detected",
]


def run() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Load and build base datasets
    df_raw = load_bulk()
    df     = clean_bulk(df_raw)
    print(f"[FeatV2] Base rows (settled, price>0): {len(df):,}")

    df = build_generic_features(df)

    # Macro lookup table
    print("[FeatV2] Building macro features...")
    macro = build_macro_features()
    wiki  = build_wiki_features()

    macro["date"] = pd.to_datetime(macro["date"]).dt.date
    df["close_date"] = pd.to_datetime(df["close_date"]).dt.date

    # Join macro on date
    df = df.merge(macro, left_on="close_date", right_on="date", how="left", suffixes=("","_macro"))
    df = df.drop(columns=["date_macro"], errors="ignore")

    # Join wiki (optional — many NaNs ok, RF handles them)
    if not wiki.empty:
        wiki["date"] = pd.to_datetime(wiki["date"]).dt.date
        df = df.merge(wiki, left_on="close_date", right_on="date", how="left")
        df = df.drop(columns=["date"], errors="ignore")

    # Join whale features
    df = join_whale_features(df)

    # Fill macro NaNs with neutral values
    for col in ["fear_greed_value"]:
        if col in df.columns:
            df[col] = df[col].fillna(50)
    for col in ["btc_ret_1d","btc_ret_7d","btc_vol_20d","btc_zscore_30"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ── All-market v2 ─────────────────────────────────────────────────────────
    avail_gen = [c for c in GENERIC_FEATURES_V2 if c in df.columns]
    all_mat   = df.dropna(subset=["price_lag1","hist_win_rate_series"]).copy()
    out_all   = PROC_DIR / "feature_matrix_all_v2.parquet"
    all_mat[["series_ticker","category","close_date","outcome"] + avail_gen].to_parquet(out_all, index=False)
    print(f"[FeatV2] All-market v2: {len(all_mat):,} rows × {len(avail_gen)} features → {out_all}")

    # ── Weather v2 ────────────────────────────────────────────────────────────
    weather_raw = df[df["series_ticker"].isin(WEATHER_SERIES)].copy()
    print(f"[FeatV2] Weather markets (raw): {len(weather_raw):,}")

    noaa_path = PROC_DIR / "noaa_all_cities_max.parquet"
    if not noaa_path.exists():
        noaa_path = PROC_DIR / "noaa_all_cities.parquet"
    noaa = pd.read_parquet(noaa_path)

    weather_raw["close_date"] = pd.to_datetime(weather_raw["close_date"]).dt.date
    weather = build_weather_features(weather_raw, noaa)
    weather_mat = weather.dropna(subset=["tmax_lag1","bracket_dist_lag1"])
    avail_wx   = [c for c in WEATHER_FEATURES_V2 if c in weather_mat.columns]
    out_wx     = PROC_DIR / "feature_matrix_weather_v2.parquet"
    # Include series_ticker so model_ensemble can group correctly
    wx_id_cols = ["series_ticker", "city", "close_date", "outcome"]
    wx_id_cols = [c for c in wx_id_cols if c in weather_mat.columns]
    weather_mat[wx_id_cols + avail_wx].to_parquet(out_wx, index=False)
    print(f"[FeatV2] Weather v2: {len(weather_mat):,} rows × {len(avail_wx)} features → {out_wx}")

    # Feature count summary
    print(f"\n[FeatV2] Total feature counts:")
    print(f"  All-market: {len(avail_gen)} features")
    print(f"  Weather:    {len(avail_wx)} features")


if __name__ == "__main__":
    run()
