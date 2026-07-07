"""
src/data/kalshi_feature_adapter.py — Map Kalshi Prod API fields to RF model features.

Native Kalshi adapter: the RF model was trained on Kalshi data, so most features
map directly rather than needing proxies/medians like the Polymarket adapter.

Feature mapping:
  days_in_market        → DIRECT: (close_time - open_time) in days
  floor_strike_pct      → TRAINING MEDIAN: 1.0 (expanding mean is complex live)
  category_enc          → DIRECT: Kalshi category field from series endpoint
  calendar              → DIRECT: from close_time date
  hist_win_rate_series  → COMPUTED: settled markets per series from prod API
  volume_rank           → COMPUTED: cross-sectional percentile of volume_fp
  macro                 → DIRECT: from CSV caches (fear_greed, binance, fred, coinalyze)
"""

import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("paper_trader")

# Authenticated MarketApi — set via set_api_client() from paper_trader
_market_api = None


def set_api_client(api_client) -> None:
    """Inject an authenticated ApiClient for prod API calls (win rate fetches).

    Called from paper_trader at startup. The api_client should already have
    set_kalshi_auth() called.
    """
    global _market_api
    from kalshi_python_sync import MarketApi
    _market_api = MarketApi(api_client)
    log.info("Kalshi MarketApi injected for win rate fetches")


# ── Model feature order (20 features, same as polymarket_feature_adapter) ────

MODEL_FEATURES = [
    "days_in_market", "floor_strike_pct", "category_enc",
    "day_of_week", "month", "day_of_year", "doy_sin", "doy_cos",
    "hist_win_rate_series", "volume_rank",
    "fear_greed_value", "btc_ret_1d", "btc_ret_7d", "btc_vol_20d", "btc_zscore_30",
    "dxy_proxy", "fed_funds_rate", "yield_10y",
    "btc_funding_rate", "btc_open_interest",
]

# ── Training-set medians (fallbacks) ─────────────────────────────────────────

TRAINING_MEDIANS = {
    "hist_win_rate_series": 0.0149,
    "volume_rank":          0.52,
    "floor_strike_pct":     1.0,
    "days_in_market":       0.04,
}

# ── Category encoding (same as training) ─────────────────────────────────────

_CAT_ENC = {
    "Climate and Weather": 0, "Companies": 1, "Crypto": 2, "Economics": 3,
    "Education": 4, "Elections": 5, "Entertainment": 6, "Exotics": 7,
    "Financials": 8, "Health": 9, "Mentions": 10, "Politics": 11,
    "Science and Technology": 12, "Social": 13, "Sports": 14,
    "Transportation": 15, "World": 16,
}

# Kalshi API category strings → training category names
_KALSHI_CAT_MAP = {
    "Climate and Weather": "Climate and Weather",
    "Companies": "Companies",
    "Crypto": "Crypto",
    "Economics": "Economics",
    "Education": "Education",
    "Elections": "Elections",
    "Entertainment": "Entertainment",
    "Exotics": "Exotics",
    "Financials": "Financials",
    "Health": "Health",
    "Mentions": "Mentions",
    "Politics": "Politics",
    "Science and Technology": "Science and Technology",
    "Social": "Social",
    "Sports": "Sports",
    "Transportation": "Transportation",
    "World": "World",
}

# For TARGET_CATS filter: Kalshi category → lowercase tag
_CAT_TO_TAG = {
    "Climate and Weather": "weather",
    "Crypto": "crypto",
    "Politics": "politics",
    "Entertainment": "entertainment",
    "Financials": "financials",
    "Sports": "sports",
    "Economics": "economics",
    "Elections": "elections",
    "World": "politics",
    "Companies": "financials",
    "Science and Technology": "entertainment",
    "Health": "economics",
}

# ── Hist win rate cache (in-memory per series_ticker) ────────────────────────

_win_rate_cache: dict[str, float] = {}


def _fetch_series_win_rate(series_ticker: str) -> float:
    """Fetch settled markets for a series and compute win rate.

    Uses authenticated MarketApi against Kalshi prod. Returns
    sum(result=="yes") / count. Cached in memory.
    """
    if series_ticker in _win_rate_cache:
        return _win_rate_cache[series_ticker]

    if _market_api is None:
        log.debug("MarketApi not set — using training median for win rate")
        _win_rate_cache[series_ticker] = TRAINING_MEDIANS["hist_win_rate_series"]
        return TRAINING_MEDIANS["hist_win_rate_series"]

    try:
        yes_count = 0
        total = 0
        cursor = None
        # Fetch up to 2 pages of settled markets
        for _ in range(2):
            resp = _market_api.get_markets(
                series_ticker=series_ticker,
                status="settled",
                limit=100,
                cursor=cursor if cursor else None,
            )
            batch = resp.markets or []
            for m in batch:
                d = m.to_dict() if hasattr(m, "to_dict") else m
                result = d.get("result", "")
                if result in ("yes", "no"):
                    total += 1
                    if result == "yes":
                        yes_count += 1
            cursor = resp.cursor
            if not cursor or not batch:
                break
            time.sleep(0.1)

        if total > 0:
            rate = yes_count / total
            _win_rate_cache[series_ticker] = rate
            return rate
    except Exception as e:
        log.debug(f"Win rate fetch failed for {series_ticker}: {e}")

    # Fallback to training median
    _win_rate_cache[series_ticker] = TRAINING_MEDIANS["hist_win_rate_series"]
    return TRAINING_MEDIANS["hist_win_rate_series"]


def clear_win_rate_cache() -> None:
    """Clear the win rate cache (useful for testing)."""
    _win_rate_cache.clear()


# ── Feature computation ──────────────────────────────────────────────────────

def compute_days_in_market(market: dict) -> float:
    """Compute market duration from open_time to close_time in days.

    Direct Kalshi fields — no proxy needed.
    """
    open_time = market.get("open_time")
    close_time = market.get("close_time")
    if not open_time or not close_time:
        return TRAINING_MEDIANS["days_in_market"]
    try:
        ot = datetime.fromisoformat(open_time.replace("Z", "+00:00"))
        ct = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        delta = (ct - ot).total_seconds() / 86400
        return max(delta, 0.001)
    except Exception:
        return TRAINING_MEDIANS["days_in_market"]


def compute_volume_rank(market: dict, all_markets: list[dict]) -> float:
    """Cross-sectional percentile of volume_fp across all active markets."""
    vol = float(market.get("volume") or market.get("volume_fp") or 0)
    if not all_markets:
        return TRAINING_MEDIANS["volume_rank"]
    all_vols = sorted(
        float(m.get("volume") or m.get("volume_fp") or 0) for m in all_markets
    )
    if not all_vols or all_vols[-1] == 0:
        return TRAINING_MEDIANS["volume_rank"]
    rank = sum(1 for v in all_vols if v <= vol) / len(all_vols)
    return round(rank, 4)


def get_category(market: dict) -> str:
    """Get Kalshi category from market dict.

    Kalshi markets have a `category` field directly. Falls back to Exotics.
    """
    cat = market.get("category") or ""
    if cat in _CAT_ENC:
        return cat
    # Try mapping from API category string
    mapped = _KALSHI_CAT_MAP.get(cat)
    if mapped:
        return mapped
    return "Exotics"


def market_passes_filter(market: dict, target_cats: set[str]) -> bool:
    """Check if market's category matches target categories."""
    cat = get_category(market)
    tag = _CAT_TO_TAG.get(cat, cat.lower())
    return tag in target_cats


def build_features(market: dict, macro: dict,
                   all_markets: list[dict] | None = None) -> np.ndarray:
    """Build 20-feature vector for the RF model from a Kalshi market.

    Returns shape (1, 20) array matching MODEL_FEATURES order.
    """
    # Calendar features from close_time or today
    close_time = market.get("close_time")
    if close_time:
        try:
            ct = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            cal_date = ct.date()
        except Exception:
            cal_date = date.today()
    else:
        cal_date = date.today()

    doy = cal_date.timetuple().tm_yday

    cat = get_category(market)
    cat_enc = _CAT_ENC.get(cat, 7)  # 7 = Exotics

    # Hist win rate from series
    series_ticker = market.get("series_ticker") or market.get("event_ticker") or ""
    if series_ticker:
        hist_wr = _fetch_series_win_rate(series_ticker)
    else:
        hist_wr = TRAINING_MEDIANS["hist_win_rate_series"]

    vals = {
        # Market structure
        "days_in_market":       compute_days_in_market(market),
        "floor_strike_pct":     TRAINING_MEDIANS["floor_strike_pct"],
        "category_enc":         cat_enc,
        # Calendar
        "day_of_week":          cal_date.weekday(),
        "month":                cal_date.month,
        "day_of_year":          doy,
        "doy_sin":              np.sin(2 * np.pi * doy / 365),
        "doy_cos":              np.cos(2 * np.pi * doy / 365),
        # Series statistics
        "hist_win_rate_series": hist_wr,
        "volume_rank":          compute_volume_rank(market, all_markets or []),
        # Macro
        "fear_greed_value":     macro.get("fear_greed_value", 50.0),
        "btc_ret_1d":           macro.get("btc_ret_1d", 0.0),
        "btc_ret_7d":           macro.get("btc_ret_7d", 0.0),
        "btc_vol_20d":          macro.get("btc_vol_20d", 0.0),
        "btc_zscore_30":        macro.get("btc_zscore_30", 0.0),
        "dxy_proxy":            macro.get("dxy_proxy", 0.0),
        "fed_funds_rate":       macro.get("fed_funds_rate", 0.0),
        "yield_10y":            macro.get("yield_10y", 0.0),
        "btc_funding_rate":     macro.get("btc_funding_rate", 0.0),
        "btc_open_interest":    macro.get("btc_open_interest", 0.0),
    }

    vec = np.array([vals.get(f, 0) for f in MODEL_FEATURES], dtype=np.float64)
    return vec.reshape(1, -1)


def diagnose_features(market: dict, macro: dict,
                      all_markets: list[dict] | None = None) -> dict:
    """Return diagnostic dict showing each feature, value, and mapping type."""
    X = build_features(market, macro, all_markets)
    vec = X[0]

    cat = get_category(market)
    mapping_types = {
        "days_in_market": "DIRECT (open_time → close_time)",
        "floor_strike_pct": "TRAINING MEDIAN",
        "category_enc": f"DIRECT (category={cat})",
        "day_of_week": "DIRECT",
        "month": "DIRECT",
        "day_of_year": "DIRECT",
        "doy_sin": "DIRECT",
        "doy_cos": "DIRECT",
        "hist_win_rate_series": "COMPUTED (settled series)",
        "volume_rank": "COMPUTED (cross-sectional)",
        "fear_greed_value": "DIRECT (CSV)",
        "btc_ret_1d": "DIRECT (CSV)",
        "btc_ret_7d": "DIRECT (CSV)",
        "btc_vol_20d": "DIRECT (CSV)",
        "btc_zscore_30": "DIRECT (CSV)",
        "dxy_proxy": "DIRECT (CSV)",
        "fed_funds_rate": "DIRECT (CSV)",
        "yield_10y": "DIRECT (CSV)",
        "btc_funding_rate": "DIRECT (CSV)",
        "btc_open_interest": "DIRECT (CSV)",
    }

    diag = {}
    for i, feat in enumerate(MODEL_FEATURES):
        v = vec[i]
        diag[feat] = {
            "value": float(v),
            "nan": bool(np.isnan(v)),
            "mapping": mapping_types.get(feat, "UNKNOWN"),
        }
    return diag
