"""
src/data/polymarket_feature_adapter.py — Map Polymarket Gamma API fields to RF model features.

Problem: The RF model was trained on Kalshi data with 29 CLEAN_FEATURES.
Polymarket's Gamma API has different fields and NO tags/category fields.
This adapter bridges the gap.

Feature mapping strategy:
  1. DIRECT MAP: Polymarket field → model feature (calendar, some market structure)
  2. PROXY: Use a related Polymarket field as reasonable approximation
  3. TRAINING MEDIAN: When no proxy exists, use training-set median (documented)

Training-set medians (from wf_clean_general.parquet):
  hist_win_rate_series:  0.0149  (most series have very low win rates)
  volume_rank:           0.52
  days_in_market:        0.0417
  floor_strike_pct:      1.0
  market_type_enc:       0       (constant in training data)
  smart_money_index:     50.0    (constant in training data — no signal)
  whale_cluster_detected: 0      (constant in training data — no signal)
"""

import json
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("paper_trader")

# ── Model feature order ───────────────────────────────────────────────────────
# The RF model was trained on these 20 features (the 8 constant/NaN features
# from CLEAN_FEATURES were excluded by model_clean.py's `nunique() > 1` filter):
#   Excluded: market_type_enc, news_count_*, smart_money_index, whale_cluster_detected

MODEL_FEATURES = [
    "days_in_market", "floor_strike_pct", "category_enc",
    "day_of_week", "month", "day_of_year", "doy_sin", "doy_cos",
    "hist_win_rate_series", "volume_rank",
    "fear_greed_value", "btc_ret_1d", "btc_ret_7d", "btc_vol_20d", "btc_zscore_30",
    "dxy_proxy", "fed_funds_rate", "yield_10y",
    "btc_funding_rate", "btc_open_interest",
]

# ── Training-set medians for features without a Polymarket proxy ──────────────

TRAINING_MEDIANS = {
    "hist_win_rate_series": 0.0149,   # most Kalshi series resolve NO
    "volume_rank":          0.52,
    "floor_strike_pct":     1.0,
    "market_type_enc":      0,        # constant in training data
    "smart_money_index":    50.0,     # constant in training data
    "whale_cluster_detected": 0,      # constant in training data
}

# ── Category encoding (same as training: Kalshi categories → int) ─────────────

_CAT_ENC = {
    "Climate and Weather": 0, "Companies": 1, "Crypto": 2, "Economics": 3,
    "Education": 4, "Elections": 5, "Entertainment": 6, "Exotics": 7,
    "Financials": 8, "Health": 9, "Mentions": 10, "Politics": 11,
    "Science and Technology": 12, "Social": 13, "Sports": 14,
    "Transportation": 15, "World": 16,
}

# ── Keyword-based category inference ──────────────────────────────────────────
# Polymarket Gamma API returns tags=None and category=None for all markets.
# We infer category from slug + question text using keyword matching.

_KEYWORD_RULES = [
    # (keywords_in_slug_or_question, kalshi_category)
    # Order matters: first match wins. More specific patterns first.
    (r"\b(bitcoin|btc|ethereum|eth|crypto|solana|sol|doge|xrp|binance|coinbase)\b", "Crypto"),
    (r"\b(nfl|nba|mlb|nhl|ncaa|fifa|world cup|champions league|premier league|"
     r"superbowl|super bowl|playoff|mvp|touchdown|goal scorer|ufc|boxing|f1|"
     r"formula.?1|grand prix|tennis|wimbledon|us open|olympics)\b", "Sports"),
    (r"\b(trump|biden|president|congress|senate|house|republican|democrat|"
     r"impeach|election|electoral|nominee|primary|caucus|governor|mayor|"
     r"political|gop|dnc|rnc|kamala|desantis|vote)\b", "Politics"),
    (r"\b(election|electoral|ballot|midterm|runoff|swing state)\b", "Elections"),
    (r"\b(fed|federal reserve|interest rate|inflation|cpi|gdp|unemployment|"
     r"recession|tariff|jobs report|non.?farm|fomc|rate cut|rate hike)\b", "Economics"),
    (r"\b(stock|s&p|nasdaq|dow|spy|qqq|vix|market cap|ipo|earnings|"
     r"wall street|bond|treasury|yield|forex|gold price)\b", "Financials"),
    (r"\b(weather|hurricane|tornado|earthquake|temperature|flood|wildfire|"
     r"storm|climate|snowfall|rainfall)\b", "Climate and Weather"),
    (r"\b(movie|oscar|grammy|emmy|album|spotify|netflix|disney|"
     r"box office|rihanna|taylor swift|drake|kanye|beyonce|gta|"
     r"game release|tv show|celebrity|tiktok|youtube)\b", "Entertainment"),
    (r"\b(covid|pandemic|vaccine|fda|who|health|disease|virus|"
     r"drug approval|clinical trial)\b", "Health"),
    (r"\b(spacex|nasa|ai |artificial intelligence|openai|google|apple|"
     r"microsoft|amazon|tesla|tech|launch|rocket|satellite)\b", "Science and Technology"),
    (r"\b(war|russia|ukraine|china|taiwan|nato|military|invasion|"
     r"ceasefire|peace|sanction|iran|north korea|israel|gaza)\b", "World"),
    (r"\b(airline|flight|aviation|ship|port|train|transport)\b", "Transportation"),
    (r"\b(university|school|student|education|college)\b", "Education"),
    (r"\b(twitter|x\.com|instagram|facebook|social media|troll|"
     r"follower|subscriber)\b", "Social"),
    (r"\b(company|acquisition|merger|ceo|startup|founder|valuation)\b", "Companies"),
]
_COMPILED_RULES = [(re.compile(pat, re.IGNORECASE), cat) for pat, cat in _KEYWORD_RULES]


def infer_category(market: dict) -> str:
    """Infer Kalshi-equivalent category from Polymarket market fields.

    Uses slug + question + event slug for keyword matching.
    Falls back to 'Exotics' (enc=7) for truly unclassifiable markets.
    """
    slug = market.get("slug") or ""
    question = market.get("question") or ""
    events = market.get("events") or []
    event_slug = events[0].get("slug", "") if events else ""
    text = f"{slug} {question} {event_slug}".lower()

    for pattern, category in _COMPILED_RULES:
        if pattern.search(text):
            return category

    # No match — use Exotics (the "other" bucket in Kalshi)
    return "Exotics"


def market_passes_filter(market: dict, target_cats: set[str]) -> bool:
    """Check if a market's inferred category matches target categories.

    Unlike the old tag-based filter, this uses keyword inference so markets
    without tags/category (i.e., all of them) can still pass.
    """
    cat = infer_category(market)
    # Map Kalshi category back to lowercase tag for comparison
    cat_to_tag = {
        "Climate and Weather": "weather", "Crypto": "crypto",
        "Politics": "politics", "Entertainment": "entertainment",
        "Financials": "financials", "Sports": "sports",
        "Economics": "economics", "Elections": "elections",
        "World": "politics",  # World conflicts → politics for filter
        "Companies": "financials",
        "Science and Technology": "entertainment",  # broad bucket
        "Health": "economics",  # broad bucket
    }
    tag = cat_to_tag.get(cat, cat.lower())
    return tag in target_cats


def compute_days_in_market(market: dict) -> float:
    """Compute days since market creation from startDate.

    MAPPING: Polymarket startDate → Kalshi days_in_market
    Polymarket provides startDate as ISO datetime.
    Training data median was 0.0417 (~1 hour).
    """
    start = market.get("startDate") or market.get("startDateIso")
    if not start:
        return TRAINING_MEDIANS.get("days_in_market", 0.04)
    try:
        if "T" in str(start):
            dt = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(str(start)[:10], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        delta = datetime.now(timezone.utc) - dt
        return max(delta.total_seconds() / 86400, 0.001)
    except Exception:
        return 0.04


def compute_volume_rank(market: dict, all_markets: list[dict]) -> float:
    """Approximate volume_rank from Polymarket volume24hr across current markets.

    MAPPING: Polymarket volume24hr → Kalshi volume_rank (percentile)
    Training data used per-series volume percentile from prior dates.
    We approximate with cross-sectional percentile of 24h volume.
    """
    vol = float(market.get("volume24hr") or market.get("volume") or 0)
    if not all_markets:
        return TRAINING_MEDIANS["volume_rank"]
    all_vols = sorted(
        float(m.get("volume24hr") or m.get("volume") or 0)
        for m in all_markets
    )
    if not all_vols or all_vols[-1] == 0:
        return TRAINING_MEDIANS["volume_rank"]
    # Percentile rank
    rank = sum(1 for v in all_vols if v <= vol) / len(all_vols)
    return round(rank, 4)


def build_features(market: dict, macro: dict,
                   all_markets: list[dict] | None = None) -> np.ndarray:
    """Build the 29-feature vector for the RF model from a Polymarket market.

    Returns shape (1, 20) array matching MODEL_FEATURES order.

    Feature mapping documentation:
      days_in_market        → PROXY: computed from Polymarket startDate
      floor_strike_pct      → TRAINING MEDIAN: 1.0 (no strike concept in Polymarket)
      category_enc          → PROXY: keyword inference from slug/question
      day_of_week           → DIRECT: today's weekday
      month                 → DIRECT: today's month
      day_of_year           → DIRECT: today's day of year
      doy_sin, doy_cos      → DIRECT: sinusoidal encoding of day_of_year
      hist_win_rate_series  → TRAINING MEDIAN: 0.0149 (no series history)
      volume_rank           → PROXY: cross-sectional percentile of 24h volume
      fear_greed_value      → DIRECT: from macro CSV (fear_greed.csv)
      btc_*                 → DIRECT: from macro CSV (binance_prices.csv)
      dxy_proxy, fed_*, yield_*  → DIRECT: from macro CSV (fred_macro.csv)
      btc_funding_rate, btc_open_interest → DIRECT: from macro CSV

    Excluded (constant/NaN in training → not in model):
      market_type_enc, news_count_*, smart_money_index, whale_cluster_detected
    """
    today = date.today()
    doy = today.timetuple().tm_yday

    cat = infer_category(market)
    cat_enc = _CAT_ENC.get(cat, 7)  # 7 = Exotics

    vals = {
        # Market structure
        "days_in_market":          compute_days_in_market(market),
        "floor_strike_pct":        TRAINING_MEDIANS["floor_strike_pct"],
        "category_enc":            cat_enc,
        # Calendar (direct — no mapping needed)
        "day_of_week":             today.weekday(),
        "month":                   today.month,
        "day_of_year":             doy,
        "doy_sin":                 np.sin(2 * np.pi * doy / 365),
        "doy_cos":                 np.cos(2 * np.pi * doy / 365),
        # Series statistics (no Polymarket equivalent)
        "hist_win_rate_series":    TRAINING_MEDIANS["hist_win_rate_series"],
        "volume_rank":             compute_volume_rank(
                                       market, all_markets or []),
        # Macro (direct from CSV caches)
        "fear_greed_value":        macro.get("fear_greed_value", 50.0),
        "btc_ret_1d":              macro.get("btc_ret_1d", 0.0),
        "btc_ret_7d":              macro.get("btc_ret_7d", 0.0),
        "btc_vol_20d":             macro.get("btc_vol_20d", 0.0),
        "btc_zscore_30":           macro.get("btc_zscore_30", 0.0),
        "dxy_proxy":               macro.get("dxy_proxy", 0.0),
        "fed_funds_rate":          macro.get("fed_funds_rate", 0.0),
        "yield_10y":               macro.get("yield_10y", 0.0),
        "btc_funding_rate":        macro.get("btc_funding_rate", 0.0),
        "btc_open_interest":       macro.get("btc_open_interest", 0.0),
    }

    vec = np.array([vals.get(f, 0) for f in MODEL_FEATURES], dtype=np.float64)
    return vec.reshape(1, -1)


def diagnose_features(market: dict, macro: dict,
                      all_markets: list[dict] | None = None) -> dict:
    """Return a diagnostic dict showing each feature, its value, and mapping type.

    Useful for debugging and verifying the adapter is working correctly.
    """
    X = build_features(market, macro, all_markets)
    vec = X[0]

    mapping_types = {
        "days_in_market": "PROXY (startDate)",
        "floor_strike_pct": "TRAINING MEDIAN",
        "category_enc": f"PROXY (keyword → {infer_category(market)})",
        "day_of_week": "DIRECT",
        "month": "DIRECT",
        "day_of_year": "DIRECT",
        "doy_sin": "DIRECT",
        "doy_cos": "DIRECT",
        "hist_win_rate_series": "TRAINING MEDIAN",
        "volume_rank": "PROXY (cross-sectional 24h vol)",
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
        is_nan = np.isnan(v) if isinstance(v, float) else False
        diag[feat] = {
            "value": float(v),
            "nan": is_nan,
            "mapping": mapping_types.get(feat, "UNKNOWN"),
        }
    return diag
