"""
src/alpha/bayes_updater.py — Bayesian sequential probability updater.

Chains evidence updates onto RF model probability before it enters
the AlphaCombinator. Each evidence source provides likelihood ratios:

  P(evidence | YES) and P(evidence | NO)

Update rule:
  posterior = P(E|YES) * prior / [P(E|YES)*prior + P(E|NO)*(1-prior)]

Evidence sources:
  1. NewsAPI headlines: negative sentiment → bearish, positive → bullish
  2. FRED macro surprise: hawkish surprise → bearish on rate-sensitive markets
  3. NOAA deviation: weather miss → shift on weather markets

When no evidence is available for a source, prior passes through unchanged.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("paper_trader")

# ── Category mappings for evidence applicability ────────────────────────

# NewsAPI categories → Kalshi market categories they inform
_NEWS_CATEGORY_MAP = {
    "crypto":   {"Crypto"},
    "weather":  {"Climate and Weather"},
    "macro":    {"Economics", "Financials"},
    "politics": {"Politics", "Elections"},
    "sports":   {"Sports"},
}

# FRED-sensitive categories (rate/inflation markets)
_RATE_SENSITIVE = {"Economics", "Financials"}

# Weather categories
_WEATHER_CATS = {"Climate and Weather"}


# ── Core Bayes update ──────────────────────────────────────────────────

def bayes_update(prior: float, p_evidence_given_true: float,
                 p_evidence_given_false: float) -> float:
    """Single Bayesian update step.

    Args:
        prior: P(YES) before seeing evidence
        p_evidence_given_true: P(E | YES)
        p_evidence_given_false: P(E | NO)
    Returns:
        posterior: P(YES | E)
    """
    if prior <= 0.0 or prior >= 1.0:
        return prior
    p_evidence = (p_evidence_given_true * prior +
                  p_evidence_given_false * (1.0 - prior))
    if p_evidence <= 0:
        return prior
    return (p_evidence_given_true * prior) / p_evidence


# ── Evidence extractors ────────────────────────────────────────────────

def _news_evidence(category: str, macro: dict) -> tuple[float, float] | None:
    """Extract news sentiment evidence from cached NewsAPI data.

    Returns (p_yes, p_no) likelihood pair, or None if no evidence.
    Negative headlines → (0.30, 0.70) — bearish
    Positive/high volume → (0.70, 0.30) — bullish
    Neutral → None (skip update)
    """
    newsapi_path = Path("data/raw/newsapi.csv")
    if not newsapi_path.exists():
        return None

    try:
        df = pd.read_csv(newsapi_path)
        if df.empty or "headline_count" not in df.columns:
            return None
    except Exception:
        return None

    # Find relevant news category for this market category
    relevant_news_cat = None
    for news_cat, market_cats in _NEWS_CATEGORY_MAP.items():
        if category in market_cats:
            relevant_news_cat = news_cat
            break
    if relevant_news_cat is None:
        return None

    row = df[df["category"] == relevant_news_cat]
    if row.empty:
        return None

    count = int(row.iloc[0].get("headline_count", 0))
    if count == 0:
        return None

    # High headline volume = uncertainty/negative sentiment (crisis coverage)
    # Threshold: >50 headlines in 7 days = elevated coverage
    if count > 50:
        return (0.30, 0.70)  # bearish — negative headlines dominate
    elif count > 20:
        return None  # neutral volume, skip
    else:
        return (0.70, 0.30)  # low coverage = positive/stable


def _fred_evidence(category: str, macro: dict) -> tuple[float, float] | None:
    """Extract FRED macro surprise evidence.

    Hawkish surprise (rate up, yield up) → bearish on rate markets.
    Only applies to Economics/Financials categories.

    Returns (p_yes, p_no) or None.
    """
    if category not in _RATE_SENSITIVE:
        return None

    # Use fed_funds_rate and yield_10y from macro dict
    fed_rate = macro.get("fed_funds_rate", 0)
    yield_10y = macro.get("yield_10y", 0)

    if fed_rate == 0 and yield_10y == 0:
        return None  # no macro data available

    # Hawkish = high rates. Check if fed rate > 4.5% (elevated)
    # or 10Y yield > 4.5% (elevated)
    if fed_rate > 4.5 or yield_10y > 4.5:
        return (0.15, 0.35)  # hawkish surprise — bearish on rate markets
    elif fed_rate < 2.0 or yield_10y < 3.0:
        return (0.35, 0.15)  # dovish — bullish on rate markets
    return None  # neutral


def _noaa_evidence(category: str, market: dict) -> tuple[float, float] | None:
    """Extract NOAA weather deviation evidence.

    Applies only to weather markets. Checks if recent temperature
    deviates significantly from historical average for the city.

    Returns (p_yes, p_no) or None.
    """
    if category not in _WEATHER_CATS:
        return None

    # Try to extract city from market title/ticker
    title = (market.get("title", "") or market.get("question", "")).lower()
    ticker = (market.get("ticker", "") or "").lower()
    text = f"{title} {ticker}"

    city_files = {
        "nyc": "noaa_nyc_max.csv", "new york": "noaa_nyc_max.csv",
        "chicago": "noaa_chicago_max.csv",
        "miami": "noaa_miami_max.csv",
        "atlanta": "noaa_atlanta_max.csv",
        "dallas": "noaa_dallas_max.csv",
        "seattle": "noaa_seattle_max.csv",
    }

    noaa_file = None
    for city_key, fname in city_files.items():
        if city_key in text:
            noaa_file = Path("data/raw") / fname
            break
    if noaa_file is None or not noaa_file.exists():
        return None

    try:
        df = pd.read_csv(noaa_file)
        if df.empty or "tavg" not in df.columns:
            return None

        # Compare last observation to 30-day rolling mean
        df["tavg"] = pd.to_numeric(df["tavg"], errors="coerce")
        df = df.dropna(subset=["tavg"])
        if len(df) < 31:
            return None

        recent = df["tavg"].iloc[-1]
        rolling_mean = df["tavg"].iloc[-31:-1].mean()
        rolling_std = df["tavg"].iloc[-31:-1].std()
        if rolling_std < 1.0:
            return None

        z_score = (recent - rolling_mean) / rolling_std

        # Significant deviation: |z| > 1.5
        if abs(z_score) > 1.5:
            return (0.20, 0.40)  # weather miss — increases uncertainty
        return None  # no significant deviation

    except Exception:
        return None


# ── Chain all evidence updates ──────────────────────────────────────────

def bayesian_update_prob(prior: float, category: str, market: dict,
                         macro: dict) -> tuple[float, list[str]]:
    """Chain Bayesian updates from all evidence sources.

    Args:
        prior: RF model probability (raw)
        category: Kalshi category string (e.g., "Crypto", "Economics")
        market: market dict from API
        macro: macro data dict

    Returns:
        (posterior, updates_applied) where updates_applied lists
        which evidence sources shifted the probability.
    """
    posterior = prior
    updates = []

    # Evidence 1: NewsAPI headlines
    news_ev = _news_evidence(category, macro)
    if news_ev is not None:
        posterior = bayes_update(posterior, news_ev[0], news_ev[1])
        direction = "bearish" if news_ev[0] < news_ev[1] else "bullish"
        updates.append(f"news({direction})")

    # Evidence 2: FRED macro surprise
    fred_ev = _fred_evidence(category, macro)
    if fred_ev is not None:
        posterior = bayes_update(posterior, fred_ev[0], fred_ev[1])
        direction = "hawkish" if fred_ev[0] < fred_ev[1] else "dovish"
        updates.append(f"fred({direction})")

    # Evidence 3: NOAA weather deviation
    noaa_ev = _noaa_evidence(category, market)
    if noaa_ev is not None:
        posterior = bayes_update(posterior, noaa_ev[0], noaa_ev[1])
        updates.append("noaa(deviation)")

    # Clip to valid probability range
    posterior = float(np.clip(posterior, 0.01, 0.99))

    return posterior, updates
