"""
src/risk/market_impact.py — Realistic Market Impact Model

impact_cost = position_size * volatility / sqrt(daily_volume)

Liquidity tiers (by daily_volume in $):
  < $100:   ILLIQUID — skip trade entirely
  $100-500: HIGH impact (5-15%) — quarter-Kelly sizing
  $500-5K:  MODERATE impact (1-5%) — half-Kelly sizing
  > $5K:    LOW impact (<1%) — full sizing

Hard reject: position_size > 10% of daily_volume

Volume assumption: when per-market volume unavailable, uses category
median volume as proxy. This is documented as an approximation.
"""

import numpy as np
import pandas as pd


# ── Liquidity tiers ──────────────────────────────────────────────────────────

TIER_ILLIQUID  = "ILLIQUID"   # < $100/day
TIER_HIGH      = "HIGH"       # $100-500
TIER_MODERATE  = "MODERATE"   # $500-5K
TIER_LOW       = "LOW"        # > $5K

TIER_KELLY_MULT = {
    TIER_ILLIQUID: 0.0,   # don't trade
    TIER_HIGH:     0.25,  # quarter-Kelly
    TIER_MODERATE: 0.50,  # half-Kelly
    TIER_LOW:      1.0,   # full sizing
}

MIN_VOLUME_DOLLARS = 100.0        # below this: skip
MAX_POSITION_PCT   = 0.10         # reject if position > 10% of daily volume
DEFAULT_VOLATILITY = 0.15         # annualized vol for binary markets (~15%)
# Daily vol ≈ annual / sqrt(252) ≈ 0.0094, but we use annualized in the formula
# and normalize by sqrt(daily_volume) which is in $ terms


def liquidity_tier(daily_volume_dollars: float) -> str:
    """Classify market by daily volume."""
    if daily_volume_dollars < 100:
        return TIER_ILLIQUID
    elif daily_volume_dollars < 500:
        return TIER_HIGH
    elif daily_volume_dollars < 5000:
        return TIER_MODERATE
    else:
        return TIER_LOW


def market_impact(position_size: float, daily_volume: float,
                  volatility: float = DEFAULT_VOLATILITY) -> float:
    """
    Almgren-Chriss linear impact estimate for prediction markets.

    impact = position_size * volatility / sqrt(daily_volume)

    Returns impact cost as fraction of $1 notional (same scale as P&L).
    """
    if daily_volume <= 0:
        return 1.0  # max impact for zero-volume
    return position_size * volatility / np.sqrt(daily_volume)


def should_trade(position_size: float, daily_volume: float) -> bool:
    """Returns False if trade should be rejected (illiquid or too large)."""
    if daily_volume < MIN_VOLUME_DOLLARS:
        return False
    if position_size > daily_volume * MAX_POSITION_PCT:
        return False
    return True


def kelly_multiplier(daily_volume: float) -> float:
    """Kelly sizing multiplier based on liquidity tier."""
    tier = liquidity_tier(daily_volume)
    return TIER_KELLY_MULT[tier]


# ── Category volume proxies ─────────────────────────────────────────────────
# When per-market volume is unavailable, use category median.
# These are computed from WF volume_fp data (contracts, not dollars).
# Contract value ≈ entry_price ($0.01-$0.99), so volume_dollars ≈ volume_fp * avg_price
# For safety, we assume avg_price ≈ $0.10 for all categories (conservative).

_CATEGORY_VOLUME_PROXY = {
    "Crypto":                   123.0,   # median volume_fp
    "Financials":               200.0,
    "Entertainment":            180.0,
    "Sports":                   160.0,
    "Climate and Weather":      100.0,
    "Economics":                150.0,
    "Elections":                300.0,
    "Politics":                 250.0,
    "Mentions":                 120.0,
    "Exotics":                   80.0,
    "Companies":                200.0,
    "Health":                   100.0,
    "Science and Technology":   100.0,
    "Social":                   100.0,
    "World":                    100.0,
    "Education":                100.0,
    "Transportation":           100.0,
}


def _estimate_daily_volume_dollars(volume_fp: float, entry_price: float,
                                    category: str = "") -> float:
    """
    Estimate daily volume in dollars from contract volume and entry price.
    volume_dollars = volume_fp * entry_price (notional value traded)
    Falls back to category proxy if volume_fp is missing.
    """
    if pd.notna(volume_fp) and volume_fp > 0:
        price = max(entry_price, 0.01)
        return volume_fp * price

    # Fallback: category proxy * assumed avg price
    proxy_vol = _CATEGORY_VOLUME_PROXY.get(category, 100.0)
    return proxy_vol * max(entry_price, 0.05)


# ── Backtest integration ────────────────────────────────────────────────────

def compute_trade_impact(entry_prices: pd.Series,
                         volumes: pd.Series,
                         categories: pd.Series | None = None,
                         position_size: float = 1.0) -> pd.DataFrame:
    """
    Compute per-trade market impact for backtest.

    Args:
        entry_prices: entry price per trade (0-1 scale)
        volumes: volume_fp per trade (contract count)
        categories: category per trade (for proxy)
        position_size: contracts per trade (default 1)

    Returns DataFrame with columns:
        daily_vol_dollars, liquidity_tier, impact_cost, kelly_mult, trade_allowed
    """
    n = len(entry_prices)
    ep = entry_prices.values.astype(float)
    vols = volumes.values.astype(float) if volumes is not None else np.full(n, np.nan)
    cats = categories.values if categories is not None else np.full(n, "")

    daily_vol = np.zeros(n)
    tiers = []
    impacts = np.zeros(n)
    kelly_mults = np.zeros(n)
    allowed = np.ones(n, dtype=bool)

    for i in range(n):
        # Estimate daily volume in dollars
        dvd = _estimate_daily_volume_dollars(vols[i], ep[i],
                                              str(cats[i]) if pd.notna(cats[i]) else "")
        daily_vol[i] = dvd

        tier = liquidity_tier(dvd)
        tiers.append(tier)
        kelly_mults[i] = TIER_KELLY_MULT[tier]

        # Check if trade should happen
        pos_dollars = position_size * ep[i]
        if not should_trade(pos_dollars, dvd):
            allowed[i] = False
            impacts[i] = 0.0  # trade won't happen
        else:
            impacts[i] = market_impact(pos_dollars, dvd)

    return pd.DataFrame({
        "daily_vol_dollars": daily_vol,
        "liquidity_tier": tiers,
        "impact_cost": impacts,
        "kelly_mult": kelly_mults,
        "trade_allowed": allowed,
    }, index=entry_prices.index)
