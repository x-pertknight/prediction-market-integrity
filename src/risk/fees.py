"""
src/risk/fees.py — Accurate fee models for Kalshi and Polymarket.

Kalshi: Parabolic taker fee  fee = 0.07 * P * (1 - P)
  - Max at P=0.50: $0.0175 (1.75¢)
  - At P=0.05:     $0.003325 (0.33¢)
  - Zero settlement fee. Fee only on trade execution (buy/sell).
  - In backtest: contracts settle at 0/1, so fee applies on ENTRY only.

Polymarket: Conditional — most markets are FREE.
  - Crypto short-term, NCAAB, some sports: fee = P * (1 - P) * fee_rate
  - Maker gets rebate (20% crypto, 25% sports)
  - US regulated (CFTC): flat 0.01% taker on everything
"""

import numpy as np
import pandas as pd


# ── Kalshi ──────────────────────────────────────────────────────────────────────

KALSHI_TAKER_COEFF = 0.07  # parabolic coefficient


def kalshi_taker_fee(price: float) -> float:
    """Kalshi taker fee per contract. Price in [0, 1]. Returns dollars."""
    p = np.clip(price, 0.0, 1.0)
    return KALSHI_TAKER_COEFF * p * (1.0 - p)


def kalshi_total_fee(price: float, n_contracts: int = 1) -> float:
    """Total Kalshi fee for n contracts at given price."""
    return kalshi_taker_fee(price) * n_contracts


def kalshi_round_trip_fee(entry_price: float, exit_price: float) -> float:
    """
    Round-trip fee: entry + exit.
    In backtest, exit = settlement (0 or 1) → zero exit fee.
    Use this for live trading where you might sell before settlement.
    """
    return kalshi_taker_fee(entry_price) + kalshi_taker_fee(exit_price)


# ── Polymarket ──────────────────────────────────────────────────────────────────

# Market types that charge fees on Polymarket
POLY_FEE_MARKETS = frozenset({
    "crypto_15m", "crypto_5m", "crypto_1h", "crypto_4h",
    "crypto_daily", "crypto_weekly",
    "ncaab",
    "sport_fee",
})

POLY_TAKER_RATE = 0.02       # ~1.56% effective max at p=0.50
POLY_US_FLAT_RATE = 0.0001   # 0.01% CFTC-regulated flat rate


def polymarket_taker_fee(price: float, market_type: str = "") -> float:
    """
    Polymarket taker fee per share. Most markets: FREE.
    Crypto/NCAAB/some sports: parabolic with 0.02 rate.
    """
    if market_type in POLY_FEE_MARKETS:
        p = np.clip(price, 0.0, 1.0)
        return p * (1.0 - p) * POLY_TAKER_RATE
    return 0.0


def polymarket_maker_rebate(price: float, market_type: str = "") -> float:
    """Polymarket maker rebate — you GET PAID for limit orders."""
    fee = polymarket_taker_fee(price, market_type)
    if fee == 0.0:
        return 0.0
    if "crypto" in market_type:
        return fee * 0.20
    if "sport" in market_type or "ncaab" in market_type:
        return fee * 0.25
    return 0.0


def polymarket_us_fee(price: float) -> float:
    """Polymarket US (CFTC-regulated): flat 0.01% taker on all markets."""
    return np.clip(price, 0.0, 1.0) * POLY_US_FLAT_RATE


# ── Backtest integration ────────────────────────────────────────────────────────

# Map category → Polymarket market_type for fee determination
# Most categories are fee-free; only crypto short-duration and some sports pay
_CATEGORY_TO_POLY_TYPE = {
    "Crypto": "crypto_daily",  # conservative: assume daily crypto fee applies
    "Sports": "sport_fee",     # some sports markets have fees
}


def compute_trade_fees(entry_prices: pd.Series,
                       exit_prices: pd.Series,
                       fee_model: str = "none",
                       categories: pd.Series | None = None) -> pd.DataFrame:
    """
    Compute per-trade fees for a Series of trades.

    fee_model:
      "none"        — zero fees
      "kalshi"      — parabolic taker on entry; settlement = free
      "polymarket"  — conditional (free for most categories)
      "polymarket_us" — CFTC flat 0.01%

    Returns DataFrame with columns: fee_entry, fee_exit, fee_total
    """
    n = len(entry_prices)
    entry = entry_prices.values.astype(float)
    exit_ = exit_prices.values.astype(float)

    fee_entry = np.zeros(n)
    fee_exit = np.zeros(n)

    if fee_model == "kalshi":
        # Parabolic fee on entry only; settlement is free
        fee_entry = KALSHI_TAKER_COEFF * entry * (1.0 - entry)
        # Exit fee only if exit is NOT settlement (0 or 1)
        # In backtest exit = outcome (0 or 1) → settlement → free
        # But clip for safety: if exit is between 0.01 and 0.99, charge fee
        not_settlement = (exit_ > 0.01) & (exit_ < 0.99)
        fee_exit = np.where(
            not_settlement,
            KALSHI_TAKER_COEFF * exit_ * (1.0 - exit_),
            0.0,
        )

    elif fee_model == "polymarket":
        if categories is not None:
            cats = categories.values
            for i in range(n):
                cat = str(cats[i]) if pd.notna(cats[i]) else ""
                mtype = _CATEGORY_TO_POLY_TYPE.get(cat, "")
                fee_entry[i] = polymarket_taker_fee(entry[i], mtype)
                not_settle = 0.01 < exit_[i] < 0.99
                fee_exit[i] = polymarket_taker_fee(exit_[i], mtype) if not_settle else 0.0
        # else: all zero (most markets are free)

    elif fee_model == "polymarket_us":
        fee_entry = entry * POLY_US_FLAT_RATE
        not_settlement = (exit_ > 0.01) & (exit_ < 0.99)
        fee_exit = np.where(not_settlement, exit_ * POLY_US_FLAT_RATE, 0.0)

    return pd.DataFrame({
        "fee_entry": fee_entry,
        "fee_exit": fee_exit,
        "fee_total": fee_entry + fee_exit,
    }, index=entry_prices.index)
