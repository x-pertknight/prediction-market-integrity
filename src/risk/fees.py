"""
src/risk/fees.py — Accurate fee models for Kalshi and Polymarket.

Kalshi: Parabolic taker fee  fee = 0.07 * P * (1 - P)
  - Max at P=0.50: $0.0175 (1.75¢)
  - At P=0.05:     $0.003325 (0.33¢)
  - Zero settlement fee. Fee only on trade execution (buy/sell).
  - In backtest: contracts settle at 0/1, so fee applies on ENTRY only.

Polymarket: Per-category taker fee (docs.polymarket.com, verified 2026-07-06).
  - fee = rate * (P * (1 - P)) ** exponent, exponent = 1 on all current schedules
  - Rates: crypto 0.07, sports 0.03, finance/politics/tech/mentions 0.04,
    economics/culture/weather/other 0.05, geopolitics 0.
  - Maker gets rebate (20-25% of taker fee); sells and makers pay no fee.
  - Markets created before 2026-03-30 in some categories are grandfathered
    fee-free; charging the fee anyway is the conservative default here.
  - Gamma API exposes the exact per-market schedule in `feeSchedule`
    (e.g. {'rate': 0.03, 'exponent': 1, 'takerOnly': True, 'rebateRate': 0.25});
    prefer polymarket_fee_from_schedule() when the market dict is available.
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

# Official per-category taker fee rates (docs.polymarket.com, 2026-07-06).
POLY_CATEGORY_FEE_RATES = {
    "crypto": 0.07,
    "sports": 0.03,
    "finance": 0.04,
    "politics": 0.04,
    "tech": 0.04,
    "mentions": 0.04,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "other": 0.05,
    "geopolitics": 0.0,
}

# infer_category() output (Kalshi-style names) → Polymarket fee category
_TRAPTRADE_CAT_TO_FEE_CAT = {
    "Crypto": "crypto",
    "Sports": "sports",
    "Politics": "politics",
    "Elections": "politics",
    "Financials": "finance",
    "Companies": "finance",
    "Economics": "economics",
    "Entertainment": "culture",
    "Social": "culture",
    "Climate and Weather": "weather",
    "Education": "other",
    "Exotics": "other",
}

POLY_US_FLAT_RATE = 0.0001   # 0.01% CFTC-regulated flat rate


def _poly_fee_category(market_type: str) -> str:
    """Resolve a market_type/category string to a Polymarket fee category.

    Accepts fee categories ("sports"), infer_category() names ("Sports"),
    and legacy strings ("crypto_daily", "ncaab", "sport_fee").
    Unknown types fall back to "other" (0.05) — conservative, since only
    geopolitics is fee-free today.
    """
    if market_type in POLY_CATEGORY_FEE_RATES:
        return market_type
    if market_type in _TRAPTRADE_CAT_TO_FEE_CAT:
        return _TRAPTRADE_CAT_TO_FEE_CAT[market_type]
    mt = market_type.lower()
    if mt in POLY_CATEGORY_FEE_RATES:
        return mt
    if mt.startswith("crypto"):
        return "crypto"
    if mt.startswith("sport") or mt in ("ncaab", "nfl", "nba", "mlb", "nhl"):
        return "sports"
    return "other"


def polymarket_taker_fee(price: float, market_type: str = "") -> float:
    """Polymarket taker fee per share: rate * P * (1 - P), rate by category."""
    rate = POLY_CATEGORY_FEE_RATES[_poly_fee_category(market_type)]
    p = np.clip(price, 0.0, 1.0)
    return p * (1.0 - p) * rate


def polymarket_fee_from_schedule(price: float, schedule: dict | None) -> float:
    """Exact per-market taker fee from the Gamma API `feeSchedule` field.

    Schedule shape: {'rate': 0.03, 'exponent': 1, 'takerOnly': True,
    'rebateRate': 0.25}. Missing/empty schedule → no fee (the API omits it
    on fee-free markets).
    """
    if not schedule:
        return 0.0
    rate = float(schedule.get("rate", 0) or 0)
    if rate == 0.0:
        return 0.0
    exponent = float(schedule.get("exponent", 1) or 1)
    p = float(np.clip(price, 0.0, 1.0))
    return rate * (p * (1.0 - p)) ** exponent


def polymarket_maker_rebate(price: float, market_type: str = "") -> float:
    """Polymarket maker rebate — you GET PAID for limit orders.

    25% of taker fee on sports, 20% elsewhere (official rebateRate values).
    """
    fee = polymarket_taker_fee(price, market_type)
    if fee == 0.0:
        return 0.0
    rebate_rate = 0.25 if _poly_fee_category(market_type) == "sports" else 0.20
    return fee * rebate_rate


def polymarket_us_fee(price: float) -> float:
    """Polymarket US (CFTC-regulated): flat 0.01% taker on all markets."""
    return np.clip(price, 0.0, 1.0) * POLY_US_FLAT_RATE


# ── Backtest integration ────────────────────────────────────────────────────────


def compute_trade_fees(entry_prices: pd.Series,
                       exit_prices: pd.Series,
                       fee_model: str = "none",
                       categories: pd.Series | None = None) -> pd.DataFrame:
    """
    Compute per-trade fees for a Series of trades.

    fee_model:
      "none"        — zero fees
      "kalshi"      — parabolic taker on entry; settlement = free
      "polymarket"  — per-category rate * P * (1-P); "other" 0.05 if no category
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
        cats = categories.values if categories is not None else np.array([""] * n, dtype=object)
        for i in range(n):
            cat = str(cats[i]) if pd.notna(cats[i]) else ""
            fee_entry[i] = polymarket_taker_fee(entry[i], cat)
            not_settle = 0.01 < exit_[i] < 0.99
            fee_exit[i] = polymarket_taker_fee(exit_[i], cat) if not_settle else 0.0

    elif fee_model == "polymarket_us":
        fee_entry = entry * POLY_US_FLAT_RATE
        not_settlement = (exit_ > 0.01) & (exit_ < 0.99)
        fee_exit = np.where(not_settlement, exit_ * POLY_US_FLAT_RATE, 0.0)

    return pd.DataFrame({
        "fee_entry": fee_entry,
        "fee_exit": fee_exit,
        "fee_total": fee_entry + fee_exit,
    }, index=entry_prices.index)
