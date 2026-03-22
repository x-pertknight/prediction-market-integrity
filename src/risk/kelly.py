"""
src/risk/kelly.py
Kelly Criterion position sizing for binary prediction markets.
Uses quarter-Kelly for conservative sizing.
"""
import numpy as np


def kelly_fraction(win_prob: float, odds_win: float, odds_lose: float = 1.0) -> float:
    """
    Full Kelly fraction for binary bet.
    win_prob: model probability of YES outcome (0–1)
    odds_win: net gain if win (e.g., if entry=0.10, win pays $0.90 → odds=9.0)
    odds_lose: loss fraction if lose (typically 1.0 = lose full stake)
    Returns: fraction of bankroll to bet (0–1). Negative = don't bet.
    """
    if win_prob <= 0 or odds_win <= 0:
        return 0.0
    lose_prob = 1.0 - win_prob
    k = (win_prob * odds_win - lose_prob * odds_lose) / odds_win
    return max(0.0, k)


def quarter_kelly(win_prob: float, entry_price: float) -> float:
    """
    Quarter-Kelly for a binary prediction market.
    entry_price: price paid per share (0–1). Win pays $1.
    Returns: fraction of bankroll (already quarter-scaled).
    """
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    odds_win = (1.0 - entry_price) / entry_price   # net gain per $ staked
    full_k   = kelly_fraction(win_prob, odds_win, 1.0)
    return full_k / 4.0    # quarter-Kelly


def size_trade(
    bankroll:    float,
    win_prob:    float,
    entry_price: float,
    max_pct:     float = 0.10,   # max 10% per trade
) -> float:
    """
    Return dollar size for a trade given bankroll.
    Caps at max_pct of bankroll.
    """
    qk   = quarter_kelly(win_prob, entry_price)
    size = qk * bankroll
    return min(size, bankroll * max_pct)


def expected_value(win_prob: float, entry_price: float) -> float:
    """EV per $1 notional (1 share at entry_price)."""
    return win_prob * (1.0 - entry_price) - (1.0 - win_prob) * entry_price


if __name__ == "__main__":
    # Example: model says 25% chance, market price is 10 cents
    prob   = 0.25
    price  = 0.10
    bank   = 1000.0
    size   = size_trade(bank, prob, price)
    ev     = expected_value(prob, price)
    qk     = quarter_kelly(prob, price)
    print(f"win_prob={prob} entry={price} bankroll=${bank}")
    print(f"  quarter-Kelly fraction: {qk:.4f} ({qk*100:.2f}%)")
    print(f"  trade size: ${size:.2f}")
    print(f"  EV per $1 notional: ${ev:.4f}")
