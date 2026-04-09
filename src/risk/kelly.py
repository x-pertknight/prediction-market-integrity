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


def kelly_with_cv(
    win_prob:    float,
    entry_price: float,
    n_paths:     int = 10_000,
    n_trades:    int = 100,
    rng_seed:    int = 42,
) -> tuple[float, float]:
    """
    Empirical Kelly with coefficient-of-variation adjustment.

    f_empirical = f_kelly x (1 - CV_edge)

    CV_edge = std(edge) / mean(edge) from Monte Carlo simulation.
    When edge is uncertain (high CV), sizing is reduced conservatively.

    Args:
        win_prob: model probability of YES outcome
        entry_price: price paid per share (0-1)
        n_paths: Monte Carlo simulation paths
        n_trades: trades per path
        rng_seed: random seed for reproducibility

    Returns:
        (f_empirical, cv_edge) tuple
    """
    f_kelly = quarter_kelly(win_prob, entry_price)
    if f_kelly <= 0:
        return 0.0, 0.0

    rng = np.random.default_rng(rng_seed)

    # Simulate outcome distributions across paths
    outcomes = rng.binomial(1, win_prob, (n_paths, n_trades))
    win_rates = outcomes.mean(axis=1)

    # Edge per path: E[payoff] = wr * (1-price) - (1-wr) * price
    edges = win_rates * (1.0 - entry_price) - (1.0 - win_rates) * entry_price

    mean_edge = edges.mean()
    std_edge = edges.std()

    if abs(mean_edge) < 1e-10:
        return 0.0, float("inf")

    cv_edge = std_edge / abs(mean_edge)
    f_empirical = f_kelly * max(0.0, 1.0 - cv_edge)

    return f_empirical, cv_edge


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
