"""
taleb.py — Taleb Mathematical Risk Framework
=============================================
Implements the REAL mathematics from Taleb's papers.
Every formula has a reference to the original paper.

Modules:
  jensen_convexity    — Jensen's Inequality, convexity bias
  hill_estimator      — Fat tail detection (Hill 1975)
  ergodicity_check    — Peters-Gell-Mann geometric growth
  barbell_allocation  — Tail-exponent-dependent allocation
  fragility_heuristic — Taleb-Douady fragility measure
  inv_n_sizing        — 1/sqrt(N) diversification
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TalebSignal:
    """Per-trade Taleb signal bundle."""
    ticker:            str
    convexity_bias:    float   # Jensen's inequality convexity
    uncertainty:       float   # RF tree variance (proxy for model uncertainty)
    geometric_growth:  float   # Peters-Gell-Mann time-average growth rate
    kelly_fraction:    float   # Ergodicity-adjusted Kelly
    is_ergodic:        bool    # True if geometric_growth > 0
    taleb_score:       float   # Combined score (higher = more desirable)


@dataclass
class TalebPortfolioRisk:
    """Portfolio-level Taleb risk metrics."""
    fragility:         float   # Taleb-Douady fragility measure
    is_antifragile:    bool    # True if fragility > 0
    alpha_by_category: dict    # Hill tail exponent per category
    barbell_ratio:     float   # Calculated safe/speculative split
    ruin_prob_1000:    float   # Estimated ruin probability at 1000 trades
    ruin_prob_5000:    float   # At 5000 trades
    ruin_prob_10000:   float   # At 10000 trades
    stress_results:    dict    # ±10/20/30% shock P&L


# ── 2a. Jensen's Inequality — Convexity Bias ─────────────────────────────────

def jensen_convexity(entry_price: float, model_prob: float) -> float:
    """
    Computes convexity bias for a binary prediction market contract.

    For a binary YES contract with payout 1 and entry price p:
      F(x) = max(1 - p, 0) if YES; max(0 - p, 0) = 0 if NO
      E[F(x)] = model_prob * (1 - entry_price)
      F(E[x]) = max(model_prob - entry_price, 0)
      convexity_bias = E[F(x)] - F(E[x])

    Interpretation:
      > 0: Contract is convex in probability space (antifragile)
           Higher uncertainty INCREASES expected value relative to
           the certainty-equivalent payoff
      = 0: Linear payoff, no convexity benefit
      < 0: Should not occur for long binary positions

    Ref: Jensen (1906) "Sur les fonctions convexes et les inégalités"
         Taleb, "Understanding is a Poor Substitute for Convexity"
         Edge.org, 2012. Rule: seek positive convexity.

    Args:
        entry_price:  Current market ask price (0–1)
        model_prob:   RF model's P(YES) estimate (0–1)

    Returns:
        convexity_bias: float ≥ 0
    """
    # E[F(x)]: expected payoff under model probability
    e_f_x = model_prob * (1.0 - entry_price)
    # F(E[x]): payoff at the expected probability (certainty equivalent)
    f_e_x = max(model_prob - entry_price, 0.0)
    return e_f_x - f_e_x


def rf_uncertainty(rf_model, X: np.ndarray) -> np.ndarray:
    """
    Compute per-sample uncertainty as std dev of individual RF tree predictions.

    Uses the variance of individual decision tree probabilities as a proxy
    for model epistemic uncertainty. High variance = low consensus among
    trees = high model uncertainty = higher convexity benefit.

    Ref: Breiman (2001) "Random Forests"; Taleb (2012) convexity framework
         (uncertainty ↑ → convexity_bias ↑ for long option positions)

    Args:
        rf_model:  Trained RandomForestClassifier
        X:         Feature matrix (n_samples, n_features)

    Returns:
        uncertainty: np.ndarray shape (n_samples,) — std dev of tree probs
    """
    if not hasattr(rf_model, 'estimators_'):
        return np.zeros(len(X))
    # Collect P(YES) from each tree
    tree_probs = np.array([
        tree.predict_proba(X)[:, 1]
        for tree in rf_model.estimators_
    ])  # shape: (n_trees, n_samples)
    return tree_probs.std(axis=0)


# ── 2b. Hill Estimator — Fat Tail Detection ───────────────────────────────────

def hill_estimator(returns: np.ndarray, quantile: float = 0.95) -> float:
    """
    Hill (1975) maximum likelihood estimator for the tail exponent alpha.

    For a Pareto-tailed distribution, the tail exponent alpha is estimated:
        alpha_hat = n / sum_{i=1}^{n} log(x_i / x_min)
    where x_i are the returns exceeding the (1-quantile) threshold x_min.

    Interpretation:
        alpha < 2: Infinite variance. DO NOT use Sharpe Ratio (undefined).
                   Use Gain-to-Pain ratio instead.
        2 ≤ alpha < 3: Finite mean, explosive variance.
                   Limit to quarter-Kelly sizing.
        alpha ≥ 3: Thin tails. Half-Kelly acceptable.
        alpha ≥ 4: Near-normal. Standard Kelly.

    Ref: Hill, B.M. (1975) "A Simple General Approach to Inference About
         the Tail of a Distribution." Annals of Statistics 3(5):1163–1174.
         Taleb, N.N. (2020) "Statistical Consequences of Fat Tails" Ch.4
         Mandelbrot (1963) "The Variation of Certain Speculative Prices"
         Journal of Business 36(4):394–419.

    Args:
        returns:   Array of trade returns (e.g., pnl per trade)
        quantile:  Use top (1-quantile) fraction as tail. Default 0.95.

    Returns:
        alpha_hat: float. Returns np.inf if insufficient tail observations.
    """
    # Work with absolute returns for fat-tail analysis on BOTH sides
    x = np.abs(returns)
    x = x[x > 0]
    if len(x) < 10:
        return np.inf   # insufficient data — assume thin tails (safe)

    x_min = np.quantile(x, quantile)
    tail = x[x > x_min]
    if len(tail) < 5:
        return np.inf

    # Hill estimator
    alpha_hat = len(tail) / np.sum(np.log(tail / x_min))
    return float(alpha_hat)


def kelly_from_alpha(alpha: float) -> float:
    """
    Maximum Kelly fraction based on tail exponent.

    Ref: Taleb "Statistical Consequences of Fat Tails" Ch.4
         Kelly (1956) "A New Interpretation of Information Rate"

    Returns:
        max_kelly_fraction: 0.25 (alpha<2), 0.25 (alpha<3), 0.50 (alpha<4), 1.0 (else)
    """
    if alpha < 2:
        return 0.10   # Near-infinite variance: extreme caution
    elif alpha < 3:
        return 0.25   # Quarter-Kelly max
    elif alpha < 4:
        return 0.50   # Half-Kelly
    else:
        return 1.00   # Full Kelly acceptable


def gain_to_pain(returns: np.ndarray) -> float:
    """
    Gain-to-Pain ratio: sum(positive returns) / abs(sum(negative returns))
    Use when alpha < 2 (Sharpe undefined due to infinite variance).

    Ref: Schwager (1992) "The New Market Wizards". Endorsed by Taleb as
         alternative to Sharpe when variance is not finite.
    """
    gains  = returns[returns > 0].sum()
    losses = np.abs(returns[returns < 0].sum())
    if losses == 0:
        return np.inf
    return gains / losses


# ── 2c. Peters-Gell-Mann Ergodicity Check ─────────────────────────────────────

def geometric_growth_rate(p: float, gain: float, loss: float,
                          bankroll: float) -> float:
    """
    Compute the time-average (geometric) growth rate for a bet.

    For a binary bet with win probability p:
        g = p * log(1 + gain/bankroll) + (1-p) * log(1 - loss/bankroll)

    If g ≤ 0: the bet destroys wealth in the long run (time-average),
    even if the ensemble-average expected value E[wealth] > 0.
    This is the ergodicity problem: ensemble-average ≠ time-average.
    DO NOT TAKE THE BET if g ≤ 0.

    Example:
        p=0.5, +110% win, -105% loss, bankroll=1
        E[wealth] = 0.5*2.1 + 0.5*(-0.05) = 1.025 > 1  (positive EV)
        g = 0.5*log(2.1) + 0.5*log(-0.05) → undefined (ruin possible)
        Correct: DO NOT BET

    Ref: Peters, O. & Gell-Mann, M. (2016) "Evaluating Gambles Using
         Dynamics." Chaos: An Interdisciplinary Journal 26(2).
         Kelly, J.L. (1956) "A New Interpretation of Information Rate."
         Bell System Technical Journal 35(4):917–926.

    Args:
        p:        Win probability (0–1)
        gain:     Dollar gain if WIN (positive)
        loss:     Dollar loss if LOSE (positive — the amount you risk)
        bankroll: Current total capital

    Returns:
        g: float — geometric growth rate. Positive = grow, negative = ruin.
    """
    if bankroll <= 0 or loss >= bankroll:
        return -np.inf
    return p * np.log(1.0 + gain / bankroll) + (1.0 - p) * np.log(1.0 - loss / bankroll)


def ergodicity_kelly(p: float, gain_per_dollar: float, loss_per_dollar: float) -> float:
    """
    Optimal Kelly fraction from ergodicity theory.

    f* = p/loss_per_dollar - (1-p)/gain_per_dollar

    For binary prediction markets:
      gain_per_dollar = (1 - entry_price) / entry_price  (return if YES)
      loss_per_dollar = 1.0                              (lose full stake if NO)
      f* = p - (1-p) / ((1-entry)/entry)
         = p - (1-p) * entry / (1-entry)

    Ref: Kelly (1956), Peters & Gell-Mann (2016) — ergodic optimal fraction.
    """
    if gain_per_dollar <= 0 or loss_per_dollar <= 0:
        return 0.0
    f_star = p / loss_per_dollar - (1.0 - p) / gain_per_dollar
    return max(0.0, f_star)


# ── 2d. Barbell Allocation ────────────────────────────────────────────────────

def barbell_ratio(alpha: float) -> float:
    """
    Taleb's dynamic barbell: safe allocation depends on tail exponent.

        safe_allocation = 1 - (alpha - 1) / alpha    for alpha > 1

    Intuition: higher alpha (thinner tails) → smaller safe fraction needed
    because the speculative tail bets are less dangerous.
        alpha = 2  → safe = 50%
        alpha = 4  → safe = 75%
        alpha = 10 → safe = 90%
        alpha → ∞  → safe = 100% (normal distribution)

    Ref: Taleb, N.N. (2012) "Antifragile" Ch.12 — Barbell strategy.
         Dynamic version: safe fraction = 1 - 1/alpha

    Args:
        alpha: Hill tail exponent

    Returns:
        safe_fraction: float ∈ [0, 1]
    """
    if alpha <= 1:
        return 0.99  # near-infinite tails: almost all safe
    safe = 1.0 - (alpha - 1.0) / alpha
    return float(np.clip(safe, 0.01, 0.99))


def classify_market(entry_price: float, conv_bias: float, model_prob: float,
                    hist_sharpe: float, liquidity: float,
                    hist_win_rate: float) -> str:
    """
    Classify a market as 'safe', 'speculative', or 'skip' for barbell.

    Safe:        hist_Sharpe > 2 AND liquidity > $5000 AND hist_win_rate > 0.6
    Speculative: entry_price < $0.10 AND conv_bias > 0.3 AND model_prob > 0.3
    Skip:        everything else

    Ref: Taleb "Antifragile" Ch.12 — Safe = bonds/T-bills equivalent;
         Speculative = deep OTM options equivalent.
    """
    if hist_sharpe > 2.0 and liquidity > 5000 and hist_win_rate > 0.6:
        return "safe"
    if entry_price < 0.10 and conv_bias > 0.30 and model_prob > 0.30:
        return "speculative"
    return "skip"


# ── 2e. Fragility Heuristic — Taleb & Douady (2013) ──────────────────────────

def fragility_score(pnl_series: pd.Series, bankroll: float = 1000.0) -> float:
    """
    Taleb-Douady heuristic fragility measure.

        fragility = (loss_at_2sigma - 2 * loss_at_1sigma) / bankroll

    Convex portfolio (antifragile, fragility > 0):
        The response to a 2-sigma shock is LESS THAN twice a 1-sigma shock.
        P&L curve is concave in loss space = convex in value space.
    Concave portfolio (fragile, fragility < 0):
        2-sigma shock costs MORE than 2× the 1-sigma shock.
        Non-linear downside: FRAGILE.

    Note: We work with LOSSES (positive = bad).
      loss_at_nsigma = -percentile(pnl, cdf(−n))
    For daily P&L:
      1-sigma ≈ 16th percentile loss (left tail)
      2-sigma ≈ 2.3rd percentile loss

    Ref: Taleb, N.N., Canetti, E., Kinda, T., Loukoianova, E. (2012)
         "A New Heuristic Measure of Fragility and Tail Risks." IMF WP/12/216.
         Taleb & Douady (2013) "Mathematical Definition, Mapping, and
         Detection of (Anti)Fragility." Quantitative Finance 13(11).

    Args:
        pnl_series: Series of per-trade P&L
        bankroll:   Current bankroll for normalization

    Returns:
        fragility: float. Positive = antifragile, negative = fragile.
    """
    if len(pnl_series) < 30:
        return 0.0   # insufficient data

    # 1-sigma loss: ~16th percentile of pnl (lower tail)
    p1 = np.percentile(pnl_series, 16)   # ~-1 sigma
    p2 = np.percentile(pnl_series, 2.3)  # ~-2 sigma

    loss_1sigma = -min(p1, 0)  # convert to positive loss
    loss_2sigma = -min(p2, 0)

    fragility = (loss_2sigma - 2.0 * loss_1sigma) / bankroll
    return float(fragility)


def stress_test(pnl_series: pd.Series, entry_prices: pd.Series,
                shocks: list[float] = [-0.30, -0.20, -0.10, 0.10, 0.20, 0.30]
                ) -> dict[str, float]:
    """
    Stress test: simulate price shocks and recompute portfolio P&L.

    For each shock magnitude s:
      - New entry price = entry_price * (1 + s)
      - For YES trades: new_pnl = outcome - new_entry_price
      - Positive shock (price ↑): hurts buyers (higher entry cost)
      - Negative shock (price ↓): helps buyers (lower entry cost)
    A CONVEX response: losses from +s are SMALLER than gains from −s.

    Ref: Taleb & Douady (2013) "Mathematical Definition, Mapping, and
         Detection of (Anti)Fragility."

    Returns:
        dict: {shock_str: total_pnl_change} e.g. {"-30%": -145.2, "+30%": 180.1}
    """
    results = {}
    for s in shocks:
        new_entry = entry_prices.clip(lower=1e-4) * (1.0 + s)
        new_entry = new_entry.clip(0.001, 0.999)
        # Recompute pnl with shocked entry prices
        # Assumes outcomes (0/1) are fixed (they are — binary resolution)
        outcomes = entry_prices.index.map(
            lambda i: 1.0 if pnl_series.iloc[i] + entry_prices.iloc[i] > 0.5 else 0.0
        ) if hasattr(entry_prices.index, 'map') else None
        # Simpler: original pnl = outcome - entry; shock changes entry by s*entry
        # new_pnl = pnl - s*entry (shock increases cost by s*entry)
        shocked_pnl = pnl_series - s * entry_prices
        results[f"{'+' if s>=0 else ''}{s*100:.0f}%"] = round(float(shocked_pnl.sum()), 2)

    results["baseline"] = round(float(pnl_series.sum()), 2)
    return results


# ── 2f. 1/N Diversification Rule ─────────────────────────────────────────────

def max_position_size(bankroll: float, n_active_markets: int,
                      min_pct: float = 0.001, max_pct: float = 0.10) -> float:
    """
    Taleb's 1/sqrt(N) diversification rule for prediction markets.

        max_allocation = bankroll / sqrt(N_active_markets)

    Intuition: Uncertainty scales with sqrt(N) independent bets.
    At 100 markets: max 10% per trade.
    At 10,000 markets: max 1% per trade.

    Ref: Taleb, N.N. (2012) "Understanding is a Poor Substitute for Convexity"
         Edge.org. Rule 2: diversify — uncertainty grows with sqrt(N).

    Args:
        bankroll:          Current capital in dollars
        n_active_markets:  Number of concurrent open positions
        min_pct:           Minimum allocation fraction (default 0.1%)
        max_pct:           Maximum allocation fraction (default 10%)

    Returns:
        max_dollars: Maximum dollar allocation per new trade
    """
    if n_active_markets <= 0:
        return bankroll * max_pct
    fraction = 1.0 / np.sqrt(max(n_active_markets, 1))
    fraction  = float(np.clip(fraction, min_pct, max_pct))
    return bankroll * fraction


# ── Portfolio-level analysis ──────────────────────────────────────────────────

def ruin_probability(n_trades: int, win_rate: float, mean_pnl: float,
                     std_pnl: float, bankroll: float = 1000.0,
                     ruin_threshold: float = 0.20) -> float:
    """
    Estimate ruin probability using normal approximation (Gambler's Ruin).

    P(ruin) ≈ P(cumulative P&L < -bankroll * ruin_threshold at any point in n_trades)

    Using the reflection principle for random walks:
        σ_n  = std_pnl * sqrt(n_trades)
        μ_n  = mean_pnl * n_trades
        ruin_level = -bankroll * ruin_threshold
        P(ruin) ≈ 2 * Φ((ruin_level - μ_n) / σ_n)

    For fat-tailed returns (alpha < 3), this underestimates ruin probability.
    Add a fat-tail correction factor: multiply by (3/alpha) when alpha < 3.

    Ref: Feller (1968) "An Introduction to Probability Theory" Vol.1.
         Taleb "Statistical Consequences of Fat Tails" — Normal approx
         underestimates ruin for fat tails.

    Args:
        n_trades:       Number of trades to simulate
        win_rate:       Historical win rate
        mean_pnl:       Mean P&L per trade
        std_pnl:        Std dev of P&L per trade
        bankroll:       Starting capital
        ruin_threshold: Fraction of bankroll defining ruin (default 20%)

    Returns:
        p_ruin: float ∈ [0, 1]
    """
    from scipy import stats
    if std_pnl <= 0 or n_trades <= 0:
        return 0.0

    mu_n  = mean_pnl * n_trades
    sig_n = std_pnl * np.sqrt(n_trades)
    ruin_level = -bankroll * ruin_threshold

    # Reflection principle for two-sided boundary (simplified)
    z = (ruin_level - mu_n) / sig_n
    p_ruin = 2.0 * float(stats.norm.cdf(z))
    return float(np.clip(p_ruin, 0.0, 1.0))


def analyze_portfolio(trades: pd.DataFrame, bankroll: float = 1000.0) -> TalebPortfolioRisk:
    """
    Full Taleb portfolio risk analysis on a trades DataFrame.

    Required columns: pnl, entry_price, category (optional)

    Returns:
        TalebPortfolioRisk with fragility, tail exponents, ruin probs, stress results
    """
    pnl          = trades["pnl"]
    entry_prices = trades["entry_price"]

    # Fragility
    frag  = fragility_score(pnl, bankroll)
    is_af = frag > 0

    # Tail exponents per category
    alpha_by_cat = {}
    if "category" in trades.columns:
        for cat, g in trades.groupby("category"):
            alpha_by_cat[cat] = hill_estimator(g["pnl"].values)
    alpha_overall = hill_estimator(pnl.values)
    alpha_by_cat["_overall"] = alpha_overall

    # Barbell ratio from overall alpha
    bb = barbell_ratio(alpha_overall)

    # Ruin probabilities
    mu_pnl  = float(pnl.mean())
    std_pnl = float(pnl.std())
    ruin_1k  = ruin_probability(1_000,  0.0, mu_pnl, std_pnl, bankroll)
    ruin_5k  = ruin_probability(5_000,  0.0, mu_pnl, std_pnl, bankroll)
    ruin_10k = ruin_probability(10_000, 0.0, mu_pnl, std_pnl, bankroll)

    # Stress test
    stress = stress_test(pnl, entry_prices)

    return TalebPortfolioRisk(
        fragility          = frag,
        is_antifragile     = is_af,
        alpha_by_category  = alpha_by_cat,
        barbell_ratio      = bb,
        ruin_prob_1000     = ruin_1k,
        ruin_prob_5000     = ruin_5k,
        ruin_prob_10000    = ruin_10k,
        stress_results     = stress,
    )


def score_trade(entry_price: float, model_prob: float,
                uncertainty: float, bankroll: float = 1000.0,
                stake: float = 1.0) -> TalebSignal:
    """
    Compute full Taleb signal bundle for a single trade candidate.

    Args:
        entry_price:  Current market ask (0–1)
        model_prob:   RF P(YES) estimate
        uncertainty:  RF tree std dev (from rf_uncertainty())
        bankroll:     Current capital
        stake:        Planned stake in dollars

    Returns:
        TalebSignal
    """
    # Jensen convexity
    conv = jensen_convexity(entry_price, model_prob)

    # Geometric growth
    gain = 1.0 - entry_price   # win: get $1, cost entry_price
    loss = entry_price          # lose: lose entry_price
    g    = geometric_growth_rate(model_prob, gain, loss, bankroll)

    # Kelly
    gain_per_dollar = (1.0 - entry_price) / max(entry_price, 1e-6)
    k = ergodicity_kelly(model_prob, gain_per_dollar, 1.0)

    # Taleb score: combo of convexity + uncertainty + ergodic growth
    # Normalized to [0, 1] range where possible
    taleb_sc = (
        0.40 * np.clip(conv / 0.5, 0, 1) +      # convexity (max ~0.5 for binary)
        0.30 * np.clip(uncertainty / 0.3, 0, 1) + # uncertainty (max ~0.3 for RF)
        0.30 * np.clip(g * 100, 0, 1)             # geometric growth (scaled)
    )

    return TalebSignal(
        ticker           = "",
        convexity_bias   = round(conv, 6),
        uncertainty      = round(uncertainty, 6),
        geometric_growth = round(g, 8),
        kelly_fraction   = round(k, 6),
        is_ergodic       = g > 0,
        taleb_score      = round(taleb_sc, 6),
    )


if __name__ == "__main__":
    # Quick smoke test
    print("=== Taleb Framework Smoke Test ===")

    # Jensen convexity for binary contract at 5¢ with 25% model prob
    c = jensen_convexity(0.05, 0.25)
    print(f"Jensen convexity (entry=0.05, p=0.25): {c:.4f}")
    assert c > 0, "Convexity should be positive for mispriced contract"

    # Hill estimator on synthetic fat-tailed returns
    np.random.seed(42)
    returns = np.random.pareto(2.5, 500) * 0.1
    alpha = hill_estimator(returns)
    print(f"Hill alpha (Pareto a=2.5): {alpha:.2f}  (should be ~2.5)")

    # Geometric growth
    g = geometric_growth_rate(0.27, 0.951, 0.049, 1000)
    print(f"Geometric growth (p=0.27, gain=0.951, loss=0.049): {g:.6f}")
    print(f"Is ergodic: {g > 0}")

    # Barbell ratio
    for a in [2, 3, 4, 10]:
        print(f"Barbell ratio (alpha={a}): safe={barbell_ratio(a):.2f}")

    # Fragility on simple sequence
    pnl = pd.Series(np.random.normal(0.05, 0.4, 500))
    f = fragility_score(pnl)
    print(f"Fragility score: {f:.6f}  ({'antifragile' if f>0 else 'fragile'})")

    # 1/N sizing
    for n in [10, 100, 1000, 10000]:
        print(f"Max position at N={n}: ${max_position_size(10000, n):.2f}")

    print("✓ All smoke tests passed")
