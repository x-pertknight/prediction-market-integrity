"""tests/test_taleb.py — Unit tests for src/risk/taleb.py"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from risk.taleb import (
    jensen_convexity, hill_estimator, kelly_from_alpha, gain_to_pain,
    geometric_growth_rate, ergodicity_kelly,
    barbell_ratio, classify_market,
    fragility_score, stress_test,
    max_position_size, ruin_probability,
    score_trade, analyze_portfolio,
    TalebSignal, TalebPortfolioRisk,
)


# ── Jensen Convexity ─────────────────────────────────────────────────────────

class TestJensenConvexity:
    def test_positive_convexity(self):
        """Mispriced contract (model_prob > entry_price) has positive convexity."""
        c = jensen_convexity(0.05, 0.25)
        assert c > 0

    def test_at_fair_value(self):
        """When entry == model_prob, convexity = p*(1-p) - 0 = p*(1-p)."""
        c = jensen_convexity(0.50, 0.50)
        # E[F(x)] = 0.5 * 0.5 = 0.25; F(E[x]) = max(0.5-0.5, 0) = 0
        assert c == pytest.approx(0.25, abs=1e-6)

    def test_extreme_mispricing(self):
        """Very cheap contract with high model_prob → positive convexity."""
        c = jensen_convexity(0.01, 0.90)
        # E[F(x)] = 0.90 * 0.99 = 0.891; F(E[x]) = max(0.90 - 0.01, 0) = 0.89
        # convexity = 0.891 - 0.89 = 0.001
        assert c > 0

    def test_zero_convexity_at_extremes(self):
        """entry=0, model=0 → convexity=0."""
        assert jensen_convexity(0.0, 0.0) == pytest.approx(0.0, abs=1e-10)

    def test_always_non_negative(self):
        """Convexity should be >= 0 for entry <= model_prob."""
        for ep in np.arange(0.01, 0.50, 0.05):
            for mp in np.arange(ep, 1.0, 0.05):
                assert jensen_convexity(ep, mp) >= -1e-10


# ── Hill Estimator ───────────────────────────────────────────────────────────

class TestHillEstimator:
    def test_pareto_returns(self):
        """Hill on Pareto(2.5) data should estimate alpha near 2.5."""
        rng = np.random.default_rng(42)
        returns = rng.pareto(2.5, 1000) * 0.1
        alpha = hill_estimator(returns)
        assert 1.5 < alpha < 4.0  # broad but reasonable

    def test_insufficient_data(self):
        """< 10 observations → returns inf (thin-tail assumption)."""
        alpha = hill_estimator(np.array([0.1, 0.2, 0.3]))
        assert alpha == np.inf

    def test_all_zeros(self):
        """All-zero returns → inf."""
        alpha = hill_estimator(np.zeros(100))
        assert alpha == np.inf

    def test_normal_returns_high_alpha(self):
        """Normal distribution has thin tails → high alpha (> 3)."""
        rng = np.random.default_rng(99)
        returns = rng.normal(0, 1, 5000)
        alpha = hill_estimator(returns)
        assert alpha > 2.5  # normal tails should give high alpha


class TestKellyFromAlpha:
    def test_extreme_fat_tails(self):
        assert kelly_from_alpha(1.5) == 0.10

    def test_fat_tails(self):
        assert kelly_from_alpha(2.5) == 0.25

    def test_moderate(self):
        assert kelly_from_alpha(3.5) == 0.50

    def test_thin_tails(self):
        assert kelly_from_alpha(5.0) == 1.0


class TestGainToPain:
    def test_positive_only(self):
        """All positive returns → infinite G/P."""
        assert gain_to_pain(np.array([0.1, 0.2, 0.3])) == np.inf

    def test_mixed(self):
        """Known G/P calculation."""
        returns = np.array([0.5, -0.2, 0.3, -0.1])
        expected = (0.5 + 0.3) / abs(-0.2 + -0.1)
        assert gain_to_pain(returns) == pytest.approx(expected, abs=1e-8)


# ── Geometric Growth Rate ────────────────────────────────────────────────────

class TestGeometricGrowth:
    def test_positive_growth(self):
        """Favorable bet has positive growth rate."""
        g = geometric_growth_rate(0.60, 1.0, 0.50, 100.0)
        assert g > 0

    def test_zero_bankroll(self):
        """Zero bankroll → -inf."""
        assert geometric_growth_rate(0.50, 1.0, 0.50, 0.0) == -np.inf

    def test_loss_exceeds_bankroll(self):
        """Loss >= bankroll → -inf (ruin)."""
        assert geometric_growth_rate(0.50, 1.0, 100.0, 100.0) == -np.inf

    def test_near_certain_win(self):
        """p near 1 with small loss → positive growth."""
        g = geometric_growth_rate(0.99, 0.10, 0.10, 1000.0)
        assert g > 0


# ── Ergodicity Kelly ─────────────────────────────────────────────────────────

class TestErgodicityKelly:
    def test_favorable_bet(self):
        """High p, good odds → positive Kelly fraction."""
        k = ergodicity_kelly(0.60, 2.0, 1.0)
        assert k > 0

    def test_unfavorable_bet(self):
        """Low p → zero Kelly (clipped)."""
        k = ergodicity_kelly(0.10, 0.5, 1.0)
        assert k == 0.0

    def test_zero_gain(self):
        """Zero gain per dollar → zero Kelly."""
        assert ergodicity_kelly(0.50, 0.0, 1.0) == 0.0


# ── Barbell Ratio ────────────────────────────────────────────────────────────

class TestBarbellRatio:
    def test_alpha_2(self):
        """alpha=2 → safe = 1 - 1/2 = 0.50."""
        assert barbell_ratio(2.0) == pytest.approx(0.50, abs=0.01)

    def test_alpha_4(self):
        """alpha=4 → safe = 1 - 3/4 = 0.25."""
        assert barbell_ratio(4.0) == pytest.approx(0.25, abs=0.01)

    def test_alpha_leq_1(self):
        """alpha <= 1 → extreme: 99% safe."""
        assert barbell_ratio(0.5) == 0.99
        assert barbell_ratio(1.0) == 0.99

    def test_clips_range(self):
        """Output always in [0.01, 0.99]."""
        for a in [0.1, 1.5, 2, 10, 100]:
            assert 0.01 <= barbell_ratio(a) <= 0.99


class TestClassifyMarket:
    def test_safe(self):
        result = classify_market(0.50, 0.1, 0.6, 3.0, 10000, 0.7)
        assert result == "safe"

    def test_speculative(self):
        result = classify_market(0.05, 0.5, 0.5, 0.5, 100, 0.3)
        assert result == "speculative"

    def test_skip(self):
        result = classify_market(0.30, 0.1, 0.2, 0.5, 200, 0.4)
        assert result == "skip"


# ── Fragility Score ──────────────────────────────────────────────────────────

class TestFragilityScore:
    def test_insufficient_data(self):
        """< 30 trades → returns 0."""
        assert fragility_score(pd.Series([0.1] * 10)) == 0.0

    def test_returns_float(self):
        """Output is a float."""
        rng = np.random.default_rng(42)
        pnl = pd.Series(rng.normal(0.05, 0.4, 500))
        f = fragility_score(pnl)
        assert isinstance(f, float)


# ── Stress Test ──────────────────────────────────────────────────────────────

class TestStressTest:
    def test_has_baseline(self):
        pnl = pd.Series([0.1, -0.05, 0.2])
        entry = pd.Series([0.20, 0.30, 0.10])
        result = stress_test(pnl, entry)
        assert "baseline" in result

    def test_negative_shock_helps_buyers(self):
        """Negative price shock → higher pnl for buyers."""
        pnl = pd.Series([0.1] * 50)
        entry = pd.Series([0.20] * 50)
        result = stress_test(pnl, entry)
        assert result["-10%"] > result["baseline"]


# ── Max Position Size ────────────────────────────────────────────────────────

class TestMaxPositionSize:
    def test_one_market(self):
        """1 market → 1/sqrt(1) = 100%, clipped to max_pct."""
        result = max_position_size(10000, 1)
        assert result == 10000 * 0.10  # clipped to 10%

    def test_hundred_markets(self):
        """100 markets → 1/sqrt(100) = 10%."""
        result = max_position_size(10000, 100)
        assert result == pytest.approx(1000, abs=1)  # 10% of 10000

    def test_zero_markets(self):
        """0 markets → max_pct."""
        result = max_position_size(10000, 0)
        assert result == 10000 * 0.10


# ── Ruin Probability ─────────────────────────────────────────────────────────

class TestRuinProbability:
    def test_zero_std(self):
        assert ruin_probability(1000, 0.6, 0.01, 0.0) == 0.0

    def test_positive_mean_low_ruin(self):
        """Strongly profitable system → low ruin prob."""
        p = ruin_probability(1000, 0.6, 0.10, 0.05, 1000.0)
        assert p < 0.01

    def test_negative_mean_high_ruin(self):
        """Losing system → high ruin prob."""
        p = ruin_probability(10000, 0.4, -0.10, 0.20, 100.0)
        assert p > 0.5


# ── Score Trade ──────────────────────────────────────────────────────────────

class TestScoreTrade:
    def test_returns_taleb_signal(self):
        sig = score_trade(0.05, 0.25, 0.10)
        assert isinstance(sig, TalebSignal)
        assert sig.convexity_bias > 0
        assert 0 <= sig.taleb_score <= 1

    def test_ergodic_favorable_bet(self):
        """Cheap contract with high model prob → ergodic."""
        sig = score_trade(0.05, 0.30, 0.10)
        assert bool(sig.is_ergodic) is True
        assert sig.kelly_fraction > 0


# ── Analyze Portfolio ────────────────────────────────────────────────────────

class TestAnalyzePortfolio:
    def test_returns_portfolio_risk(self):
        rng = np.random.default_rng(42)
        trades = pd.DataFrame({
            "pnl": rng.normal(0.02, 0.1, 100),
            "entry_price": rng.uniform(0.05, 0.50, 100),
            "category": ["Crypto"] * 50 + ["Politics"] * 50,
        })
        result = analyze_portfolio(trades)
        assert isinstance(result, TalebPortfolioRisk)
        assert "baseline" in result.stress_results
        assert "_overall" in result.alpha_by_category
