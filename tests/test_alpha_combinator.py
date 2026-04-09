"""tests/test_alpha_combinator.py — Unit tests for AlphaCombinator engine."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from alpha.combinator import AlphaCombinator, CombinedResult


class TestCombineEmptySignals:
    def test_returns_neutral_probability(self):
        ac = AlphaCombinator(window=10)
        result = ac.combine({})
        assert result.probability == 0.5

    def test_returns_empty_weights(self):
        ac = AlphaCombinator(window=10)
        result = ac.combine({})
        assert result.weights == {}

    def test_fallback_flag_set(self):
        ac = AlphaCombinator(window=10)
        result = ac.combine({})
        assert result.diagnostics["fallback"] is True


class TestCombineSingleSignal:
    def test_single_signal_returns_that_value(self):
        ac = AlphaCombinator(window=10)
        result = ac.combine({"rf_model": 0.8})
        assert result.probability == pytest.approx(0.8, abs=0.01)

    def test_single_signal_weight_is_one(self):
        ac = AlphaCombinator(window=10)
        result = ac.combine({"rf_model": 0.8})
        assert result.weights["rf_model"] == pytest.approx(1.0, abs=1e-6)


class TestEqualWeightFallback:
    def test_below_window_uses_equal_weights(self):
        ac = AlphaCombinator(window=200)
        signals = {"rf": 0.6, "hmm": 0.4, "cal": 0.8}
        result = ac.combine(signals)
        assert result.diagnostics["fallback"] is True
        for w in result.weights.values():
            assert w == pytest.approx(1.0 / 3, abs=1e-6)

    def test_equal_weight_probability_is_average(self):
        ac = AlphaCombinator(window=200)
        signals = {"rf": 0.6, "hmm": 0.4, "cal": 0.8}
        result = ac.combine(signals)
        expected = (0.6 + 0.4 + 0.8) / 3
        assert result.probability == pytest.approx(expected, abs=0.01)


class TestFullCombineAfterWindow:
    def _fill_history(self, ac, n_signals=3, window=50):
        """Fill combinator with random signal observations."""
        rng = np.random.default_rng(42)
        names = [f"sig_{i}" for i in range(n_signals)]
        for _ in range(window):
            signals = {name: float(rng.uniform(0.2, 0.8)) for name in names}
            ac.combine(signals)

    def test_full_combine_after_window(self):
        ac = AlphaCombinator(window=50)
        self._fill_history(ac, n_signals=3, window=50)
        result = ac.combine({"sig_0": 0.7, "sig_1": 0.3, "sig_2": 0.5})
        assert result.diagnostics["fallback"] is False

    def test_full_combine_has_n_effective(self):
        ac = AlphaCombinator(window=50)
        self._fill_history(ac, n_signals=3, window=50)
        result = ac.combine({"sig_0": 0.7, "sig_1": 0.3, "sig_2": 0.5})
        assert "n_effective" in result.diagnostics
        assert result.diagnostics["n_effective"] > 0

    def test_full_combine_has_combined_z(self):
        ac = AlphaCombinator(window=50)
        self._fill_history(ac, n_signals=3, window=50)
        result = ac.combine({"sig_0": 0.7, "sig_1": 0.3, "sig_2": 0.5})
        assert "combined_z" in result.diagnostics


class TestOutputProbabilityBounded:
    def test_probability_in_valid_range(self):
        ac = AlphaCombinator(window=10)
        for _ in range(15):
            result = ac.combine({"a": 0.99, "b": 0.01})
        assert 0.01 <= result.probability <= 0.99

    def test_extreme_signals_still_bounded(self):
        ac = AlphaCombinator(window=5)
        for _ in range(10):
            result = ac.combine({"a": 1.0, "b": 0.0})
        assert 0.01 <= result.probability <= 0.99


class TestWeightsSumToOne:
    def test_fallback_weights_sum(self):
        ac = AlphaCombinator(window=200)
        result = ac.combine({"a": 0.3, "b": 0.7, "c": 0.5})
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_full_combine_weights_sum(self):
        ac = AlphaCombinator(window=20)
        rng = np.random.default_rng(7)
        for _ in range(20):
            ac.combine({
                "a": float(rng.uniform(0.2, 0.8)),
                "b": float(rng.uniform(0.2, 0.8)),
                "c": float(rng.uniform(0.2, 0.8)),
            })
        result = ac.combine({"a": 0.6, "b": 0.4, "c": 0.5})
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)


class TestIREstimate:
    def test_ir_nonnegative(self):
        ac = AlphaCombinator(window=20)
        rng = np.random.default_rng(99)
        for _ in range(20):
            ac.combine({
                "a": float(rng.uniform(0.2, 0.8)),
                "b": float(rng.uniform(0.2, 0.8)),
            })
        result = ac.combine({"a": 0.6, "b": 0.4})
        assert result.ir_estimate >= 0.0


class TestRollingWindowTrim:
    def test_history_capped_at_window(self):
        ac = AlphaCombinator(window=10)
        for i in range(50):
            ac.combine({"x": float(i) / 50})
        assert len(ac.history) == 10

    def test_ready_after_window_filled(self):
        ac = AlphaCombinator(window=10)
        for i in range(10):
            ac.combine({"x": float(i) / 10})
        assert ac.ready is True

    def test_not_ready_before_window(self):
        ac = AlphaCombinator(window=10)
        for i in range(9):
            ac.combine({"x": float(i) / 10})
        assert ac.ready is False


class TestCorrelatedSignals:
    def test_correlated_signals_lower_n_effective(self):
        """Perfectly correlated signals should yield N_effective ~ 1."""
        ac_corr = AlphaCombinator(window=50)
        ac_indep = AlphaCombinator(window=50)
        rng = np.random.default_rng(42)

        for _ in range(50):
            v = float(rng.uniform(0.2, 0.8))
            # Correlated: both signals are the same value
            ac_corr.combine({"a": v, "b": v})
            # Independent: signals are uncorrelated
            ac_indep.combine({
                "a": float(rng.uniform(0.2, 0.8)),
                "b": float(rng.uniform(0.2, 0.8)),
            })

        r_corr = ac_corr.combine({"a": 0.6, "b": 0.6})
        r_indep = ac_indep.combine({"a": 0.6, "b": 0.6})

        # Correlated signals should have lower N_effective
        assert r_corr.diagnostics["n_effective"] < r_indep.diagnostics["n_effective"]
