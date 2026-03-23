"""tests/test_backtest.py — Unit tests for backtest_clean.py"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from backtest_clean import apply_fees, bootstrap_sharpe_ci, EDGE_MULT


# ── apply_fees ───────────────────────────────────────────────────────────────

class TestApplyFees:
    def test_none_returns_original(self):
        """fee_model='none' returns original pnl unchanged."""
        pnl = pd.Series([0.1, -0.05, 0.2])
        entry = pd.Series([0.10, 0.30, 0.05])
        exit_ = pd.Series([1.0, 0.0, 1.0])
        pnl_net, fees = apply_fees(pnl, entry, exit_, fee_model="none")
        pd.testing.assert_series_equal(pnl_net, pnl)
        assert (fees == 0.0).all()

    def test_kalshi_reduces_pnl(self):
        """Kalshi fees make net P&L lower than gross."""
        pnl = pd.Series([0.5, 0.3, 0.8])
        entry = pd.Series([0.10, 0.20, 0.05])
        exit_ = pd.Series([1.0, 1.0, 1.0])
        pnl_net, fees = apply_fees(pnl, entry, exit_, fee_model="kalshi")
        assert (fees > 0).all()
        assert (pnl_net < pnl).all()

    def test_returns_tuple(self):
        """apply_fees returns a (pnl_net, fee_total) tuple."""
        pnl = pd.Series([0.1])
        entry = pd.Series([0.10])
        exit_ = pd.Series([1.0])
        result = apply_fees(pnl, entry, exit_, fee_model="kalshi")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_net_plus_fees_equals_gross(self):
        """net + fees = gross (conservation of money)."""
        pnl = pd.Series([0.5, -0.1, 0.3])
        entry = pd.Series([0.10, 0.50, 0.20])
        exit_ = pd.Series([1.0, 0.0, 1.0])
        pnl_net, fees = apply_fees(pnl, entry, exit_, fee_model="kalshi")
        np.testing.assert_array_almost_equal(
            (pnl_net + fees).values, pnl.values, decimal=10
        )


# ── bootstrap_sharpe_ci ─────────────────────────────────────────────────────

class TestBootstrapSharpeCI:
    def test_returns_dict_keys(self):
        returns = pd.Series(np.random.default_rng(42).normal(0.01, 0.1, 100))
        result = bootstrap_sharpe_ci(returns, n=100)
        assert "sharpe" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_ci_order(self):
        """ci_lower <= sharpe <= ci_upper."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.02, 0.1, 200))
        result = bootstrap_sharpe_ci(returns, n=500)
        assert result["ci_lower"] <= result["sharpe"] <= result["ci_upper"]

    def test_insufficient_data(self):
        """Very few returns → still returns a dict."""
        returns = pd.Series([0.01, 0.02])
        result = bootstrap_sharpe_ci(returns)
        assert isinstance(result, dict)

    def test_positive_sharpe_for_positive_mean(self):
        """Positive mean returns → positive Sharpe estimate."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.05, 0.1, 500))
        result = bootstrap_sharpe_ci(returns, n=200)
        assert result["sharpe"] > 0


# ── Constants ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_edge_mult(self):
        """EDGE_MULT should be 0.70 (enter when price <= pred_prob * 0.70)."""
        assert EDGE_MULT == pytest.approx(0.70, abs=1e-6)
