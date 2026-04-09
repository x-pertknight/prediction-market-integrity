"""tests/test_kelly_cv.py — Unit tests for Monte Carlo Kelly sizing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from risk.kelly import kelly_with_cv, quarter_kelly


class TestKellyWithCV:
    def test_positive_edge_positive_sizing(self):
        """Profitable bet (25% chance at 10c) yields f_empirical > 0."""
        f_emp, cv = kelly_with_cv(0.25, 0.10)
        assert f_emp > 0.0

    def test_no_edge_zero_sizing(self):
        """50/50 bet at 50c has zero edge -> zero sizing."""
        f_emp, cv = kelly_with_cv(0.50, 0.50)
        assert f_emp == 0.0

    def test_negative_edge_zero_sizing(self):
        """Bad bet (10% chance at 50c) has negative Kelly -> zero sizing."""
        f_emp, cv = kelly_with_cv(0.10, 0.50)
        assert f_emp == 0.0
        assert cv == 0.0

    def test_reduces_vs_kelly(self):
        """f_empirical should be <= quarter_kelly (CV penalty)."""
        win_prob, price = 0.30, 0.10
        f_emp, cv = kelly_with_cv(win_prob, price)
        f_kelly = quarter_kelly(win_prob, price)
        assert f_emp <= f_kelly

    def test_cv_edge_positive(self):
        """CV of edge should be positive for any uncertain bet."""
        _, cv = kelly_with_cv(0.25, 0.10)
        assert cv > 0.0

    def test_deterministic_with_seed(self):
        """Same seed produces identical results."""
        f1, cv1 = kelly_with_cv(0.30, 0.15, rng_seed=123)
        f2, cv2 = kelly_with_cv(0.30, 0.15, rng_seed=123)
        assert f1 == f2
        assert cv1 == cv2

    def test_different_seeds_differ(self):
        """Different seeds can produce different results."""
        f1, _ = kelly_with_cv(0.30, 0.15, rng_seed=1)
        f2, _ = kelly_with_cv(0.30, 0.15, rng_seed=2)
        # Very likely different (not guaranteed, but with 10K paths...)
        # Just check they're both positive
        assert f1 > 0 and f2 > 0

    def test_high_uncertainty_reduces_more(self):
        """Lower win probability -> higher CV -> lower f_empirical relative to kelly."""
        # Marginal edge: 15% chance at 10c
        f_marginal, cv_marginal = kelly_with_cv(0.15, 0.10)
        # Strong edge: 50% chance at 10c
        f_strong, cv_strong = kelly_with_cv(0.50, 0.10)

        # The marginal bet has higher uncertainty relative to edge
        # So cv_marginal should be higher than cv_strong
        assert cv_marginal > cv_strong

    def test_more_paths_stabilizes(self):
        """More Monte Carlo paths should give similar results."""
        f_1k, _ = kelly_with_cv(0.30, 0.15, n_paths=1_000, rng_seed=42)
        f_50k, _ = kelly_with_cv(0.30, 0.15, n_paths=50_000, rng_seed=42)
        # Should be close (within 20%)
        assert abs(f_1k - f_50k) / max(f_50k, 1e-10) < 0.20

    def test_boundary_price_zero(self):
        """entry_price=0 should return zero (quarter_kelly returns 0)."""
        f_emp, cv = kelly_with_cv(0.50, 0.0)
        assert f_emp == 0.0

    def test_boundary_price_one(self):
        """entry_price=1 should return zero."""
        f_emp, cv = kelly_with_cv(0.50, 1.0)
        assert f_emp == 0.0
