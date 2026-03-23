"""tests/test_risk.py — Unit tests for circuit_breaker.py and market_impact.py"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

from risk.circuit_breaker import (
    CircuitBreaker, ROLLING_WINDOW, MIN_ROLLING_SHARPE,
    MAX_CONSECUTIVE_LOSSES, SIZING_FULL, SIZING_REDUCED,
)
from risk.market_impact import (
    liquidity_tier, market_impact, should_trade, kelly_multiplier,
    compute_trade_impact, TIER_ILLIQUID, TIER_HIGH, TIER_MODERATE, TIER_LOW,
    TIER_KELLY_MULT, MIN_VOLUME_DOLLARS, MAX_POSITION_PCT,
)


# ── Circuit Breaker ──────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_initial_state_allowed(self):
        """Fresh circuit breaker allows trading."""
        cb = CircuitBreaker()
        result = cb.check()
        assert result["allowed"] is True
        assert result["sizing_mult"] == SIZING_FULL

    def test_three_consecutive_losses_halt(self):
        """3 consecutive losses trigger halt."""
        cb = CircuitBreaker()
        cb.record_trade(-0.01)
        cb.record_trade(-0.02)
        assert cb.check()["allowed"] is True  # only 2 losses
        cb.record_trade(-0.01)  # 3rd loss
        assert cb.check()["allowed"] is False
        assert "consecutive" in cb.halt_reason.lower()

    def test_win_resets_consecutive_losses(self):
        """A winning trade resets the consecutive loss counter."""
        cb = CircuitBreaker()
        cb.record_trade(-0.01)
        cb.record_trade(-0.01)
        cb.record_trade(0.10)  # win
        cb.record_trade(-0.01)
        cb.record_trade(-0.01)
        # Only 2 consecutive losses (after the win), should still be allowed
        assert cb.check()["allowed"] is True

    def test_manual_reset(self):
        """Manual reset clears halt state."""
        cb = CircuitBreaker()
        for _ in range(MAX_CONSECUTIVE_LOSSES):
            cb.record_trade(-0.01)
        assert cb.check()["allowed"] is False
        cb.reset()
        assert cb.check()["allowed"] is True

    def test_record_trade_updates_state(self):
        """record_trade appends to trades list."""
        cb = CircuitBreaker()
        cb.record_trade(0.05)
        cb.record_trade(-0.02)
        assert len(cb.trades) == 2

    def test_status_summary(self):
        """status_summary returns expected keys."""
        cb = CircuitBreaker()
        cb.record_trade(0.05)
        summary = cb.status_summary()
        expected_keys = {
            "halted", "halt_reason", "sizing_mult", "total_trades",
            "consecutive_losses", "rolling_sharpe_30", "total_pnl",
        }
        assert set(summary.keys()) == expected_keys
        assert summary["total_trades"] == 1
        assert summary["total_pnl"] == 0.05

    def test_rolling_sharpe_halt(self):
        """Rolling Sharpe below threshold triggers halt after enough trades."""
        cb = CircuitBreaker()
        # Feed 30 trades with negative mean (low Sharpe)
        rng = np.random.default_rng(42)
        for _ in range(ROLLING_WINDOW):
            cb.record_trade(rng.normal(-0.05, 0.01))
        assert cb.check()["allowed"] is False


# ── Market Impact ────────────────────────────────────────────────────────────

class TestLiquidityTier:
    def test_illiquid(self):
        assert liquidity_tier(50) == TIER_ILLIQUID

    def test_high(self):
        assert liquidity_tier(200) == TIER_HIGH

    def test_moderate(self):
        assert liquidity_tier(1000) == TIER_MODERATE

    def test_low(self):
        assert liquidity_tier(10000) == TIER_LOW

    def test_boundaries(self):
        assert liquidity_tier(99.99) == TIER_ILLIQUID
        assert liquidity_tier(100) == TIER_HIGH
        assert liquidity_tier(500) == TIER_MODERATE
        assert liquidity_tier(5000) == TIER_LOW


class TestMarketImpact:
    def test_zero_volume(self):
        """Zero volume → max impact of 1.0."""
        assert market_impact(1.0, 0.0) == 1.0

    def test_large_volume_low_impact(self):
        """High volume → low impact."""
        impact = market_impact(1.0, 1_000_000)
        assert impact < 0.001

    def test_scales_with_position_size(self):
        """Impact increases with position size."""
        small = market_impact(1.0, 1000)
        large = market_impact(10.0, 1000)
        assert large > small

    def test_formula(self):
        """Verify formula: impact = pos * vol / sqrt(daily_vol)."""
        pos, vol, daily = 5.0, 0.15, 10000.0
        expected = pos * vol / np.sqrt(daily)
        assert market_impact(pos, daily, vol) == pytest.approx(expected, abs=1e-10)


class TestShouldTrade:
    def test_illiquid_rejected(self):
        assert should_trade(1.0, 50.0) is False

    def test_too_large_rejected(self):
        """Position > 10% of daily volume → reject."""
        assert should_trade(600.0, 1000.0) is False

    def test_normal_trade_allowed(self):
        assert should_trade(10.0, 5000.0) is True


class TestKellyMultiplier:
    def test_illiquid_zero(self):
        assert kelly_multiplier(50) == 0.0

    def test_low_full(self):
        assert kelly_multiplier(10000) == 1.0

    def test_all_tiers(self):
        """Each tier maps to expected Kelly mult."""
        for tier, mult in TIER_KELLY_MULT.items():
            vol = {TIER_ILLIQUID: 50, TIER_HIGH: 200,
                   TIER_MODERATE: 1000, TIER_LOW: 10000}[tier]
            assert kelly_multiplier(vol) == mult


class TestComputeTradeImpact:
    def test_output_columns(self):
        entry = pd.Series([0.10, 0.50])
        vols = pd.Series([500.0, 10000.0])
        result = compute_trade_impact(entry, vols)
        expected_cols = {"daily_vol_dollars", "liquidity_tier", "impact_cost",
                         "kelly_mult", "trade_allowed"}
        assert set(result.columns) == expected_cols

    def test_output_length(self):
        entry = pd.Series([0.10, 0.20, 0.30])
        vols = pd.Series([100.0, 500.0, 1000.0])
        result = compute_trade_impact(entry, vols)
        assert len(result) == 3
