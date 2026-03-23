"""tests/test_fees.py — Unit tests for src/risk/fees.py"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from risk.fees import (
    kalshi_taker_fee, kalshi_total_fee, kalshi_round_trip_fee,
    polymarket_taker_fee, polymarket_maker_rebate, polymarket_us_fee,
    compute_trade_fees, KALSHI_TAKER_COEFF,
)


# ── Kalshi fees ──────────────────────────────────────────────────────────────

class TestKalshiFees:
    def test_fee_at_50_cents(self):
        """Maximum fee at p=0.50: 0.07 * 0.5 * 0.5 = 0.0175"""
        assert kalshi_taker_fee(0.50) == pytest.approx(0.0175, abs=1e-6)

    def test_fee_at_5_cents(self):
        """Fee at p=0.05: 0.07 * 0.05 * 0.95 = 0.003325"""
        assert kalshi_taker_fee(0.05) == pytest.approx(0.003325, abs=1e-6)

    def test_fee_at_boundaries(self):
        """Fee at 0 and 1 should be zero."""
        assert kalshi_taker_fee(0.0) == 0.0
        assert kalshi_taker_fee(1.0) == 0.0

    def test_fee_symmetry(self):
        """Fee is symmetric: fee(p) == fee(1-p)."""
        for p in [0.1, 0.2, 0.3, 0.4]:
            assert kalshi_taker_fee(p) == pytest.approx(kalshi_taker_fee(1.0 - p), abs=1e-10)

    def test_fee_clips_negative(self):
        """Negative prices get clipped to 0."""
        assert kalshi_taker_fee(-0.5) == 0.0

    def test_fee_clips_above_one(self):
        """Prices > 1 get clipped to 1."""
        assert kalshi_taker_fee(1.5) == 0.0

    def test_total_fee_multiple_contracts(self):
        """Total fee scales linearly with contract count."""
        single = kalshi_taker_fee(0.3)
        assert kalshi_total_fee(0.3, 5) == pytest.approx(single * 5, abs=1e-10)

    def test_round_trip_fee(self):
        """Round-trip = entry + exit fee."""
        entry_fee = kalshi_taker_fee(0.20)
        exit_fee = kalshi_taker_fee(0.80)
        assert kalshi_round_trip_fee(0.20, 0.80) == pytest.approx(entry_fee + exit_fee, abs=1e-10)

    def test_parabolic_shape(self):
        """Fee should peak at p=0.5 and decrease toward 0 and 1."""
        fee_50 = kalshi_taker_fee(0.5)
        fee_30 = kalshi_taker_fee(0.3)
        fee_10 = kalshi_taker_fee(0.1)
        assert fee_50 > fee_30 > fee_10


# ── Polymarket fees ──────────────────────────────────────────────────────────

class TestPolymarketFees:
    def test_free_market(self):
        """Most markets on Polymarket are fee-free."""
        assert polymarket_taker_fee(0.50, "politics") == 0.0
        assert polymarket_taker_fee(0.50, "") == 0.0

    def test_crypto_fee(self):
        """Crypto short-term markets: P*(1-P)*0.02"""
        fee = polymarket_taker_fee(0.50, "crypto_daily")
        expected = 0.50 * 0.50 * 0.02
        assert fee == pytest.approx(expected, abs=1e-8)

    def test_ncaab_fee(self):
        """NCAAB markets have fees."""
        assert polymarket_taker_fee(0.50, "ncaab") > 0

    def test_maker_rebate_crypto(self):
        """Crypto maker rebate is 20% of taker fee."""
        fee = polymarket_taker_fee(0.50, "crypto_daily")
        rebate = polymarket_maker_rebate(0.50, "crypto_daily")
        assert rebate == pytest.approx(fee * 0.20, abs=1e-10)

    def test_maker_rebate_sport(self):
        """Sport maker rebate is 25% of taker fee."""
        fee = polymarket_taker_fee(0.50, "sport_fee")
        rebate = polymarket_maker_rebate(0.50, "sport_fee")
        assert rebate == pytest.approx(fee * 0.25, abs=1e-10)

    def test_us_fee_flat(self):
        """US CFTC fee: flat 0.01% of price."""
        assert polymarket_us_fee(0.50) == pytest.approx(0.50 * 0.0001, abs=1e-10)


# ── compute_trade_fees integration ───────────────────────────────────────────

class TestComputeTradeFees:
    def test_none_model(self):
        """fee_model='none' returns all zeros."""
        entry = pd.Series([0.10, 0.50, 0.90])
        exit_ = pd.Series([1.0, 0.0, 1.0])
        result = compute_trade_fees(entry, exit_, fee_model="none")
        assert (result["fee_total"] == 0.0).all()

    def test_kalshi_settlement_free(self):
        """Settlement exits (0 or 1) should have zero exit fee."""
        entry = pd.Series([0.10, 0.50])
        exit_ = pd.Series([1.0, 0.0])  # settlement
        result = compute_trade_fees(entry, exit_, fee_model="kalshi")
        assert (result["fee_exit"] == 0.0).all()
        assert (result["fee_entry"] > 0).all()

    def test_kalshi_entry_fee_matches(self):
        """Entry fees should match kalshi_taker_fee."""
        prices = [0.05, 0.20, 0.50, 0.80, 0.95]
        entry = pd.Series(prices)
        exit_ = pd.Series([1.0] * len(prices))
        result = compute_trade_fees(entry, exit_, fee_model="kalshi")
        for i, p in enumerate(prices):
            assert result["fee_entry"].iloc[i] == pytest.approx(
                kalshi_taker_fee(p), abs=1e-8
            )

    def test_output_shape(self):
        """Result has correct columns and length."""
        entry = pd.Series([0.10, 0.20, 0.30])
        exit_ = pd.Series([1.0, 0.0, 1.0])
        result = compute_trade_fees(entry, exit_, fee_model="kalshi")
        assert list(result.columns) == ["fee_entry", "fee_exit", "fee_total"]
        assert len(result) == 3

    def test_fee_total_is_sum(self):
        """fee_total = fee_entry + fee_exit."""
        entry = pd.Series([0.30, 0.50])
        exit_ = pd.Series([0.60, 0.0])  # first is non-settlement
        result = compute_trade_fees(entry, exit_, fee_model="kalshi")
        np.testing.assert_array_almost_equal(
            result["fee_total"].values,
            (result["fee_entry"] + result["fee_exit"]).values,
        )
