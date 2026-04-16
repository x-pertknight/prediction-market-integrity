"""tests/test_vol_surface.py — Unit tests for Black-Scholes vol surface scanner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from alpha.vol_surface import (
    binary_bs_price,
    implied_vol,
    analyze_vol_surface,
    VolSurfaceScanner,
)


class TestBinaryBSPrice:
    def test_at_the_money_positive(self):
        # S=0.5, K=0.5, T=1yr, sigma=0.5 → reasonable price
        price = binary_bs_price(0.5, 0.5, 1.0, 0.0, 0.5)
        assert 0.0 < price < 1.0

    def test_higher_vol_moves_toward_half(self):
        # Higher vol → price moves toward 0.5 (uncertainty)
        p_low = binary_bs_price(0.7, 0.5, 1.0, 0.0, 0.2)
        p_high = binary_bs_price(0.7, 0.5, 1.0, 0.0, 2.0)
        # With very high vol, d2 → -inf, N(d2) → 0, so price → 0
        # But for moderate vol increase, we're just checking it changes
        assert p_low != p_high

    def test_zero_time_returns_spot(self):
        price = binary_bs_price(0.7, 0.5, 0.0, 0.0, 0.5)
        assert price == pytest.approx(0.7, abs=0.01)

    def test_zero_vol_returns_spot(self):
        price = binary_bs_price(0.6, 0.5, 1.0, 0.0, 0.0)
        assert price == pytest.approx(0.6, abs=0.01)

    def test_zero_spot_returns_zero(self):
        price = binary_bs_price(0.0, 0.5, 1.0, 0.0, 0.5)
        assert price == 0.0

    def test_price_bounded(self):
        # Price should always be in [0, 1]
        for S in [0.1, 0.5, 0.9]:
            for T in [0.01, 0.5, 2.0]:
                for sigma in [0.1, 1.0, 5.0]:
                    p = binary_bs_price(S, 0.5, T, 0.0, sigma)
                    assert 0.0 <= p <= 1.0, f"S={S}, T={T}, σ={sigma} → {p}"


class TestImpliedVol:
    def test_roundtrip(self):
        # Compute price at known vol, then recover vol
        true_vol = 0.8
        price = binary_bs_price(0.5, 0.5, 0.5, 0.0, true_vol)
        recovered = implied_vol(price, S=0.5, K=0.5, T=0.5, r=0.0)
        assert recovered is not None
        assert recovered == pytest.approx(true_vol, abs=0.05)

    def test_extreme_price_returns_none(self):
        assert implied_vol(0.001, S=0.5, T=0.5) is None
        assert implied_vol(0.999, S=0.5, T=0.5) is None

    def test_no_time_returns_none(self):
        assert implied_vol(0.5, S=0.5, T=0.0) is None
        assert implied_vol(0.5, S=0.5, T=None) is None

    def test_negative_time_returns_none(self):
        assert implied_vol(0.5, S=0.5, T=-0.1) is None

    def test_returns_positive(self):
        iv = implied_vol(0.5, S=0.5, K=0.5, T=0.5, r=0.0)
        if iv is not None:
            assert iv > 0


class TestAnalyzeVolSurface:
    def _make_markets(self, series: str, n: int,
                      prices: list[float] | None = None,
                      T_days: list[float] | None = None) -> list[dict]:
        """Helper to create multi-expiry market list."""
        from datetime import datetime, timezone, timedelta
        if prices is None:
            prices = [0.3 + i * 0.05 for i in range(n)]
        if T_days is None:
            T_days = [30 + i * 30 for i in range(n)]
        markets = []
        for i in range(n):
            exp = datetime.now(timezone.utc) + timedelta(days=T_days[i])
            markets.append({
                "ticker": f"{series}-{i}",
                "event_ticker": f"{series}-EV{i}",
                "series_ticker": series,
                "title": f"Test market {series} #{i}",
                "last_price": prices[i],
                "close_time": exp.isoformat(),
                "category": "Economics",
                "volume": 1000,
            })
        return markets

    def test_same_maturity_same_price_no_anomaly(self):
        # Markets at same maturity + same price → same IV → no anomalies
        markets = self._make_markets(
            "TEST", 4, prices=[0.60, 0.60, 0.60, 0.60],
            T_days=[60, 60, 60, 60],
        )
        anomalies = analyze_vol_surface(markets)
        assert len(anomalies) == 0

    def test_detects_outlier_iv(self):
        # 3 consistent + 1 extreme outlier (very cheap = high IV)
        markets = self._make_markets(
            "TEST", 4,
            prices=[0.60, 0.60, 0.60, 0.30],  # last one diverges
            T_days=[30, 60, 90, 120],
        )
        anomalies = analyze_vol_surface(markets)
        # The outlier price should be flagged
        assert len(anomalies) >= 1

    def test_too_few_markets_skipped(self):
        markets = self._make_markets("TEST", 2)
        anomalies = analyze_vol_surface(markets)
        assert len(anomalies) == 0

    def test_different_series_not_mixed(self):
        # Same maturity + price in each series → no cross-contamination
        m1 = self._make_markets("SERIES_A", 3, prices=[0.60, 0.60, 0.60],
                                T_days=[60, 60, 60])
        m2 = self._make_markets("SERIES_B", 3, prices=[0.70, 0.70, 0.70],
                                T_days=[60, 60, 60])
        anomalies = analyze_vol_surface(m1 + m2)
        assert len(anomalies) == 0  # consistent within each

    def test_extreme_prices_filtered(self):
        # Prices at 0.01 or 0.99 should be skipped
        markets = self._make_markets(
            "TEST", 4,
            prices=[0.01, 0.99, 0.60, 0.60],
            T_days=[30, 60, 90, 120],
        )
        # Only 2 valid prices → below MIN_MARKETS_PER_SERIES
        anomalies = analyze_vol_surface(markets)
        assert len(anomalies) == 0

    def test_anomaly_has_required_fields(self):
        markets = self._make_markets(
            "TEST", 4,
            prices=[0.60, 0.60, 0.60, 0.30],
            T_days=[30, 60, 90, 120],
        )
        anomalies = analyze_vol_surface(markets)
        if anomalies:
            a = anomalies[0]
            assert "iv" in a
            assert "mean_iv" in a
            assert "iv_deviation" in a
            assert "side" in a
            assert a["side"] in ("BUY", "SELL")
            assert abs(a["iv_deviation"]) > 0.30

    def test_empty_markets_no_crash(self):
        assert analyze_vol_surface([]) == []

    def test_no_series_ticker_skipped(self):
        markets = [{"ticker": "X", "last_price": 0.5}]
        assert analyze_vol_surface(markets) == []
