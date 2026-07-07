"""tests/test_kalshi_adapter.py — Unit tests for kalshi_feature_adapter.py"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from data.kalshi_feature_adapter import (
    MODEL_FEATURES,
    TRAINING_MEDIANS,
    _CAT_ENC,
    build_features,
    compute_days_in_market,
    compute_volume_rank,
    get_category,
    market_passes_filter,
    diagnose_features,
    clear_win_rate_cache,
    _win_rate_cache,
    _fetch_series_win_rate,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_market():
    now = datetime.now(timezone.utc)
    return {
        "ticker": "KXHIGHNY-26MAR25-T50",
        "title": "NYC High Temp: 50°F or above on March 26?",
        "series_ticker": "KXHIGHNY",
        "event_ticker": "KXHIGHNY-26MAR25",
        "category": "Climate and Weather",
        "status": "open",
        "open_time": (now - timedelta(days=2)).isoformat(),
        "close_time": (now + timedelta(days=1)).isoformat(),
        "last_price_dollars": "0.65",
        "yes_ask_dollars": "0.68",
        "yes_bid_dollars": "0.62",
        "volume": 1500,
        "volume_fp": 1500,
    }


@pytest.fixture
def sample_macro():
    return {
        "fear_greed_value": 45.0,
        "btc_ret_1d": 0.02,
        "btc_ret_7d": -0.05,
        "btc_vol_20d": 0.03,
        "btc_zscore_30": 1.2,
        "dxy_proxy": 104.5,
        "fed_funds_rate": 4.5,
        "yield_10y": 4.2,
        "btc_funding_rate": -0.001,
        "btc_open_interest": 75000.0,
    }


@pytest.fixture
def all_markets(sample_market):
    """Multiple markets for volume_rank computation."""
    markets = []
    for i, vol in enumerate([100, 500, 1500, 3000, 10000]):
        m = dict(sample_market)
        m["ticker"] = f"MKT-{i}"
        m["volume"] = vol
        m["volume_fp"] = vol
        markets.append(m)
    return markets


@pytest.fixture(autouse=True)
def clean_cache():
    """Clear win rate cache before each test."""
    clear_win_rate_cache()
    yield
    clear_win_rate_cache()


# ── Tests ────────────────────────────────────────────────────────────────────

class TestModelFeatures:
    def test_model_features_count(self):
        """Model expects exactly 20 features."""
        assert len(MODEL_FEATURES) == 20

    def test_no_duplicates(self):
        assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES))


class TestBuildFeatures:
    def test_shape(self, sample_market, sample_macro):
        X = build_features(sample_market, sample_macro)
        assert X.shape == (1, 20)

    def test_no_nan(self, sample_market, sample_macro):
        X = build_features(sample_market, sample_macro)
        assert not np.isnan(X).any(), f"NaN at indices: {np.where(np.isnan(X))}"

    def test_dtype(self, sample_market, sample_macro):
        X = build_features(sample_market, sample_macro)
        assert X.dtype == np.float64

    def test_macro_values_propagated(self, sample_market, sample_macro):
        """Macro values should appear in the feature vector."""
        X = build_features(sample_market, sample_macro)
        vec = X[0]
        fg_idx = MODEL_FEATURES.index("fear_greed_value")
        assert vec[fg_idx] == 45.0
        dxy_idx = MODEL_FEATURES.index("dxy_proxy")
        assert vec[dxy_idx] == 104.5

    def test_empty_macro_uses_defaults(self, sample_market):
        X = build_features(sample_market, {})
        vec = X[0]
        fg_idx = MODEL_FEATURES.index("fear_greed_value")
        assert vec[fg_idx] == 50.0  # default


class TestDaysInMarket:
    def test_from_open_close_time(self):
        now = datetime.now(timezone.utc)
        m = {
            "open_time": (now - timedelta(days=3)).isoformat(),
            "close_time": (now + timedelta(days=4)).isoformat(),
        }
        days = compute_days_in_market(m)
        assert abs(days - 7.0) < 0.1  # ~7 days

    def test_missing_times_uses_median(self):
        days = compute_days_in_market({})
        assert days == TRAINING_MEDIANS["days_in_market"]

    def test_short_duration(self):
        now = datetime.now(timezone.utc)
        m = {
            "open_time": (now - timedelta(hours=1)).isoformat(),
            "close_time": (now + timedelta(hours=1)).isoformat(),
        }
        days = compute_days_in_market(m)
        assert 0.05 < days < 0.15  # ~0.083 days


class TestVolumeRank:
    def test_cross_sectional(self, all_markets):
        # Market with volume=1500 is 3rd of 5 (100, 500, 1500, 3000, 10000)
        m = all_markets[2]  # volume=1500
        rank = compute_volume_rank(m, all_markets)
        assert 0.4 <= rank <= 0.8  # should be around 0.6

    def test_highest_volume(self, all_markets):
        m = all_markets[4]  # volume=10000
        rank = compute_volume_rank(m, all_markets)
        assert rank == 1.0

    def test_empty_all_markets(self, sample_market):
        rank = compute_volume_rank(sample_market, [])
        assert rank == TRAINING_MEDIANS["volume_rank"]


class TestCategoryEncoding:
    def test_known_category(self):
        m = {"category": "Crypto"}
        assert get_category(m) == "Crypto"

    def test_weather_category(self):
        m = {"category": "Climate and Weather"}
        assert get_category(m) == "Climate and Weather"

    def test_unknown_category(self):
        m = {"category": "SomethingNew"}
        assert get_category(m) == "Exotics"

    def test_empty_category(self):
        m = {}
        assert get_category(m) == "Exotics"

    def test_encoding_values(self):
        """All category encodings should be in expected range."""
        for cat, enc in _CAT_ENC.items():
            assert 0 <= enc <= 16


class TestMarketFilter:
    def test_weather_passes(self):
        m = {"category": "Climate and Weather"}
        cats = {"weather", "crypto", "politics"}
        assert market_passes_filter(m, cats)

    def test_crypto_passes(self):
        m = {"category": "Crypto"}
        assert market_passes_filter(m, {"crypto"})

    def test_exotics_blocked(self):
        m = {"category": "Exotics"}
        cats = {"weather", "crypto", "politics"}
        assert not market_passes_filter(m, cats)


class TestHistWinRate:
    @patch("data.kalshi_feature_adapter._market_api")
    def test_cache_populated(self, mock_api):
        """After fetching, cache should contain the series."""
        mock_market = lambda result: MagicMock(to_dict=lambda: {"result": result})
        mock_resp = MagicMock()
        mock_resp.markets = [
            mock_market("yes"), mock_market("no"), mock_market("yes"),
            mock_market("no"), mock_market("no"),
        ]
        mock_resp.cursor = ""
        mock_api.get_markets.return_value = mock_resp

        rate = _fetch_series_win_rate("TEST_SERIES")
        assert rate == 2 / 5
        assert "TEST_SERIES" in _win_rate_cache

    @patch("data.kalshi_feature_adapter._market_api")
    def test_cache_hit(self, mock_api):
        """Second call should use cache, not API."""
        _win_rate_cache["CACHED"] = 0.75
        rate = _fetch_series_win_rate("CACHED")
        assert rate == 0.75
        mock_api.get_markets.assert_not_called()

    @patch("data.kalshi_feature_adapter._market_api")
    def test_api_failure_uses_median(self, mock_api):
        mock_api.get_markets.side_effect = Exception("timeout")
        rate = _fetch_series_win_rate("FAIL_SERIES")
        assert rate == TRAINING_MEDIANS["hist_win_rate_series"]


class TestCalendarFeatures:
    def test_calendar_from_close_time(self, sample_macro):
        m = {
            "close_time": "2026-03-25T12:00:00Z",
            "open_time": "2026-03-24T12:00:00Z",
            "category": "Crypto",
        }
        X = build_features(m, sample_macro)
        vec = X[0]

        dow_idx = MODEL_FEATURES.index("day_of_week")
        month_idx = MODEL_FEATURES.index("month")
        doy_idx = MODEL_FEATURES.index("day_of_year")

        # March 25, 2026 = Wednesday (2), month=3, doy=84
        assert vec[dow_idx] == 2  # Wednesday
        assert vec[month_idx] == 3
        assert vec[doy_idx] == 84

    def test_doy_sin_cos_range(self, sample_market, sample_macro):
        X = build_features(sample_market, sample_macro)
        vec = X[0]
        sin_idx = MODEL_FEATURES.index("doy_sin")
        cos_idx = MODEL_FEATURES.index("doy_cos")
        assert -1 <= vec[sin_idx] <= 1
        assert -1 <= vec[cos_idx] <= 1


class TestDiagnose:
    def test_diagnose_returns_all_features(self, sample_market, sample_macro):
        diag = diagnose_features(sample_market, sample_macro)
        assert len(diag) == 20
        for feat in MODEL_FEATURES:
            assert feat in diag
            assert "value" in diag[feat]
            assert "nan" in diag[feat]
            assert "mapping" in diag[feat]
