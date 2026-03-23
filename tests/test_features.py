"""tests/test_features.py — Unit tests for features_clean.py"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from features_clean import CLEAN_FEATURES, LOOKAHEAD_FEATURES


class TestCleanFeatures:
    def test_no_lookahead_in_clean(self):
        """CLEAN_FEATURES must not contain any lookahead feature."""
        for f in CLEAN_FEATURES:
            assert f not in LOOKAHEAD_FEATURES, f"{f} is a lookahead feature"

    def test_market_price_excluded(self):
        assert "market_price" not in CLEAN_FEATURES

    def test_price_lag1_excluded(self):
        assert "price_lag1" not in CLEAN_FEATURES

    def test_minimum_feature_count(self):
        """Should have at least 20 features (currently 27)."""
        assert len(CLEAN_FEATURES) >= 20

    def test_no_duplicates(self):
        assert len(CLEAN_FEATURES) == len(set(CLEAN_FEATURES))

    def test_key_features_present(self):
        """Core features that should always exist."""
        required = [
            "days_in_market", "category_enc", "market_type_enc",
            "day_of_week", "month", "fear_greed_value",
            "btc_ret_1d", "hist_win_rate_series",
        ]
        for feat in required:
            assert feat in CLEAN_FEATURES, f"Missing required feature: {feat}"

    def test_lookahead_features_defined(self):
        """LOOKAHEAD_FEATURES should include the known lookahead columns."""
        assert "market_price" in LOOKAHEAD_FEATURES
        assert "price_lag1" in LOOKAHEAD_FEATURES
