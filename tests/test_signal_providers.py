"""tests/test_signal_providers.py — Unit tests for all 7 signal providers."""

import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from alpha.combinator import (
    RFSignalProvider,
    HMMSignalProvider,
    CrossVenueSignalProvider,
    MomentumSignalProvider,
    CalibrationSignalProvider,
    OBISignalProvider,
    WhaleSignalProvider,
    SignalResult,
)


# ── RF Signal Provider ───────────────────────────────────────────────────

class MockRFModel:
    """Mock sklearn classifier for testing."""
    def __init__(self, prob: float = 0.7):
        self._prob = prob

    def predict_proba(self, X):
        return np.array([[1 - self._prob, self._prob]])


class TestRFSignalProvider:
    def test_returns_model_probability(self):
        model = MockRFModel(prob=0.75)
        provider = RFSignalProvider(model)
        result = provider.score({}, {"features": np.zeros((1, 20))})
        assert result.confidence == pytest.approx(0.75, abs=1e-6)
        assert result.name == "rf_model"

    def test_no_model_returns_neutral(self):
        provider = RFSignalProvider(None)
        result = provider.score({}, {"features": np.zeros((1, 20))})
        assert result.confidence == 0.5

    def test_no_features_returns_neutral(self):
        model = MockRFModel(prob=0.8)
        provider = RFSignalProvider(model)
        result = provider.score({}, {})
        assert result.confidence == 0.5

    def test_confidence_bounded(self):
        model = MockRFModel(prob=0.999)
        provider = RFSignalProvider(model)
        result = provider.score({}, {"features": np.zeros((1, 20))})
        assert 0.0 <= result.confidence <= 1.0


# ── HMM Signal Provider ─────────────────────────────────────────────────

class MockHMMDetector:
    """Mock HMMRegimeDetector for testing."""
    def __init__(self, regime: str = "TRENDING", fitted: bool = True):
        self._regime = regime
        self.fitted = fitted

    def predict_current(self, returns):
        return self._regime

    def kelly_multiplier(self, regime):
        return {"TRENDING": 1.0, "MEAN_REVERTING": 1.5, "VOLATILE": 0.5}.get(regime, 1.0)

    def confidence_score(self, returns):
        return {"MEAN_REVERTING": 0.7, "TRENDING": 0.5, "VOLATILE": 0.2}.get(self._regime, 0.5)


class TestHMMSignalProvider:
    def test_trending_regime(self):
        detector = MockHMMDetector(regime="TRENDING")
        provider = HMMSignalProvider(detector)
        result = provider.score({}, {"returns": pd.Series([0.01, -0.02, 0.01])})
        assert result.confidence == pytest.approx(0.5, abs=0.01)

    def test_mean_reverting_regime(self):
        detector = MockHMMDetector(regime="MEAN_REVERTING")
        provider = HMMSignalProvider(detector)
        result = provider.score({}, {"returns": pd.Series([0.01, -0.02])})
        assert result.confidence == pytest.approx(0.7, abs=0.01)

    def test_volatile_regime(self):
        detector = MockHMMDetector(regime="VOLATILE")
        provider = HMMSignalProvider(detector)
        result = provider.score({}, {"returns": pd.Series([0.01])})
        assert result.confidence == pytest.approx(0.2, abs=0.01)

    def test_no_detector_returns_neutral(self):
        provider = HMMSignalProvider(None)
        result = provider.score({}, {"returns": pd.Series([0.01])})
        assert result.confidence == 0.5

    def test_unfitted_detector_returns_neutral(self):
        detector = MockHMMDetector(fitted=False)
        provider = HMMSignalProvider(detector)
        result = provider.score({}, {"returns": pd.Series([0.01])})
        assert result.confidence == 0.5

    def test_no_returns_in_context(self):
        detector = MockHMMDetector()
        provider = HMMSignalProvider(detector)
        result = provider.score({}, {})
        assert result.confidence == 0.5


# ── Cross-Venue Signal Provider ──────────────────────────────────────────

class TestCrossVenueSignalProvider:
    def test_no_cross_prices_returns_neutral(self):
        provider = CrossVenueSignalProvider()
        result = provider.score({"title": "Test"}, {})
        assert result.confidence == 0.5
        assert result.metadata["matched"] is False

    def test_high_similarity_match(self):
        provider = CrossVenueSignalProvider()
        market = {"title": "Will Bitcoin exceed 100000 by December", "last_price": 0.45}
        cross = [{"title": "Will Bitcoin exceed 100000 by December", "price": 0.44}]
        result = provider.score(market, {"cross_venue_prices": cross})
        assert result.metadata["matched"] is True
        # Divergence = 0.01 -> confidence = 1 - 0.01/0.10 = 0.90
        assert result.confidence == pytest.approx(0.90, abs=0.01)

    def test_below_similarity_threshold_no_match(self):
        provider = CrossVenueSignalProvider()
        market = {"title": "Bitcoin price prediction for 2026"}
        cross = [{"title": "Ethereum staking rewards Q4", "price": 0.5}]
        result = provider.score(market, {"cross_venue_prices": cross})
        assert result.metadata["matched"] is False
        assert result.confidence == 0.5

    def test_large_divergence_low_confidence(self):
        provider = CrossVenueSignalProvider()
        market = {"title": "Will Bitcoin exceed 100000", "last_price": 0.30}
        cross = [{"title": "Will Bitcoin exceed 100000", "price": 0.50}]
        result = provider.score(market, {"cross_venue_prices": cross})
        # Divergence = 0.20, confidence = max(0, 1 - 0.20/0.10) = 0.0
        assert result.confidence == pytest.approx(0.0, abs=0.01)

    def test_similarity_threshold_exact_boundary(self):
        """90% similarity required. Test just below boundary."""
        provider = CrossVenueSignalProvider()
        # Jaccard with these words (removing stop words):
        # a = {"bitcoin", "price"}, b = {"bitcoin", "value"} -> 1/3 = 0.33
        market = {"title": "bitcoin price"}
        cross = [{"title": "bitcoin value", "price": 0.5}]
        result = provider.score(market, {"cross_venue_prices": cross})
        assert result.metadata["matched"] is False


# ── Momentum Signal Provider ────────────────────────────────────────────

class TestMomentumSignalProvider:
    def test_no_history_returns_neutral(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "momentum.json"
            provider = MomentumSignalProvider(buffer_path=path)
            result = provider.score({"ticker": "TEST-001", "last_price": 0.5}, {})
            # First observation recorded, but only 1 data point -> neutral
            assert result.confidence == 0.5

    def test_positive_slope_above_neutral(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "momentum.json"
            # Pre-populate buffer with rising prices
            buffer = {"TEST-001": [
                {"price": 0.40, "ts": 1000.0},
                {"price": 0.42, "ts": 1300.0},
                {"price": 0.44, "ts": 1600.0},
                {"price": 0.46, "ts": 1900.0},
                {"price": 0.48, "ts": 2200.0},
            ]}
            path.write_text(json.dumps(buffer))
            provider = MomentumSignalProvider(buffer_path=path)
            result = provider.score({"ticker": "TEST-001", "last_price": 0.50}, {})
            assert result.confidence > 0.5
            assert result.metadata["hourly_slope"] > 0

    def test_negative_slope_below_neutral(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "momentum.json"
            buffer = {"TEST-002": [
                {"price": 0.60, "ts": 1000.0},
                {"price": 0.58, "ts": 1300.0},
                {"price": 0.56, "ts": 1600.0},
                {"price": 0.54, "ts": 1900.0},
                {"price": 0.52, "ts": 2200.0},
            ]}
            path.write_text(json.dumps(buffer))
            provider = MomentumSignalProvider(buffer_path=path)
            result = provider.score({"ticker": "TEST-002", "last_price": 0.50}, {})
            assert result.confidence < 0.5
            assert result.metadata["hourly_slope"] < 0

    def test_buffer_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "momentum.json"
            provider = MomentumSignalProvider(buffer_path=path)
            provider.score({"ticker": "PERSIST-1", "last_price": 0.5}, {})
            assert path.exists()
            data = json.loads(path.read_text())
            assert "PERSIST-1" in data
            assert len(data["PERSIST-1"]) == 1

    def test_buffer_max_history_trim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "momentum.json"
            # Pre-fill with MAX_HISTORY entries
            buffer = {"TRIM-1": [
                {"price": 0.5, "ts": float(i)} for i in range(300)
            ]}
            path.write_text(json.dumps(buffer))
            provider = MomentumSignalProvider(buffer_path=path)
            provider.score({"ticker": "TRIM-1", "last_price": 0.5}, {})
            assert len(provider._tracker.buffer["TRIM-1"]) <= provider._tracker.MAX_HISTORY


# ── Calibration Signal Provider ──────────────────────────────────────────

class MockCalibrationAnalyzer:
    """Mock CalibrationAnalyzer for testing."""
    def adjusted_prob(self, raw_prob, category=None):
        # Simulate slight adjustment
        return raw_prob * 0.95

    def confidence_score(self, raw_prob, category=None):
        return self.adjusted_prob(raw_prob, category)


class TestCalibrationSignalProvider:
    def test_no_analyzer_returns_neutral(self):
        provider = CalibrationSignalProvider(None)
        result = provider.score({}, {"model_prob": 0.7})
        assert result.confidence == 0.5

    def test_with_analyzer_returns_adjusted(self):
        analyzer = MockCalibrationAnalyzer()
        provider = CalibrationSignalProvider(analyzer)
        result = provider.score({"category": "Crypto"}, {"model_prob": 0.8})
        # 0.8 * 0.95 = 0.76
        assert result.confidence == pytest.approx(0.76, abs=0.01)

    def test_metadata_contains_raw_prob(self):
        analyzer = MockCalibrationAnalyzer()
        provider = CalibrationSignalProvider(analyzer)
        result = provider.score({"category": "Sports"}, {"model_prob": 0.6})
        assert result.metadata["raw_prob"] == 0.6
        assert result.metadata["category"] == "Sports"

    def test_confidence_bounded(self):
        analyzer = MockCalibrationAnalyzer()
        provider = CalibrationSignalProvider(analyzer)
        result = provider.score({}, {"model_prob": 1.5})
        assert 0.0 <= result.confidence <= 1.0


# ── OBI Signal Provider ────────────────────────────────────────────────

class MockOBIEngine:
    """Mock OrderBookImbalance for testing."""
    def __init__(self, obi: float = 0.0):
        self._obi = obi

    def fetch_and_compute(self, ticker: str) -> dict:
        ir = (self._obi + 1.0) / 2.0
        signal = ("BULLISH" if self._obi > 0.35 else
                  "BEARISH" if self._obi < -0.35 else "NEUTRAL")
        return {"obi": self._obi, "ir": ir, "signal": signal, "source": "mock"}


class TestOBISignalProvider:
    def test_neutral_obi(self):
        engine = MockOBIEngine(obi=0.0)
        provider = OBISignalProvider(obi_engine=engine)
        result = provider.score({"ticker": "TEST"}, {})
        assert result.confidence == pytest.approx(0.5, abs=0.01)
        assert result.name == "obi"

    def test_bullish_obi(self):
        engine = MockOBIEngine(obi=0.6)
        provider = OBISignalProvider(obi_engine=engine)
        result = provider.score({"ticker": "TEST"}, {})
        # (0.6 + 1) / 2 = 0.8
        assert result.confidence == pytest.approx(0.8, abs=0.01)
        assert result.metadata["signal"] == "BULLISH"

    def test_bearish_obi(self):
        engine = MockOBIEngine(obi=-0.6)
        provider = OBISignalProvider(obi_engine=engine)
        result = provider.score({"ticker": "TEST"}, {})
        # (-0.6 + 1) / 2 = 0.2
        assert result.confidence == pytest.approx(0.2, abs=0.01)
        assert result.metadata["signal"] == "BEARISH"

    def test_no_ticker_returns_neutral(self):
        engine = MockOBIEngine(obi=0.5)
        provider = OBISignalProvider(obi_engine=engine)
        result = provider.score({}, {})
        assert result.confidence == 0.5

    def test_confidence_bounded(self):
        engine = MockOBIEngine(obi=1.0)
        provider = OBISignalProvider(obi_engine=engine)
        result = provider.score({"ticker": "TEST"}, {})
        assert 0.0 <= result.confidence <= 1.0


# ── Whale Signal Provider ──────────────────────────────────────────────

class TestWhaleSignalProvider:
    def test_no_data_returns_neutral(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = WhaleSignalProvider(whale_data_dir=tmpdir)
            result = provider.score({"title": "Test market"}, {})
            assert result.confidence == 0.5
            assert result.name == "whale_smi"

    def test_matched_signal_returns_smi(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            whale_dir = Path(tmpdir)
            latest = {
                "top_signals": [{
                    "market_slug": "bitcoin-above-100k",
                    "title": "Will Bitcoin exceed 100000",
                    "smart_money_idx": 75.0,
                    "whale_count": 5,
                    "conviction_score": 80.0,
                    "direction": "BULLISH",
                    "signal_strength": "STRONG",
                }],
            }
            (whale_dir / "whale_latest.json").write_text(json.dumps(latest))
            provider = WhaleSignalProvider(whale_data_dir=tmpdir)
            result = provider.score(
                {"title": "Will Bitcoin exceed 100000", "slug": "bitcoin-above-100k"},
                {},
            )
            assert result.metadata["matched"] is True
            # SMI 75 -> confidence 0.75
            assert result.confidence == pytest.approx(0.75, abs=0.01)

    def test_unmatched_market_returns_neutral(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            whale_dir = Path(tmpdir)
            latest = {
                "top_signals": [{
                    "market_slug": "bitcoin-above-100k",
                    "title": "Will Bitcoin exceed 100000",
                    "smart_money_idx": 80.0,
                    "whale_count": 4,
                    "conviction_score": 70.0,
                    "direction": "BULLISH",
                    "signal_strength": "STRONG",
                }],
            }
            (whale_dir / "whale_latest.json").write_text(json.dumps(latest))
            provider = WhaleSignalProvider(whale_data_dir=tmpdir)
            result = provider.score(
                {"title": "Weather in Miami tomorrow", "slug": "miami-weather"},
                {},
            )
            assert result.metadata["matched"] is False
            assert result.confidence == 0.5

    def test_slug_match_takes_priority(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            whale_dir = Path(tmpdir)
            latest = {
                "top_signals": [{
                    "market_slug": "eth-above-5000",
                    "title": "Ethereum hits 5000",
                    "smart_money_idx": 30.0,
                    "whale_count": 3,
                    "conviction_score": 55.0,
                    "direction": "BEARISH",
                    "signal_strength": "MODERATE",
                }],
            }
            (whale_dir / "whale_latest.json").write_text(json.dumps(latest))
            provider = WhaleSignalProvider(whale_data_dir=tmpdir)
            result = provider.score(
                {"title": "ETH 5000", "slug": "eth-above-5000"},
                {},
            )
            assert result.metadata["matched"] is True
            # SMI 30 -> confidence 0.30
            assert result.confidence == pytest.approx(0.30, abs=0.01)
