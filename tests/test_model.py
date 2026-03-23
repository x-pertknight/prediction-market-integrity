"""tests/test_model.py — Unit tests for model loading and prediction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
import joblib

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
CLEAN_GENERAL = MODEL_DIR / "rf_clean_general.joblib"
CLEAN_WEATHER = MODEL_DIR / "rf_clean_weather.joblib"


class TestModelLoading:
    @pytest.mark.skipif(not CLEAN_GENERAL.exists(), reason="Model file not found")
    def test_load_general(self):
        """rf_clean_general.joblib loads without error."""
        model = joblib.load(CLEAN_GENERAL)
        assert hasattr(model, "predict_proba")

    @pytest.mark.skipif(not CLEAN_WEATHER.exists(), reason="Model file not found")
    def test_load_weather(self):
        """rf_clean_weather.joblib loads without error."""
        model = joblib.load(CLEAN_WEATHER)
        assert hasattr(model, "predict_proba")


class TestModelPrediction:
    @pytest.mark.skipif(not CLEAN_GENERAL.exists(), reason="Model file not found")
    def test_predict_shape(self):
        """Prediction output shape matches input."""
        model = joblib.load(CLEAN_GENERAL)
        n_features = model.n_features_in_
        X = np.zeros((5, n_features))
        proba = model.predict_proba(X)
        assert proba.shape == (5, 2)  # binary classifier: 2 classes

    @pytest.mark.skipif(not CLEAN_GENERAL.exists(), reason="Model file not found")
    def test_proba_range(self):
        """Probabilities should be in [0, 1]."""
        model = joblib.load(CLEAN_GENERAL)
        n_features = model.n_features_in_
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (10, n_features))
        proba = model.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    @pytest.mark.skipif(not CLEAN_GENERAL.exists(), reason="Model file not found")
    def test_proba_sums_to_one(self):
        """Each row of predict_proba should sum to ~1.0."""
        model = joblib.load(CLEAN_GENERAL)
        n_features = model.n_features_in_
        X = np.zeros((3, n_features))
        proba = model.predict_proba(X)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0, decimal=6)

    @pytest.mark.skipif(not CLEAN_GENERAL.exists(), reason="Model file not found")
    def test_feature_count(self):
        """Model should expect ≥ 20 features (CLEAN_FEATURES has 27)."""
        model = joblib.load(CLEAN_GENERAL)
        assert model.n_features_in_ >= 20

    @pytest.mark.skipif(not CLEAN_GENERAL.exists(), reason="Model file not found")
    def test_estimators_exist(self):
        """RF model should have estimators (trees)."""
        model = joblib.load(CLEAN_GENERAL)
        assert hasattr(model, "estimators_")
        assert len(model.estimators_) > 0
