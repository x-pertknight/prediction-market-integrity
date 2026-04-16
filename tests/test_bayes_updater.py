"""tests/test_bayes_updater.py — Unit tests for Bayesian sequential updater."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from alpha.bayes_updater import (
    bayes_update,
    bayesian_update_prob,
    _news_evidence,
    _fred_evidence,
    _noaa_evidence,
)


class TestBayesUpdate:
    def test_neutral_evidence_no_change(self):
        # P(E|YES) == P(E|NO) → no update
        assert bayes_update(0.5, 0.5, 0.5) == pytest.approx(0.5)

    def test_strong_positive_evidence(self):
        # P(E|YES) >> P(E|NO) → posterior increases
        posterior = bayes_update(0.3, 0.9, 0.1)
        assert posterior > 0.3
        assert posterior == pytest.approx(0.794, abs=0.01)

    def test_strong_negative_evidence(self):
        # P(E|YES) << P(E|NO) → posterior decreases
        posterior = bayes_update(0.7, 0.1, 0.9)
        assert posterior < 0.7

    def test_bearish_evidence_from_article(self):
        # Article spec: negative headlines → (0.30, 0.70)
        posterior = bayes_update(0.5, 0.30, 0.70)
        assert posterior < 0.5
        assert posterior == pytest.approx(0.30, abs=0.01)

    def test_bullish_evidence_from_article(self):
        posterior = bayes_update(0.5, 0.70, 0.30)
        assert posterior > 0.5
        assert posterior == pytest.approx(0.70, abs=0.01)

    def test_hawkish_fred_from_article(self):
        # Article spec: hawkish → (0.15, 0.35)
        posterior = bayes_update(0.5, 0.15, 0.35)
        assert posterior < 0.5
        assert posterior == pytest.approx(0.30, abs=0.01)

    def test_prior_zero_unchanged(self):
        assert bayes_update(0.0, 0.9, 0.1) == 0.0

    def test_prior_one_unchanged(self):
        assert bayes_update(1.0, 0.1, 0.9) == 1.0

    def test_chained_updates_monotonic(self):
        # Two bearish updates should move further down
        p1 = bayes_update(0.5, 0.30, 0.70)
        p2 = bayes_update(p1, 0.15, 0.35)
        assert p2 < p1 < 0.5

    def test_posterior_bounded(self):
        # Even extreme evidence can't push past 0/1
        posterior = bayes_update(0.01, 0.001, 0.999)
        assert 0.0 <= posterior <= 1.0


class TestBayesianUpdateProb:
    def test_no_evidence_returns_prior(self):
        # No relevant data → prior unchanged
        posterior, updates = bayesian_update_prob(
            0.5, "Exotics", {}, {}
        )
        assert posterior == pytest.approx(0.5, abs=0.01)
        assert updates == []

    def test_posterior_bounded(self):
        posterior, _ = bayesian_update_prob(0.99, "Crypto", {}, {})
        assert 0.01 <= posterior <= 0.99

    def test_weather_market_with_no_noaa(self):
        # Weather market but no city match → no NOAA update
        posterior, updates = bayesian_update_prob(
            0.5, "Climate and Weather",
            {"title": "Will it rain in Timbuktu?"},
            {},
        )
        # No NOAA evidence for Timbuktu
        assert "noaa(deviation)" not in updates

    def test_economics_with_hawkish_rates(self):
        # High fed rate → hawkish evidence on economics markets
        posterior, updates = bayesian_update_prob(
            0.5, "Economics", {},
            {"fed_funds_rate": 5.5, "yield_10y": 5.0},
        )
        assert posterior < 0.5
        assert "fred(hawkish)" in updates

    def test_economics_with_dovish_rates(self):
        posterior, updates = bayesian_update_prob(
            0.5, "Economics", {},
            {"fed_funds_rate": 1.5, "yield_10y": 2.5},
        )
        assert posterior > 0.5
        assert "fred(dovish)" in updates

    def test_crypto_ignores_fred(self):
        # Crypto markets should not be affected by FRED
        posterior, updates = bayesian_update_prob(
            0.5, "Crypto", {},
            {"fed_funds_rate": 5.5, "yield_10y": 5.0},
        )
        assert "fred(hawkish)" not in updates


class TestNewsEvidence:
    def test_no_file_returns_none(self):
        assert _news_evidence("Crypto", {}) is None

    def test_non_matching_category_returns_none(self):
        assert _news_evidence("Transportation", {}) is None


class TestFredEvidence:
    def test_non_rate_sensitive_returns_none(self):
        assert _fred_evidence("Sports", {"fed_funds_rate": 5.0}) is None

    def test_hawkish(self):
        result = _fred_evidence("Economics", {"fed_funds_rate": 5.0, "yield_10y": 5.0})
        assert result is not None
        assert result[0] < result[1]  # bearish: p_yes < p_no

    def test_dovish(self):
        result = _fred_evidence("Economics", {"fed_funds_rate": 1.0, "yield_10y": 2.5})
        assert result is not None
        assert result[0] > result[1]  # bullish: p_yes > p_no

    def test_neutral(self):
        result = _fred_evidence("Economics", {"fed_funds_rate": 3.0, "yield_10y": 4.0})
        assert result is None


class TestNoaaEvidence:
    def test_non_weather_returns_none(self):
        assert _noaa_evidence("Crypto", {"title": "Bitcoin"}) is None

    def test_no_city_match_returns_none(self):
        assert _noaa_evidence(
            "Climate and Weather",
            {"title": "Will it be hot in London?"},
        ) is None
