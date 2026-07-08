"""tests/test_uma_dispute.py — Unit tests for src/alpha/uma_dispute.py

All synthetic data, zero network access.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

from alpha.uma_dispute import (
    VoterStats, DisputeForecast, UmaDisputeForecaster, compute_voter_stats,
)


# ── VoterStats.conflict_rate ─────────────────────────────────────────────────

class TestVoterStatsConflictRate:
    def test_conflict_rate_basic(self):
        vs = VoterStats(voter="0xabc", total_votes=10, votes_with_position=5,
                         voted_own_book=4, avg_stake=100.0)
        assert vs.conflict_rate == pytest.approx(0.8)

    def test_conflict_rate_zero_when_never_had_position(self):
        """No linked position ever -> 0.0 (no signal), not a ZeroDivisionError/NaN."""
        vs = VoterStats(voter="0xabc", total_votes=10, votes_with_position=0,
                         voted_own_book=0, avg_stake=100.0)
        assert vs.conflict_rate == 0.0

    def test_conflict_rate_full_alignment(self):
        vs = VoterStats(voter="0xabc", total_votes=3, votes_with_position=3,
                         voted_own_book=3, avg_stake=50.0)
        assert vs.conflict_rate == pytest.approx(1.0)


# ── compute_voter_stats ──────────────────────────────────────────────────────

class TestComputeVoterStats:
    def test_aggregates_per_voter(self):
        votes = pd.DataFrame([
            {"voter": "0xA", "has_position": True,  "voted_with_own_position": True,  "stake": 100.0},
            {"voter": "0xA", "has_position": True,  "voted_with_own_position": False, "stake": 200.0},
            {"voter": "0xA", "has_position": False, "voted_with_own_position": False, "stake": 50.0},
            {"voter": "0xB", "has_position": True,  "voted_with_own_position": True,  "stake": 300.0},
        ])
        stats = compute_voter_stats(votes)
        assert stats["0xA"].total_votes == 3
        assert stats["0xA"].votes_with_position == 2
        assert stats["0xA"].voted_own_book == 1
        assert stats["0xA"].conflict_rate == pytest.approx(0.5)
        assert stats["0xA"].avg_stake == pytest.approx((100 + 200 + 50) / 3)
        assert stats["0xB"].conflict_rate == pytest.approx(1.0)

    def test_empty_input(self):
        votes = pd.DataFrame(columns=["voter", "has_position", "voted_with_own_position", "stake"])
        stats = compute_voter_stats(votes)
        assert stats == {}


# ── UmaDisputeForecaster.forecast ────────────────────────────────────────────

class TestForecastStakeWeighting:
    def test_stake_weighted_conflict_rate_drives_p_yes(self):
        voter_stats = {
            "0xA": VoterStats("0xA", total_votes=5, votes_with_position=5, voted_own_book=5, avg_stake=100.0),  # conflict_rate 1.0
            "0xB": VoterStats("0xB", total_votes=5, votes_with_position=5, voted_own_book=1, avg_stake=100.0),  # conflict_rate 0.2
        }
        forecaster = UmaDisputeForecaster(voter_stats=voter_stats, measured_base_rate=0.85)
        dispute_voters = pd.DataFrame([
            {"voter": "0xA", "side": "YES", "stake": 1000.0, "link_confidence": "high"},
            {"voter": "0xB", "side": "NO",  "stake": 1000.0, "link_confidence": "high"},
        ])
        result = forecaster.forecast(dispute_voters)
        assert result.p_yes == pytest.approx(1000.0 / 1200.0)
        assert result.confidence == "measured_voters"
        assert len(result.drivers) == 2

    def test_excludes_voters_below_min_historical_votes(self):
        voter_stats = {
            "0xA": VoterStats("0xA", total_votes=2, votes_with_position=2, voted_own_book=2, avg_stake=100.0),
        }
        forecaster = UmaDisputeForecaster(voter_stats=voter_stats, measured_base_rate=0.7)
        dispute_voters = pd.DataFrame([
            {"voter": "0xA", "side": "YES", "stake": 1000.0, "link_confidence": "high"},
        ])
        result = forecaster.forecast(dispute_voters)
        assert result.confidence == "base_rate_fallback"
        assert result.p_yes == pytest.approx(0.7)
        assert result.drivers == []

    def test_excludes_low_confidence_links(self):
        voter_stats = {
            "0xA": VoterStats("0xA", total_votes=5, votes_with_position=5, voted_own_book=5, avg_stake=100.0),
        }
        forecaster = UmaDisputeForecaster(voter_stats=voter_stats, measured_base_rate=0.6)
        dispute_voters = pd.DataFrame([
            {"voter": "0xA", "side": "YES", "stake": 1000.0, "link_confidence": "low"},
        ])
        result = forecaster.forecast(dispute_voters)
        assert result.confidence == "base_rate_fallback"
        assert result.p_yes == pytest.approx(0.6)

    def test_fallback_uses_measured_base_rate_not_hardcoded(self):
        """The fallback must come from the backfill measurement, never an invented constant."""
        empty = pd.DataFrame(columns=["voter", "side", "stake", "link_confidence"])
        forecaster_a = UmaDisputeForecaster(voter_stats={}, measured_base_rate=0.42)
        forecaster_b = UmaDisputeForecaster(voter_stats={}, measured_base_rate=0.91)
        assert forecaster_a.forecast(empty).p_yes == pytest.approx(0.42)
        assert forecaster_b.forecast(empty).p_yes == pytest.approx(0.91)

    def test_unknown_voter_not_in_stats_is_ignored(self):
        forecaster = UmaDisputeForecaster(voter_stats={}, measured_base_rate=0.5)
        dispute_voters = pd.DataFrame([
            {"voter": "0xGHOST", "side": "YES", "stake": 1000.0, "link_confidence": "high"},
        ])
        result = forecaster.forecast(dispute_voters)
        assert result.confidence == "base_rate_fallback"
        assert result.p_yes == pytest.approx(0.5)

    def test_zero_conflict_rate_voters_contribute_no_weight(self):
        """A voter who never votes their own book (conflict_rate 0) shouldn't move p_yes off the fallback."""
        voter_stats = {
            "0xA": VoterStats("0xA", total_votes=5, votes_with_position=5, voted_own_book=0, avg_stake=100.0),
        }
        forecaster = UmaDisputeForecaster(voter_stats=voter_stats, measured_base_rate=0.33)
        dispute_voters = pd.DataFrame([
            {"voter": "0xA", "side": "YES", "stake": 1000.0, "link_confidence": "high"},
        ])
        result = forecaster.forecast(dispute_voters)
        assert result.confidence == "base_rate_fallback"
        assert result.p_yes == pytest.approx(0.33)
