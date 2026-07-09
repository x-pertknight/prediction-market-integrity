"""tests/test_uma_dispute.py — Unit tests for src/alpha/uma_dispute.py

All synthetic data, zero network access.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

from alpha.uma_dispute import (
    PRICE_NO, PRICE_VOID, PRICE_YES, UMA_IGNORE_PRICE,
    DisputeForecast, DisputeOutcome, UmaDisputeForecaster, VoterStats,
    classify_outcome, compute_voter_stats, expected_payouts, link_votes_to_disputes,
    question_id, stamped_ancillary_hash, summarize_disputes, vote_side,
    annotate_votes_with_positions,
)


# ── Outcome classification: sentinel is a RESET, not a void ──────────────────
# Verified against the deployed UmaCtfAdapter (_resolve / _constructPayouts).

class TestClassifyOutcome:
    def test_sentinel_is_reset_not_void(self):
        """The ignore price re-requests the question; it does not cancel the market."""
        assert classify_outcome(UMA_IGNORE_PRICE) is DisputeOutcome.RESET

    def test_half_ether_is_the_only_true_void(self):
        assert classify_outcome(PRICE_VOID) is DisputeOutcome.VOID_5050

    def test_yes_and_no(self):
        assert classify_outcome(PRICE_YES) is DisputeOutcome.YES
        assert classify_outcome(PRICE_NO) is DisputeOutcome.NO

    def test_accepts_string_price_from_csv(self):
        assert classify_outcome(str(UMA_IGNORE_PRICE)) is DisputeOutcome.RESET

    def test_rejects_price_the_adapter_would_revert_on(self):
        with pytest.raises(ValueError, match="InvalidOOPrice"):
            classify_outcome(1)


class TestExpectedPayouts:
    def test_mirrors_construct_payouts(self):
        """Adapter reports payouts as [YES, NO]."""
        assert expected_payouts(PRICE_NO) == (0, 1)
        assert expected_payouts(PRICE_YES) == (1, 0)
        assert expected_payouts(PRICE_VOID) == (1, 1)

    def test_sentinel_has_no_payout_array(self):
        """The adapter resets rather than constructing payouts, so this must not
        silently return a 50/50 -- that would misprice every reset dispute."""
        with pytest.raises(ValueError, match="resets the question"):
            expected_payouts(UMA_IGNORE_PRICE)

    def test_invalid_price_rejected(self):
        with pytest.raises(ValueError, match="InvalidOOPrice"):
            expected_payouts(123)


class TestVoteSide:
    def test_ignore_vote_is_not_yes_or_no(self):
        assert vote_side(UMA_IGNORE_PRICE) == "IGNORE"

    def test_yes_no_void(self):
        assert vote_side(PRICE_YES) == "YES"
        assert vote_side(PRICE_NO) == "NO"
        assert vote_side(PRICE_VOID) == "VOID"


class TestSummarizeDisputes:
    def _frame(self, rows):
        return pd.DataFrame(rows, columns=["settled_price", "proposedPrice"])

    def test_reset_excluded_from_base_rate(self):
        """A reset is not a ruling: it belongs in neither numerator nor denominator."""
        b = summarize_disputes(self._frame([
            {"settled_price": UMA_IGNORE_PRICE, "proposedPrice": PRICE_YES},
            {"settled_price": PRICE_YES, "proposedPrice": PRICE_YES},   # confirmed
            {"settled_price": PRICE_NO, "proposedPrice": PRICE_YES},    # overturned
        ]))
        assert (b.n_disputes, b.n_reset, b.n_decided) == (3, 1, 2)
        assert b.n_confirmed == 1
        assert b.base_rate == pytest.approx(0.5)
        assert b.reset_rate == pytest.approx(1 / 3)

    def test_void_counts_as_decided(self):
        b = summarize_disputes(self._frame([
            {"settled_price": PRICE_VOID, "proposedPrice": PRICE_VOID},
        ]))
        assert b.n_void_5050 == 1 and b.n_decided == 1 and b.n_confirmed == 1

    def test_base_rate_none_when_all_reset(self):
        b = summarize_disputes(self._frame([
            {"settled_price": UMA_IGNORE_PRICE, "proposedPrice": PRICE_NO},
        ]))
        assert b.n_decided == 0 and b.base_rate is None

    def test_pending_disputes_ignored(self):
        b = summarize_disputes(self._frame([
            {"settled_price": None, "proposedPrice": PRICE_YES},
            {"settled_price": PRICE_YES, "proposedPrice": PRICE_YES},
        ]))
        assert b.n_disputes == 1 and b.base_rate == pytest.approx(1.0)


# ── The join ─────────────────────────────────────────────────────────────────

class TestStampedAncillaryHash:
    def test_matches_onchain_value(self):
        """Regression anchor: dispute 0x3fac9158..., whose OracleChildTunnel log
        carries ancillaryDataHash:177c2bb4...  Computed offline it must agree."""
        anc = b"q: title: T, res_data: p1: 0, p2: 1."
        requester = "0x2f5e3684cb1f318ec51b00edba38d79ac2c0aa9d"
        h = stamped_ancillary_hash(anc, requester)
        from eth_utils import keccak
        assert h == keccak(anc + b",ooRequester:" + requester[2:].encode()).hex()

    def test_requester_case_insensitive(self):
        anc = b"x"
        assert stamped_ancillary_hash(anc, "0xAABB" + "cc" * 18) == \
               stamped_ancillary_hash(anc, "0xaabb" + "cc" * 18)

    def test_rejects_malformed_requester(self):
        with pytest.raises(ValueError, match="20-byte address"):
            stamped_ancillary_hash(b"x", "0xdeadbeef")

    def test_question_id_is_keccak_of_raw_ancillary(self):
        from eth_utils import keccak
        assert question_id(b"abc") == keccak(b"abc").hex()


class TestLinkVotesToDisputes:
    def test_reset_rounds_are_not_conflated(self):
        """Same ancillary data disputed twice -> same stamped_hash, different DVM
        request time. Joining on the hash alone would fan each round's votes onto
        both rounds; the two-key join must keep them apart."""
        disputes = pd.DataFrame([
            {"stamped_hash": "aa", "dvm_request_time": 100, "dispute_tx_hash": "0xr1"},
            {"stamped_hash": "aa", "dvm_request_time": 200, "dispute_tx_hash": "0xr2"},
        ])
        votes = pd.DataFrame([
            {"stamped_hash": "aa", "dvm_request_time": 100, "voter": "0xA", "vote_price": PRICE_YES, "stake": 1.0},
            {"stamped_hash": "aa", "dvm_request_time": 200, "voter": "0xB", "vote_price": PRICE_NO, "stake": 2.0},
        ])
        out = link_votes_to_disputes(disputes, votes)
        assert len(out) == 2
        assert set(out[out["dispute_tx_hash"] == "0xr1"]["voter"]) == {"0xA"}
        assert set(out[out["dispute_tx_hash"] == "0xr2"]["voter"]) == {"0xB"}

    def test_side_derived_from_vote_price(self):
        disputes = pd.DataFrame([{"stamped_hash": "aa", "dvm_request_time": 1}])
        votes = pd.DataFrame([
            {"stamped_hash": "aa", "dvm_request_time": 1, "voter": "0xA", "vote_price": UMA_IGNORE_PRICE, "stake": 1.0},
        ])
        assert link_votes_to_disputes(disputes, votes)["side"].tolist() == ["IGNORE"]

    def test_unmatched_votes_dropped(self):
        disputes = pd.DataFrame([{"stamped_hash": "aa", "dvm_request_time": 1}])
        votes = pd.DataFrame([
            {"stamped_hash": "zz", "dvm_request_time": 9, "voter": "0xA", "vote_price": PRICE_YES, "stake": 1.0},
        ])
        assert link_votes_to_disputes(disputes, votes).empty


class TestAnnotateVotesWithPositions:
    def _votes(self, side, voter="0xa"):
        return pd.DataFrame([{"dispute_tx_hash": "0xd", "voter": voter, "side": side, "stake": 1.0}])

    def _pos(self, yes, no, voter="0xa"):
        return pd.DataFrame([{"dispute_tx_hash": "0xd", "voter": voter, "yes_shares": yes, "no_shares": no}])

    def test_long_yes_voting_yes_is_own_book(self):
        out = annotate_votes_with_positions(self._votes("YES"), self._pos(100, 0))
        assert bool(out["has_position"][0]) and bool(out["voted_with_own_position"][0])

    def test_long_yes_voting_no_is_against_own_book(self):
        out = annotate_votes_with_positions(self._votes("NO"), self._pos(100, 0))
        assert bool(out["has_position"][0]) and not bool(out["voted_with_own_position"][0])

    def test_net_exposure_decides_direction(self):
        """Holding both sides, the NET is what pays; 40 YES vs 100 NO is a NO book."""
        out = annotate_votes_with_positions(self._votes("NO"), self._pos(40, 100))
        assert bool(out["voted_with_own_position"][0])

    def test_complete_set_is_not_a_position(self):
        """Equal YES and NO redeems for $1 either way -- hedged, no conflict."""
        out = annotate_votes_with_positions(self._votes("YES"), self._pos(100, 100))
        assert not bool(out["has_position"][0])
        assert not bool(out["voted_with_own_position"][0])

    def test_ignore_vote_never_counts_as_own_book(self):
        out = annotate_votes_with_positions(self._votes("IGNORE"), self._pos(100, 0))
        assert bool(out["has_position"][0]) and not bool(out["voted_with_own_position"][0])

    def test_voter_without_position_gets_zeros(self):
        out = annotate_votes_with_positions(self._votes("YES", voter="0xzz"), self._pos(100, 0, voter="0xa"))
        assert not bool(out["has_position"][0])

    def test_voter_case_insensitive_join(self):
        votes = self._votes("YES", voter="0xAaBb")
        out = annotate_votes_with_positions(votes, self._pos(50, 0, voter="0xaabb"))
        assert bool(out["voted_with_own_position"][0])

    def test_empty_positions_yields_no_conflict(self):
        out = annotate_votes_with_positions(self._votes("YES"), pd.DataFrame())
        assert not bool(out["has_position"][0])


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
