"""tests/test_uma_whale_edge.py — the preregistered trading rule.

Synthetic data, zero network. These pin the rule as preregistered: if a constant
or a sign convention drifts, the backtest silently stops testing what was written
down, which is worse than it failing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from alpha.uma_dispute import PRICE_NO, PRICE_VOID, PRICE_YES, UMA_IGNORE_PRICE, VoterStats
from alpha.uma_whale_edge import (
    BASE_YES, ENTRY_THRESHOLD, K_TILT, NOTIONAL,
    concentration, dispute_tilt, forecast_from_tilt, simulate_trade, split_halves,
    terminal_outcomes, vwap, whale_voters,
)

NO_FEE = lambda p: 0.0


class TestTerminalOutcomes:
    def _req(self, rows):
        return pd.DataFrame(rows, columns=["question_id", "settled_price"])

    def test_reset_is_not_an_outcome(self):
        """A RESET sends the question back to the oracle; it resolves nothing."""
        out = terminal_outcomes(self._req([{"question_id": "q1", "settled_price": str(UMA_IGNORE_PRICE)}]))
        assert out == {}

    def test_decided_round_wins_over_reset_rounds(self):
        out = terminal_outcomes(self._req([
            {"question_id": "q1", "settled_price": str(UMA_IGNORE_PRICE)},
            {"question_id": "q1", "settled_price": str(PRICE_YES)},
        ]))
        assert out == {"q1": 1.0}

    def test_payout_mapping(self):
        out = terminal_outcomes(self._req([
            {"question_id": "y", "settled_price": str(PRICE_YES)},
            {"question_id": "n", "settled_price": str(PRICE_NO)},
            {"question_id": "v", "settled_price": str(PRICE_VOID)},
        ]))
        assert out == {"y": 1.0, "n": 0.0, "v": 0.5}

    def test_pending_dispute_ignored(self):
        out = terminal_outcomes(self._req([{"question_id": "q", "settled_price": None}]))
        assert out == {}


class TestWhaleVoters:
    def _s(self, total, linked):
        return VoterStats("0xa", total_votes=total, votes_with_position=linked,
                          voted_own_book=linked, avg_stake=1.0)

    def test_requires_three_linked_votes(self):
        stats = {"a": self._s(100, 2), "b": self._s(100, 3)}
        assert whale_voters(stats) == {"b"}

    def test_single_linked_vote_excluded(self):
        """The whole point of the floor: conflict_rate 1.00 off one vote is noise."""
        assert whale_voters({"a": self._s(500, 1)}) == set()


class TestDisputeTilt:
    def test_sign_only_not_position_size(self):
        """A whale with a 1000x bigger book must not dominate: only sign enters."""
        equal_stakes = {"a": 10.0, "b": 10.0}
        assert dispute_tilt({"a": 1.0, "b": -1_000_000.0}, equal_stakes) == pytest.approx(0.0)

    def test_stake_weights_the_sign(self):
        tilt = dispute_tilt({"a": 5.0, "b": -5.0}, {"a": 30.0, "b": 10.0})
        assert tilt == pytest.approx((30 - 10) / 40)

    def test_complete_set_contributes_nothing(self):
        """net == 0 is hedged inventory, not a directional book."""
        assert dispute_tilt({"a": 0.0, "b": 3.0}, {"a": 99.0, "b": 1.0}) == pytest.approx(1.0)

    def test_none_when_no_signal(self):
        assert dispute_tilt({"a": 0.0}, {"a": 5.0}) is None
        assert dispute_tilt({}, {}) is None

    def test_zero_stake_voter_ignored(self):
        assert dispute_tilt({"a": 1.0, "b": -1.0}, {"a": 5.0, "b": 0.0}) == pytest.approx(1.0)


class TestForecast:
    def test_neutral_tilt_gives_base_rate(self):
        assert forecast_from_tilt(0.0) == pytest.approx(BASE_YES)

    def test_bounds(self):
        assert 0.02 <= forecast_from_tilt(1.0) <= 0.98
        assert forecast_from_tilt(1.0) == pytest.approx(BASE_YES + K_TILT)
        assert forecast_from_tilt(-1.0) == pytest.approx(BASE_YES - K_TILT)

    def test_preregistered_constants_unchanged(self):
        """Guards the preregistration: these were fixed before the backtest ran."""
        assert (BASE_YES, K_TILT, ENTRY_THRESHOLD, NOTIONAL) == (0.4606, 0.25, 0.10, 100.0)


class TestSimulateTrade:
    def _t(self, forecast, price, payout, fee=NO_FEE):
        return simulate_trade("0xd", pd.Timestamp("2026-01-01"), forecast, price, payout, fee)

    def test_no_trade_below_threshold(self):
        assert self._t(0.55, 0.50, 1.0) is None

    def test_buys_yes_when_forecast_above_price_and_wins(self):
        t = self._t(0.71, 0.50, 1.0)
        assert t.side == "YES" and t.entry_price == 0.50
        assert t.shares == pytest.approx(200.0)
        assert t.net_pnl == pytest.approx(100.0)      # 200 shares * $1 - $100

    def test_buys_no_when_forecast_below_price(self):
        t = self._t(0.20, 0.50, 0.0)
        assert t.side == "NO" and t.entry_price == pytest.approx(0.50)
        assert t.payout_per_share == pytest.approx(1.0)
        assert t.net_pnl == pytest.approx(100.0)

    def test_losing_trade(self):
        t = self._t(0.71, 0.50, 0.0)   # bought YES, resolved NO
        assert t.net_pnl == pytest.approx(-100.0)

    def test_void_pays_half_to_both_sides(self):
        assert self._t(0.71, 0.50, 0.5).net_pnl == pytest.approx(0.0)
        assert self._t(0.20, 0.50, 0.5).net_pnl == pytest.approx(0.0)

    def test_fee_charged_on_entry_only_and_reduces_pnl(self):
        free = self._t(0.71, 0.50, 1.0)
        fee = self._t(0.71, 0.50, 1.0, fee=lambda p: 0.05 * p * (1 - p))
        assert fee.fee == pytest.approx(0.05 * 0.25 * 200.0)
        assert fee.net_pnl == pytest.approx(free.net_pnl - fee.fee)

    def test_degenerate_prices_rejected(self):
        assert self._t(0.9, 0.0, 1.0) is None
        assert self._t(0.1, 1.0, 0.0) is None

    def test_net_edge_is_per_dollar_of_notional(self):
        assert self._t(0.71, 0.50, 1.0).net_edge == pytest.approx(1.0)


class TestVwap:
    def test_size_weighted(self):
        assert vwap(np.array([0.4, 0.6]), np.array([3.0, 1.0])) == pytest.approx(0.45)

    def test_zero_size_raises(self):
        with pytest.raises(ValueError):
            vwap(np.array([0.5]), np.array([0.0]))


class TestConcentration:
    def test_top3_share(self):
        assert concentration(np.array([1.0, 1.0, 10.0, 10.0, 10.0])) == pytest.approx(30 / 32)

    def test_infinite_when_total_not_positive(self):
        """'top 3 = 200% of PnL' is meaningless when the total is negative."""
        assert np.isinf(concentration(np.array([5.0, -20.0, 1.0, 1.0])))


class TestSplitHalves:
    def test_chronological_and_balanced(self):
        trades = [simulate_trade("d", pd.Timestamp(f"2026-01-{d:02d}"), 0.9, 0.5, 1.0, NO_FEE)
                  for d in (5, 1, 3, 2)]
        a, b = split_halves(trades)
        assert [t.dispute_time.day for t in a] == [1, 2]
        assert [t.dispute_time.day for t in b] == [3, 5]
