"""tests/test_negrisk_paper.py — Unit tests for src/alpha/negrisk_paper.py

All offline: simulate_entry and resolve_position are pure.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from alpha.negrisk_paper import simulate_entry, resolve_position, log_trade, TRADE_COLUMNS


def _leg(token: str, *, closed: bool = False, yes_paid: float | None = None) -> dict:
    leg = {
        "clobTokenIds": json.dumps([token, token + "_no"]),
        "feesEnabled": False,
        "closed": closed,
    }
    if yes_paid is not None:
        leg["outcomePrices"] = json.dumps([str(yes_paid), str(1 - yes_paid)])
    return leg


def _book(bid: float, ask: float, depth: float = 500.0) -> dict:
    return {"best_bid": bid, "best_ask": ask, "spread": ask - bid,
            "bid_depth": depth, "ask_depth": depth}


def _row(event_id: str = "42") -> dict:
    return {"event_id": event_id, "title": "Tournament Winner",
            "exhaustive": True, "exhaustive_reason": "whitelist",
            "margin_long_net": 0.03, "margin_short_net": -0.01}


NO_FEE = lambda price, leg: 0.0


class TestSimulateEntry:
    def test_long_set_locked_pnl(self):
        """Buy 3 asks summing 0.95 → locks 5c/set before fees."""
        legs = [_leg("a"), _leg("b"), _leg("c")]
        books = {"a": _book(0.39, 0.40), "b": _book(0.29, 0.30), "c": _book(0.24, 0.25)}
        pos = simulate_entry(_row(), legs, books, "long", NO_FEE)
        assert pos["cost_per_set"] == pytest.approx(0.95, abs=1e-9)
        assert pos["payout_per_set"] == 1.0
        assert pos["locked_pnl_per_set"] == pytest.approx(0.05, abs=1e-9)

    def test_short_set_locked_pnl(self):
        """Bids sum 1.06 → NO set costs 3−1.06=1.94, pays 2 → locks 6c."""
        legs = [_leg("a"), _leg("b"), _leg("c")]
        books = {"a": _book(0.45, 0.46), "b": _book(0.35, 0.36), "c": _book(0.26, 0.27)}
        pos = simulate_entry(_row(), legs, books, "short", NO_FEE)
        assert pos["cost_per_set"] == pytest.approx(3 - 1.06, abs=1e-9)
        assert pos["payout_per_set"] == 2.0
        assert pos["locked_pnl_per_set"] == pytest.approx(0.06, abs=1e-9)

    def test_sets_capped_by_depth(self):
        legs = [_leg("a"), _leg("b"), _leg("c")]
        books = {"a": _book(0.39, 0.40, depth=37), "b": _book(0.29, 0.30),
                 "c": _book(0.24, 0.25)}
        pos = simulate_entry(_row(), legs, books, "long", NO_FEE)
        assert pos["sets"] == 37

    def test_fees_included_in_cost(self):
        legs = [_leg("a"), _leg("b"), _leg("c")]
        books = {"a": _book(0.39, 0.40), "b": _book(0.29, 0.30), "c": _book(0.24, 0.25)}
        fee = lambda price, leg: 0.01
        pos = simulate_entry(_row(), legs, books, "long", fee)
        assert pos["cost_per_set"] == pytest.approx(0.95 + 0.03, abs=1e-9)

    def test_missing_book_returns_none(self):
        legs = [_leg("a"), _leg("b"), _leg("c")]
        books = {"a": _book(0.39, 0.40), "b": _book(0.29, 0.30)}  # c missing
        assert simulate_entry(_row(), legs, books, "long", NO_FEE) is None

    def test_empty_book_returns_none(self):
        """A leg with no real quotes (bid 0 / ask 1) can't be filled."""
        legs = [_leg("a"), _leg("b"), _leg("c")]
        books = {"a": _book(0.39, 0.40), "b": _book(0.29, 0.30),
                 "c": _book(0.0, 1.0)}
        assert simulate_entry(_row(), legs, books, "long", NO_FEE) is None


class TestResolvePosition:
    def _pos(self, side: str, cost: float) -> dict:
        return {"event_id": "42", "title": "T", "side": side, "n_legs": 3,
                "sets": 10, "tokens": ["a", "b", "c"], "cost_per_set": cost,
                "payout_per_set": 1.0 if side == "long" else 2.0}

    def test_long_set_wins_exactly_one(self):
        """Exhaustive set: one owned leg pays $1 → PnL = 1 − cost."""
        event = {"markets": [_leg("a", closed=True, yes_paid=0),
                             _leg("b", closed=True, yes_paid=1),
                             _leg("c", closed=True, yes_paid=0)]}
        res = resolve_position(self._pos("long", 0.95), event)
        assert res["payout_per_set"] == 1.0
        assert res["total_pnl"] == pytest.approx(0.05 * 10, abs=1e-6)

    def test_exhaustiveness_failure_realizes_loss(self):
        """A bad whitelist entry: NO owned leg pays → full cost lost.
        This is the failure mode the paper run exists to measure."""
        event = {"markets": [_leg("a", closed=True, yes_paid=0),
                             _leg("b", closed=True, yes_paid=0),
                             _leg("c", closed=True, yes_paid=0)]}
        res = resolve_position(self._pos("long", 0.95), event)
        assert res["payout_per_set"] == 0.0
        assert res["total_pnl"] == pytest.approx(-0.95 * 10, abs=1e-6)

    def test_short_set_payout(self):
        """Short 3-leg set with one winner: NO pays on 2 legs → payout 2."""
        event = {"markets": [_leg("a", closed=True, yes_paid=1),
                             _leg("b", closed=True, yes_paid=0),
                             _leg("c", closed=True, yes_paid=0)]}
        res = resolve_position(self._pos("short", 1.94), event)
        assert res["payout_per_set"] == 2.0
        assert res["total_pnl"] == pytest.approx(0.06 * 10, abs=1e-6)

    def test_unresolved_leg_keeps_position_open(self):
        event = {"markets": [_leg("a", closed=True, yes_paid=0),
                             _leg("b", closed=False),
                             _leg("c", closed=True, yes_paid=0)]}
        assert resolve_position(self._pos("long", 0.95), event) is None


class TestPositionsPersistence:
    def test_atomic_roundtrip(self, tmp_path):
        from alpha.negrisk_paper import save_positions, load_positions
        p = tmp_path / "positions.json"
        save_positions({"42:long": {"sets": 10}}, p)
        assert load_positions(p) == {"42:long": {"sets": 10}}
        assert not p.with_suffix(".tmp").exists()

    def test_corrupt_file_moved_aside_not_overwritten(self, tmp_path):
        """A truncated state file must be preserved for recovery, and the
        loader must not silently pretend there were no open positions
        without leaving a trace."""
        from alpha.negrisk_paper import load_positions
        p = tmp_path / "positions.json"
        p.write_text('{"42:long": {"se')  # truncated mid-write
        assert load_positions(p) == {}
        assert not p.exists()  # moved aside, not left in place
        corrupt = list(tmp_path.glob("positions.corrupt-*.json"))
        assert len(corrupt) == 1
        assert corrupt[0].read_text().startswith('{"42:long"')

    def test_missing_file_returns_empty(self, tmp_path):
        from alpha.negrisk_paper import load_positions
        assert load_positions(tmp_path / "nope.json") == {}


class TestLedger:
    def test_log_trade_creates_csv(self, tmp_path):
        out = tmp_path / "trades.csv"
        log_trade({c: "x" for c in TRADE_COLUMNS}, out)
        log_trade({c: "y" for c in TRADE_COLUMNS}, out)
        lines = out.read_text().strip().splitlines()
        assert lines[0] == ",".join(TRADE_COLUMNS)
        assert len(lines) == 3
