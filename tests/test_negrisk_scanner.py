"""tests/test_negrisk_scanner.py — Unit tests for src/alpha/negrisk_scanner.py

All tests are offline: analyze_event / live_legs / has_catchall are pure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from alpha.negrisk_scanner import (
    analyze_event, live_legs, has_catchall, append_rows, CSV_COLUMNS,
)


def _leg(title: str, bid: float, ask: float, *, closed: bool = False,
         fees: bool = False, rate: float = 0.03) -> dict:
    leg = {
        "groupItemTitle": title,
        "question": f"Will {title} win?",
        "active": True,
        "closed": closed,
        "acceptingOrders": not closed,
        "bestBid": bid,
        "bestAsk": ask,
        "feesEnabled": fees,
    }
    if fees:
        leg["feeSchedule"] = {"rate": rate, "exponent": 1,
                              "takerOnly": True, "rebateRate": 0.25}
    return leg


def _event(legs: list[dict], event_id: str = "42",
           title: str = "Tournament Winner") -> dict:
    return {"id": event_id, "title": title, "negRisk": True,
            "volume24hr": 1000.0, "markets": legs}


# ── Leg filtering ────────────────────────────────────────────────────────────

class TestLiveLegs:
    def test_excludes_closed_legs(self):
        """Eliminated outcomes (closed legs) must not enter the sums."""
        legs = [_leg("A", 0.30, 0.31), _leg("B", 0.68, 0.69),
                _leg("Eliminated", 0.0, 0.001, closed=True)]
        assert len(live_legs(_event(legs))) == 2

    def test_excludes_legs_without_quotes(self):
        legs = [_leg("A", 0.30, 0.31)]
        legs.append({"groupItemTitle": "B", "active": True, "closed": False,
                     "acceptingOrders": True})  # no bestBid/bestAsk
        assert len(live_legs(_event(legs))) == 1


# ── Exhaustiveness gate ──────────────────────────────────────────────────────

class TestExhaustivenessGate:
    def test_catchall_detected(self):
        legs = [_leg("A", 0.4, 0.41), _leg("B", 0.3, 0.31),
                _leg("Other candidate", 0.2, 0.21)]
        assert has_catchall(legs) is True

    def test_no_catchall(self):
        legs = [_leg("Spain", 0.1, 0.11), _leg("France", 0.3, 0.31),
                _leg("England", 0.14, 0.15)]
        assert has_catchall(legs) is False

    def test_incomplete_set_not_tradable_long(self):
        """The Dem-Nominee false positive: cheap sum-of-asks on a set with
        unlisted outcomes must NOT be flagged tradable long."""
        legs = [_leg("A", 0.40, 0.41), _leg("B", 0.30, 0.31),
                _leg("C", 0.20, 0.21)]  # sum ask 0.93 — looks like 7c profit
        row = analyze_event(_event(legs))
        assert row["margin_long_net"] > 0.02
        assert row["exhaustive"] is False
        assert row["tradable_long"] is False

    def test_whitelist_enables_long(self):
        legs = [_leg("A", 0.40, 0.41), _leg("B", 0.30, 0.31),
                _leg("C", 0.20, 0.21)]
        row = analyze_event(_event(legs, event_id="42"), whitelist={"42"})
        assert row["exhaustive"] is True
        assert row["exhaustive_reason"] == "whitelist"
        assert row["tradable_long"] is True

    def test_catchall_enables_long(self):
        legs = [_leg("A", 0.40, 0.41), _leg("B", 0.30, 0.31),
                _leg("Field (any other)", 0.20, 0.21)]
        row = analyze_event(_event(legs))
        assert row["tradable_long"] is True
        assert row["exhaustive_reason"] == "catchall"


# ── Margin math ──────────────────────────────────────────────────────────────

class TestMargins:
    def test_long_margin_gross(self):
        """1 − Σasks, no fees."""
        legs = [_leg("A", 0.40, 0.41), _leg("B", 0.30, 0.31),
                _leg("Other", 0.20, 0.21)]
        row = analyze_event(_event(legs))
        assert row["margin_long_gross"] == pytest.approx(1 - 0.93, abs=1e-9)
        assert row["margin_long_net"] == row["margin_long_gross"]  # feesEnabled=False

    def test_short_margin_gross(self):
        """Σbids − 1: overpriced set is shortable without any gate."""
        legs = [_leg("A", 0.45, 0.46), _leg("B", 0.35, 0.36),
                _leg("C", 0.26, 0.27)]  # sum bid 1.06
        row = analyze_event(_event(legs))
        assert row["margin_short_gross"] == pytest.approx(0.06, abs=1e-9)
        assert row["exhaustive"] is False
        assert row["tradable_short"] is True  # no exhaustiveness needed

    def test_fees_reduce_net_margin(self):
        """Per-leg feeSchedule is netted from both sides."""
        legs = [_leg("A", 0.40, 0.41, fees=True), _leg("B", 0.30, 0.31, fees=True),
                _leg("Other", 0.20, 0.21, fees=True)]
        row = analyze_event(_event(legs))
        expected_fee = sum(0.03 * p * (1 - p) for p in (0.41, 0.31, 0.21))
        assert row["margin_long_net"] == pytest.approx(
            row["margin_long_gross"] - expected_fee, abs=1e-4)
        assert row["margin_long_net"] < row["margin_long_gross"]

    def test_balanced_book_no_trigger(self):
        """A fairly priced set triggers nothing."""
        legs = [_leg("A", 0.49, 0.50), _leg("B", 0.29, 0.30),
                _leg("Other", 0.19, 0.20)]  # asks sum 1.00, bids 0.97
        row = analyze_event(_event(legs))
        assert row["tradable_long"] is False
        assert row["tradable_short"] is False

    def test_too_few_legs_returns_none(self):
        legs = [_leg("A", 0.49, 0.50), _leg("B", 0.49, 0.50)]
        assert analyze_event(_event(legs)) is None


# ── Evidence log ─────────────────────────────────────────────────────────────

class TestAppendRows:
    def test_creates_csv_with_header(self, tmp_path):
        legs = [_leg("A", 0.40, 0.41), _leg("B", 0.30, 0.31),
                _leg("Other", 0.20, 0.21)]
        row = analyze_event(_event(legs))
        out = tmp_path / "negrisk_log.csv"
        append_rows([row], out)
        append_rows([row], out)  # second append: no duplicate header
        lines = out.read_text().strip().splitlines()
        assert lines[0] == ",".join(CSV_COLUMNS)
        assert len(lines) == 3

    def test_empty_rows_no_file(self, tmp_path):
        out = tmp_path / "negrisk_log.csv"
        append_rows([], out)
        assert not out.exists()
