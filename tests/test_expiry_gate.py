"""tests/test_expiry_gate.py — hours_to_expiry helper (incident 2026-07-07).

The trader bought expiry-day dust markets that its own exit rule dumped at
bid on the next scan (-10.5% bankroll in 10 minutes). The entry gate relies
on this helper; these tests pin its behavior.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from execution.polymarket_trader import hours_to_expiry, EXPIRY_EXIT_H


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class TestHoursToExpiry:
    def test_future_market(self):
        m = {"endDate": _iso(datetime.now(timezone.utc) + timedelta(hours=100))}
        assert hours_to_expiry(m) == pytest.approx(100, abs=0.1)

    def test_expiry_day_market_below_exit_threshold(self):
        """The incident case: a market inside the exit window must be
        detectable so the entry gate can skip it."""
        m = {"endDate": _iso(datetime.now(timezone.utc) + timedelta(hours=5))}
        h = hours_to_expiry(m)
        assert h is not None and h <= EXPIRY_EXIT_H

    def test_already_expired_market_is_negative(self):
        """'Hong Kong high temp on July 6' bought on July 7: past end date."""
        m = {"endDate": _iso(datetime.now(timezone.utc) - timedelta(days=1))}
        h = hours_to_expiry(m)
        assert h is not None and h < 0

    def test_missing_end_date_returns_none(self):
        assert hours_to_expiry({}) is None

    def test_malformed_end_date_returns_none(self):
        assert hours_to_expiry({"endDate": "not-a-date"}) is None

    def test_end_date_iso_fallback(self):
        m = {"endDateIso": _iso(datetime.now(timezone.utc) + timedelta(hours=48))}
        assert hours_to_expiry(m) == pytest.approx(48, abs=0.1)
