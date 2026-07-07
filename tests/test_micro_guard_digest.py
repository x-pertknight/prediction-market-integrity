"""tests/test_micro_guard_digest.py — microstructure entry guard + digest."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from execution.polymarket_trader import entry_microstructure_ok
from alpha.negrisk_scanner import compose_digest


class TestMicrostructureGuard:
    def test_dust_price_rejected(self):
        """The incident pattern: sub-cent entries rejected."""
        m = {"bestBid": 0.002, "bestAsk": 0.003}
        ok, reason = entry_microstructure_ok(m, 0.0025)
        assert not ok and reason == "dust_price"

    def test_no_bid_rejected(self):
        m = {"bestBid": 0, "bestAsk": 0.30}
        ok, reason = entry_microstructure_ok(m, 0.30)
        assert not ok and reason == "no_bid"

    def test_wide_spread_rejected(self):
        """Spread 0.05-0.10 on mid 0.075 = 67% of mid → reject."""
        m = {"bestBid": 0.05, "bestAsk": 0.10}
        ok, reason = entry_microstructure_ok(m, 0.075)
        assert not ok and reason == "wide_spread"

    def test_healthy_book_accepted(self):
        m = {"bestBid": 0.30, "bestAsk": 0.31}
        ok, reason = entry_microstructure_ok(m, 0.305)
        assert ok and reason == "ok"

    def test_missing_book_fields_rejected(self):
        """Gamma omitting bestBid means no exit path — reject."""
        ok, reason = entry_microstructure_ok({}, 0.30)
        assert not ok and reason == "no_bid"


class TestComposeDigest:
    def _rows(self):
        return [
            {"event_id": "1", "title": "World Cup Winner", "exhaustive": "True",
             "margin_long_net": "-0.011", "margin_short_net": "-0.048",
             "tradable_long": "False", "tradable_short": "False"},
            {"event_id": "2", "title": "Nobel Prize", "exhaustive": "False",
             "margin_long_net": "0.38", "margin_short_net": "-0.1",
             "tradable_long": "False", "tradable_short": "False"},
            {"event_id": "3", "title": "Norway Elimination", "exhaustive": "True",
             "margin_long_net": "0.021", "margin_short_net": "0.005",
             "tradable_long": "True", "tradable_short": "False"},
        ]

    def test_contains_counts_and_best(self):
        text = compose_digest(self._rows(), [], 2, "2026-07-07")
        assert "3 osservazioni su 3 eventi" in text
        assert "Norway Elimination" in text        # best exhaustive long
        assert "+0.0210" in text
        assert "Finestre sopra soglia: 1" in text
        assert "2 set aperti" in text

    def test_gated_candidate_shown_but_not_best_long(self):
        """Nobel (+38c phantom) must appear as gated, never as best long."""
        text = compose_digest(self._rows(), [], 0, "2026-07-07")
        best_line = [l for l in text.splitlines() if "Best LONG" in l][0]
        assert "Nobel" not in best_line
        assert any("Gated" in l and "Nobel" in l for l in text.splitlines())

    def test_paper_summary(self):
        trades = [
            {"action": "OPEN", "total_pnl": "0.5"},
            {"action": "RESOLVE", "total_pnl": "1.25"},
            {"action": "RESOLVE", "total_pnl": "-0.25"},
        ]
        text = compose_digest([], trades, 1, "2026-07-07")
        assert "1 aperture, 2 risolte" in text
        assert "+1.00" in text

    def test_empty_day(self):
        text = compose_digest([], [], 0, "2026-07-07")
        assert "0 osservazioni" in text
