"""tests/test_polymarket_collector.py — Unit tests for polymarket_collector.py"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from data.polymarket_collector import (
    parse_prices,
    snapshot_to_row,
    monthly_filename,
    append_snapshot,
    detect_resolutions,
    load_active_ids,
    save_active_ids,
    SNAPSHOT_COLS,
    RESOLUTION_COLS,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_gamma_market():
    return {
        "id": "0x123abc",
        "conditionId": "0xcond456",
        "question": "Will BTC reach $100K by June 2026?",
        "slug": "will-btc-reach-100k",
        "outcomePrices": '[0.35, 0.65]',
        "volume": 50000,
        "volume24hr": 2500,
        "liquidity": 12000,
        "startDate": "2026-01-15T00:00:00Z",
        "endDate": "2026-06-30T23:59:59Z",
        "active": True,
        "closed": False,
    }


@pytest.fixture
def ts():
    return datetime(2026, 3, 25, 14, 0, 0, tzinfo=timezone.utc)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestParsePrices:
    def test_json_string(self):
        m = {"outcomePrices": '[0.55, 0.45]'}
        yes, no = parse_prices(m)
        assert abs(yes - 0.55) < 1e-6
        assert abs(no - 0.45) < 1e-6

    def test_list_input(self):
        m = {"outcomePrices": [0.30, 0.70]}
        yes, no = parse_prices(m)
        assert abs(yes - 0.30) < 1e-6
        assert abs(no - 0.70) < 1e-6

    def test_missing_prices(self):
        yes, no = parse_prices({})
        assert yes == 0.0
        assert no == 0.0

    def test_malformed_json(self):
        m = {"outcomePrices": "not json"}
        yes, no = parse_prices(m)
        assert yes == 0.0
        assert no == 0.0

    def test_single_price(self):
        m = {"outcomePrices": '[0.80]'}
        yes, no = parse_prices(m)
        assert abs(yes - 0.80) < 1e-6
        assert abs(no - 0.20) < 1e-6


class TestSnapshotToRow:
    def test_all_fields_present(self, sample_gamma_market, ts):
        row = snapshot_to_row(sample_gamma_market, ts)
        for col in SNAPSHOT_COLS:
            assert col in row, f"Missing column: {col}"

    def test_prices_parsed(self, sample_gamma_market, ts):
        row = snapshot_to_row(sample_gamma_market, ts)
        assert abs(row["yes_price"] - 0.35) < 1e-6
        assert abs(row["no_price"] - 0.65) < 1e-6

    def test_timestamp_set(self, sample_gamma_market, ts):
        row = snapshot_to_row(sample_gamma_market, ts)
        assert row["snapshot_ts"] == ts.isoformat()


class TestMonthlyFilename:
    def test_format(self):
        ts = datetime(2026, 3, 25, tzinfo=timezone.utc)
        assert monthly_filename(ts) == "markets_2026-03.csv"

    def test_january(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert monthly_filename(ts) == "markets_2026-01.csv"


class TestAppendSnapshot:
    def test_creates_header(self, tmp_path, sample_gamma_market, ts):
        rows = [snapshot_to_row(sample_gamma_market, ts)]
        path = append_snapshot(rows, tmp_path, ts)
        assert path.exists()
        content = path.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == ",".join(SNAPSHOT_COLS)
        assert len(lines) == 2  # header + 1 data row

    def test_append_no_duplicate_header(self, tmp_path, sample_gamma_market, ts):
        rows = [snapshot_to_row(sample_gamma_market, ts)]
        append_snapshot(rows, tmp_path, ts)
        append_snapshot(rows, tmp_path, ts)
        path = tmp_path / monthly_filename(ts)
        content = path.read_text()
        lines = content.strip().split("\n")
        # 1 header + 2 data rows
        assert len(lines) == 3
        # Only first line should be header
        header_count = sum(1 for l in lines if l.startswith("snapshot_ts,"))
        assert header_count == 1


class TestResolutionDetection:
    @patch("data.polymarket_collector.fetch_market_detail")
    def test_vanished_markets_detected(self, mock_fetch, tmp_path):
        mock_fetch.return_value = {
            "question": "Test market?",
            "outcome": "Yes",
            "volume": 10000,
        }

        prev_ids = {"mkt_1": "Q1?", "mkt_2": "Q2?", "mkt_3": "Q3?"}
        current_ids = {"mkt_1", "mkt_3"}  # mkt_2 vanished

        count = detect_resolutions(prev_ids, current_ids, tmp_path)
        assert count == 1

        res_file = tmp_path / "resolutions.csv"
        assert res_file.exists()
        content = res_file.read_text()
        assert "mkt_2" in content

    def test_no_vanished(self, tmp_path):
        prev_ids = {"mkt_1": "Q1?", "mkt_2": "Q2?"}
        current_ids = {"mkt_1", "mkt_2"}
        count = detect_resolutions(prev_ids, current_ids, tmp_path)
        assert count == 0


class TestStateFile:
    def test_save_and_load(self, tmp_path):
        state_file = tmp_path / ".active_ids.json"
        data = {"mkt_1": "Q1?", "mkt_2": "Q2?"}
        save_active_ids(state_file, data)
        loaded = load_active_ids(state_file)
        assert loaded == data

    def test_load_missing_file(self, tmp_path):
        state_file = tmp_path / "nonexistent.json"
        assert load_active_ids(state_file) == {}
