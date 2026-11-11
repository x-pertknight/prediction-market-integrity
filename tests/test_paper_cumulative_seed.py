"""tests/test_paper_cumulative_seed.py — cumulative_pnl seeded from CSV (FIX 3).

The trades CSV is appended across restarts; the in-memory counter must be
seeded from it so a restart continues the lifetime total instead of resetting
to zero.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

from execution.paper_trader import _seed_cumulative_pnl, _CSV_COLS


def _write_trades(path, pnls):
    """Write a paper_trades.csv-shaped file. Each entry: float => closed
    (PAPER_SELL with pnl_net) ; None => open row (empty pnl_net)."""
    rows = []
    for i, p in enumerate(pnls):
        row = {c: "" for c in _CSV_COLS}
        row["market_id"] = f"MKT{i}"
        if p is None:
            row["side"] = "PAPER_BUY"
        else:
            row["side"] = "PAPER_SELL"
            row["pnl_net"] = p
        rows.append(row)
    pd.DataFrame(rows, columns=_CSV_COLS).to_csv(path, index=False)


def test_seed_zero_when_no_csv(tmp_path):
    assert _seed_cumulative_pnl(tmp_path / "nope.csv") == 0.0


def test_seed_sums_closed_trades_skipping_open_rows(tmp_path):
    csv = tmp_path / "paper_trades.csv"
    _write_trades(csv, [0.10, None, 0.25, None, -0.05])   # 3 closed, 2 open
    assert _seed_cumulative_pnl(csv) == pytest.approx(0.30)


def test_restart_midhistory_continues_not_resets(tmp_path):
    """Simulated restart at mid-history: seeding recovers the lifetime total
    instead of restarting from zero, then keeps accumulating. Mirrors the
    real divergence (April run +2.28, then first post-restart close +0.1129)."""
    csv = tmp_path / "paper_trades.csv"
    _write_trades(csv, [0.0416, 0.4389, 0.0867, 0.2263, 0.2576, 0.6428, 0.5858])
    seeded = _seed_cumulative_pnl(csv)
    assert seeded == pytest.approx(2.2797, abs=1e-4)
    assert seeded != 0.0                      # the bug would have restarted at 0
    running = seeded + 0.1129                 # a post-restart close continues it
    assert running == pytest.approx(2.3926, abs=1e-4)
