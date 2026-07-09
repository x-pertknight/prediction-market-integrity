"""tests/test_position_manager.py — daily-halt rollover semantics (FIX 2).

Covers the mandatory cases:
  (a) a live process that crosses midnight with a DAILY halt unblocks —
      can_trade() itself performs the rollover, no reconstruction needed;
  (b) a STICKY halt stays blocked even after the day rolls over.
STATE_FILE is redirected to a tmp path so these tests never touch the live
data/risk_state.json.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

import risk.position_manager as pm_mod
from risk.position_manager import PositionManager


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    f = tmp_path / "risk_state.json"
    monkeypatch.setattr(pm_mod, "STATE_FILE", f)
    return f


def _pm_with(**state):
    """Build a PositionManager, then overwrite its state for a scenario."""
    pm = PositionManager()
    for k, v in state.items():
        setattr(pm.state, k, v)
    return pm


def test_daily_drawdown_halt_clears_when_day_rolls_over_mid_run(tmp_state):
    """(a) daily-drawdown halt from 'yesterday' clears on the next can_trade()."""
    pm = _pm_with(
        halted=True, halt_scope="daily", halt_reason="daily drawdown limit hit",
        daily_start="2000-01-01", daily_pnl=-100.0,
    )
    can, reason = pm.can_trade()          # a loop iteration on the new day
    assert can is True
    assert reason == "ok"
    assert pm.state.halted is False
    assert pm.state.halt_scope == ""
    assert pm.state.daily_pnl == 0.0
    assert pm.state.daily_start != "2000-01-01"


def test_daily_loss_cap_halt_also_clears_on_rollover(tmp_state):
    """The daily loss-cap halt is daily-scoped too → clears at rollover."""
    pm = _pm_with(
        halted=True, halt_scope="daily", halt_reason="daily loss cap $50.0",
        daily_start="2000-01-01", daily_pnl=-60.0,
    )
    can, _ = pm.can_trade()
    assert can is True
    assert pm.state.halted is False


def test_sticky_halt_stays_blocked_after_rollover(tmp_state):
    """(b) a sticky halt (consecutive losses / manual / equity) persists."""
    pm = _pm_with(
        halted=True, halt_scope="sticky", halt_reason="consecutive losses: 3",
        daily_start="2000-01-01", consecutive_losses=3,
    )
    can, reason = pm.can_trade()
    assert can is False
    assert reason == "consecutive losses: 3"
    assert pm.state.halted is True
    # daily fields still roll over, but the halt itself stays put
    assert pm.state.daily_start != "2000-01-01"
    assert pm.state.daily_pnl == 0.0


def test_legacy_halt_without_scope_is_treated_as_sticky(tmp_state):
    """Back-compat: a pre-existing halt with no halt_scope must NOT auto-clear
    (scope is explicit; we never infer 'daily' from the reason text)."""
    pm = _pm_with(
        halted=True, halt_scope="", halt_reason="daily drawdown limit hit",
        daily_start="2000-01-01",
    )
    can, _ = pm.can_trade()
    assert can is False
    assert pm.state.halted is True


def test_daily_drawdown_halt_is_tagged_daily(tmp_state):
    """A fresh daily-drawdown halt is tagged halt_scope='daily'."""
    pm = _pm_with(bankroll=1000.0, daily_pnl=-60.0)   # -6% > 5% limit
    can, reason = pm.can_trade()
    assert can is False
    assert "daily drawdown" in reason
    assert pm.state.halt_scope == "daily"


def test_consecutive_loss_halt_is_tagged_sticky(tmp_state):
    """A fresh consecutive-loss halt is tagged sticky so it won't auto-reset."""
    pm = _pm_with(consecutive_losses=3)
    can, _ = pm.can_trade()
    assert can is False
    assert pm.state.halt_scope == "sticky"
