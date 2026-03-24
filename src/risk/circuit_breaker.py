"""
src/risk/circuit_breaker.py — Mechanical trading halt rules.

Calibrated for 19% win rate (81% loss rate).

With 19% win rate, consecutive loss probabilities:
  3 losses:  (0.81)^3  = 53% — happens every 2 days (NORMAL)
  5 losses:  (0.81)^5  = 35%
  8 losses:  (0.81)^8  = 17%
  10 losses: (0.81)^10 = 12%
  15 losses: (0.81)^15 =  4%
  20 losses: (0.81)^20 =  1.2%

NEVER use loss-count as primary halt trigger with <30% win rate.
Use drawdown-based rules ONLY.

Rules (non-discretionary):
  1. Rolling Sharpe (100-trade) < 0.0     → HALT (need 100+ trades for significance)
  2. Drawdown > 15% of peak equity        → HALT
  3. Daily drawdown > 8% of equity         → reduce sizing 50%
  4. Monthly Sharpe (min 50 trades) < 0.0  → HALT until manual review

Every trigger is logged to data/circuit_breaker_log.csv.
Paper trader and executor MUST call check() before every trade.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("data/circuit_breaker_log.csv")

# Calibrated for 19% win rate
ROLLING_WINDOW       = 100    # need 100+ trades for statistically meaningful Sharpe
MIN_ROLLING_SHARPE   = 0.0    # halt only when rolling Sharpe goes negative
DRAWDOWN_HALT_PCT    = 0.15   # 15% drawdown from peak equity → HALT
DAILY_DD_REDUCE_PCT  = 0.08   # 8% daily drawdown → reduce sizing (not halt)
MONTHLY_SHARPE_HALT  = 0.0    # monthly Sharpe < 0 with enough trades → HALT
MIN_MONTHLY_TRADES   = 50     # need 50 trades for monthly Sharpe to be meaningful

SIZING_FULL    = 1.0
SIZING_REDUCED = 0.5


class CircuitBreaker:
    """
    Stateful circuit breaker. Feed it trades and it tells you
    whether to proceed, reduce sizing, or halt.

    Calibrated for low win rate (19%) strategies where long loss
    streaks are expected. Does NOT use consecutive loss count.
    """

    def __init__(self, initial_capital: float = 100.0):
        self.trades: list[dict] = []
        self.halted = False
        self.halt_reason = ""
        self.sizing_mult = SIZING_FULL
        self._daily_pnl: dict[str, float] = {}
        self._initial_capital = initial_capital
        self._peak_equity = initial_capital

    def record_trade(self, pnl: float, timestamp: datetime | None = None) -> None:
        """Record a completed trade. Updates all circuit breaker state."""
        ts = timestamp or datetime.now(timezone.utc)
        self.trades.append({"pnl": pnl, "ts": ts})

        # Track daily P&L
        day_key = ts.strftime("%Y-%m-%d")
        self._daily_pnl[day_key] = self._daily_pnl.get(day_key, 0.0) + pnl

        # Update peak equity
        equity = self._current_equity()
        if equity > self._peak_equity:
            self._peak_equity = equity

        self._check_rules()

    def check(self) -> dict:
        """
        Check if trading is allowed. Call BEFORE every trade.

        Returns:
            {"allowed": bool, "sizing_mult": float, "reason": str}
        """
        if self.halted:
            return {
                "allowed": False,
                "sizing_mult": 0.0,
                "reason": self.halt_reason,
            }
        return {
            "allowed": True,
            "sizing_mult": self.sizing_mult,
            "reason": "",
        }

    def reset(self) -> None:
        """Manual reset after review. Clears halt state but keeps history."""
        self.halted = False
        self.halt_reason = ""
        self.sizing_mult = SIZING_FULL
        self._log("MANUAL_RESET", "Circuit breaker manually reset")

    def _current_equity(self) -> float:
        return self._initial_capital + sum(t["pnl"] for t in self.trades)

    def _check_rules(self) -> None:
        """Evaluate all circuit breaker rules. Drawdown-based, not loss-count."""
        # Rule 1: Rolling Sharpe (100-trade window) < 0
        if len(self.trades) >= ROLLING_WINDOW:
            recent = [t["pnl"] for t in self.trades[-ROLLING_WINDOW:]]
            arr = np.array(recent)
            if arr.std() > 0:
                rolling_sharpe = arr.mean() / arr.std() * np.sqrt(252)
                if rolling_sharpe < MIN_ROLLING_SHARPE:
                    self._halt(f"Rolling Sharpe ({ROLLING_WINDOW}-trade) = "
                               f"{rolling_sharpe:.2f} < {MIN_ROLLING_SHARPE}")
                    return

        # Rule 2: Drawdown from peak equity > 15%
        equity = self._current_equity()
        if self._peak_equity > 0:
            dd_pct = (self._peak_equity - equity) / self._peak_equity
            if dd_pct > DRAWDOWN_HALT_PCT:
                self._halt(f"Drawdown {dd_pct:.1%} > {DRAWDOWN_HALT_PCT:.0%} "
                           f"(equity=${equity:.2f}, peak=${self._peak_equity:.2f})")
                return

        # Rule 3: Daily drawdown > 8% → reduce sizing (not halt)
        if self._daily_pnl:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_pnl = self._daily_pnl.get(today, 0.0)
            if equity > 0 and daily_pnl < 0:
                daily_dd = abs(daily_pnl) / equity
                if daily_dd > DAILY_DD_REDUCE_PCT:
                    self.sizing_mult = SIZING_REDUCED
                    self._log("SIZING_REDUCED",
                              f"Daily drawdown {daily_dd:.1%} > {DAILY_DD_REDUCE_PCT:.0%}")
                elif self.sizing_mult == SIZING_REDUCED:
                    self.sizing_mult = SIZING_FULL

        # Rule 4: Monthly Sharpe < 0 with enough trades → HALT
        if len(self.trades) >= MIN_MONTHLY_TRADES:
            month_cutoff = datetime.now(timezone.utc).timestamp() - 30 * 86400
            month_trades = [t["pnl"] for t in self.trades
                           if t["ts"].timestamp() > month_cutoff]
            if len(month_trades) >= MIN_MONTHLY_TRADES:
                arr = np.array(month_trades)
                if arr.std() > 0:
                    monthly_sharpe = arr.mean() / arr.std() * np.sqrt(252)
                    if monthly_sharpe < MONTHLY_SHARPE_HALT:
                        self._halt(f"Monthly Sharpe ({len(month_trades)} trades) = "
                                   f"{monthly_sharpe:.2f} < {MONTHLY_SHARPE_HALT} "
                                   f"— needs manual review")
                        return

        # All clear
        if not self.halted:
            self.halt_reason = ""

    def _halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason
        self.sizing_mult = 0.0
        self._log("HALT", reason)

    def _log(self, event: str, reason: str) -> None:
        """Append event to circuit_breaker_log.csv."""
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        equity = self._current_equity()
        dd_pct = (self._peak_equity - equity) / self._peak_equity \
                 if self._peak_equity > 0 else 0.0
        row = pd.DataFrame([{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "reason": reason,
            "n_trades": len(self.trades),
            "equity": round(equity, 2),
            "peak_equity": round(self._peak_equity, 2),
            "drawdown_pct": round(dd_pct, 4),
            "halted": self.halted,
            "sizing_mult": self.sizing_mult,
        }])
        row.to_csv(LOG_PATH, mode="a",
                   header=not LOG_PATH.exists(), index=False)

    def status_summary(self) -> dict:
        """Current state for dashboard/Telegram."""
        equity = self._current_equity()
        dd_pct = (self._peak_equity - equity) / self._peak_equity \
                 if self._peak_equity > 0 else 0.0

        recent_pnl = [t["pnl"] for t in self.trades[-ROLLING_WINDOW:]] \
                     if self.trades else []
        arr = np.array(recent_pnl) if recent_pnl else np.array([0.0])
        rolling_sharpe = arr.mean() / arr.std() * np.sqrt(252) \
                         if arr.std() > 0 else 0.0

        return {
            "halted":            self.halted,
            "halt_reason":       self.halt_reason,
            "sizing_mult":       self.sizing_mult,
            "total_trades":      len(self.trades),
            "equity":            round(equity, 2),
            "peak_equity":       round(self._peak_equity, 2),
            "drawdown_pct":      round(dd_pct, 4),
            "rolling_sharpe":    round(rolling_sharpe, 2),
            "total_pnl":         round(sum(t["pnl"] for t in self.trades), 2),
        }
