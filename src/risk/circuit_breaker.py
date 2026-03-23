"""
src/risk/circuit_breaker.py — Mechanical trading halt rules.

Rules (non-discretionary):
  1. Rolling Sharpe (30-trade) < 0.5  → HALT
  2. Daily drawdown > 5%              → HALT
  3. 3 consecutive losses             → HALT
  4. Weekly Sharpe < 1.0              → reduce sizing 50%
  5. Monthly Sharpe < 0.5             → HALT until manual review

Every trigger is logged to data/circuit_breaker_log.csv.
Paper trader and executor MUST call check() before every trade.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("data/circuit_breaker_log.csv")

ROLLING_WINDOW     = 30     # trades for rolling Sharpe
MIN_ROLLING_SHARPE = 0.5    # below this: HALT
DAILY_DD_LIMIT     = 0.05   # 5% daily drawdown
MAX_CONSECUTIVE_LOSSES = 3
WEEKLY_SHARPE_WARN = 1.0    # below: reduce sizing 50%
MONTHLY_SHARPE_HALT = 0.5   # below: HALT

SIZING_FULL    = 1.0
SIZING_REDUCED = 0.5


class CircuitBreaker:
    """
    Stateful circuit breaker. Feed it trades and it tells you
    whether to proceed, reduce sizing, or halt.
    """

    def __init__(self):
        self.trades: list[dict] = []
        self.halted = False
        self.halt_reason = ""
        self.sizing_mult = SIZING_FULL
        self._daily_pnl: dict[str, float] = {}  # date_str → cumulative pnl
        self._consecutive_losses = 0

    def record_trade(self, pnl: float, timestamp: datetime | None = None) -> None:
        """Record a completed trade. Updates all circuit breaker state."""
        ts = timestamp or datetime.now(timezone.utc)
        self.trades.append({"pnl": pnl, "ts": ts})

        # Track consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Track daily P&L
        day_key = ts.strftime("%Y-%m-%d")
        self._daily_pnl[day_key] = self._daily_pnl.get(day_key, 0.0) + pnl

        # Check all rules
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
        self._consecutive_losses = 0
        self._log("MANUAL_RESET", "Circuit breaker manually reset")

    def _check_rules(self) -> None:
        """Evaluate all circuit breaker rules."""
        # Rule 1: Rolling Sharpe < 0.5
        if len(self.trades) >= ROLLING_WINDOW:
            recent = [t["pnl"] for t in self.trades[-ROLLING_WINDOW:]]
            arr = np.array(recent)
            if arr.std() > 0:
                rolling_sharpe = arr.mean() / arr.std() * np.sqrt(252)
                if rolling_sharpe < MIN_ROLLING_SHARPE:
                    self._halt(f"Rolling Sharpe ({ROLLING_WINDOW}-trade) = "
                               f"{rolling_sharpe:.2f} < {MIN_ROLLING_SHARPE}")
                    return

        # Rule 2: Daily drawdown > 5%
        if self._daily_pnl:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_pnl = self._daily_pnl.get(today, 0.0)
            # Approximate capital as cumulative P&L (starting at $100 nominal)
            total_capital = 100.0 + sum(t["pnl"] for t in self.trades)
            if total_capital > 0 and daily_pnl < 0:
                daily_dd = abs(daily_pnl) / total_capital
                if daily_dd > DAILY_DD_LIMIT:
                    self._halt(f"Daily drawdown {daily_dd:.1%} > {DAILY_DD_LIMIT:.0%}")
                    return

        # Rule 3: 3 consecutive losses
        if self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self._halt(f"{self._consecutive_losses} consecutive losses")
            return

        # Rule 4: Weekly Sharpe < 1.0 → reduce sizing
        if len(self.trades) >= 7:
            week_cutoff = datetime.now(timezone.utc).timestamp() - 7 * 86400
            week_trades = [t["pnl"] for t in self.trades
                           if t["ts"].timestamp() > week_cutoff]
            if len(week_trades) >= 5:
                arr = np.array(week_trades)
                if arr.std() > 0:
                    weekly_sharpe = arr.mean() / arr.std() * np.sqrt(252)
                    if weekly_sharpe < WEEKLY_SHARPE_WARN:
                        self.sizing_mult = SIZING_REDUCED
                        self._log("SIZING_REDUCED",
                                  f"Weekly Sharpe {weekly_sharpe:.2f} < {WEEKLY_SHARPE_WARN}")
                    else:
                        self.sizing_mult = SIZING_FULL

        # Rule 5: Monthly Sharpe < 0.5 → HALT
        if len(self.trades) >= 20:
            month_cutoff = datetime.now(timezone.utc).timestamp() - 30 * 86400
            month_trades = [t["pnl"] for t in self.trades
                            if t["ts"].timestamp() > month_cutoff]
            if len(month_trades) >= 15:
                arr = np.array(month_trades)
                if arr.std() > 0:
                    monthly_sharpe = arr.mean() / arr.std() * np.sqrt(252)
                    if monthly_sharpe < MONTHLY_SHARPE_HALT:
                        self._halt(f"Monthly Sharpe {monthly_sharpe:.2f} < "
                                   f"{MONTHLY_SHARPE_HALT} — needs manual review")
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
        row = pd.DataFrame([{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "reason": reason,
            "n_trades": len(self.trades),
            "consecutive_losses": self._consecutive_losses,
            "halted": self.halted,
            "sizing_mult": self.sizing_mult,
        }])
        row.to_csv(LOG_PATH, mode="a",
                   header=not LOG_PATH.exists(), index=False)

    def status_summary(self) -> dict:
        """Current state for dashboard/Telegram."""
        recent_pnl = [t["pnl"] for t in self.trades[-ROLLING_WINDOW:]] \
                     if self.trades else []
        arr = np.array(recent_pnl) if recent_pnl else np.array([0.0])
        rolling_sharpe = arr.mean() / arr.std() * np.sqrt(252) \
                         if arr.std() > 0 else 0.0

        return {
            "halted":             self.halted,
            "halt_reason":        self.halt_reason,
            "sizing_mult":        self.sizing_mult,
            "total_trades":       len(self.trades),
            "consecutive_losses": self._consecutive_losses,
            "rolling_sharpe_30":  round(rolling_sharpe, 2),
            "total_pnl":          round(sum(t["pnl"] for t in self.trades), 2),
        }
