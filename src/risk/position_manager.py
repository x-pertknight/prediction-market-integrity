"""
src/risk/position_manager.py
Risk management: Kelly sizing, drawdown limits, consecutive loss stops,
max exposure per market/category.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import date

from risk.kelly import size_trade, expected_value

STATE_FILE = Path("data/risk_state.json")


@dataclass
class RiskState:
    bankroll:           float = 1000.0
    peak_bankroll:      float = 1000.0
    daily_start:        str   = ""
    daily_pnl:          float = 0.0
    consecutive_losses: int   = 0
    halted:             bool  = False
    halt_reason:        str   = ""
    halt_scope:         str   = ""    # "daily" clears at day rollover; "sticky" never auto-resets
    open_positions:     dict  = field(default_factory=dict)   # market_id → {size, price}
    category_exposure:  dict  = field(default_factory=dict)   # category → total_size

    def to_dict(self):
        return asdict(self)


# ── Limits ────────────────────────────────────────────────────────────────────
MAX_DAILY_DRAWDOWN_PCT   = 0.05   # 5% of bankroll
MAX_CONSEC_LOSSES        = 3
MAX_MARKET_EXPOSURE_PCT  = 0.10   # 10% per market
MAX_CATEGORY_EXPOSURE_PCT= 0.30   # 30% per category
MAX_OPEN_POSITIONS       = 50     # max concurrent open positions
MAX_DAILY_LOSS_DOLLARS   = 50.0   # hard dollar cap on daily loss


def load_state() -> RiskState:
    if STATE_FILE.exists():
        d = json.loads(STATE_FILE.read_text())
        return RiskState(**{k: v for k, v in d.items() if k in RiskState.__dataclass_fields__})
    return RiskState()


def save_state(state: RiskState) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state.to_dict(), indent=2))


class PositionManager:
    def __init__(self):
        self.state = load_state()
        self._reset_daily_if_needed()

    def _reset_daily_if_needed(self):
        today = date.today().isoformat()
        if self.state.daily_start != today:
            self.state.daily_start = today
            self.state.daily_pnl   = 0.0
            # Only daily-scoped halts (daily drawdown, daily loss cap) clear
            # at the day rollover; sticky halts (consecutive losses, manual,
            # equity) never auto-reset. Scope is an explicit field, not text.
            if self.state.halted and self.state.halt_scope == "daily":
                self.state.halted      = False
                self.state.halt_reason = ""
                self.state.halt_scope  = ""
            save_state(self.state)

    def can_trade(self) -> tuple[bool, str]:
        # Roll the day over even in a long-running process. The reset is no
        # longer confined to __init__, so a daily halt clears when the date
        # changes mid-run instead of persisting across midnight until restart.
        self._reset_daily_if_needed()
        if self.state.halted:
            return False, self.state.halt_reason
        # Daily drawdown check (percentage)
        max_loss = self.state.bankroll * MAX_DAILY_DRAWDOWN_PCT
        if self.state.daily_pnl <= -max_loss:
            self.state.halted     = True
            self.state.halt_reason = "daily drawdown limit hit"
            self.state.halt_scope  = "daily"
            save_state(self.state)
            return False, self.state.halt_reason
        # Daily loss dollar cap
        if self.state.daily_pnl <= -MAX_DAILY_LOSS_DOLLARS:
            self.state.halted     = True
            self.state.halt_reason = f"daily loss cap ${MAX_DAILY_LOSS_DOLLARS}"
            self.state.halt_scope  = "daily"
            save_state(self.state)
            return False, self.state.halt_reason
        # Max open positions
        if len(self.state.open_positions) >= MAX_OPEN_POSITIONS:
            return False, f"max open positions ({MAX_OPEN_POSITIONS})"
        # Consecutive losses check
        if self.state.consecutive_losses >= MAX_CONSEC_LOSSES:
            self.state.halted     = True
            self.state.halt_reason = f"consecutive losses: {self.state.consecutive_losses}"
            self.state.halt_scope  = "sticky"
            save_state(self.state)
            return False, self.state.halt_reason
        return True, "ok"

    def size_order(
        self,
        market_id: str,
        category:  str,
        win_prob:  float,
        entry_price: float,
    ) -> tuple[float, dict]:
        """
        Compute allowed order size given risk limits.
        Returns (size_dollars, metadata_dict).
        """
        can, reason = self.can_trade()
        if not can:
            return 0.0, {"blocked": reason}

        ev = expected_value(win_prob, entry_price)
        if ev <= 0:
            return 0.0, {"blocked": "negative EV", "ev": ev}

        # Kelly size
        raw_size = size_trade(self.state.bankroll, win_prob, entry_price,
                              max_pct=MAX_MARKET_EXPOSURE_PCT)

        # Market exposure cap
        existing_market = self.state.open_positions.get(market_id, {}).get("size", 0)
        max_market_size = self.state.bankroll * MAX_MARKET_EXPOSURE_PCT
        size = min(raw_size, max_market_size - existing_market)

        # Category exposure cap
        cat_used = self.state.category_exposure.get(category, 0)
        max_cat  = self.state.bankroll * MAX_CATEGORY_EXPOSURE_PCT
        size     = min(size, max_cat - cat_used)

        if size <= 0:
            return 0.0, {"blocked": "exposure limit", "category": category}

        return round(size, 2), {
            "kelly_size":   round(raw_size, 2),
            "final_size":   round(size, 2),
            "win_prob":     win_prob,
            "entry_price":  entry_price,
            "ev":           round(ev, 4),
        }

    def record_fill(self, market_id: str, category: str, size: float, price: float):
        self.state.open_positions[market_id] = {"size": size, "price": price}
        self.state.category_exposure[category] = (
            self.state.category_exposure.get(category, 0) + size
        )
        save_state(self.state)

    def record_close(self, market_id: str, category: str, pnl: float):
        pos = self.state.open_positions.pop(market_id, {})
        size = pos.get("size", 0)
        self.state.category_exposure[category] = max(
            0, self.state.category_exposure.get(category, 0) - size
        )
        self.state.bankroll   += pnl
        self.state.peak_bankroll = max(self.state.peak_bankroll, self.state.bankroll)
        self.state.daily_pnl  += pnl
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        save_state(self.state)

    def status(self) -> dict:
        can, reason = self.can_trade()
        return {
            "bankroll":           self.state.bankroll,
            "daily_pnl":         self.state.daily_pnl,
            "consecutive_losses": self.state.consecutive_losses,
            "can_trade":         can,
            "halt_reason":       reason,
            "open_positions":    len(self.state.open_positions),
            "drawdown_pct":      (self.state.bankroll - self.state.peak_bankroll) / self.state.peak_bankroll,
        }


if __name__ == "__main__":
    pm = PositionManager()
    print("Risk status:", json.dumps(pm.status(), indent=2))
    size, meta = pm.size_order("mkt_123", "weather", 0.25, 0.10)
    print(f"Order size: ${size} | meta: {meta}")
