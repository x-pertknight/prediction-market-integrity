"""
src/execution/paper_trader.py
Paper trading engine — simulates orders, tracks fills, computes slippage.
Prepares interface for future live trading (Polymarket CLOB + Polygon wallet).
NO real trades. NO wallet. NO real money.

Future live trading path:
  1. Replace PaperBroker with LiveBroker (uses py-clob-client)
  2. Add wallet.sign_order(order) before submission
  3. Remove paper simulation logic
"""
import json
import uuid
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PAPER_DIR = Path("data/paper_trades")

SLIPPAGE_BPS = 50    # 50 basis points simulated slippage for liquid markets
FEE_PER_DOLLAR = 0.07 / 100   # Kalshi/Polymarket ~$0.07 per $1 notional


# ── Order models ──────────────────────────────────────────────────────────────

@dataclass
class Order:
    market_id:    str
    market_title: str
    side:         str    # "buy_yes" | "buy_no"
    size_dollars: float
    limit_price:  float  # 0–1 scale
    category:     str    = ""
    model_prob:   float  = 0.0
    order_id:     str    = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status:       str    = "pending"   # pending | filled | cancelled
    filled_price: float  = 0.0
    fee:          float  = 0.0
    timestamp:    str    = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Position:
    order_id:     str
    market_id:    str
    market_title: str
    side:         str
    size_dollars: float
    entry_price:  float
    category:     str
    model_prob:   float
    open_time:    str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    exit_price:   float = 0.0
    pnl:          float = 0.0
    status:       str   = "open"   # open | closed


# ── Paper broker ─────────────────────────────────────────────────────────────

class PaperBroker:
    """
    Simulates order execution for paper trading.
    Applies realistic slippage and fees.
    Persists trade history to data/paper_trades/.
    """
    def __init__(self):
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path = PAPER_DIR / "paper_trades.jsonl"
        self._pos_path = PAPER_DIR / "open_positions.json"
        self.open_positions: dict[str, Position] = self._load_positions()

    def _load_positions(self) -> dict[str, Position]:
        if self._pos_path.exists():
            try:
                raw = json.loads(self._pos_path.read_text())
                return {k: Position(**v) for k, v in raw.items()}
            except Exception:
                pass
        return {}

    def _save_positions(self):
        data = {k: asdict(v) for k, v in self.open_positions.items()}
        self._pos_path.write_text(json.dumps(data, indent=2))

    def _log_trade(self, event: dict):
        with open(self._log_path, "a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **event}) + "\n")

    def _simulate_fill_price(self, limit_price: float, side: str) -> float:
        """Apply slippage: buy fills slightly above limit, sell slightly below."""
        slippage = SLIPPAGE_BPS / 10_000
        if side == "buy_yes":
            return min(limit_price * (1 + slippage), 0.999)
        return max(limit_price * (1 - slippage), 0.001)

    def submit_order(self, order: Order) -> Order:
        """Simulate order fill."""
        fill_price = self._simulate_fill_price(order.limit_price, order.side)
        fee        = order.size_dollars * FEE_PER_DOLLAR
        # Simulate partial fill probability for illiquid markets (10% chance)
        if random.random() < 0.10:
            order.status = "cancelled"
            order.filled_price = 0.0
            print(f"  [Paper] Order {order.order_id} CANCELLED (simulated illiquidity)")
        else:
            order.status       = "filled"
            order.filled_price = round(fill_price, 4)
            order.fee          = round(fee, 4)
            # Open position
            pos = Position(
                order_id     = order.order_id,
                market_id    = order.market_id,
                market_title = order.market_title,
                side         = order.side,
                size_dollars = order.size_dollars,
                entry_price  = fill_price,
                category     = order.category,
                model_prob   = order.model_prob,
            )
            self.open_positions[order.market_id] = pos
            self._save_positions()
            print(f"  [Paper] FILLED {order.market_title[:40]} | "
                  f"size=${order.size_dollars:.2f} @ {fill_price:.3f} | fee=${fee:.3f}")

        self._log_trade({"event": "order", **order.to_dict()})
        return order

    def close_position(self, market_id: str, outcome: int) -> float:
        """
        Close position at binary resolution.
        outcome: 1 = YES won, 0 = NO won.
        Returns PnL.
        """
        pos = self.open_positions.pop(market_id, None)
        if not pos:
            return 0.0

        shares     = pos.size_dollars / pos.entry_price
        exit_price = 1.0 if outcome == 1 else 0.0
        gross_pnl  = shares * (exit_price - pos.entry_price)
        fee        = pos.size_dollars * FEE_PER_DOLLAR
        net_pnl    = gross_pnl - fee

        pos.exit_price = exit_price
        pos.pnl        = round(net_pnl, 4)
        pos.status     = "closed"

        self._save_positions()
        self._log_trade({"event": "close", **asdict(pos)})

        outcome_str = "✅ WIN" if net_pnl > 0 else "❌ LOSS"
        print(f"  [Paper] CLOSE {pos.market_title[:40]} | {outcome_str} | "
              f"P&L=${net_pnl:+.2f}")
        return net_pnl

    def portfolio_summary(self) -> dict:
        all_trades = []
        if self._log_path.exists():
            for line in self._log_path.read_text().splitlines():
                try:
                    all_trades.append(json.loads(line))
                except Exception:
                    pass
        closes = [t for t in all_trades if t.get("event") == "close"]
        pnls   = [t.get("pnl", 0) for t in closes]
        return {
            "total_trades":    len(closes),
            "open_positions":  len(self.open_positions),
            "total_pnl":       round(sum(pnls), 2),
            "win_rate":        sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0,
            "avg_pnl":         sum(pnls) / len(pnls) if pnls else 0,
        }


# ── Future live trading stub ──────────────────────────────────────────────────

class LiveBroker:
    """
    PLACEHOLDER — future live trading via Polymarket CLOB.
    NOT IMPLEMENTED. Replace PaperBroker with this when ready.
    Requires:
      - py-clob-client
      - Polygon wallet with USDC balance
      - CLOB_API_KEY, CLOB_SECRET env vars
    """
    def __init__(self):
        raise NotImplementedError(
            "Live trading not implemented. "
            "Use PaperBroker for simulation. "
            "See docs/live_trading_setup.md for implementation plan."
        )


if __name__ == "__main__":
    broker = PaperBroker()

    # Test buy
    order = broker.submit_order(Order(
        market_id    = "test_mkt_001",
        market_title = "Will NYC high temp exceed 75°F on Apr 1, 2026?",
        side         = "buy_yes",
        size_dollars = 20.0,
        limit_price  = 0.12,
        category     = "weather",
        model_prob   = 0.28,
    ))
    print(f"\nOrder: {order.status} @ {order.filled_price}")

    # Simulate resolution
    pnl = broker.close_position("test_mkt_001", outcome=1)
    print(f"PnL: ${pnl:+.2f}")
    print(f"\nPortfolio: {json.dumps(broker.portfolio_summary(), indent=2)}")
