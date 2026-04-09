"""
src/data/momentum_tracker.py — Price momentum tracking with disk persistence.

Tracks per-market price history and computes hourly slope via linear regression.
Persists buffer to data/momentum_history.json so state survives restarts.
"""

import json

import numpy as np
from pathlib import Path

DEFAULT_PATH = Path("data/momentum_history.json")


class MomentumTracker:
    """Track price history for momentum signal computation.

    Persists to disk so momentum state survives restarts.
    Stores up to 288 observations per market (24h at 5-min intervals).
    """

    MAX_HISTORY = 288

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path
        self.buffer: dict[str, list[dict]] = {}
        self.load()

    def record(self, market_id: str, price: float, ts: float) -> None:
        """Record a price observation for a market."""
        if market_id not in self.buffer:
            self.buffer[market_id] = []
        self.buffer[market_id].append({"price": price, "ts": ts})
        if len(self.buffer[market_id]) > self.MAX_HISTORY:
            self.buffer[market_id] = self.buffer[market_id][-self.MAX_HISTORY:]
        self.save()

    def get_history(self, market_id: str) -> list[dict]:
        """Get price history for a market."""
        return self.buffer.get(market_id, [])

    def compute_momentum(self, market_id: str,
                         lookback: int = 12) -> float | None:
        """Compute hourly price slope from recent history.

        Returns None if insufficient data (< 2 observations or < 60s span).
        Returns slope in price-per-hour units.
        """
        history = self.get_history(market_id)
        if len(history) < 2:
            return None
        recent = history[-lookback:]
        if len(recent) < 2:
            return None
        prices = [h["price"] for h in recent]
        times = [h["ts"] for h in recent]
        if times[-1] - times[0] < 60:
            return None
        t = np.array(times) - times[0]
        p = np.array(prices)
        slope = float(np.polyfit(t, p, 1)[0])
        return slope * 3600  # convert to per-hour

    def load(self) -> None:
        """Load buffer from disk."""
        if self.path.exists():
            try:
                self.buffer = json.loads(self.path.read_text())
            except (json.JSONDecodeError, Exception):
                self.buffer = {}

    def save(self) -> None:
        """Persist buffer to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.buffer))
