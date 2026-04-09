"""
src/data/polymarket_clob.py
Polymarket CLOB API — orderbook, best bid/ask, market depth.
No auth for public data.
https://clob.polymarket.com
"""
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

BASE    = "https://clob.polymarket.com"
RAW_DIR = Path("data/raw")


def fetch_markets(limit: int = 200) -> pd.DataFrame:
    """Paginate CLOB markets to get orderbook metadata."""
    all_markets = []
    cursor = ""
    pages  = 0
    while pages < 20:
        params = {"limit": 100}
        if cursor:
            params["next_cursor"] = cursor
        try:
            r = requests.get(f"{BASE}/markets", params=params, timeout=15)
            if r.status_code != 200:
                break
            data   = r.json()
            batch  = data.get("data", [])
            cursor = data.get("next_cursor", "")
            all_markets.extend(batch)
            pages += 1
            time.sleep(0.1)
            if not cursor or not batch or len(all_markets) >= limit:
                break
        except Exception as e:
            print(f"  [PolyCLOB] {e}")
            break

    if not all_markets:
        return pd.DataFrame()

    rows = []
    for m in all_markets:
        rows.append({
            "condition_id":        m.get("condition_id",""),
            "question_id":         m.get("question_id",""),
            "active":              m.get("active", False),
            "closed":              m.get("closed", False),
            "accepting_orders":    m.get("accepting_orders", False),
            "minimum_order_size":  float(m.get("minimum_order_size", 0) or 0),
            "minimum_tick_size":   float(m.get("minimum_tick_size", 0) or 0),
        })
    return pd.DataFrame(rows)


def fetch_orderbook(token_id: str) -> dict:
    """Fetch best bid/ask and spread from CLOB for a token."""
    try:
        r = requests.get(f"{BASE}/book", params={"token_id": token_id}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            best_bid = float(bids[0]["price"]) if bids else 0.0
            best_ask = float(asks[0]["price"]) if asks else 1.0
            return {
                "token_id":  token_id,
                "best_bid":  best_bid,
                "best_ask":  best_ask,
                "spread":    best_ask - best_bid,
                "bid_depth": sum(float(b["size"]) for b in bids[:5]),
                "ask_depth": sum(float(a["size"]) for a in asks[:5]),
            }
    except Exception:
        pass
    return {}


def fetch_and_save() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("[PolyCLOB] Fetching CLOB markets...")
    df = fetch_markets(200)
    if df.empty:
        print("  [PolyCLOB] No data")
        return df
    df.to_csv(RAW_DIR / "polymarket_clob.csv", index=False)
    active = df[df["active"] == True]
    print(f"  [PolyCLOB] {len(df)} markets | {len(active)} active | accepting_orders={df['accepting_orders'].sum()}")
    return df


def validate(df: pd.DataFrame) -> None:
    assert "condition_id" in df.columns


MOMENTUM_PATH = Path("data/momentum_history.json")


class MomentumTracker:
    """Track price history for momentum signal computation.

    Persists to disk so momentum state survives restarts.
    Stores up to 288 observations per market (24h at 5-min intervals).
    """

    MAX_HISTORY = 288

    def __init__(self, path: Path = MOMENTUM_PATH):
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


if __name__ == "__main__":
    df = fetch_and_save()
    validate(df)
