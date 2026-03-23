"""
src/alpha/obi.py
Order Book Imbalance (OBI) signal for prediction markets.

Ref: Saguillo et al. (arXiv 2025), Cont et al. (2014).
OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)  in [-1, 1]
  OBI > 0 → more bids → upward price pressure
  OBI < 0 → more asks → downward price pressure

OBI explains ~65% of short-term price variance.
IR > 0.65 predicts price increase in 15-30 min (58% accuracy).

Usage:
    from alpha.obi import OrderBookImbalance
    obi = OrderBookImbalance()
    df  = obi.annotate_wf(wf_df, mock=True)
"""

import numpy as np
import pandas as pd
import requests
import time

DEMO_BASE            = "https://demo-api.kalshi.co/trade-api/v2"
OBI_SIGNAL_THRESHOLD = 0.35
IR_THRESHOLD         = 0.65


def compute_obi(bid_volume: float, ask_volume: float) -> float:
    total = bid_volume + ask_volume
    return 0.0 if total <= 0 else (bid_volume - ask_volume) / total


def compute_ir(bid_volume: float, ask_volume: float) -> float:
    total = bid_volume + ask_volume
    return 0.5 if total <= 0 else bid_volume / total


class OrderBookImbalance:
    """Computes OBI for prediction market order books (live or mock)."""

    def fetch_orderbook(self, ticker: str) -> dict | None:
        try:
            r = requests.get(f"{DEMO_BASE}/markets/{ticker}/orderbook", timeout=10)
            if r.status_code == 200:
                return r.json().get("orderbook", {})
        except Exception:
            pass
        return None

    def orderbook_to_obi(self, orderbook: dict) -> dict:
        """Parse Kalshi orderbook: yes=[price,size,...], no=[price,size,...]"""
        yes_vol = sum(sz for _, sz in orderbook.get("yes", [])
                      if isinstance(sz, (int, float)))
        no_vol  = sum(sz for _, sz in orderbook.get("no", [])
                      if isinstance(sz, (int, float)))
        obi = compute_obi(yes_vol, no_vol)
        ir  = compute_ir(yes_vol, no_vol)
        signal = ("BULLISH" if obi > OBI_SIGNAL_THRESHOLD else
                  "BEARISH" if obi < -OBI_SIGNAL_THRESHOLD else "NEUTRAL")
        return {"bid_volume": yes_vol, "ask_volume": no_vol,
                "obi": round(obi, 4), "ir": round(ir, 4),
                "signal": signal, "strong_ir": abs(ir - 0.5) > 0.15}

    def fetch_and_compute(self, ticker: str) -> dict:
        ob = self.fetch_orderbook(ticker)
        if ob:
            result = self.orderbook_to_obi(ob)
            result["ticker"] = ticker
            result["source"] = "live"
            return result
        return self._mock_obi(ticker)

    def annotate_wf(self, wf: pd.DataFrame, mock: bool = True) -> pd.DataFrame:
        """
        Add OBI-derived feature columns to walk-forward DataFrame.
        mock=True: derives proxy from volume_fp (fast, no API calls).
        Adds: obi_value, obi_bullish, obi_bearish, obi_strong
        """
        out = wf.copy()
        if mock and "volume_fp" in out.columns:
            vol = out["volume_fp"].fillna(0)
            vol_norm = (vol - vol.median()) / (vol.std() + 1e-8)
            rng   = np.random.default_rng(42)
            noise = rng.normal(0, 0.1, len(out))
            out["obi_value"]   = (vol_norm * 0.1 + noise).clip(-1, 1)
        else:
            out["obi_value"]   = 0.0
        out["obi_bullish"] = out["obi_value"] > OBI_SIGNAL_THRESHOLD
        out["obi_bearish"] = out["obi_value"] < -OBI_SIGNAL_THRESHOLD
        out["obi_strong"]  = out["obi_value"].abs() > IR_THRESHOLD
        return out

    def _mock_obi(self, ticker: str) -> dict:
        import hashlib
        seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)
        rng  = np.random.default_rng(seed)
        obi  = float(np.clip(rng.normal(0, 0.2), -1, 1))
        ir   = (obi + 1) / 2
        return {"ticker": ticker, "bid_volume": float(rng.integers(100, 10000)),
                "ask_volume": float(rng.integers(100, 10000)),
                "obi": round(obi, 4), "ir": round(ir, 4),
                "signal": ("BULLISH" if obi > OBI_SIGNAL_THRESHOLD else
                           "BEARISH" if obi < -OBI_SIGNAL_THRESHOLD else "NEUTRAL"),
                "strong_ir": abs(ir - 0.5) > 0.15, "source": "mock"}


if __name__ == "__main__":
    obi = OrderBookImbalance()
    print("[OBI] OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)")
    print("  |OBI| > 0.35: directional signal | IR > 0.65: strong bullish (Saguillo 2025)")
    sample = pd.DataFrame({"volume_fp": [100, 5000, 50000, 200]})
    annotated = obi.annotate_wf(sample, mock=True)
    print(annotated[["obi_value", "obi_bullish", "obi_bearish"]].to_string())
