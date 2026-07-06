"""
src/data/polymarket_clob.py
Polymarket CLOB API — orderbook, best bid/ask, market depth.
No auth for public data.
https://clob.polymarket.com
"""
import time
import requests
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
    """Fetch best bid/ask and spread from CLOB for a token.

    The /book endpoint returns levels sorted worst→best (best level LAST,
    verified live 2026-07-06), so sort by price instead of trusting array
    order. Depth = size within the 5 levels closest to the touch.
    """
    try:
        r = requests.get(f"{BASE}/book", params={"token_id": token_id}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            bids = sorted((float(b["price"]), float(b["size"])) for b in data.get("bids", []))
            asks = sorted((float(a["price"]), float(a["size"])) for a in data.get("asks", []))
            best_bid = bids[-1][0] if bids else 0.0
            best_ask = asks[0][0] if asks else 1.0
            return {
                "token_id":  token_id,
                "best_bid":  best_bid,
                "best_ask":  best_ask,
                "spread":    best_ask - best_bid,
                "bid_depth": sum(sz for _, sz in bids[-5:]),
                "ask_depth": sum(sz for _, sz in asks[:5]),
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


from data.momentum_tracker import MomentumTracker  # noqa: F401 (re-export)


if __name__ == "__main__":
    df = fetch_and_save()
    validate(df)
