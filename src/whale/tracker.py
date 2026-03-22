"""
src/whale/tracker.py
Polymarket Whale Tracker.
- Fetches top wallets from Polymarket Data API leaderboard
- Tracks active positions per wallet
- Computes Smart Money Index (0–100) per market
- Detects whale cluster alerts (3+ whales same direction)
- Saves to data/whale/

Reference: ~/references/polymarket-full-bot/src/connectors/polymarket_data.py
API: https://data-api.polymarket.com (no auth for public endpoints)
"""
import json
import time
import requests
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, date
from typing import Any

POLY_DATA  = "https://data-api.polymarket.com"
POLY_GAMMA = "https://gamma-api.polymarket.com"
WHALE_DIR  = Path("data/whale")
HDR        = {"Accept": "application/json", "User-Agent": "traptrade/1.0"}

WHALE_THRESHOLD_USD = 1000    # min position value to qualify as whale
TOP_N_WALLETS       = 100


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class WalletPosition:
    proxy_wallet:  str   = ""
    market_slug:   str   = ""
    title:         str   = ""
    outcome:       str   = ""
    size:          float = 0.0
    avg_price:     float = 0.0
    cur_price:     float = 0.0
    current_value: float = 0.0
    cash_pnl:      float = 0.0


@dataclass
class WhaleAlert:
    market_slug:    str
    direction:      str       # "yes" | "no"
    whale_count:    int
    whale_wallets:  list[str] = field(default_factory=list)
    total_exposure: float     = 0.0
    avg_price:      float     = 0.0
    smart_money_idx: float    = 0.0   # 0-100

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_slug":    self.market_slug,
            "direction":      self.direction,
            "whale_count":    self.whale_count,
            "whale_wallets":  self.whale_wallets[:5],   # top 5
            "total_exposure": round(self.total_exposure, 2),
            "avg_price":      round(self.avg_price, 4),
            "smart_money_idx": round(self.smart_money_idx, 1),
            "cluster_detected": self.whale_count >= 3,
        }


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(url: str, params: dict = {}, retries: int = 3) -> Any:
    delay = 1.0
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, headers=HDR, timeout=15)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 30)
            else:
                return None
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 15)
    return None


# ── Leaderboard ───────────────────────────────────────────────────────────────

def fetch_leaderboard(limit: int = TOP_N_WALLETS) -> list[str]:
    """Fetch top trader wallet addresses from Polymarket leaderboard."""
    # Try leaderboard endpoint
    data = _get(f"{POLY_DATA}/leaderboard", {"limit": limit, "window": "allTime"})
    if data and isinstance(data, list):
        wallets = [w.get("proxyWallet", w.get("address","")) for w in data[:limit]]
        return [w for w in wallets if w]

    # Fallback: try rankings endpoint
    data = _get(f"{POLY_DATA}/rankings", {"limit": limit})
    if data and isinstance(data, list):
        return [w.get("proxyWallet","") for w in data[:limit] if w.get("proxyWallet")]

    print("  [Whale] Leaderboard not available via public API — using hardcoded sample wallets")
    # Hardcoded top trader addresses (public, from Polymarket explorer)
    return [
        "0x0000000000000000000000000000000000000001",  # placeholder
    ]


# ── Position fetcher ──────────────────────────────────────────────────────────

def fetch_wallet_positions(address: str) -> list[WalletPosition]:
    data = _get(f"{POLY_DATA}/positions", {"user": address.lower(), "limit": 100, "sortBy": "CURRENT"})
    if not data:
        return []
    items = data if isinstance(data, list) else data.get("positions", data.get("data", []))
    positions = []
    for item in items:
        cur_val = float(item.get("currentValue", item.get("current_value", 0)) or 0)
        if cur_val < WHALE_THRESHOLD_USD:
            continue    # skip small positions
        positions.append(WalletPosition(
            proxy_wallet  = address,
            market_slug   = str(item.get("slug", item.get("market_slug", ""))),
            title         = str(item.get("title", "")),
            outcome       = str(item.get("outcome", "")).lower(),
            size          = float(item.get("size", 0) or 0),
            avg_price     = float(item.get("avgPrice", item.get("avg_price", 0)) or 0),
            cur_price     = float(item.get("curPrice", item.get("cur_price", 0)) or 0),
            current_value = cur_val,
            cash_pnl      = float(item.get("cashPnl", item.get("cash_pnl", 0)) or 0),
        ))
    return positions


# ── Smart Money Index ─────────────────────────────────────────────────────────

def compute_smart_money_index(positions: list[WalletPosition], market_slug: str) -> float:
    """
    SMI 0–100 for a given market:
      - 0   = strong bearish whale consensus (all on NO, large size)
      - 50  = neutral / mixed
      - 100 = strong bullish whale consensus (all on YES, large size)
    Weighted by position value.
    """
    mkt_pos = [p for p in positions if p.market_slug == market_slug]
    if not mkt_pos:
        return 50.0

    total_val = sum(p.current_value for p in mkt_pos)
    if total_val == 0:
        return 50.0

    yes_val = sum(p.current_value for p in mkt_pos if "yes" in p.outcome)
    no_val  = sum(p.current_value for p in mkt_pos if "no"  in p.outcome)
    score   = (yes_val / total_val) * 100
    return round(score, 1)


# ── Cluster detection ─────────────────────────────────────────────────────────

def detect_whale_clusters(all_positions: list[WalletPosition]) -> list[WhaleAlert]:
    """
    Group positions by market + outcome.
    Alert when 3+ whales on same direction with total exposure > threshold.
    """
    from collections import defaultdict
    market_dir: dict[tuple, list[WalletPosition]] = defaultdict(list)

    for pos in all_positions:
        if pos.market_slug and pos.outcome in ("yes","no"):
            market_dir[(pos.market_slug, pos.outcome)].append(pos)

    alerts = []
    for (slug, direction), positions in market_dir.items():
        if len(positions) < 3:
            continue
        total_exp = sum(p.current_value for p in positions)
        if total_exp < WHALE_THRESHOLD_USD * 3:
            continue
        avg_price = (
            sum(p.avg_price * p.current_value for p in positions) / total_exp
            if total_exp > 0 else 0.5
        )
        smi = compute_smart_money_index(positions, slug)
        alerts.append(WhaleAlert(
            market_slug    = slug,
            direction      = direction,
            whale_count    = len(positions),
            whale_wallets  = [p.proxy_wallet[:10] for p in positions],
            total_exposure = total_exp,
            avg_price      = avg_price,
            smart_money_idx = smi,
        ))

    return sorted(alerts, key=lambda a: -a.total_exposure)


# ── Main scan ─────────────────────────────────────────────────────────────────

def run_scan(top_n: int = TOP_N_WALLETS) -> dict:
    WHALE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Whale] Fetching top {top_n} wallets from leaderboard...")
    wallets = fetch_leaderboard(top_n)
    print(f"[Whale] Got {len(wallets)} wallets. Fetching positions...")

    all_positions: list[WalletPosition] = []
    for i, wallet in enumerate(wallets):
        if not wallet or wallet.startswith("0x000"):
            continue
        positions = fetch_wallet_positions(wallet)
        all_positions.extend(positions)
        if i % 10 == 0:
            print(f"  [{i+1}/{len(wallets)}] {wallet[:12]}... → {len(positions)} positions")
        time.sleep(0.1)

    print(f"[Whale] Total whale positions: {len(all_positions)}")

    # Save positions
    pos_rows = [
        {
            "proxy_wallet":  p.proxy_wallet,
            "market_slug":   p.market_slug,
            "title":         p.title,
            "outcome":       p.outcome,
            "size":          p.size,
            "avg_price":     p.avg_price,
            "cur_price":     p.cur_price,
            "current_value": p.current_value,
            "cash_pnl":      p.cash_pnl,
            "scan_date":     date.today().isoformat(),
        }
        for p in all_positions
    ]
    df_pos = pd.DataFrame(pos_rows)
    pos_path = WHALE_DIR / f"whale_positions_{date.today().isoformat()}.csv"
    df_pos.to_csv(pos_path, index=False)

    # Detect clusters
    alerts = detect_whale_clusters(all_positions)
    print(f"[Whale] Cluster alerts: {len(alerts)}")
    for a in alerts[:10]:
        print(f"  {a.market_slug[:40]:<40} dir={a.direction} n={a.whale_count} "
              f"exp=${a.total_exposure:,.0f} SMI={a.smart_money_idx:.0f} "
              f"{'⚡ CLUSTER' if a.whale_count >= 3 else ''}")

    alerts_path = WHALE_DIR / f"whale_alerts_{date.today().isoformat()}.json"
    with open(alerts_path, "w") as f:
        json.dump([a.to_dict() for a in alerts], f, indent=2)

    # Latest snapshot (for dashboard)
    latest_path = WHALE_DIR / "whale_latest.json"
    with open(latest_path, "w") as f:
        json.dump({
            "scan_date":      date.today().isoformat(),
            "wallets_scanned": len(wallets),
            "total_positions": len(all_positions),
            "cluster_alerts":  len([a for a in alerts if a.whale_count >= 3]),
            "top_alerts":      [a.to_dict() for a in alerts[:20]],
        }, f, indent=2)

    return {"positions": df_pos, "alerts": alerts}


if __name__ == "__main__":
    run_scan(20)   # quick scan with top 20 for testing
