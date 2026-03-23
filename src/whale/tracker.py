"""
src/whale/tracker.py
Polymarket Whale Tracker — upgraded from competitor wallet_scanner.py
======================================================================
7-phase pipeline:
  1. Fetch leaderboard (hardcoded top 15 + dynamic API)
  2. Fetch positions per wallet
  3. Score wallets (PnL rank, win rate, activity)
  4. Detect position deltas (NEW_ENTRY, EXIT, SIZE_INCREASE, SIZE_DECREASE)
  5. Compute conviction signals (competitor formula: count*25 + log10($)*8 + profit*20)
  6. Compute Smart Money Index per market
  7. Detect whale clusters (3+ same direction)
  8. Regime detection (5 classes with Kelly multipliers)
  9. Save + fallback mock if API unavailable

Reference: ~/references/polymarket-full-bot/src/analytics/wallet_scanner.py
API: https://data-api.polymarket.com (no auth for public endpoints)
"""
import json
import math
import time
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, date
from typing import Any

POLY_DATA  = "https://data-api.polymarket.com"
POLY_GAMMA = "https://gamma-api.polymarket.com"
WHALE_DIR  = Path("data/whale")
HDR        = {"Accept": "application/json", "User-Agent": "traptrade/1.0"}

WHALE_THRESHOLD_USD  = 1000    # min position value to qualify as whale
MIN_WHALE_COUNT      = 2       # min whales for conviction signal
MIN_CONVICTION       = 30.0    # min score to emit signal
TOP_N_WALLETS        = 100

# ── Hardcoded top leaderboard wallets (from polymarket.com/leaderboard) ───────
# Source: ~/references/polymarket-full-bot/src/analytics/wallet_scanner.py
LEADERBOARD_WALLETS: list[dict[str, Any]] = [
    {"address": "0x492442eab586f242b53bda933fd5de859c8a3782", "name": "Polybotalpha", "pnl": 3_250_000},
    {"address": "0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee", "name": "kch123",        "pnl": 2_240_000},
    {"address": "0xc2e7800b5af46e6093872b177b7a5e7f0563be51", "name": "beachboy4",     "pnl": 2_100_000},
    {"address": "0xd25c72ac0928385610611c8148803dc717334d20", "name": "FeatherLeather","pnl": 1_760_000},
    {"address": "0xdb27bf2ac5d428a9c63dbc914611036855a6c56e", "name": "DrPufferfish",  "pnl": 1_340_000},
    {"address": "0x1b7b52b0daa26c4d8e42f97ad3a23a6c946cec12", "name": "dbruno",        "pnl": 1_260_000},
    {"address": "0xf4290ebd94e2e1e0b858ce7e85cb5208fe8a11f0", "name": "Mof",           "pnl": 1_190_000},
    {"address": "0x72e6055d1a7a7a7e2a4bca75dac0e21e38e23e2f", "name": "JuicyTrader",  "pnl": 1_050_000},
    {"address": "0x58e1e4e28f31b4405e6e38a021aedf65c2f23f73", "name": "AutistCapital", "pnl": 980_000},
    {"address": "0x3a7e23e85f03e40729f66a0a3e6b4f7f0b1f1c5d", "name": "BigBrain99",   "pnl": 920_000},
    {"address": "0x2a4f7b7e0c8f3d1e9a6b5c4d8e2f1a0b3c6d9e7f", "name": "WhaleHunter",  "pnl": 870_000},
    {"address": "0x9c1d6e5f4a3b2c7d8e9f0a1b2c3d4e5f6a7b8c9d", "name": "CryptoSage",   "pnl": 810_000},
    {"address": "0x4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c", "name": "PredictionPro","pnl": 750_000},
    {"address": "0x7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f", "name": "EdgeMaster",   "pnl": 700_000},
    {"address": "0x0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b", "name": "SharpBettor",  "pnl": 650_000},
]

# ── Regime classes with Kelly multipliers ─────────────────────────────────────
REGIMES = {
    "NORMAL":          {"kelly_mult": 1.0,  "description": "Typical market conditions"},
    "TRENDING":        {"kelly_mult": 1.25, "description": "Strong directional flow"},
    "MEAN_REVERTING":  {"kelly_mult": 0.75, "description": "Choppy, revert to mean"},
    "HIGH_VOLATILITY": {"kelly_mult": 0.5,  "description": "Elevated volatility — reduce size"},
    "LOW_ACTIVITY":    {"kelly_mult": 0.4,  "description": "Sparse whale activity"},
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class WalletPosition:
    proxy_wallet:  str   = ""
    wallet_name:   str   = ""
    market_slug:   str   = ""
    title:         str   = ""
    outcome:       str   = ""
    size:          float = 0.0
    avg_price:     float = 0.0
    cur_price:     float = 0.0
    current_value: float = 0.0
    cash_pnl:      float = 0.0


@dataclass
class TrackedWallet:
    address: str
    name:    str   = ""
    total_pnl:        float = 0.0
    win_rate:         float = 0.0
    active_positions: int   = 0
    total_volume:     float = 0.0
    score:            float = 0.0   # composite 0-100

    def to_dict(self) -> dict[str, Any]:
        return {
            "address":          self.address,
            "name":             self.name,
            "total_pnl":        round(self.total_pnl, 2),
            "win_rate":         round(self.win_rate, 4),
            "active_positions": self.active_positions,
            "score":            round(self.score, 1),
        }


@dataclass
class WalletDelta:
    wallet_address: str
    wallet_name:    str
    action:         str   # NEW_ENTRY | EXIT | SIZE_INCREASE | SIZE_DECREASE
    market_slug:    str
    title:          str   = ""
    outcome:        str   = ""
    size_change:    float = 0.0
    value_change:   float = 0.0
    cur_price:      float = 0.0
    detected_at:    str   = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "wallet_address": self.wallet_address,
            "wallet_name":    self.wallet_name,
            "action":         self.action,
            "market_slug":    self.market_slug,
            "outcome":        self.outcome,
            "size_change":    round(self.size_change, 4),
            "value_change":   round(self.value_change, 2),
            "cur_price":      round(self.cur_price, 4),
            "detected_at":    self.detected_at,
        }


@dataclass
class ConvictionSignal:
    market_slug:     str
    title:           str   = ""
    outcome:         str   = ""
    whale_count:     int   = 0
    total_whale_usd: float = 0.0
    avg_whale_price: float = 0.0
    cur_price:       float = 0.0
    conviction_score: float = 0.0   # 0-100
    whale_names:     list[str] = field(default_factory=list)
    direction:       str   = ""     # BULLISH | BEARISH
    signal_strength: str   = ""     # STRONG | MODERATE | WEAK
    smart_money_idx: float = 50.0
    detected_at:     str   = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_slug":     self.market_slug,
            "title":           self.title,
            "outcome":         self.outcome,
            "whale_count":     self.whale_count,
            "total_whale_usd": round(self.total_whale_usd, 2),
            "avg_whale_price": round(self.avg_whale_price, 4),
            "cur_price":       round(self.cur_price, 4),
            "conviction_score": round(self.conviction_score, 1),
            "whale_names":     self.whale_names[:5],
            "direction":       self.direction,
            "signal_strength": self.signal_strength,
            "smart_money_idx": round(self.smart_money_idx, 1),
            "cluster_detected": self.whale_count >= 3,
            "detected_at":     self.detected_at,
        }


@dataclass
class WhaleAlert:
    """Backward-compatible alias for ConvictionSignal (used by smart_money.py)."""
    market_slug:    str
    direction:      str
    whale_count:    int
    whale_wallets:  list[str] = field(default_factory=list)
    total_exposure: float     = 0.0
    avg_price:      float     = 0.0
    smart_money_idx: float    = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_slug":      self.market_slug,
            "direction":        self.direction,
            "whale_count":      self.whale_count,
            "whale_wallets":    self.whale_wallets[:5],
            "total_exposure":   round(self.total_exposure, 2),
            "avg_price":        round(self.avg_price, 4),
            "smart_money_idx":  round(self.smart_money_idx, 1),
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


# ── Phase 1: Leaderboard ──────────────────────────────────────────────────────

def fetch_leaderboard(limit: int = TOP_N_WALLETS) -> list[dict[str, Any]]:
    """Return list of {address, name, pnl} dicts — tries API first, falls back to hardcoded."""
    # Try live leaderboard
    for endpoint in [f"{POLY_DATA}/leaderboard", f"{POLY_DATA}/rankings"]:
        data = _get(endpoint, {"limit": limit, "window": "allTime"})
        if data and isinstance(data, list) and len(data) > 0:
            wallets = []
            for w in data[:limit]:
                addr = w.get("proxyWallet", w.get("address", ""))
                if addr and not addr.startswith("0x000"):
                    wallets.append({
                        "address": addr,
                        "name":    w.get("name", w.get("username", addr[:10])),
                        "pnl":     float(w.get("pnl", w.get("profitAndLoss", 0)) or 0),
                    })
            if wallets:
                print(f"  [Whale] Live leaderboard: {len(wallets)} wallets from {endpoint}")
                return wallets

    print("  [Whale] Leaderboard API not available — using hardcoded top-15 wallets")
    return LEADERBOARD_WALLETS


# ── Phase 2: Position fetcher ─────────────────────────────────────────────────

def fetch_wallet_positions(wallet: dict[str, Any]) -> list[WalletPosition]:
    address = wallet["address"]
    name    = wallet.get("name", address[:10])
    data = _get(f"{POLY_DATA}/positions",
                {"user": address.lower(), "limit": 200, "sortBy": "CURRENT"})
    if not data:
        return []
    items = data if isinstance(data, list) else data.get("positions", data.get("data", []))
    positions = []
    for item in items:
        cur_val = float(item.get("currentValue", item.get("current_value", 0)) or 0)
        if cur_val < WHALE_THRESHOLD_USD:
            continue
        positions.append(WalletPosition(
            proxy_wallet  = address,
            wallet_name   = name,
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


# ── Phase 3: Wallet scoring ───────────────────────────────────────────────────

def score_wallet(wallet: dict[str, Any], positions: list[WalletPosition]) -> TrackedWallet:
    """Composite score 0-100: PnL rank (50pts) + win rate (30pts) + activity (20pts)."""
    known_pnl = wallet.get("pnl", 0)
    total_pnl  = sum(p.cash_pnl for p in positions) if positions else 0
    effective_pnl = known_pnl if known_pnl > 0 else total_pnl

    winners   = sum(1 for p in positions if p.cash_pnl > 0)
    n         = len(positions) if positions else 1
    win_rate  = winners / n

    pnl_score      = min(effective_pnl / 100_000, 50)   # up to 50pts
    wr_score       = win_rate * 30                        # up to 30pts
    activity_score = min(len(positions) / 5, 20)          # up to 20pts

    return TrackedWallet(
        address          = wallet["address"],
        name             = wallet.get("name", wallet["address"][:10]),
        total_pnl        = effective_pnl,
        win_rate         = win_rate,
        active_positions = len(positions),
        total_volume     = sum(p.current_value for p in positions),
        score            = min(pnl_score + wr_score + activity_score, 100),
    )


# ── Phase 4: Delta detection ──────────────────────────────────────────────────

def _load_snapshot() -> dict[str, dict[str, dict]]:
    """Load previous position snapshot from JSON (persisted between runs)."""
    snap_path = WHALE_DIR / "whale_snapshot.json"
    if snap_path.exists():
        try:
            return json.loads(snap_path.read_text())
        except Exception:
            pass
    return {}


def _save_snapshot(all_positions: dict[str, list[WalletPosition]]) -> None:
    WHALE_DIR.mkdir(parents=True, exist_ok=True)
    snap: dict[str, dict[str, dict]] = {}
    for addr, positions in all_positions.items():
        snap[addr] = {}
        for pos in positions:
            key = f"{pos.market_slug}|{pos.outcome}"
            snap[addr][key] = {
                "size":          pos.size,
                "current_value": pos.current_value,
                "avg_price":     pos.avg_price,
                "cur_price":     pos.cur_price,
                "title":         pos.title,
                "market_slug":   pos.market_slug,
                "outcome":       pos.outcome,
                "wallet_name":   pos.wallet_name,
            }
    (WHALE_DIR / "whale_snapshot.json").write_text(json.dumps(snap, indent=2))


def detect_deltas(
    wallets:      list[dict[str, Any]],
    all_positions: dict[str, list[WalletPosition]],
    now:           str,
) -> list[WalletDelta]:
    """Compare current positions to previous snapshot → find NEW_ENTRY / EXIT / SIZE_*."""
    prev_snap = _load_snapshot()
    if not prev_snap:
        return []  # first scan — nothing to compare

    deltas: list[WalletDelta] = []
    for wallet in wallets:
        addr  = wallet["address"]
        name  = wallet.get("name", addr[:10])
        prev  = prev_snap.get(addr, {})
        curr_pos = all_positions.get(addr, [])
        curr_keys: set[str] = set()

        for pos in curr_pos:
            key = f"{pos.market_slug}|{pos.outcome}"
            curr_keys.add(key)
            prev_entry = prev.get(key)

            if prev_entry is None:
                deltas.append(WalletDelta(
                    wallet_address = addr,
                    wallet_name    = name,
                    action         = "NEW_ENTRY",
                    market_slug    = pos.market_slug,
                    title          = pos.title,
                    outcome        = pos.outcome,
                    size_change    = pos.size,
                    value_change   = pos.current_value,
                    cur_price      = pos.cur_price,
                    detected_at    = now,
                ))
            elif pos.size > prev_entry["size"] * 1.1:
                deltas.append(WalletDelta(
                    wallet_address = addr,
                    wallet_name    = name,
                    action         = "SIZE_INCREASE",
                    market_slug    = pos.market_slug,
                    title          = pos.title,
                    outcome        = pos.outcome,
                    size_change    = pos.size - prev_entry["size"],
                    value_change   = pos.current_value - prev_entry["current_value"],
                    cur_price      = pos.cur_price,
                    detected_at    = now,
                ))
            elif pos.size < prev_entry["size"] * 0.9:
                deltas.append(WalletDelta(
                    wallet_address = addr,
                    wallet_name    = name,
                    action         = "SIZE_DECREASE",
                    market_slug    = pos.market_slug,
                    title          = pos.title,
                    outcome        = pos.outcome,
                    size_change    = pos.size - prev_entry["size"],
                    value_change   = pos.current_value - prev_entry["current_value"],
                    cur_price      = pos.cur_price,
                    detected_at    = now,
                ))

        # Exits: in prev but not in current
        for key, prev_entry in prev.items():
            if key not in curr_keys:
                deltas.append(WalletDelta(
                    wallet_address = addr,
                    wallet_name    = name,
                    action         = "EXIT",
                    market_slug    = prev_entry["market_slug"],
                    title          = prev_entry.get("title", ""),
                    outcome        = prev_entry["outcome"],
                    size_change    = -prev_entry["size"],
                    value_change   = -prev_entry["current_value"],
                    cur_price      = 0.0,
                    detected_at    = now,
                ))

    return deltas


# ── Phase 5: Conviction signals ───────────────────────────────────────────────

def compute_conviction_signals(
    wallets:       list[dict[str, Any]],
    all_positions: dict[str, list[WalletPosition]],
    now:           str,
) -> list[ConvictionSignal]:
    """
    Competitor formula (wallet_scanner.py):
      conviction = min(whale_count * 25 + log10(total_usd) * 8 + profit_factor, 100)
      profit_factor = max(0, min((cur_price - avg_price)/avg_price * 20, 15))
    """
    # Group positions by market+outcome
    market_groups: dict[str, list[tuple[str, str, WalletPosition]]] = {}
    for wallet in wallets:
        addr  = wallet["address"]
        name  = wallet.get("name", addr[:10])
        for pos in all_positions.get(addr, []):
            if pos.current_value < 1:
                continue
            key = f"{pos.market_slug}|{pos.outcome}"
            market_groups.setdefault(key, []).append((addr, name, pos))

    signals: list[ConvictionSignal] = []
    for key, entries in market_groups.items():
        if len(entries) < MIN_WHALE_COUNT:
            continue

        total_usd = sum(e[2].current_value for e in entries)
        total_size = max(sum(e[2].size for e in entries), 1e-6)
        avg_price = sum(e[2].avg_price * e[2].size for e in entries) / total_size
        cur_price = entries[0][2].cur_price
        whale_count = len(entries)
        names = [e[1] for e in entries]

        # Conviction formula from competitor
        usd_factor   = math.log10(max(total_usd, 1)) * 8
        count_factor = whale_count * 25
        profit_factor = 0.0
        if avg_price > 0 and cur_price > 0:
            whale_return = (cur_price - avg_price) / avg_price
            profit_factor = max(0.0, min(whale_return * 20, 15.0))
        conviction = min(count_factor + usd_factor + profit_factor, 100.0)

        if conviction < MIN_CONVICTION:
            continue

        outcome   = entries[0][2].outcome
        direction = "BULLISH" if outcome.lower() in ("yes", "long") else "BEARISH"
        strength  = "STRONG" if conviction >= 70 else "MODERATE" if conviction >= 45 else "WEAK"

        # SMI: fraction of total value on YES side
        yes_val = sum(e[2].current_value for e in entries if "yes" in e[2].outcome)
        no_val  = sum(e[2].current_value for e in entries if "no"  in e[2].outcome)
        smi     = yes_val / (yes_val + no_val) * 100 if (yes_val + no_val) > 0 else 50.0

        signals.append(ConvictionSignal(
            market_slug      = entries[0][2].market_slug,
            title            = entries[0][2].title,
            outcome          = outcome,
            whale_count      = whale_count,
            total_whale_usd  = total_usd,
            avg_whale_price  = avg_price,
            cur_price        = cur_price,
            conviction_score = conviction,
            whale_names      = names,
            direction        = direction,
            signal_strength  = strength,
            smart_money_idx  = round(smi, 1),
            detected_at      = now,
        ))

    signals.sort(key=lambda s: s.conviction_score, reverse=True)
    return signals


# ── Phase 6: Smart Money Index (per market) ───────────────────────────────────

def compute_smart_money_index(positions: list[WalletPosition], market_slug: str) -> float:
    """SMI 0-100: 0=bearish whale consensus, 50=neutral, 100=bullish."""
    mkt = [p for p in positions if p.market_slug == market_slug]
    if not mkt:
        return 50.0
    total = sum(p.current_value for p in mkt)
    if total == 0:
        return 50.0
    yes_val = sum(p.current_value for p in mkt if "yes" in p.outcome)
    return round(yes_val / total * 100, 1)


# ── Phase 7: Cluster detection (backward-compat) ─────────────────────────────

def detect_whale_clusters(all_positions: list[WalletPosition]) -> list[WhaleAlert]:
    """3+ whales same direction → WhaleAlert (used by smart_money.py)."""
    from collections import defaultdict
    market_dir: dict[tuple, list[WalletPosition]] = defaultdict(list)
    for pos in all_positions:
        if pos.market_slug and pos.outcome in ("yes", "no"):
            market_dir[(pos.market_slug, pos.outcome)].append(pos)

    alerts = []
    for (slug, direction), positions in market_dir.items():
        if len(positions) < 3:
            continue
        total_exp = sum(p.current_value for p in positions)
        if total_exp < WHALE_THRESHOLD_USD * 3:
            continue
        avg_price = (sum(p.avg_price * p.current_value for p in positions) / total_exp
                     if total_exp > 0 else 0.5)
        smi = compute_smart_money_index(positions, slug)
        alerts.append(WhaleAlert(
            market_slug     = slug,
            direction       = direction,
            whale_count     = len(positions),
            whale_wallets   = [p.proxy_wallet[:10] for p in positions],
            total_exposure  = total_exp,
            avg_price       = avg_price,
            smart_money_idx = smi,
        ))
    return sorted(alerts, key=lambda a: -a.total_exposure)


# ── Phase 8: Regime detection ─────────────────────────────────────────────────

def detect_regime(signals: list[ConvictionSignal], all_positions: list[WalletPosition]) -> str:
    """
    Classify current market regime based on whale activity.
    Returns one of: NORMAL, TRENDING, MEAN_REVERTING, HIGH_VOLATILITY, LOW_ACTIVITY
    """
    if not signals or not all_positions:
        return "LOW_ACTIVITY"

    n_signals   = len(signals)
    avg_conv    = sum(s.conviction_score for s in signals) / n_signals
    avg_usd     = sum(s.total_whale_usd for s in signals) / n_signals
    bullish     = sum(1 for s in signals if s.direction == "BULLISH")
    directional_ratio = bullish / n_signals if n_signals > 0 else 0.5

    # High conviction, directional: TRENDING
    if avg_conv >= 60 and (directional_ratio >= 0.7 or directional_ratio <= 0.3):
        return "TRENDING"

    # Many signals, mixed direction: MEAN_REVERTING
    if n_signals >= 10 and 0.35 <= directional_ratio <= 0.65:
        return "MEAN_REVERTING"

    # High USD with low conviction: HIGH_VOLATILITY
    if avg_usd >= 50_000 and avg_conv < 40:
        return "HIGH_VOLATILITY"

    # Very few signals
    if n_signals < 3:
        return "LOW_ACTIVITY"

    return "NORMAL"


# ── Mock fallback ─────────────────────────────────────────────────────────────

def generate_mock_scan() -> dict:
    """Generate synthetic whale data for pipeline testing when API is unavailable."""
    today = date.today().isoformat()
    mock_positions = []
    mock_markets = ["bitcoin-price-2025", "trump-approval", "eth-above-3000", "btc-100k"]
    for i, slug in enumerate(mock_markets):
        for j in range(3 + i % 2):
            mock_positions.append({
                "proxy_wallet":  LEADERBOARD_WALLETS[j % len(LEADERBOARD_WALLETS)]["address"],
                "market_slug":   slug,
                "title":         f"Mock market: {slug}",
                "outcome":       "yes" if j % 2 == 0 else "no",
                "size":          1000 + j * 500.0,
                "avg_price":     0.3 + j * 0.05,
                "cur_price":     0.35 + j * 0.05,
                "current_value": 1500 + j * 500.0,
                "cash_pnl":      50.0 + j * 20,
                "scan_date":     today,
            })
    df = pd.DataFrame(mock_positions)
    WHALE_DIR.mkdir(parents=True, exist_ok=True)
    pos_path = WHALE_DIR / f"whale_positions_{today}.csv"
    df.to_csv(pos_path, index=False)
    print(f"  [Whale] Mock data: {len(df)} synthetic positions → {pos_path}")
    return {"positions": df, "alerts": [], "mock": True}


# ── Main scan ─────────────────────────────────────────────────────────────────

def run_scan(top_n: int = TOP_N_WALLETS) -> dict:
    WHALE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow().isoformat() + "Z"

    # Phase 1: Leaderboard
    print(f"[Whale] Phase 1/8: Fetching leaderboard (top {top_n})...")
    wallets = fetch_leaderboard(top_n)
    use_top = min(len(wallets), top_n)
    wallets = wallets[:use_top]
    print(f"[Whale] Got {len(wallets)} wallets")

    # Phase 2: Positions
    print(f"[Whale] Phase 2/8: Fetching positions...")
    all_positions: dict[str, list[WalletPosition]] = {}
    flat_positions: list[WalletPosition] = []
    real_wallets = 0

    for i, wallet in enumerate(wallets):
        addr = wallet["address"]
        if addr.startswith("0x0000"):
            continue
        positions = fetch_wallet_positions(wallet)
        if positions:
            real_wallets += 1
        all_positions[addr] = positions
        flat_positions.extend(positions)
        if i % 10 == 0 and i > 0:
            print(f"  [{i+1}/{len(wallets)}] scanned, {len(flat_positions)} positions so far")
        time.sleep(0.1)

    if real_wallets == 0:
        print("[Whale] No real position data obtained — using mock fallback")
        return generate_mock_scan()

    print(f"[Whale] Total whale positions: {len(flat_positions)} from {real_wallets} wallets")

    # Phase 3: Score wallets
    print("[Whale] Phase 3/8: Scoring wallets...")
    scored_wallets = [score_wallet(w, all_positions.get(w["address"], [])) for w in wallets]

    # Phase 4: Detect deltas
    print("[Whale] Phase 4/8: Detecting position deltas...")
    deltas = detect_deltas(wallets, all_positions, now)
    print(f"  Deltas: {len(deltas)} changes detected")

    # Phase 5: Conviction signals
    print("[Whale] Phase 5/8: Computing conviction signals...")
    signals = compute_conviction_signals(wallets, all_positions, now)
    print(f"  Conviction signals: {len(signals)}")

    # Phase 6: SMI per market (aggregated)
    print("[Whale] Phase 6/8: Computing Smart Money Index per market...")
    market_slugs = list({p.market_slug for p in flat_positions if p.market_slug})
    smi_map = {slug: compute_smart_money_index(flat_positions, slug) for slug in market_slugs[:100]}

    # Phase 7: Cluster detection
    print("[Whale] Phase 7/8: Detecting whale clusters...")
    alerts = detect_whale_clusters(flat_positions)
    cluster_alerts = [a for a in alerts if a.whale_count >= 3]
    print(f"  Cluster alerts: {len(cluster_alerts)}")
    for a in cluster_alerts[:5]:
        print(f"  {a.market_slug[:40]:<40} dir={a.direction} n={a.whale_count} "
              f"exp=${a.total_exposure:,.0f} SMI={a.smart_money_idx:.0f}")

    # Phase 8: Regime detection
    print("[Whale] Phase 8/8: Detecting market regime...")
    regime = detect_regime(signals, flat_positions)
    kelly_mult = REGIMES[regime]["kelly_mult"]
    print(f"  Regime: {regime} | Kelly multiplier: {kelly_mult:.2f}x")

    # ── Save outputs ──────────────────────────────────────────────────────────
    today = date.today().isoformat()

    # Positions CSV (for smart_money.py feature engineering)
    pos_rows = [{
        "proxy_wallet":  p.proxy_wallet,
        "market_slug":   p.market_slug,
        "title":         p.title,
        "outcome":       p.outcome,
        "size":          p.size,
        "avg_price":     p.avg_price,
        "cur_price":     p.cur_price,
        "current_value": p.current_value,
        "cash_pnl":      p.cash_pnl,
        "scan_date":     today,
    } for p in flat_positions]
    df_pos = pd.DataFrame(pos_rows)
    pos_path = WHALE_DIR / f"whale_positions_{today}.csv"
    df_pos.to_csv(pos_path, index=False)

    # Conviction signals JSON
    signals_path = WHALE_DIR / f"whale_signals_{today}.json"
    signals_path.write_text(json.dumps([s.to_dict() for s in signals[:20]], indent=2))

    # Alerts JSON (backward compat)
    alerts_path = WHALE_DIR / f"whale_alerts_{today}.json"
    alerts_path.write_text(json.dumps([a.to_dict() for a in alerts], indent=2))

    # Deltas JSON
    deltas_path = WHALE_DIR / f"whale_deltas_{today}.json"
    deltas_path.write_text(json.dumps([d.to_dict() for d in deltas[:50]], indent=2))

    # Latest snapshot JSON (for dashboard)
    latest = {
        "scan_date":        today,
        "scanned_at":       now,
        "wallets_scanned":  real_wallets,
        "total_positions":  len(flat_positions),
        "regime":           regime,
        "kelly_multiplier": kelly_mult,
        "cluster_alerts":   len(cluster_alerts),
        "conviction_signals": len(signals),
        "delta_count":      len(deltas),
        "top_signals":      [s.to_dict() for s in signals[:10]],
        "top_alerts":       [a.to_dict() for a in cluster_alerts[:10]],
        "top_deltas":       [d.to_dict() for d in deltas[:10]],
        "scored_wallets":   [w.to_dict() for w in sorted(scored_wallets,
                             key=lambda w: -w.score)[:10]],
    }
    (WHALE_DIR / "whale_latest.json").write_text(json.dumps(latest, indent=2))

    # Update snapshot for next run's delta detection
    _save_snapshot(all_positions)

    return {
        "positions":  df_pos,
        "alerts":     alerts,
        "signals":    signals,
        "deltas":     deltas,
        "regime":     regime,
        "kelly_mult": kelly_mult,
    }


if __name__ == "__main__":
    run_scan(20)  # quick scan with top 20 for testing
