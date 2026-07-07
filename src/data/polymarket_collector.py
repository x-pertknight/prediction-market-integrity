"""
src/data/polymarket_collector.py — Collect Polymarket market snapshots for future retraining.

Snapshots all active markets hourly. Detects resolutions by comparing
current market IDs against previous state. Stores data in monthly CSVs.

Usage:
  python src/data/polymarket_collector.py              # single run
  python src/data/polymarket_collector.py --daemon      # hourly loop

Output:
  data/raw/polymarket_history/markets_YYYY-MM.csv      # hourly snapshots
  data/raw/polymarket_history/resolutions.csv           # resolved markets
  data/raw/polymarket_history/.active_ids.json          # state file
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
HISTORY_DIR = Path("data/raw/polymarket_history")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("polymarket_collector")

# ── Schema ───────────────────────────────────────────────────────────────────

SNAPSHOT_COLS = [
    "snapshot_ts", "market_id", "condition_id", "question", "slug",
    "yes_price", "no_price", "volume", "volume_24hr", "liquidity",
    "start_date", "end_date", "active", "closed",
]

RESOLUTION_COLS = [
    "resolution_ts", "market_id", "question", "outcome", "final_volume",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_prices(market: dict) -> tuple[float, float]:
    """Parse outcomePrices JSON string → (yes_price, no_price)."""
    prices = market.get("outcomePrices")
    if isinstance(prices, str):
        try:
            parsed = json.loads(prices)
            if len(parsed) >= 2:
                return float(parsed[0]), float(parsed[1])
            if len(parsed) == 1:
                return float(parsed[0]), 1.0 - float(parsed[0])
        except (json.JSONDecodeError, IndexError, ValueError):
            pass
    if isinstance(prices, list) and len(prices) >= 2:
        return float(prices[0]), float(prices[1])
    return 0.0, 0.0


def monthly_filename(ts: datetime) -> str:
    """Generate monthly CSV filename: markets_YYYY-MM.csv"""
    return f"markets_{ts.strftime('%Y-%m')}.csv"


def snapshot_to_row(market: dict, ts: datetime) -> dict:
    """Convert a Gamma API market dict to a snapshot row."""
    yes_price, no_price = parse_prices(market)
    return {
        "snapshot_ts":  ts.isoformat(),
        "market_id":    market.get("id", ""),
        "condition_id": market.get("conditionId", ""),
        "question":     market.get("question", ""),
        "slug":         market.get("slug", ""),
        "yes_price":    round(yes_price, 6),
        "no_price":     round(no_price, 6),
        "volume":       float(market.get("volume", 0) or 0),
        "volume_24hr":  float(market.get("volume24hr", 0) or 0),
        "liquidity":    float(market.get("liquidity", 0) or 0),
        "start_date":   market.get("startDate", ""),
        "end_date":     market.get("endDate", ""),
        "active":       market.get("active", False),
        "closed":       market.get("closed", False),
    }


# ── API fetch ────────────────────────────────────────────────────────────────

def fetch_all_active(limit: int = 2000) -> list[dict]:
    """Fetch all active Polymarket markets via Gamma API pagination."""
    all_markets = []
    offset = 0

    while len(all_markets) < limit:
        try:
            r = requests.get(
                f"{GAMMA_API}/markets",
                params={"active": "true", "limit": 100, "offset": offset},
                timeout=15,
            )
            if r.status_code != 200:
                log.warning(f"Gamma API returned {r.status_code}")
                break
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break
            all_markets.extend(batch)
            if len(batch) < 100:
                break
            offset += len(batch)
            time.sleep(0.2)
        except Exception as e:
            log.warning(f"Gamma API error: {e}")
            break

    return all_markets


def fetch_market_detail(market_id: str) -> dict | None:
    """Fetch a single market to get resolution outcome."""
    try:
        r = requests.get(f"{GAMMA_API}/markets/{market_id}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ── State management ─────────────────────────────────────────────────────────

def load_active_ids(state_file: Path) -> dict[str, str]:
    """Load previous active market IDs → questions mapping."""
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_active_ids(state_file: Path, active: dict[str, str]) -> None:
    """Save current active market IDs."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(active, indent=2))


# ── Core logic ───────────────────────────────────────────────────────────────

def append_snapshot(rows: list[dict], out_dir: Path, ts: datetime) -> Path:
    """Append snapshot rows to monthly CSV. Creates header on first write."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = monthly_filename(ts)
    path = out_dir / fname
    is_new = not path.exists()

    with open(path, "a") as f:
        if is_new:
            f.write(",".join(SNAPSHOT_COLS) + "\n")
        for row in rows:
            vals = [str(row.get(col, "")) for col in SNAPSHOT_COLS]
            f.write(",".join(vals) + "\n")

    return path


def detect_resolutions(prev_ids: dict[str, str],
                       current_ids: set[str],
                       out_dir: Path) -> int:
    """Detect markets that vanished (resolved). Fetch outcome and log."""
    vanished = set(prev_ids.keys()) - current_ids
    if not vanished:
        return 0

    res_path = out_dir / "resolutions.csv"
    is_new = not res_path.exists()
    now = datetime.now(timezone.utc)
    count = 0

    with open(res_path, "a") as f:
        if is_new:
            f.write(",".join(RESOLUTION_COLS) + "\n")
        for mid in vanished:
            detail = fetch_market_detail(mid)
            outcome = ""
            final_vol = 0.0
            question = prev_ids.get(mid, "")
            if detail:
                outcome = detail.get("outcome", "")
                final_vol = float(detail.get("volume", 0) or 0)
                question = detail.get("question", question)
            vals = [
                now.isoformat(), mid, question,
                str(outcome), str(final_vol),
            ]
            f.write(",".join(vals) + "\n")
            count += 1
            time.sleep(0.1)  # be gentle with API

    return count


def run_once(out_dir: Path | None = None) -> int:
    """Single collection run. Returns number of markets snapshotted."""
    out_dir = out_dir or HISTORY_DIR
    state_file = out_dir / ".active_ids.json"

    markets = fetch_all_active()
    if not markets:
        log.warning("No markets fetched")
        return 0

    ts = datetime.now(timezone.utc)
    rows = [snapshot_to_row(m, ts) for m in markets]
    path = append_snapshot(rows, out_dir, ts)

    # Resolution detection
    prev_ids = load_active_ids(state_file)
    current_ids = {m.get("id", "") for m in markets}
    current_map = {m.get("id", ""): (m.get("question") or "")[:100] for m in markets}
    resolved = detect_resolutions(prev_ids, current_ids, out_dir)

    # Save current state
    save_active_ids(state_file, current_map)

    log.info(f"Snapshot: {len(rows)} markets → {path.name} | "
             f"{resolved} resolutions detected")
    return len(rows)


def daemon_loop(interval: int = 3600) -> None:
    """Run collection every `interval` seconds (default: 1 hour)."""
    log.info(f"Collector daemon started | interval={interval}s")
    while True:
        try:
            run_once()
        except KeyboardInterrupt:
            log.info("Stopped by user")
            break
        except Exception as e:
            log.error(f"Collection error: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket data collector")
    parser.add_argument("--daemon", action="store_true",
                        help="Run in daemon mode (hourly loop)")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Daemon interval in seconds (default: 3600)")
    args = parser.parse_args()

    if args.daemon:
        daemon_loop(args.interval)
    else:
        count = run_once()
        print(f"Collected {count} markets")
