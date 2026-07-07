"""
src/alpha/negrisk_scanner.py — READ-ONLY scanner for negRisk (multi-outcome)
sum-of-prices arbitrage on Polymarket.

Mechanism (arXiv 2508.03474, replicated live 2026-07-06): on an event whose
outcomes are mutually exclusive, sum(best YES asks) < $1 means buying the
full outcome set locks a risk-free $1 payout at resolution; sum(best YES
bids) > $1 means the set is overpriced and shorting it (buying the NO set)
locks the excess.

The long side is ONLY safe when the outcome set is provably exhaustive.
On an incomplete set the missing probability mass belongs to UNLISTED
outcomes, not to profit (live false positive: "Dem Nominee 2028" showed
sum-asks 0.93 with major candidates unlisted). Gate: a catch-all leg
("Other", "Field", ...) or an event id manually whitelisted in
data/negrisk_whitelist.json after human review. The short side needs no
gate — mutual exclusivity alone caps the set payout at $1.

Fees: each leg's exact Gamma `feeSchedule` is applied on taker BUYS
(long side = YES buys, short side = NO buys). Sells and makers are free.

This module places NO orders and needs NO keys. It appends one row per
analyzed event to data/negrisk_log.csv so 1-2 weeks of margin evidence
exists before any executor is considered.

Usage:
    python src/alpha/negrisk_scanner.py --once
    python src/alpha/negrisk_scanner.py --loop 60
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk.fees import polymarket_fee_from_schedule  # noqa: E402

GAMMA_BASE = "https://gamma-api.polymarket.com"
CSV_PATH = Path("data/negrisk_log.csv")
WHITELIST_PATH = Path("data/negrisk_whitelist.json")

MIN_LEGS = 3          # below this, binary/degenerate — not a multi-outcome set
DEFAULT_THRESHOLD = 0.02  # 2c net margin per $1 set, per the verified paper bound

# Legs whose title marks the set as exhaustive by construction
_CATCHALL_RE = re.compile(
    r"\b(other|another|field|none of|no one|nobody|anyone else|any other)\b",
    re.IGNORECASE,
)

CSV_COLUMNS = [
    "timestamp", "event_id", "title", "n_legs",
    "sum_ask", "sum_bid",
    "margin_long_gross", "margin_long_net",
    "margin_short_gross", "margin_short_net",
    "fee_long", "fee_short", "n_fee_legs",
    "exhaustive", "exhaustive_reason",
    "tradable_long", "tradable_short",
    "volume24hr", "min_leg_ask", "max_leg_ask",
]


# ── Data access (read-only) ─────────────────────────────────────────────────────

def fetch_negrisk_events(max_events: int = 300) -> list[dict]:
    """Fetch active negRisk events from the Gamma API, most-traded first."""
    events: list[dict] = []
    offset = 0
    while offset < max_events:
        try:
            r = requests.get(
                f"{GAMMA_BASE}/events",
                params={
                    "active": "true", "closed": "false",
                    "order": "volume24hr", "ascending": "false",
                    "limit": 100, "offset": offset,
                },
                timeout=15,
            )
            if r.status_code != 200:
                break
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break
            events.extend(e for e in batch if e.get("negRisk"))
            if len(batch) < 100:
                break
            offset += len(batch)
            time.sleep(0.2)
        except Exception as e:
            print(f"  [NegRisk] fetch error: {e}")
            break
    return events


def load_whitelist(path: Path = WHITELIST_PATH) -> set[str]:
    """Event ids verified as exhaustive outcome sets (see
    docs/negrisk-whitelist-dossier.md for the review protocol)."""
    try:
        return {str(x) for x in json.loads(path.read_text())}
    except Exception:
        return set()


def fetch_event(event_id: str) -> dict | None:
    """Fetch a single event by id (used by the paper resolver)."""
    try:
        r = requests.get(f"{GAMMA_BASE}/events/{event_id}", timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ── Pure analysis (no network — unit tested) ────────────────────────────────────

def live_legs(event: dict) -> list[dict]:
    """Outcome legs still tradable: active, not closed, accepting orders."""
    return [
        m for m in event.get("markets", [])
        if m.get("active") and not m.get("closed") and m.get("acceptingOrders")
        and m.get("bestAsk") is not None and m.get("bestBid") is not None
    ]


def has_catchall(legs: list[dict]) -> bool:
    """True if any leg is an explicit catch-all outcome ('Other', 'Field', …)."""
    for m in legs:
        text = f"{m.get('groupItemTitle', '')} {m.get('question', '')}"
        if _CATCHALL_RE.search(text):
            return True
    return False


def _leg_fee(price: float, leg: dict) -> float:
    """Taker-buy fee for one leg from its exact Gamma feeSchedule."""
    if not leg.get("feesEnabled"):
        return 0.0
    return polymarket_fee_from_schedule(price, leg.get("feeSchedule"))


def analyze_event(event: dict, whitelist: set[str] | None = None,
                  threshold: float = DEFAULT_THRESHOLD) -> dict | None:
    """Compute set-arbitrage margins for one negRisk event.

    Long side:  buy YES on every leg → cost = Σ(ask + fee(ask)).
                margin = 1 − cost. Tradable only if the set is exhaustive.
    Short side: buy NO on every leg → cost = Σ((1 − bid) + fee(1 − bid)).
                A complete NO set pays n_legs − 1, so
                margin = Σbid − 1 − Σfee. Safe under mutual exclusivity alone.
    """
    legs = live_legs(event)
    if len(legs) < MIN_LEGS:
        return None

    asks = [float(m["bestAsk"]) for m in legs]
    bids = [float(m["bestBid"]) for m in legs]
    sum_ask, sum_bid = sum(asks), sum(bids)

    fee_long = sum(_leg_fee(a, m) for a, m in zip(asks, legs))
    fee_short = sum(_leg_fee(1.0 - b, m) for b, m in zip(bids, legs))

    margin_long_gross = 1.0 - sum_ask
    margin_long_net = margin_long_gross - fee_long
    margin_short_gross = sum_bid - 1.0
    margin_short_net = margin_short_gross - fee_short

    wl = whitelist or set()
    if str(event.get("id", "")) in wl:
        exhaustive, reason = True, "whitelist"
    elif has_catchall(legs):
        exhaustive, reason = True, "catchall"
    else:
        exhaustive, reason = False, "unknown"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event_id": str(event.get("id", "")),
        "title": (event.get("title") or "").strip(),
        "n_legs": len(legs),
        "sum_ask": round(sum_ask, 4),
        "sum_bid": round(sum_bid, 4),
        "margin_long_gross": round(margin_long_gross, 4),
        "margin_long_net": round(margin_long_net, 4),
        "margin_short_gross": round(margin_short_gross, 4),
        "margin_short_net": round(margin_short_net, 4),
        "fee_long": round(fee_long, 4),
        "fee_short": round(fee_short, 4),
        "n_fee_legs": sum(1 for m in legs if m.get("feesEnabled")),
        "exhaustive": exhaustive,
        "exhaustive_reason": reason,
        "tradable_long": exhaustive and margin_long_net >= threshold,
        "tradable_short": margin_short_net >= threshold,
        "volume24hr": round(float(event.get("volume24hr", 0) or 0), 0),
        "min_leg_ask": round(min(asks), 4),
        "max_leg_ask": round(max(asks), 4),
    }


# ── Daily Telegram digest ───────────────────────────────────────────────────────

DIGEST_STATE_PATH = Path("data/negrisk_digest_state.json")
DIGEST_HOUR_UTC = 7  # 09:00 CEST


def compose_digest(log_rows, paper_trades, open_positions: int,
                   today: str) -> str:
    """One-message daily summary of the negRisk evidence (pure, testable).

    log_rows / paper_trades: lists of dicts (today's rows)."""
    def _f(row, key):
        try:
            return float(row.get(key, 0))
        except (TypeError, ValueError):
            return 0.0

    n_rows = len(log_rows)
    events = {r.get("event_id") for r in log_rows}
    best_long = max((r for r in log_rows if str(r.get("exhaustive")) == "True"),
                    key=lambda r: _f(r, "margin_long_net"), default=None)
    best_short = max(log_rows, key=lambda r: _f(r, "margin_short_net"),
                     default=None)
    windows = sum(1 for r in log_rows
                  if str(r.get("tradable_long")) == "True"
                  or str(r.get("tradable_short")) == "True")
    gated = max((r for r in log_rows if str(r.get("exhaustive")) == "False"
                 and _f(r, "margin_long_net") > 0),
                key=lambda r: _f(r, "margin_long_net"), default=None)

    opens = [t for t in paper_trades if t.get("action") == "OPEN"]
    resolves = [t for t in paper_trades if t.get("action") == "RESOLVE"]
    resolved_pnl = sum(_f(t, "total_pnl") for t in resolves)

    lines = [f"🧭 <b>NEGRISK DIGEST</b> — {today}",
             f"📊 {n_rows} osservazioni su {len(events)} eventi"]
    if best_long is not None:
        lines.append(f"🟢 Best LONG net (esaustivo): "
                     f"{_f(best_long, 'margin_long_net'):+.4f} — "
                     f"{str(best_long.get('title'))[:40]}")
    if best_short is not None:
        lines.append(f"🔴 Best SHORT net: "
                     f"{_f(best_short, 'margin_short_net'):+.4f} — "
                     f"{str(best_short.get('title'))[:40]}")
    lines.append(f"⚡ Finestre sopra soglia: {windows}")
    lines.append(f"📄 Paper: {len(opens)} aperture, {len(resolves)} risolte "
                 f"(PnL {resolved_pnl:+.2f}), {open_positions} set aperti")
    if gated is not None:
        lines.append(f"🚧 Gated (non esaustivo): "
                     f"{_f(gated, 'margin_long_net'):+.4f} — "
                     f"{str(gated.get('title'))[:40]}")
    return "\n".join(lines)


def maybe_send_daily_digest(csv_path: Path = CSV_PATH,
                            state_path: Path = DIGEST_STATE_PATH) -> bool:
    """Send one digest per day after DIGEST_HOUR_UTC. Returns True if sent."""
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    if now.hour < DIGEST_HOUR_UTC:
        return False
    try:
        last = json.loads(state_path.read_text()).get("last_sent", "")
    except Exception:
        last = ""
    if last == today:
        return False

    import csv as _csv
    log_rows = []
    if csv_path.exists():
        with csv_path.open() as f:
            log_rows = [r for r in _csv.DictReader(f)
                        if str(r.get("timestamp", "")).startswith(today)]
    trades_path = Path("data/negrisk_paper_trades.csv")
    paper_trades = []
    if trades_path.exists():
        with trades_path.open() as f:
            paper_trades = [r for r in _csv.DictReader(f)
                            if str(r.get("timestamp", "")).startswith(today)]
    try:
        positions_path = Path("data/negrisk_paper_positions.json")
        open_positions = len(json.loads(positions_path.read_text()))
    except Exception:
        open_positions = 0

    text = compose_digest(log_rows, paper_trades, open_positions, today)
    try:
        from alerts.telegram_bot import _send
        sent = _send(text)
    except Exception as e:
        print(f"  [NegRisk] digest send failed: {e}")
        return False
    if sent:
        state_path.write_text(json.dumps({"last_sent": today}))
        print(f"  [NegRisk] daily digest sent ({today})")
    return bool(sent)


# ── Logging + CLI ───────────────────────────────────────────────────────────────

def append_rows(rows: list[dict], csv_path: Path = CSV_PATH) -> None:
    """Append analyzed rows to the evidence log, writing a header if new.

    If the on-disk header doesn't match CSV_COLUMNS (schema changed), the
    old file is rotated aside — never mixed with the new schema."""
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        with csv_path.open() as f:
            header = f.readline().strip()
        if header != ",".join(CSV_COLUMNS):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
            csv_path.rename(csv_path.with_suffix(f".pre-{stamp}.csv"))
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def scan_once(threshold: float = DEFAULT_THRESHOLD,
              max_events: int = 300,
              csv_path: Path = CSV_PATH,
              paper: bool = False) -> list[dict]:
    """One full scan: fetch → analyze → log (→ paper-trade if enabled)."""
    whitelist = load_whitelist()
    events = fetch_negrisk_events(max_events)
    rows = []
    for e in events:
        try:
            r = analyze_event(e, whitelist, threshold)
        except Exception as ex:
            # one malformed event must never kill the collection loop
            print(f"  [NegRisk] skipping event {e.get('id', '?')}: {ex}")
            continue
        if r:
            rows.append(r)
    append_rows(rows, csv_path)

    if paper:
        from alpha.negrisk_paper import paper_cycle
        from data.polymarket_clob import fetch_orderbook
        events_by_id = {str(e.get("id", "")): e for e in events}
        try:
            paper_cycle(rows, events_by_id, live_legs, _leg_fee,
                        fetch_orderbook, fetch_event)
        except Exception as e:
            print(f"  [NegRisk] paper cycle error: {e}")
        try:
            maybe_send_daily_digest()
        except Exception as e:
            print(f"  [NegRisk] digest error: {e}")

    triggered = [r for r in rows if r["tradable_long"] or r["tradable_short"]]
    print(f"[NegRisk] {len(events)} negRisk events | {len(rows)} analyzed | "
          f"{len(triggered)} above {threshold:.2%} net")
    for r in sorted(triggered, key=lambda x: -max(x["margin_long_net"],
                                                  x["margin_short_net"])):
        side = "LONG" if r["tradable_long"] else "SHORT"
        margin = r["margin_long_net"] if r["tradable_long"] else r["margin_short_net"]
        print(f"  {side} {margin:+.4f} net | {r['n_legs']} legs | "
              f"{r['exhaustive_reason']:9s} | {r['title'][:60]}")
    # Near-misses blocked only by the exhaustiveness gate — whitelist candidates
    for r in rows:
        if not r["exhaustive"] and r["margin_long_net"] >= threshold:
            print(f"  GATED {r['margin_long_net']:+.4f} net | {r['n_legs']} legs | "
                  f"NOT exhaustive — review for whitelist | {r['title'][:60]}")
    return triggered


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only Polymarket negRisk scanner")
    ap.add_argument("--once", action="store_true", help="single scan and exit")
    ap.add_argument("--loop", type=int, metavar="SECONDS",
                    help="rescan every N seconds")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--max-events", type=int, default=300)
    ap.add_argument("--paper", action="store_true",
                    help="simulate set entries at the CLOB touch (no real orders)")
    args = ap.parse_args()

    if args.loop:
        while True:
            try:
                scan_once(args.threshold, args.max_events, paper=args.paper)
            except Exception as e:
                # keep the launchd service alive; next cycle retries
                print(f"  [NegRisk] scan failed, retrying next cycle: {e}")
            time.sleep(args.loop)
    else:
        scan_once(args.threshold, args.max_events, paper=args.paper)


if __name__ == "__main__":
    main()
