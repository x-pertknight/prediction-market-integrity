"""
src/alpha/negrisk_paper.py — PAPER executor for negRisk set arbitrage.

Places NO real orders. When the scanner flags a tradable set it simulates
buying it at the executable CLOB touch (best ask per leg for long sets,
NO at 1−bid for short sets), nets each leg's exact feeSchedule, and holds
the position until the event resolves. Realized PnL is computed from actual
leg outcomes — so an exhaustiveness mistake (no owned leg pays) shows up as
a real simulated loss instead of staying a theory.

Fill model is OPTIMISTIC by design and documented as such: top-of-book
price, size capped by top-5-level depth, no queue, no adverse selection,
all legs assumed filled atomically. Paper results are therefore an UPPER
BOUND on live performance; the point is to measure margin persistence and
gate correctness, not to forecast PnL.

State:  data/negrisk_paper_positions.json   (open positions)
Ledger: data/negrisk_paper_trades.csv       (entries + resolutions)
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

POSITIONS_PATH = Path("data/negrisk_paper_positions.json")
TRADES_PATH = Path("data/negrisk_paper_trades.csv")

MAX_SETS = 100            # cap per position, before depth cap
PAPER_THRESHOLD = 0.005   # enter anything above 0.5c net — we're collecting data

TRADE_COLUMNS = [
    "timestamp", "event_id", "title", "side", "action", "n_legs", "sets",
    "cost_per_set", "payout_per_set", "locked_pnl_per_set", "total_pnl",
    "exhaustive_reason", "note",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Pure fill simulation (unit tested) ──────────────────────────────────────────

def simulate_entry(row: dict, legs: list[dict], books: dict, side: str,
                   leg_fee_fn, max_sets: int = MAX_SETS) -> dict | None:
    """Simulate buying one full outcome set at the executable CLOB touch.

    legs:  live leg market dicts (must carry clobTokenIds, feeSchedule).
    books: {yes_token_id: fetch_orderbook() dict} for every leg.
    side:  "long" (buy YES set, payout $1) or "short" (buy NO set at 1−bid,
           payout $(n−1)).
    Returns a position dict, or None if any leg has no usable book.
    """
    prices, fees, depths, tokens = [], [], [], []
    for leg in legs:
        token = json.loads(leg["clobTokenIds"])[0]  # YES token
        book = books.get(token)
        if not book:
            return None
        if side == "long":
            price, depth = book["best_ask"], book["ask_depth"]
        else:
            bid = book["best_bid"]
            price, depth = 1.0 - bid, book["bid_depth"]
        if price <= 0.0 or price >= 1.0 or depth <= 0:
            return None
        prices.append(price)
        fees.append(leg_fee_fn(price, leg))
        depths.append(depth)
        tokens.append(token)

    n = len(legs)
    sets = int(min(min(depths), max_sets))
    if sets < 1:
        return None
    cost_per_set = sum(prices) + sum(fees)
    payout_per_set = 1.0 if side == "long" else float(n - 1)
    locked = payout_per_set - cost_per_set
    return {
        "event_id": row["event_id"],
        "title": row["title"],
        "side": side,
        "n_legs": n,
        "sets": sets,
        "tokens": tokens,
        "leg_prices": [round(p, 4) for p in prices],
        "cost_per_set": round(cost_per_set, 6),
        "payout_per_set": payout_per_set,
        "locked_pnl_per_set": round(locked, 6),
        "exhaustive_reason": row["exhaustive_reason"],
        "opened_at": _now(),
    }


def resolve_position(pos: dict, event: dict) -> dict | None:
    """Realize a position from actual leg outcomes once the event resolved.

    Uses outcomePrices ('["1","0"]' = YES paid) per leg market matched by
    YES token id. Returns a resolution record, or None if any owned leg is
    still unresolved. The guaranteed-payout assumption is NOT reused here —
    payout comes from what actually resolved, so a non-exhaustive set that
    slipped through shows its true loss.
    """
    yes_paid: dict[str, float] = {}
    for m in event.get("markets", []):
        try:
            token = json.loads(m.get("clobTokenIds") or "[]")[0]
        except (IndexError, ValueError):
            continue
        if not m.get("closed"):
            continue
        try:
            yes_paid[token] = float(json.loads(m.get("outcomePrices") or "[]")[0])
        except (IndexError, ValueError):
            continue

    if any(t not in yes_paid for t in pos["tokens"]):
        return None  # still open

    winners = sum(yes_paid[t] for t in pos["tokens"])
    if pos["side"] == "long":
        payout = winners                       # $1 per set iff one owned leg won
    else:
        payout = pos["n_legs"] - winners       # each losing leg's NO pays $1
    realized_per_set = payout - pos["cost_per_set"]
    return {
        "payout_per_set": payout,
        "realized_pnl_per_set": round(realized_per_set, 6),
        "total_pnl": round(realized_per_set * pos["sets"], 4),
        "owned_winners": winners,
        "closed_at": _now(),
    }


# ── State + ledger ──────────────────────────────────────────────────────────────

def load_positions(path: Path = POSITIONS_PATH) -> dict:
    """Load open positions. A corrupt file is moved aside (never silently
    overwritten as {}) so open sets can be recovered manually."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        corrupt = path.with_suffix(f".corrupt-{stamp}.json")
        path.rename(corrupt)
        print(f"  [PAPER] positions file corrupt ({e}) — moved to {corrupt}")
        return {}


def save_positions(positions: dict, path: Path = POSITIONS_PATH) -> None:
    """Atomic write (tmp + os.replace): a crash mid-write can never leave
    a truncated file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(positions, indent=1))
    os.replace(tmp, path)


def log_trade(record: dict, path: Path = TRADES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(record)


# ── Cycle driver (called by the scanner loop) ───────────────────────────────────

def paper_cycle(rows: list[dict], events_by_id: dict, live_legs_fn, leg_fee_fn,
                fetch_book_fn, fetch_event_fn,
                threshold: float = PAPER_THRESHOLD) -> None:
    """One paper pass: open new set positions, resolve finished ones."""
    positions = load_positions()

    # 1. Entries
    for row in rows:
        candidates = []
        if row["exhaustive"] and row["margin_long_net"] >= threshold:
            candidates.append("long")
        if row["margin_short_net"] >= threshold:
            candidates.append("short")
        for side in candidates:
            key = f"{row['event_id']}:{side}"
            if key in positions:
                continue
            event = events_by_id.get(row["event_id"])
            if not event:
                continue
            legs = live_legs_fn(event)
            books = {}
            for leg in legs:
                token = json.loads(leg["clobTokenIds"])[0]
                books[token] = fetch_book_fn(token)
            pos = simulate_entry(row, legs, books, side, leg_fee_fn)
            if not pos:
                continue
            if pos["locked_pnl_per_set"] < threshold:
                continue  # executable touch was worse than the Gamma snapshot
            positions[key] = pos
            log_trade({
                "timestamp": pos["opened_at"], "event_id": pos["event_id"],
                "title": pos["title"], "side": side, "action": "OPEN",
                "n_legs": pos["n_legs"], "sets": pos["sets"],
                "cost_per_set": pos["cost_per_set"],
                "payout_per_set": pos["payout_per_set"],
                "locked_pnl_per_set": pos["locked_pnl_per_set"],
                "total_pnl": round(pos["locked_pnl_per_set"] * pos["sets"], 4),
                "exhaustive_reason": pos["exhaustive_reason"],
                "note": "paper fill at CLOB touch (optimistic)",
            })
            print(f"  [PAPER] OPEN {side.upper()} {pos['sets']} sets | "
                  f"locked {pos['locked_pnl_per_set']:+.4f}/set | {pos['title'][:50]}")

    # 2. Resolutions
    for key in list(positions):
        pos = positions[key]
        event = fetch_event_fn(pos["event_id"])
        if not event:
            continue
        res = resolve_position(pos, event)
        if res is None:
            continue
        log_trade({
            "timestamp": res["closed_at"], "event_id": pos["event_id"],
            "title": pos["title"], "side": pos["side"], "action": "RESOLVE",
            "n_legs": pos["n_legs"], "sets": pos["sets"],
            "cost_per_set": pos["cost_per_set"],
            "payout_per_set": res["payout_per_set"],
            "locked_pnl_per_set": res["realized_pnl_per_set"],
            "total_pnl": res["total_pnl"],
            "exhaustive_reason": pos["exhaustive_reason"],
            "note": f"owned_winners={res['owned_winners']}",
        })
        print(f"  [PAPER] RESOLVE {pos['side'].upper()} {pos['title'][:50]} | "
              f"PnL {res['total_pnl']:+.2f}")
        del positions[key]

    save_positions(positions)
