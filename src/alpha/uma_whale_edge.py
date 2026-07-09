"""
src/alpha/uma_whale_edge.py — Trading rule for the UMA whale-conflict edge.

Preregistered in docs/experiments/2026-07-uma-whale-edge-backtest-preregistration.md.
Every constant here was fixed BEFORE the backtest ran; none may be tuned to the
result without relabelling the result post-hoc.

Pure functions. No network, no I/O.

The chain being tested:
    whale holds a directional position
      -> whale votes their own book        (measured: 63.2%, p = 1.5e-05)
      -> that vote moves the DVM outcome   (untested, doubtful: tiny stake share)
      -> the market had not priced it      (untested, doubtful)
Links two and three are what this backtest actually probes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.uma_dispute import UMA_IGNORE_PRICE, DisputeOutcome, classify_outcome

# ── Preregistered constants (do not tune) ────────────────────────────────────
BASE_YES = 0.4606          # P(YES | DVM decided) = 146/317, RESET excluded
K_TILT = 0.25              # forecast sensitivity to whale tilt
ENTRY_THRESHOLD = 0.10     # |forecast - price| required to trade
MIN_LINKED_VOTES = 3       # whale floor: position-linked votes across the dataset
MIN_TRADES_IN_WINDOW = 3   # else the dispute has no usable price
PRICE_WINDOW_HOURS = 24
NOTIONAL = 100.0
MIN_TRADABLE_DISPUTES = 20  # below this: NOT TESTABLE
CONCENTRATION_FLAG = 0.50   # top-3 share of net PnL that triggers a tail warning


def terminal_outcomes(requests: pd.DataFrame) -> dict[str, float]:
    """question_id -> payout of one YES share at the question's final resolution.

    A question is resolved at most once: a RESET sends it back to the oracle, and
    `resolved` is only set on a decided price. So the unique non-RESET round is
    the resolution. Questions that never reach a decided round are absent, and
    must be dropped rather than assumed to have gone either way.
    """
    payout = {DisputeOutcome.YES: 1.0, DisputeOutcome.NO: 0.0, DisputeOutcome.VOID_5050: 0.5}
    out: dict[str, float] = {}
    for _, row in requests.iterrows():
        if pd.isna(row["settled_price"]):
            continue
        outcome = classify_outcome(row["settled_price"])
        if outcome is DisputeOutcome.RESET:
            continue
        out[row["question_id"]] = payout[outcome]
    return out


def whale_voters(voter_stats: dict, min_linked: int = MIN_LINKED_VOTES) -> set[str]:
    """Voters defensible as 'whales': enough position-linked votes that their
    conflict_rate is not 1.00-or-0.00 by construction."""
    return {v for v, s in voter_stats.items()
            if s.total_votes >= min_linked and s.votes_with_position >= min_linked}


def dispute_tilt(nets: dict[str, float], stakes: dict[str, float]) -> float | None:
    """Stake-weighted sign of whale net exposure, in [-1, 1]. None if no signal.

    Only the SIGN of each whale's book enters, not its size: a whale with a huge
    position is not more likely to vote their book than one with a small one, and
    letting position size in would let a single large holder dominate a signal
    that is really about voting behaviour.
    """
    num = den = 0.0
    for voter, net in nets.items():
        if net == 0:
            continue          # complete set: hedged, no directional interest
        stake = stakes.get(voter, 0.0)
        if stake <= 0:
            continue
        num += stake * np.sign(net)
        den += stake
    if den == 0:
        return None
    return num / den


def forecast_from_tilt(tilt: float, base_yes: float = BASE_YES, k: float = K_TILT) -> float:
    return float(np.clip(base_yes + k * tilt, 0.02, 0.98))


def vwap(prices: np.ndarray, sizes: np.ndarray) -> float:
    total = sizes.sum()
    if total <= 0:
        raise ValueError("cannot VWAP zero total size")
    return float((prices * sizes).sum() / total)


@dataclass
class Trade:
    dispute_tx_hash: str
    dispute_time: pd.Timestamp
    forecast: float
    price: float
    side: str            # "YES" | "NO"
    entry_price: float   # price paid per share of the side taken
    shares: float
    fee: float
    payout_per_share: float
    gross_pnl: float
    net_pnl: float

    @property
    def net_edge(self) -> float:
        """PnL per $1 of notional risked."""
        return self.net_pnl / NOTIONAL


def simulate_trade(dispute_tx_hash: str, dispute_time, forecast: float, yes_price: float,
                   yes_payout: float, fee_per_share, notional: float = NOTIONAL,
                   threshold: float = ENTRY_THRESHOLD) -> Trade | None:
    """One flat-size trade held to resolution. None when the edge is below threshold.

    `fee_per_share` is a callable price -> fee, so the caller supplies the real
    per-category Polymarket schedule. Fees are charged on entry only: redeeming a
    resolved position pays no taker fee.
    """
    if abs(forecast - yes_price) < threshold:
        return None
    buy_yes = forecast > yes_price
    entry_price = yes_price if buy_yes else 1.0 - yes_price
    if not 0.0 < entry_price < 1.0:
        return None
    shares = notional / entry_price
    fee = fee_per_share(entry_price) * shares
    payout_per_share = yes_payout if buy_yes else 1.0 - yes_payout
    gross = shares * payout_per_share - notional
    return Trade(
        dispute_tx_hash=dispute_tx_hash, dispute_time=dispute_time,
        forecast=forecast, price=yes_price, side="YES" if buy_yes else "NO",
        entry_price=entry_price, shares=shares, fee=fee,
        payout_per_share=payout_per_share, gross_pnl=gross, net_pnl=gross - fee,
    )


def concentration(pnls: np.ndarray, top_n: int = 3) -> float:
    """Share of NET total PnL contributed by the top-n trades.

    Returns inf when total net PnL <= 0, because "the top 3 made 200% of a
    negative total" is not a meaningful percentage and must not read as 2.0.
    """
    total = pnls.sum()
    if total <= 0:
        return float("inf")
    return float(np.sort(pnls)[-top_n:].sum() / total)


def split_halves(trades: list[Trade]) -> tuple[list[Trade], list[Trade]]:
    """Chronological split at the median dispute_time (preregistered)."""
    ordered = sorted(trades, key=lambda t: t.dispute_time)
    mid = len(ordered) // 2
    return ordered[:mid], ordered[mid:]
