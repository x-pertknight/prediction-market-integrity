"""
scripts/uma_dvm.py: UMA dispute and DVM vote semantics used by the offline
build step and the offline recompute.

Pure functions over pandas frames. No network, no hashing library.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

# UMA optimistic oracle price semantics, checked against the deployed Polymarket
# UmaCtfAdapter source (Polygon 0x2F5e3684cb1F318ec51b00Edba38d79Ac2c0aA9d,
# `_resolve` and `_constructPayouts`):
#
#   if (price == _ignorePrice()) return _reset(...);   // NOT a void
#   _constructPayouts: 0 -> [0,1] NO, 0.5e18 -> [1,1] 50/50, 1e18 -> [1,0] YES
#
# The ignore price sentinel ("too early") means the DVM declined to settle and
# the question is asked again. The only genuine void is the 0.5e18 tie.
UMA_IGNORE_PRICE = -(2 ** 255)   # type(int256).min
PRICE_NO = 0
PRICE_VOID = 5 * 10 ** 17        # 0.5 ether
PRICE_YES = 10 ** 18             # 1 ether


class DisputeOutcome(str, Enum):
    """What the DVM did with a disputed price request."""
    RESET = "reset"          # ignore price: question asked again, market lives on
    YES = "yes"              # payouts [1, 0]
    NO = "no"                # payouts [0, 1]
    VOID_5050 = "void_5050"  # payouts [1, 1]


def classify_outcome(settled_price: int) -> DisputeOutcome:
    """Map a settled price onto the adapter's behaviour. Raises on a price
    the adapter would reject."""
    price = int(settled_price)
    if price == UMA_IGNORE_PRICE:
        return DisputeOutcome.RESET
    if price == PRICE_NO:
        return DisputeOutcome.NO
    if price == PRICE_YES:
        return DisputeOutcome.YES
    if price == PRICE_VOID:
        return DisputeOutcome.VOID_5050
    raise ValueError(f"InvalidOOPrice: {price}")


def vote_side(vote_price: int) -> str:
    """The side a DVM voter revealed. IGNORE is a vote to reset."""
    price = int(vote_price)
    if price == UMA_IGNORE_PRICE:
        return "IGNORE"
    if price == PRICE_YES:
        return "YES"
    if price == PRICE_NO:
        return "NO"
    if price == PRICE_VOID:
        return "VOID"
    return "OTHER"


@dataclass
class DisputeBreakdown:
    n_disputes: int
    n_reset: int
    n_yes: int
    n_no: int
    n_void_5050: int
    n_confirmed: int = 0

    @property
    def n_decided(self) -> int:
        """Disputes where the DVM issued a payout bearing price."""
        return self.n_yes + self.n_no + self.n_void_5050

    @property
    def base_rate(self) -> float | None:
        """P(DVM confirms the proposer | the DVM decided). RESET is excluded
        from numerator and denominator: the DVM issued no ruling."""
        if self.n_decided == 0:
            return None
        return self.n_confirmed / self.n_decided

    @property
    def reset_rate(self) -> float | None:
        if self.n_disputes == 0:
            return None
        return self.n_reset / self.n_disputes


def summarize_disputes(disputes: pd.DataFrame) -> DisputeBreakdown:
    """Break settled disputes down by what the DVM did.

    Columns: settled_price, proposedPrice (as strings: the sentinel overflows
    int64). Rows with a null settled_price are still pending and ignored.
    """
    settled = disputes[disputes["settled_price"].notna()]
    counts = {o: 0 for o in DisputeOutcome}
    n_confirmed = 0
    for _, row in settled.iterrows():
        outcome = classify_outcome(row["settled_price"])
        counts[outcome] += 1
        if outcome is not DisputeOutcome.RESET and int(row["settled_price"]) == int(row["proposedPrice"]):
            n_confirmed += 1
    return DisputeBreakdown(
        n_disputes=len(settled),
        n_reset=counts[DisputeOutcome.RESET],
        n_yes=counts[DisputeOutcome.YES],
        n_no=counts[DisputeOutcome.NO],
        n_void_5050=counts[DisputeOutcome.VOID_5050],
        n_confirmed=n_confirmed,
    )


def annotate_votes_with_positions(votes: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Attach `has_position` and `voted_with_own_position` to each vote.

    votes:     dispute_id, voter, side, stake
    positions: dispute_id, voter, yes_shares, no_shares (at the dispute block)

    "Own book" is the NET directional exposure. Equal YES and NO is a complete
    set that redeems for $1 either way, so it is not a position. Only YES and NO
    votes can match a book; an IGNORE vote never counts as voting one's book.
    """
    if votes.empty:
        return votes.assign(has_position=[], voted_with_own_position=[])
    pos = positions[["dispute_id", "voter", "yes_shares", "no_shares"]]
    merged = votes.merge(pos, on=["dispute_id", "voter"], how="left")
    merged[["yes_shares", "no_shares"]] = merged[["yes_shares", "no_shares"]].fillna(0.0)

    net = merged["yes_shares"] - merged["no_shares"]
    merged["has_position"] = net != 0
    own_side = net.map(lambda n: "YES" if n > 0 else ("NO" if n < 0 else None))
    merged["voted_with_own_position"] = merged["has_position"] & (merged["side"] == own_side)
    return merged


@dataclass
class VoterStats:
    """Per voter record on the DVM. votes_with_position counts votes where the
    voter held a position on that market; voted_own_book is the subset where
    the vote matched the position's direction."""
    voter: str
    total_votes: int
    votes_with_position: int
    voted_own_book: int
    avg_stake: float

    @property
    def conflict_rate(self) -> float:
        if self.votes_with_position == 0:
            return 0.0
        return self.voted_own_book / self.votes_with_position


def compute_voter_stats(votes: pd.DataFrame) -> dict[str, VoterStats]:
    """Aggregate annotated votes into per voter VoterStats."""
    stats: dict[str, VoterStats] = {}
    for voter, g in votes.groupby("voter", observed=True):
        with_pos = g[g["has_position"]]
        stats[voter] = VoterStats(
            voter=voter,
            total_votes=len(g),
            votes_with_position=len(with_pos),
            voted_own_book=int(with_pos["voted_with_own_position"].sum()),
            avg_stake=float(g["stake"].mean()),
        )
    return stats
