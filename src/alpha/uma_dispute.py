"""
src/alpha/uma_dispute.py — UMA DVM dispute forecasting via voter conflict-of-interest.

Hypothesis (WSJ investigation, 2026-05): UMA DVM voters holding a Polymarket
position on the disputed market vote their own book (~1 dispute in 5 has a
voter in conflict). Preregistered predictions this is tested against live in
docs/experiments/2026-07-uma-dispute-preregistration.md.

Pure math module — no network, no I/O. Historical votes/positions come from
scripts/fetch_votes_web3.py and scripts/fetch_positions_web3.py.

Usage:
    from alpha.uma_dispute import compute_voter_stats, UmaDisputeForecaster

    voter_stats = compute_voter_stats(votes_df)
    forecaster = UmaDisputeForecaster(voter_stats, measured_base_rate=0.83)
    result = forecaster.forecast(dispute_voters_df)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
from eth_utils import keccak

MIN_HISTORICAL_VOTES = 3
HIGH_LINK_CONFIDENCE = "high"

# ── UMA OO price semantics ───────────────────────────────────────────────────
# Verified against the deployed Polymarket UmaCtfAdapter source (Polygon
# 0x2F5e3684cb1F318ec51b00Edba38d79Ac2c0aA9d, `_resolve` / `_constructPayouts`).
#
#   if (price == _ignorePrice()) return _reset(...);   // <- NOT a void
#   _constructPayouts: 0 -> [0,1] NO, 0.5e18 -> [1,1] 50/50, 1e18 -> [1,0] YES
#                      anything else -> revert InvalidOOPrice
#
# The ignore-price sentinel means the DVM declined to settle and the question is
# RE-REQUESTED, not cancelled or refunded. The only genuine "void" is the 0.5e18
# tie, which pays every share 0.50. Conflating the two badly misstates both the
# base rate and the payout a position would have received.
UMA_IGNORE_PRICE = -(2 ** 255)   # type(int256).min
PRICE_NO = 0
PRICE_VOID = 5 * 10 ** 17        # 0.5 ether
PRICE_YES = 10 ** 18             # 1 ether


class DisputeOutcome(str, Enum):
    """What the DVM actually did with a disputed price request."""
    RESET = "reset"          # ignore-price sentinel: question re-requested, market lives on
    YES = "yes"              # payouts [1, 0]
    NO = "no"                # payouts [0, 1]
    VOID_5050 = "void_5050"  # payouts [1, 1]: the only true void/refund


def classify_outcome(settled_price: int) -> DisputeOutcome:
    """Map an OO settled price onto the adapter's actual behaviour.

    Raises ValueError on a price the adapter would reject with InvalidOOPrice.
    """
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


def expected_payouts(settled_price: int) -> tuple[int, int]:
    """Mirror of UmaCtfAdapter._constructPayouts. Returns (YES, NO).

    Raises ValueError for the ignore-price sentinel: the adapter never builds a
    payout array for it, it resets the question instead. Callers must branch on
    classify_outcome() rather than assume every settled dispute paid out.
    """
    price = int(settled_price)
    if price == UMA_IGNORE_PRICE:
        raise ValueError("ignore-price sentinel resets the question; no payout array exists")
    if price == PRICE_NO:
        return (0, 1)
    if price == PRICE_VOID:
        return (1, 1)
    if price == PRICE_YES:
        return (1, 0)
    raise ValueError(f"InvalidOOPrice: {price}")


def vote_side(vote_price: int) -> str:
    """The side a DVM voter revealed. 'IGNORE' is a vote to reset, not a YES/NO."""
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
        """Disputes where the DVM actually issued a payout-bearing price."""
        return self.n_yes + self.n_no + self.n_void_5050

    @property
    def base_rate(self) -> float | None:
        """P(DVM confirms the proposer | the DVM decided). None when N == 0.

        RESET disputes are excluded from both numerator and denominator: the DVM
        issued no ruling on the proposal, it sent the question back out.
        """
        if self.n_decided == 0:
            return None
        return self.n_confirmed / self.n_decided

    @property
    def reset_rate(self) -> float | None:
        """P(a dispute buys delay rather than a ruling). None when N == 0."""
        if self.n_disputes == 0:
            return None
        return self.n_reset / self.n_disputes


def summarize_disputes(disputes: pd.DataFrame) -> DisputeBreakdown:
    """Break a settled-dispute frame down by what the DVM actually did.

    Columns required: settled_price, proposedPrice. Rows with a null
    settled_price (still pending at the DVM) are ignored.
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


# ── Dispute <-> DVM vote linkage ─────────────────────────────────────────────
# A Polygon OO dispute escalates to the Ethereum-mainnet DVM through two
# rewrites of the ancillary data, which is why a raw-text match finds nothing:
#
#   1. OptimisticOracleV2._stampAncillaryData appends ",ooRequester:<adapter>"
#      (address as lowercase utf8 hex, no 0x) before calling the oracle.
#   2. OracleChildTunnel then COMPRESSES that, so what the DVM (and every
#      VoteRevealed event) carries is
#         "ancillaryDataHash:<keccak of the stamped bytes>,childBlockNumber:...,
#          childOracle:...,childRequester:...,childChainId:137"
#
# So the join key is keccak(ancillaryData || ",ooRequester:" || requester).
# That alone is NOT sufficient: a question that is disputed, reset, and disputed
# again reuses byte-identical ancillaryData across rounds, so the hash maps to
# several distinct DVM price requests. A timestamp disambiguates them -- but NOT
# the OO request timestamp.
#
# Polymarket's requests are event-based, and on dispute OOv2 escalates to the DVM
# under the PROPOSAL time, not the original request timestamp. Verified on-chain:
# dispute 0xaba67ff5... carries DisputePrice.timestamp = 1750183833 while the
# OracleChildTunnel PriceRequestAdded emitted in that same transaction carries
# time = 1750780863, and it is the latter that every VoteRevealed event echoes.
# Keying on request_timestamp therefore matches nothing.
#
# The authoritative (hash, time) pair for a dispute is emitted inside the dispute's
# own Polygon transaction, so it is read from chain rather than inferred. See
# scripts/fetch_votes_web3.py.

def question_id(ancillary_data: bytes) -> str:
    """Polymarket's questionID = keccak(ancillaryData). Stable across resets."""
    return keccak(ancillary_data).hex()


def stamped_ancillary_hash(ancillary_data: bytes, requester: str) -> str:
    """The `ancillaryDataHash` the mainnet DVM sees for a Polygon OO request.

    UMA's AncillaryData.appendKeyValueAddress writes the address as lowercase
    utf8 hex with no 0x prefix.
    """
    addr = requester[2:] if requester.startswith("0x") else requester
    if len(addr) != 40:
        raise ValueError(f"requester must be a 20-byte address, got {requester!r}")
    stamped = ancillary_data + b",ooRequester:" + addr.lower().encode()
    return keccak(stamped).hex()


def link_votes_to_disputes(disputes: pd.DataFrame, votes: pd.DataFrame) -> pd.DataFrame:
    """Exact inner join of DVM votes onto the dispute round they were cast on.

    disputes: stamped_hash, dvm_request_time (+ any carried columns)
    votes:    stamped_hash, dvm_request_time, voter, vote_price, stake

    Joining on stamped_hash alone silently fans a reset question's votes across
    every one of its rounds; dvm_request_time is what makes each DVM price request
    unique, so both keys are required. `dvm_request_time` is the OracleChildTunnel
    request time, NOT the OO request_timestamp -- see the note above.
    """
    keys = ["stamped_hash", "dvm_request_time"]
    left = disputes.copy()
    right = votes.copy()
    for frame in (left, right):
        frame["dvm_request_time"] = frame["dvm_request_time"].astype("int64")
    merged = left.merge(right, on=keys, how="inner", validate="one_to_many")
    if not merged.empty:
        merged["side"] = merged["vote_price"].map(vote_side)
    return merged


def annotate_votes_with_positions(votes: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Attach `has_position` / `voted_with_own_position` to each linked vote.

    votes:     dispute_tx_hash, voter, side, stake
    positions: dispute_tx_hash, voter, yes_shares, no_shares (at the dispute block)

    "Own book" is the NET directional exposure. A voter holding equal YES and NO
    is holding a complete set, which redeems for $1 whichever way the DVM rules,
    so they have no directional interest and `has_position` is False -- counting
    them as conflicted would inflate conflict_rate with hedged inventory.

    Only YES/NO votes can align with a book. An IGNORE vote (a vote to reset) is
    not a directional vote, so it never counts as voting one's own book, even
    though a large holder may well have delay-motivated reasons to cast it.
    """
    if votes.empty:
        return votes.assign(has_position=[], voted_with_own_position=[])

    pos = positions.copy()
    if pos.empty:
        merged = votes.copy()
        merged["yes_shares"] = 0.0
        merged["no_shares"] = 0.0
    else:
        pos["voter"] = pos["voter"].str.lower()
        merged = votes.assign(voter=votes["voter"].str.lower()).merge(
            pos[["dispute_tx_hash", "voter", "yes_shares", "no_shares"]],
            on=["dispute_tx_hash", "voter"], how="left",
        )
        merged[["yes_shares", "no_shares"]] = merged[["yes_shares", "no_shares"]].fillna(0.0)

    net = merged["yes_shares"] - merged["no_shares"]
    merged["has_position"] = net != 0
    own_side = net.map(lambda n: "YES" if n > 0 else ("NO" if n < 0 else None))
    merged["voted_with_own_position"] = merged["has_position"] & (merged["side"] == own_side)
    return merged


@dataclass
class VoterStats:
    """Per-wallet voting record on the UMA DVM.

    votes_with_position: historical votes where the voter held a Polymarket
      position on the market being voted on.
    voted_own_book: subset of votes_with_position where the revealed vote
      matched the direction of the voter's own position.
    """
    voter: str
    total_votes: int
    votes_with_position: int
    voted_own_book: int
    avg_stake: float

    @property
    def conflict_rate(self) -> float:
        """Fraction of position-linked votes voted in the position's direction.

        0.0 when the voter never had a linked position — absence of signal,
        not a divide-by-zero.
        """
        if self.votes_with_position == 0:
            return 0.0
        return self.voted_own_book / self.votes_with_position


@dataclass
class DisputeForecast:
    p_yes: float
    confidence: str            # "measured_voters" | "base_rate_fallback"
    drivers: list[dict] = field(default_factory=list)
    hypothesis: str = ""


def compute_voter_stats(votes: pd.DataFrame) -> dict[str, VoterStats]:
    """Aggregate a per-vote history into per-voter VoterStats.

    votes columns: voter, has_position (bool), voted_with_own_position (bool),
    stake (float). One row per (voter, historical dispute vote).
    """
    stats: dict[str, VoterStats] = {}
    if votes.empty:
        return stats
    for voter, g in votes.groupby("voter"):
        with_pos = g[g["has_position"]]
        stats[voter] = VoterStats(
            voter=voter,
            total_votes=len(g),
            votes_with_position=len(with_pos),
            voted_own_book=int(with_pos["voted_with_own_position"].sum()),
            avg_stake=float(g["stake"].mean()),
        )
    return stats


class UmaDisputeForecaster:
    """Forecasts DVM dispute outcomes from stake-weighted voter conflict-of-interest.

    measured_base_rate: the "DVM confirms the original proposal" rate measured
    from the backfill (see docs/experiments/2026-07-uma-dispute-preregistration.md).
    Never a hardcoded/invented constant — passed in by the caller after the
    backfill runs.
    """

    def __init__(self, voter_stats: dict[str, VoterStats], measured_base_rate: float,
                 min_historical_votes: int = MIN_HISTORICAL_VOTES):
        self.voter_stats = voter_stats
        self.measured_base_rate = measured_base_rate
        self.min_historical_votes = min_historical_votes

    def _fallback(self) -> DisputeForecast:
        return DisputeForecast(
            p_yes=self.measured_base_rate,
            confidence="base_rate_fallback",
            drivers=[],
            hypothesis=(
                f"No eligible conflicted voter (>= {self.min_historical_votes} "
                "historical votes, high-confidence position link, nonzero "
                "conflict_rate) on either side — using the measured DVM "
                "base-rate fallback."
            ),
        )

    def forecast(self, dispute_voters: pd.DataFrame) -> DisputeForecast:
        """
        dispute_voters columns: voter, side ('YES'|'NO'), stake (float),
        link_confidence ('high'|'low'|None — confidence that the voter's
        wallet is correctly linked to a Polymarket account).

        P(YES) = sum(stake * conflict_rate) on the YES side /
                 sum(stake * conflict_rate) over YES + NO,
        restricted to voters with >= min_historical_votes and
        link_confidence == 'high'. Falls back to measured_base_rate when no
        voter clears the bar or all weights are zero.
        """
        if dispute_voters.empty:
            return self._fallback()

        drivers: list[dict] = []
        yes_weight = 0.0
        no_weight = 0.0

        for _, row in dispute_voters.iterrows():
            if row.get("link_confidence") != HIGH_LINK_CONFIDENCE:
                continue
            vs = self.voter_stats.get(row["voter"])
            if vs is None or vs.total_votes < self.min_historical_votes:
                continue
            weight = row["stake"] * vs.conflict_rate
            if weight <= 0:
                continue
            if row["side"] == "YES":
                yes_weight += weight
            elif row["side"] == "NO":
                no_weight += weight
            else:
                continue
            drivers.append({
                "voter": row["voter"],
                "side": row["side"],
                "stake": row["stake"],
                "conflict_rate": vs.conflict_rate,
                "weight": weight,
            })

        total = yes_weight + no_weight
        if total <= 0:
            return self._fallback()

        return DisputeForecast(
            p_yes=yes_weight / total,
            confidence="measured_voters",
            drivers=drivers,
            hypothesis=(
                f"{len(drivers)} conflicted voter(s) with >= {self.min_historical_votes} "
                "historical votes and high-confidence position link, "
                "stake-weighted by their own conflict_rate."
            ),
        )
