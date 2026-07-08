"""
src/alpha/uma_dispute.py — UMA DVM dispute forecasting via voter conflict-of-interest.

Hypothesis (WSJ investigation, 2026-05): UMA DVM voters holding a Polymarket
position on the disputed market vote their own book (~1 dispute in 5 has a
voter in conflict). Preregistered predictions this is tested against live in
docs/experiments/2026-07-uma-dispute-preregistration.md.

Pure math module — no network, no I/O. Historical votes/positions come from
scripts/backfill_uma_disputes.py.

Usage:
    from alpha.uma_dispute import compute_voter_stats, UmaDisputeForecaster

    voter_stats = compute_voter_stats(votes_df)
    forecaster = UmaDisputeForecaster(voter_stats, measured_base_rate=0.83)
    result = forecaster.forecast(dispute_voters_df)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

MIN_HISTORICAL_VOTES = 3
HIGH_LINK_CONFIDENCE = "high"


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
