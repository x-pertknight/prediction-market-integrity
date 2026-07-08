"""tests/test_cross_platform_arb.py — Unit tests for src/alpha/cross_platform_arb.py

Synthetic market dicts only, zero network access.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from alpha.cross_platform_arb import (
    CrossPlatformArb,
    MarketRules,
    kalshi_market_to_rules,
    polymarket_market_to_rules,
    diff_rules,
    fuzzy_candidates,
    compute_exact_pair_arb,
    _extract_outcome_bucket,
    _extract_threshold,
    _extract_edge_cases,
)
from risk.fees import polymarket_fee_from_schedule, kalshi_taker_fee


# ── Field extraction ─────────────────────────────────────────────────────────

class TestOutcomeBucketExtraction:
    def test_advance_keyword(self):
        assert _extract_outcome_bucket("Team Falcon advances to the semifinals") == "ADVANCE"

    def test_win_keyword(self):
        assert _extract_outcome_bucket("Team Falcon wins the tournament and is crowned champion") == "WIN"

    def test_neither_keyword_returns_none(self):
        assert _extract_outcome_bucket("Will it rain in Barcelona tomorrow?") is None


class TestThresholdExtraction:
    def test_goals_threshold(self):
        assert _extract_threshold("scores at least 2+ goals in the game") == pytest.approx(2.0)

    def test_no_threshold(self):
        assert _extract_threshold("Team Falcon wins the championship") is None


class TestEdgeCaseExtraction:
    def test_detects_cancellation_and_tie(self):
        flags = _extract_edge_cases("If the match is cancelled, this market voids. A draw counts as a tie.")
        assert "cancellation" in flags
        assert "tie" in flags

    def test_no_edge_cases(self):
        assert _extract_edge_cases("Team Falcon wins the championship") == frozenset()


# ── Market dict -> MarketRules ───────────────────────────────────────────────

def kalshi_dict(title, rules_primary, expiration_time="2026-08-01T00:00:00Z", ticker="KXTEST-1"):
    return {
        "ticker": ticker,
        "title": title,
        "rules_primary": rules_primary,
        "rules_secondary": "",
        "expiration_time": expiration_time,
        "event_ticker": "KXTEST",
        "floor_strike": None,
    }


def poly_dict(question, description, end_date="2026-08-01T00:00:00Z", condition_id="0xabc"):
    return {
        "id": "1",
        "conditionId": condition_id,
        "question": question,
        "description": description,
        "endDate": end_date,
        "resolutionSource": "",
    }


class TestMarketDictConversion:
    def test_kalshi_conversion_extracts_fields(self):
        m = kalshi_dict(
            "Will Team Falcon advance to the semifinals?",
            "If Team Falcon advances to the semifinals of the tournament, this market resolves Yes.",
        )
        rules = kalshi_market_to_rules(m, settlement_sources=[{"name": "League Official Site"}])
        assert rules.source == "kalshi"
        assert rules.outcome_bucket == "ADVANCE"
        assert rules.settlement_source == "League Official Site"
        assert rules.threshold is None

    def test_polymarket_conversion_extracts_fields(self):
        m = poly_dict(
            "Will Team Falcon win it all?",
            "This market resolves Yes if Team Falcon wins the tournament and is crowned champion. Otherwise No.",
        )
        rules = polymarket_market_to_rules(m)
        assert rules.source == "polymarket"
        assert rules.outcome_bucket == "WIN"


# ── diff_rules: the "advanced vs won" trap MUST be DIVERGENT ────────────────

class TestDiffRulesTrap:
    def test_advanced_vs_won_is_divergent(self):
        """The exact trap the spec calls out: same team/event, textually
        similar question, but semifinal-advance != tournament-win."""
        kalshi = kalshi_market_to_rules(kalshi_dict(
            "Will Team Falcon advance to the semifinals?",
            "If Team Falcon advances to the semifinals of the tournament, this market resolves Yes.",
        ))
        poly = polymarket_market_to_rules(poly_dict(
            "Will Team Falcon advance to the semifinals?",
            "This market resolves Yes if Team Falcon wins the tournament and is crowned champion. Otherwise No.",
        ))
        result = diff_rules(poly, kalshi, fuzzy_score=90.0)
        assert result.verdict == "DIVERGENT"
        outcome_field = next(f for f in result.fields if f.field == "outcome_definition")
        assert outcome_field.verdict == "DIVERGENT"

    def test_never_exact_from_fuzzy_score_alone(self):
        """A near-perfect fuzzy score must not produce EXACT if the
        structured fields can't confirm agreement."""
        kalshi = kalshi_market_to_rules(kalshi_dict(
            "Will it happen?", "Vague rules text with no clear outcome wording.",
        ))
        poly = polymarket_market_to_rules(poly_dict(
            "Will it happen?", "Vague rules text with no clear outcome wording.",
        ))
        result = diff_rules(poly, kalshi, fuzzy_score=100.0)
        assert result.verdict != "EXACT"
        assert result.verdict == "UNKNOWN"

    def test_genuine_match_is_exact(self):
        kalshi = kalshi_market_to_rules(kalshi_dict(
            "Will Team Falcon win the championship?",
            "If Team Falcon wins the championship of the tournament, this market resolves Yes.",
            expiration_time="2026-08-01T00:00:00Z",
        ))
        poly = polymarket_market_to_rules(poly_dict(
            "Will Team Falcon win the championship?",
            "This market resolves Yes if Team Falcon wins the tournament and is crowned champion. Otherwise No.",
            end_date="2026-08-02T12:00:00Z",   # 36h apart, within the 48h tolerance
        ))
        result = diff_rules(poly, kalshi, fuzzy_score=95.0)
        assert result.verdict == "EXACT"

    def test_differing_threshold_is_divergent(self):
        kalshi = kalshi_market_to_rules(kalshi_dict(
            "Player X: 2+ goals", "If Player X scores 2+ goals in the game, this market resolves Yes.",
        ))
        poly = polymarket_market_to_rules(poly_dict(
            "Player X goals prop", "This market resolves Yes if Player X scores 4+ goals in the game. Otherwise No.",
        ))
        # both ambiguous on outcome bucket (no advance/win keywords) but threshold differs
        result = diff_rules(poly, kalshi, fuzzy_score=90.0)
        threshold_field = next(f for f in result.fields if f.field == "threshold")
        assert threshold_field.verdict == "DIVERGENT"
        assert result.verdict == "DIVERGENT"


# ── fuzzy_candidates: stage 1 only narrows, never decides ───────────────────

class TestFuzzyCandidates:
    def test_filters_below_threshold(self):
        target = MarketRules(source="polymarket", market_id="p1",
                              question="Will Team Falcon win the championship?", rules_text="")
        far = MarketRules(source="kalshi", market_id="k1",
                           question="Will it rain in Barcelona tomorrow?", rules_text="")
        near = MarketRules(source="kalshi", market_id="k2",
                            question="Will Team Falcon win the championship?", rules_text="")
        results = fuzzy_candidates(target, [far, near], threshold=75.0)
        ids = [c.market_id for c, _ in results]
        assert "k2" in ids
        assert "k1" not in ids

    def test_respects_max_candidates(self):
        target = MarketRules(source="polymarket", market_id="p1", question="Team Falcon wins", rules_text="")
        candidates = [
            MarketRules(source="kalshi", market_id=f"k{i}", question="Team Falcon wins", rules_text="")
            for i in range(10)
        ]
        results = fuzzy_candidates(target, candidates, threshold=50.0, max_candidates=5)
        assert len(results) == 5


# ── Pricing: only EXACT pairs, real fees, no flat estimates ─────────────────

class TestComputeExactPairArb:
    def test_matches_real_fee_functions(self):
        poly_price = 0.40
        kalshi_no_price = 0.55
        schedule = {"rate": 0.04, "exponent": 1, "takerOnly": True, "rebateRate": 0.20}
        result = compute_exact_pair_arb(poly_price, schedule, kalshi_no_price)

        expected_poly_fee = polymarket_fee_from_schedule(poly_price, schedule)
        expected_kalshi_fee = kalshi_taker_fee(kalshi_no_price)
        expected_gross = 1.0 - (poly_price + kalshi_no_price)
        expected_net = expected_gross - expected_poly_fee - expected_kalshi_fee - 0.01

        assert result.poly_fee == pytest.approx(expected_poly_fee)
        assert result.kalshi_fee == pytest.approx(expected_kalshi_fee)
        assert result.net_spread == pytest.approx(expected_net)

    def test_is_arb_true_when_net_spread_positive(self):
        # Deep, cheap legs: 0.30 + 0.30 = 0.60, well under $1 even after fees/slippage.
        result = compute_exact_pair_arb(0.30, {"rate": 0.04, "exponent": 1}, 0.30)
        assert result.net_spread > 0
        assert result.is_arb is True

    def test_is_arb_false_when_fees_eat_the_spread(self):
        # Legs sum to 0.99 -- a razor-thin 1c gross edge, fees alone exceed it.
        result = compute_exact_pair_arb(0.50, {"rate": 0.07, "exponent": 1}, 0.49)
        assert result.is_arb is False

    def test_no_flat_fee_estimate_fee_free_schedule(self):
        """Empty/missing feeSchedule -> zero Polymarket fee (matches
        risk.fees.polymarket_fee_from_schedule's own contract), not a
        guessed flat rate."""
        result = compute_exact_pair_arb(0.40, None, 0.55)
        assert result.poly_fee == 0.0


# ── Backward compatibility: old class still works, but warns ───────────────

class TestBackwardCompatibility:
    def test_check_within_platform_unchanged(self):
        import pandas as pd
        arb = CrossPlatformArb()
        df = pd.DataFrame({"yes_price": [0.40], "no_price": [0.55]})
        out = arb.check_within_platform(df)
        assert out.loc[0, "arb_signal"] == "BUY_BOTH"

    def test_check_cross_platform_still_works_but_warns(self):
        import pandas as pd
        arb = CrossPlatformArb()
        kalshi_df = pd.DataFrame({"event_slug": ["e1"], "yes_price": [0.40]})
        poly_df = pd.DataFrame({"event_slug": ["e1"], "yes_price": [0.50]})
        with pytest.deprecated_call():
            arb.check_cross_platform(kalshi_df, poly_df)

    def test_scan_live_kalshi_warns(self, monkeypatch):
        import requests as req

        class _FakeResp:
            status_code = 500
            def json(self):
                return {}

        monkeypatch.setattr(req, "get", lambda *a, **k: _FakeResp())
        arb = CrossPlatformArb()
        with pytest.deprecated_call():
            arb.scan_live_kalshi(limit=1)
