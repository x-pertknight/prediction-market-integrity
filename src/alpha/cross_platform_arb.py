"""
src/alpha/cross_platform_arb.py
Cross-platform arbitrage detector for prediction markets.

Ref: Clinton & Huang (Vanderbilt 2025) — on 62/65 days before the 2024 election,
YES + NO prices did not sum to $1 on Kalshi/Polymarket.

Checks:
  1. Within-platform arb: YES_price + NO_price != $1
     If sum < $0.98: buy both sides (guaranteed profit)
     If sum > $1.02: prices inflated, avoid
  2. Cross-platform: same event on Kalshi vs Polymarket
     If prices diverge > 3¢: directional signal

Usage:
    from alpha.cross_platform_arb import CrossPlatformArb
    arb = CrossPlatformArb()
    signals = arb.check_within_platform(markets_df)
    cross_signals = arb.check_cross_platform(kalshi_df, poly_df)
"""

import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path
import requests
import time
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from risk.fees import polymarket_fee_from_schedule, kalshi_taker_fee  # noqa: E402

DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
POLY_BASE = "https://gamma-api.polymarket.com"
ARB_THRESHOLD_LOW  = 0.98  # buy both if sum < this
ARB_THRESHOLD_HIGH = 1.02  # avoid if sum > this
CROSS_DIVERGENCE   = 0.03  # 3¢ divergence signals cross-platform arb


class CrossPlatformArb:
    """
    Detects arbitrage opportunities in prediction markets.
    
    Signals:
      arb_within:  YES+NO < $0.98 (buy both for risk-free profit)
      arb_inflated: YES+NO > $1.02 (avoid, liquidity manipulation)
      arb_cross:   same event diverges > 3¢ across platforms
    """

    def check_within_platform(self, markets: pd.DataFrame,
                               yes_col: str = "yes_price",
                               no_col: str = "no_price") -> pd.DataFrame:
        """
        Check if YES+NO sums to $1.
        
        markets: DataFrame with yes_price, no_price (0-1 scale)
        Returns: markets with added columns:
          sum_price, arb_gap, arb_signal ('BUY_BOTH'|'INFLATED'|'FAIR')
        """
        out = markets.copy()
        if yes_col not in out.columns or no_col not in out.columns:
            # Try to infer from 'market_price' (YES price, NO = 1 - YES)
            if "market_price" in out.columns:
                out["yes_price"] = out["market_price"]
                out["no_price"]  = 1 - out["market_price"]
                yes_col, no_col  = "yes_price", "no_price"
            else:
                out["sum_price"]  = 1.0
                out["arb_gap"]    = 0.0
                out["arb_signal"] = "FAIR"
                return out

        out["sum_price"] = out[yes_col] + out[no_col]
        out["arb_gap"]   = out["sum_price"] - 1.0

        def classify(s):
            if s < ARB_THRESHOLD_LOW:
                return "BUY_BOTH"     # risk-free arb
            elif s > ARB_THRESHOLD_HIGH:
                return "INFLATED"     # liquidity manipulation, avoid
            else:
                return "FAIR"

        out["arb_signal"] = out["sum_price"].map(classify)
        return out

    def check_cross_platform(self, kalshi_markets: pd.DataFrame,
                              poly_markets: pd.DataFrame,
                              key_col: str = "event_slug") -> pd.DataFrame:
        """
        Compare prices for the same event across Kalshi and Polymarket.

        Both DataFrames should have: key_col, yes_price
        Returns: merged DataFrame with price divergence signal

        .. deprecated::
            Price divergence is NOT an arb signal by itself -- it just means
            the two markets disagree, or (more often) that ``key_col`` matched
            two markets whose rules aren't actually equivalent. Use
            :class:`RulesFirstCrossArb` instead, which only prices pairs whose
            settlement rules have been structurally confirmed EXACT.
        """
        warnings.warn(
            "check_cross_platform() treats price divergence as an arb signal, "
            "which conflates disagreement with mispricing. Use RulesFirstCrossArb "
            "for a rules-verified matcher.",
            DeprecationWarning, stacklevel=2,
        )
        if kalshi_markets.empty or poly_markets.empty:
            return pd.DataFrame()

        k = kalshi_markets[[key_col, "yes_price"]].rename(
            columns={"yes_price": "kalshi_yes"})
        p = poly_markets[[key_col, "yes_price"]].rename(
            columns={"yes_price": "poly_yes"})

        merged = k.merge(p, on=key_col, how="inner")
        if merged.empty:
            return pd.DataFrame()

        merged["price_divergence"] = abs(merged["kalshi_yes"] - merged["poly_yes"])
        merged["arb_direction"] = np.where(
            merged["price_divergence"] > CROSS_DIVERGENCE,
            np.where(merged["kalshi_yes"] < merged["poly_yes"], "BUY_KALSHI", "BUY_POLY"),
            "NO_ARB"
        )
        return merged[merged["price_divergence"] > CROSS_DIVERGENCE].reset_index(drop=True)

    def scan_live_kalshi(self, limit: int = 200) -> pd.DataFrame:
        """
        Fetch live Kalshi markets and check within-platform arb.
        Returns markets with arb_signal column.

        .. deprecated::
            Reads Kalshi's DEMO API -- quotes there are not representative of
            real tradable prices. Use :class:`KalshiProdClient` (PROD, public,
            no auth required) instead.
        """
        warnings.warn(
            "scan_live_kalshi() reads Kalshi's DEMO API, whose quotes are not "
            "real prices. Use KalshiProdClient against the PROD API instead.",
            DeprecationWarning, stacklevel=2,
        )
        markets = []
        cursor  = ""
        fetched = 0

        while fetched < limit:
            params: dict = {"limit": min(100, limit - fetched), "status": "open"}
            if cursor:
                params["cursor"] = cursor
            try:
                r = requests.get(f"{DEMO_BASE}/markets", params=params, timeout=10)
                if r.status_code != 200:
                    break
                data   = r.json()
                batch  = data.get("markets", [])
                cursor = data.get("cursor", "")
                for m in batch:
                    markets.append({
                        "ticker":    m.get("ticker"),
                        "title":     m.get("title", ""),
                        "category":  m.get("category", ""),
                        "yes_price": m.get("yes_ask_dollars", m.get("last_price_dollars", 0.5)),
                        "no_price":  1 - m.get("yes_ask_dollars", m.get("last_price_dollars", 0.5)),
                    })
                fetched += len(batch)
                time.sleep(0.05)
                if not cursor or not batch:
                    break
            except Exception as e:
                print(f"[CrossArb] Kalshi scan error: {e}")
                break

        if not markets:
            # Return mock data for testing
            markets = _mock_markets()

        df = pd.DataFrame(markets)
        return self.check_within_platform(df)

    def arb_summary(self, df: pd.DataFrame) -> dict:
        """Return summary stats of arbitrage scan."""
        if df.empty or "arb_signal" not in df.columns:
            return {}
        return {
            "total_markets":  len(df),
            "buy_both":       (df["arb_signal"] == "BUY_BOTH").sum(),
            "inflated":       (df["arb_signal"] == "INFLATED").sum(),
            "fair":           (df["arb_signal"] == "FAIR").sum(),
            "avg_gap":        round(df["arb_gap"].mean(), 4),
            "max_gap":        round(df["arb_gap"].abs().max(), 4),
        }

    def confidence_score(self, own_price: float,
                         other_price: float | None) -> float:
        """Return confidence score based on cross-venue price agreement.

        High confidence (near 1.0) when prices agree (low divergence).
        Low confidence (near 0.0) when prices diverge > 10 cents.
        Returns 0.5 (neutral) when no cross-venue match available.
        """
        if other_price is None:
            return 0.5
        divergence = abs(own_price - other_price)
        return float(np.clip(1.0 - divergence / 0.10, 0.0, 1.0))



# ── Rules-first matcher (added 2026-07-08) ──────────────────────────────────
#
# Fixes three problems in the class above: (1) DEMO API quotes aren't real
# prices -- use KalshiProdClient (PROD, public, no auth needed, verified
# 2026-07-08: /markets, /events/{ticker}, /markets/{ticker}/orderbook all
# return 200 without credentials); (2) title similarity is not rules
# equivalence -- matching is now two-stage, fuzzy candidates then a
# structured rules diff; (3) price divergence is not arb -- only pairs whose
# rules diff to EXACT get priced, using real fees (risk.fees) and a
# documented conservative slippage haircut, never a flat spread guess.

KALSHI_PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"

FUZZY_CANDIDATE_THRESHOLD = 75.0   # rapidfuzz 0-100 scale == spec's 0.75
# WRatio (not token_sort_ratio): question+rules text differs a lot in length
# and boilerplate between Kalshi and Polymarket for the same real event --
# verified 2026-07-08 on a real pair (Kalshi "Will the Spain win the 2026
# Men's World Cup?" vs Polymarket "Will Spain win the 2026 FIFA World Cup?"):
# token_sort_ratio scored 47 (would have been filtered out entirely) while
# WRatio scored 85. WRatio is rapidfuzz's composite scorer built for exactly
# this differing-length/format comparison.
MAX_CANDIDATES = 5

# Conservative fixed haircut for both legs combined. This is NOT a market-
# impact model -- src/risk/market_impact.py sizes impact against a position
# size and daily volume, neither of which exists yet for an unplaced pair.
# Sizing/execution is explicitly out of scope here (no wiring, see module
# docstring of the calling script); this constant only keeps the scan-time
# net-spread estimate from being overstated by a full tick's worth on paper.
CONSERVATIVE_SLIPPAGE = 0.01

ADVANCE_KEYWORDS = (
    "advance", "advances", "advancing", "qualify", "qualifies", "qualifying",
    "reach the", "move on", "progress to", "make it to", "clinch a spot",
)
WIN_KEYWORDS = (
    "champion", "championship winner", "win it all", "clinch the title", "tournament winner",
)
# Tolerates realistic real-world insertions between "win(s) the" and the
# prize noun -- verified 2026-07-08 against real Kalshi text: "wins the 2026
# Men's World Cup" does NOT contain the plain substring "wins the world cup",
# so a fixed-phrase list alone under-detects it (a safe miss -> UNKNOWN, but
# a miss all the same).
WIN_PATTERN = re.compile(
    r"wins?\s+the\s+(?:\d{4}\s+)?(?:men'?s\s+|women'?s\s+)?"
    r"(?:world\s+cup|tournament|championship|title|election|presidency|nomination|cup)\b",
    re.I,
)

EDGE_CASE_PATTERNS = {
    "cancellation": re.compile(r"\bcancel", re.I),
    "postponement": re.compile(r"\bpostpon|reschedul", re.I),
    "tie":          re.compile(r"\btie\b|\bdraw\b|50-50|50/50", re.I),
    "void":         re.compile(r"\bvoid\b|refund", re.I),
}

THRESHOLD_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:goals?|points?|wins?|votes?|seats?)", re.I)


def _extract_outcome_bucket(text: str) -> str | None:
    """Classify outcome wording into a coarse bucket. Returns None (unknown)
    rather than guessing when neither bucket's keywords appear -- this is
    the field that must catch "advance" vs "won" traps, so it never
    defaults to a match."""
    t = text.lower()
    if any(k in t for k in WIN_KEYWORDS) or WIN_PATTERN.search(t):
        return "WIN"
    if any(k in t for k in ADVANCE_KEYWORDS):
        return "ADVANCE"
    return None


def _extract_threshold(text: str) -> float | None:
    m = THRESHOLD_PATTERN.search(text)
    return float(m.group(1)) if m else None


def _extract_edge_cases(text: str) -> frozenset:
    return frozenset(name for name, pat in EDGE_CASE_PATTERNS.items() if pat.search(text))


@dataclass
class MarketRules:
    """Normalized, comparable rules fields extracted from one market."""
    source: str            # "kalshi" | "polymarket"
    market_id: str
    question: str
    rules_text: str
    deadline_utc: "datetime | None" = None
    settlement_source: "str | None" = None
    threshold: "float | None" = None
    outcome_bucket: "str | None" = None   # "ADVANCE" | "WIN" | None (unknown)
    edge_cases: frozenset = field(default_factory=frozenset)

    @property
    def match_text(self) -> str:
        return f"{self.question} {self.rules_text}"


def kalshi_market_to_rules(market: dict, settlement_sources: "list[dict] | None" = None) -> MarketRules:
    """Build MarketRules from a Kalshi PROD /markets entry. Pass the parent
    event's `settlement_sources` (from GET /events/{ticker}) when available."""
    rules_text = " ".join(filter(None, [market.get("rules_primary", ""), market.get("rules_secondary", "")]))
    settlement_source = None
    if settlement_sources:
        names = [s.get("name", "") for s in settlement_sources if s.get("name")]
        settlement_source = "; ".join(names) if names else None
    # expected_expiration_time (when the market is actually expected to
    # resolve) beats close_time (trading close) beats expiration_time (a
    # far-future backstop -- verified 2026-07-08 on KXMENWORLDCUP: set ~2
    # years out, unrelated to the actual tournament final date).
    deadline_raw = (market.get("expected_expiration_time") or market.get("close_time")
                    or market.get("expiration_time"))
    deadline = None
    if deadline_raw:
        try:
            deadline = datetime.fromisoformat(deadline_raw.replace("Z", "+00:00"))
        except ValueError:
            deadline = None
    threshold = market.get("floor_strike")
    if threshold is None:
        threshold = _extract_threshold(rules_text)
    full_text = f"{market.get('title', '')} {rules_text}"
    return MarketRules(
        source="kalshi",
        market_id=market.get("ticker", ""),
        question=market.get("title", ""),
        rules_text=rules_text,
        deadline_utc=deadline,
        settlement_source=settlement_source,
        threshold=threshold,
        outcome_bucket=_extract_outcome_bucket(full_text),
        edge_cases=_extract_edge_cases(rules_text),
    )


def polymarket_market_to_rules(market: dict) -> MarketRules:
    """Build MarketRules from a Polymarket Gamma /markets entry."""
    question = market.get("question", "")
    description = market.get("description", "") or ""
    settlement_source = market.get("resolutionSource") or None
    if not settlement_source:
        m = re.search(r"resolution source[^.]*\.", description, re.I)
        settlement_source = m.group(0) if m else None
    deadline = None
    end_date = market.get("endDate")
    if end_date:
        try:
            deadline = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            deadline = None
    full_text = f"{question} {description}"
    return MarketRules(
        source="polymarket",
        market_id=market.get("conditionId") or market.get("id", ""),
        question=question,
        rules_text=description,
        deadline_utc=deadline,
        settlement_source=settlement_source,
        threshold=_extract_threshold(full_text),
        outcome_bucket=_extract_outcome_bucket(full_text),
        edge_cases=_extract_edge_cases(description),
    )


@dataclass
class FieldVerdict:
    field: str
    verdict: str     # "MATCH" | "DIVERGENT" | "UNKNOWN"
    detail: str = ""


@dataclass
class RuleDiffResult:
    poly_market_id: str
    kalshi_ticker: str
    fuzzy_score: float
    fields: list        # list[FieldVerdict]
    verdict: str         # "EXACT" | "DIVERGENT" | "UNKNOWN"


_CORE_FIELDS = ("outcome_definition", "threshold", "deadline_utc")


def diff_rules(poly: MarketRules, kalshi: MarketRules, fuzzy_score: float) -> RuleDiffResult:
    """Conservative, rule-based field-by-field comparison.

    EXACT requires every core field (outcome_definition, threshold,
    deadline_utc) to confirm MATCH -- never granted from fuzzy_score alone.
    A single confidently-contradicting field forces DIVERGENT overall.
    Fields this repo cannot judge without an LLM (settlement_source prose,
    resolution_process) always report UNKNOWN, and UNKNOWN never escalates
    to EXACT.
    """
    fields = []

    # outcome_definition -- the "advanced" vs "won" trap lives here.
    if poly.outcome_bucket and kalshi.outcome_bucket:
        if poly.outcome_bucket == kalshi.outcome_bucket:
            fields.append(FieldVerdict("outcome_definition", "MATCH", f"both {poly.outcome_bucket}"))
        else:
            fields.append(FieldVerdict("outcome_definition", "DIVERGENT",
                                        f"poly={poly.outcome_bucket} kalshi={kalshi.outcome_bucket}"))
    else:
        fields.append(FieldVerdict("outcome_definition", "UNKNOWN",
                                    "outcome wording not confidently classified on one or both sides"))

    # threshold -- absence on BOTH sides means "not a threshold market" on
    # both, which is a legitimate match, not missing data.
    if poly.threshold is None and kalshi.threshold is None:
        fields.append(FieldVerdict("threshold", "MATCH", "neither market is threshold-based"))
    elif poly.threshold is not None and kalshi.threshold is not None:
        if abs(poly.threshold - kalshi.threshold) < 1e-6:
            fields.append(FieldVerdict("threshold", "MATCH", str(poly.threshold)))
        else:
            fields.append(FieldVerdict("threshold", "DIVERGENT",
                                        f"poly={poly.threshold} kalshi={kalshi.threshold}"))
    else:
        fields.append(FieldVerdict("threshold", "UNKNOWN",
                                    "threshold present on only one side -- ambiguous"))

    # deadline_utc -- tolerate up to 48h apart (source clock/rounding noise)
    if poly.deadline_utc and kalshi.deadline_utc:
        delta_h = abs((poly.deadline_utc - kalshi.deadline_utc).total_seconds()) / 3600
        if delta_h <= 48:
            fields.append(FieldVerdict("deadline_utc", "MATCH", f"{delta_h:.1f}h apart"))
        else:
            fields.append(FieldVerdict("deadline_utc", "DIVERGENT", f"{delta_h:.1f}h apart"))
    else:
        fields.append(FieldVerdict("deadline_utc", "UNKNOWN", "deadline missing on one or both sides"))

    # edge_cases -- shared mentions are a MATCH signal; we don't attempt to
    # detect genuine same-category contradictions without an LLM, so this
    # never reports DIVERGENT (would risk false rejects on mere silence).
    shared = poly.edge_cases & kalshi.edge_cases
    if shared:
        fields.append(FieldVerdict("edge_cases", "MATCH", f"both mention {sorted(shared)}"))
    else:
        fields.append(FieldVerdict("edge_cases", "UNKNOWN", "no shared edge-case language detected"))

    # Prose fields with no rule-based equivalence check available.
    fields.append(FieldVerdict("settlement_source", "UNKNOWN",
                                "prose comparison needs an LLM judge, not available in-repo"))
    fields.append(FieldVerdict("resolution_process", "UNKNOWN",
                                "prose comparison needs an LLM judge, not available in-repo"))

    if any(f.verdict == "DIVERGENT" for f in fields):
        verdict = "DIVERGENT"
    else:
        core = {f.field: f.verdict for f in fields if f.field in _CORE_FIELDS}
        verdict = "EXACT" if all(core.get(cf) == "MATCH" for cf in _CORE_FIELDS) else "UNKNOWN"

    return RuleDiffResult(
        poly_market_id=poly.market_id,
        kalshi_ticker=kalshi.market_id,
        fuzzy_score=fuzzy_score,
        fields=fields,
        verdict=verdict,
    )


def fuzzy_candidates(target: MarketRules, candidates: "list[MarketRules]",
                      threshold: float = FUZZY_CANDIDATE_THRESHOLD,
                      max_candidates: int = MAX_CANDIDATES) -> "list[tuple[MarketRules, float]]":
    """Stage 1: fuzzy text similarity narrows candidates. This NEVER decides
    the match verdict by itself -- see diff_rules() for stage 2."""
    scored = []
    for c in candidates:
        score = fuzz.WRatio(target.match_text, c.match_text)
        if score >= threshold:
            scored.append((c, score))
    scored.sort(key=lambda pair: -pair[1])
    return scored[:max_candidates]


@dataclass
class ArbResult:
    poly_market_id: str
    kalshi_ticker: str
    poly_leg_price: float
    kalshi_leg_price: float
    poly_fee: float
    kalshi_fee: float
    slippage: float
    net_spread: float
    is_arb: bool


def compute_exact_pair_arb(poly_yes_price: float, poly_fee_schedule: "dict | None",
                            kalshi_no_price: float, poly_market_id: str = "",
                            kalshi_ticker: str = "") -> ArbResult:
    """Price an EXACT-matched pair: buy Polymarket YES + Kalshi NO on the
    same underlying event -- exactly one leg pays $1 at resolution.

    net_spread = 1 - (poly_yes_price + kalshi_no_price) - poly_fee -
                 kalshi_fee - CONSERVATIVE_SLIPPAGE

    Fees are the real per-market Polymarket schedule and Kalshi's parabolic
    taker fee (src/risk/fees.py) -- never a flat estimate.
    """
    poly_fee = polymarket_fee_from_schedule(poly_yes_price, poly_fee_schedule)
    kalshi_fee = kalshi_taker_fee(kalshi_no_price)
    gross = 1.0 - (poly_yes_price + kalshi_no_price)
    net_spread = gross - poly_fee - kalshi_fee - CONSERVATIVE_SLIPPAGE
    return ArbResult(
        poly_market_id=poly_market_id,
        kalshi_ticker=kalshi_ticker,
        poly_leg_price=poly_yes_price,
        kalshi_leg_price=kalshi_no_price,
        poly_fee=poly_fee,
        kalshi_fee=kalshi_fee,
        slippage=CONSERVATIVE_SLIPPAGE,
        net_spread=float(net_spread),
        is_arb=bool(net_spread > 0),
    )


class KalshiProdClient:
    """Minimal client for Kalshi's PROD public market data.

    Verified 2026-07-08: /markets, /events/{ticker}, and
    /markets/{ticker}/orderbook all return 200 without an API key -- this is
    public market data, unlike the trading/portfolio endpoints.
    """

    def __init__(self, base: str = KALSHI_PROD_BASE):
        self.base = base

    def get_markets(self, status: str = "open", limit: int = 200, max_pages: int = 20,
                    exclude_combo: bool = True, series_ticker: "str | None" = None) -> list:
        """Fetch single-event markets. Kalshi's "MVE" (Multi-Value Event)
        combo/parlay products are excluded by default -- they bundle several
        unrelated legs under one ticker and have no single-event analog on
        Polymarket to match against.

        As of 2026-07-08 the unfiltered open-markets feed is almost entirely
        MVE combo products (verified: 8000/8000 sampled markets were MVE) --
        pass `series_ticker` to target a specific single-market series
        instead of paging through the combo firehose looking for needles.
        """
        markets, cursor, pages = [], "", 0
        while pages < max_pages and len(markets) < limit:
            params = {"limit": 200, "status": status}
            if series_ticker:
                params["series_ticker"] = series_ticker
            if cursor:
                params["cursor"] = cursor
            r = requests.get(f"{self.base}/markets", params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            batch = data.get("markets", [])
            if exclude_combo:
                batch = [m for m in batch if "MVE" not in m.get("ticker", "")]
            markets.extend(batch)
            cursor = data.get("cursor", "")
            pages += 1
            if not cursor or not data.get("markets", []):
                break
        return markets[:limit]

    def get_event_settlement_sources(self, event_ticker: str) -> list:
        r = requests.get(f"{self.base}/events/{event_ticker}", timeout=10)
        r.raise_for_status()
        return r.json().get("event", {}).get("settlement_sources", [])


def fetch_polymarket_markets(limit: int = 200) -> list:
    """Fetch active, open Polymarket markets from the Gamma API.

    The Gamma /markets endpoint caps each response at 100 rows regardless of
    the requested `limit`, so this paginates via `offset`.
    """
    markets: list = []
    offset = 0
    page_size = 100
    while len(markets) < limit:
        r = requests.get(f"{POLY_BASE}/markets",
                          params={"limit": page_size, "active": "true", "closed": "false", "offset": offset},
                          timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        markets.extend(batch)
        offset += page_size
    return markets[:limit]


class RulesFirstCrossArb:
    """Rules-first cross-platform matcher + arb scorer.

    match() runs both stages and returns every candidate pair with its
    verdict (EXACT/DIVERGENT/UNKNOWN) for logging/watchlist purposes.
    price_exact_pairs() prices ONLY the EXACT pairs -- DIVERGENT and UNKNOWN
    never produce a trading signal.
    """

    def __init__(self, kalshi_client: "KalshiProdClient | None" = None):
        self.kalshi = kalshi_client or KalshiProdClient()
        self._settlement_source_cache: dict = {}

    def _settlement_sources_for(self, event_ticker: str) -> list:
        if not event_ticker:
            return []
        if event_ticker not in self._settlement_source_cache:
            try:
                self._settlement_source_cache[event_ticker] = \
                    self.kalshi.get_event_settlement_sources(event_ticker)
            except requests.RequestException:
                self._settlement_source_cache[event_ticker] = []
        return self._settlement_source_cache[event_ticker]

    def match(self, poly_markets: list, kalshi_markets: list) -> "list[RuleDiffResult]":
        poly_rules = [polymarket_market_to_rules(m) for m in poly_markets]
        kalshi_rules = [
            kalshi_market_to_rules(m, self._settlement_sources_for(m.get("event_ticker", "")))
            for m in kalshi_markets
        ]
        results = []
        for p in poly_rules:
            for k, score in fuzzy_candidates(p, kalshi_rules):
                results.append(diff_rules(p, k, score))
        return results

    def price_exact_pairs(self, results: "list[RuleDiffResult]",
                           poly_by_id: dict, kalshi_by_ticker: dict) -> "list[ArbResult]":
        priced = []
        for r in results:
            if r.verdict != "EXACT":
                continue
            poly_m = poly_by_id.get(r.poly_market_id)
            kalshi_m = kalshi_by_ticker.get(r.kalshi_ticker)
            if not poly_m or not kalshi_m:
                continue
            try:
                poly_yes = float(json.loads(poly_m.get("outcomePrices", "[]"))[0])
            except (IndexError, ValueError, json.JSONDecodeError):
                continue
            kalshi_no = kalshi_m.get("no_ask_dollars")
            if kalshi_no is None:
                continue
            priced.append(compute_exact_pair_arb(
                poly_yes, poly_m.get("feeSchedule"), float(kalshi_no),
                poly_market_id=r.poly_market_id, kalshi_ticker=r.kalshi_ticker,
            ))
        return priced


def run_rules_first_scan(poly_limit: int = 200, kalshi_limit: int = 200,
                          kalshi_series_tickers: "list[str] | None" = None) -> dict:
    """End-to-end scan: fetch live Polymarket + Kalshi PROD markets, match,
    price EXACT pairs. Read-only -- places no orders, does no sizing.

    kalshi_series_tickers: fetch these specific series instead of the
    unfiltered open-markets feed. Recommended -- see get_markets() docstring
    on why the generic feed is currently useless for single-event matching.
    """
    poly_markets = fetch_polymarket_markets(limit=poly_limit)
    kalshi = KalshiProdClient()
    if kalshi_series_tickers:
        kalshi_markets = []
        per_series = max(1, kalshi_limit // len(kalshi_series_tickers))
        for ticker in kalshi_series_tickers:
            kalshi_markets.extend(kalshi.get_markets(status="open", limit=per_series, series_ticker=ticker))
    else:
        kalshi_markets = kalshi.get_markets(status="open", limit=kalshi_limit)

    matcher = RulesFirstCrossArb(kalshi_client=kalshi)
    results = matcher.match(poly_markets, kalshi_markets)

    poly_by_id = {m.get("conditionId") or m.get("id", ""): m for m in poly_markets}
    kalshi_by_ticker = {m.get("ticker", ""): m for m in kalshi_markets}
    priced = matcher.price_exact_pairs(results, poly_by_id, kalshi_by_ticker)

    counts = {"EXACT": 0, "DIVERGENT": 0, "UNKNOWN": 0}
    for r in results:
        counts[r.verdict] += 1

    return {
        "n_poly_markets": len(poly_markets),
        "n_kalshi_markets": len(kalshi_markets),
        "n_candidate_pairs": len(results),
        "verdict_counts": counts,
        "results": results,
        "priced_arb": priced,
    }


def _mock_markets(n: int = 50) -> list[dict]:
    """Generate mock market data for testing."""
    import random
    random.seed(42)
    markets = []
    for i in range(n):
        yes = random.uniform(0.05, 0.95)
        # Occasionally inject arb (10% chance)
        if random.random() < 0.1:
            no = random.uniform(0.01, 0.04)  # sum < 1 → arb
        else:
            no = round(1 - yes + random.uniform(-0.01, 0.01), 3)
        markets.append({
            "ticker":    f"MOCK-{i:03d}",
            "title":     f"Mock Market {i}",
            "category":  random.choice(["Crypto", "Financials", "Sports"]),
            "yes_price": yes,
            "no_price":  max(0, no),
        })
    return markets


def run() -> None:
    arb = CrossPlatformArb()
    print("[CrossArb] Scanning live Kalshi markets for within-platform arbitrage...")
    df = arb.scan_live_kalshi(limit=200)
    summary = arb.arb_summary(df)
    print(f"[CrossArb] Scanned {summary.get('total_markets', 0)} markets")
    print(f"  BUY_BOTH (arb):  {summary.get('buy_both', 0)}")
    print(f"  INFLATED:        {summary.get('inflated', 0)}")
    print(f"  FAIR:            {summary.get('fair', 0)}")
    print(f"  Avg YES+NO gap:  {summary.get('avg_gap', 0):+.4f}")

    arb_markets = df[df["arb_signal"] == "BUY_BOTH"]
    if not arb_markets.empty:
        print("\n[CrossArb] Arbitrage opportunities:")
        for _, row in arb_markets.head(5).iterrows():
            print(f"  {row['ticker']}: YES={row.get('yes_price',0):.3f} + "
                  f"NO={row.get('no_price',0):.3f} = {row['sum_price']:.3f} → {row['arb_signal']}")


if __name__ == "__main__":
    run()
