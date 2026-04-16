"""
src/alpha/combinator.py — Alpha Combination Framework.

IR = IC x sqrt(N). Combines multiple signals into a single probability estimate
using an 11-step Grinold-Kahn alpha combination procedure.

11-step procedure:
  1.  Collect raw signal scores for this observation
  2.  Append to rolling history buffer (window=200)
  3.  Demean each signal (subtract its rolling mean)
  4.  Normalize by rolling std -> z-scores
  5.  Cross-sectional demean (subtract mean across signals)
  6.  Compute correlation matrix from rolling window
  7.  Sequential regression: orthogonalize signals (Gram-Schmidt)
  8.  Compute residual std per signal
  9.  Weight: w(i) = |residual_i| / std(residual_i)
  10. Normalize weights to sum to 1
  11. Combined score = sigmoid(weighted z-sum) -> [0, 1]

Signals:
  1. RF model probability (predict_proba)
  2. HMM regime confidence
  3. Cross-venue pricing (Kalshi vs Polymarket spread)
  4. CLOB momentum (rate of price change)
  5. Calibration (historical resolution rate by category)
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("paper_trader")


# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    """Output from a single signal provider."""
    name: str
    confidence: float  # 0.0 - 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class CombinedResult:
    """Output from the AlphaCombinator."""
    probability: float         # combined probability 0.0-1.0
    weights: dict[str, float]  # per-signal weights (sum to 1)
    ir_estimate: float         # IR = IC x sqrt(N_effective)
    diagnostics: dict = field(default_factory=dict)


# ── Signal Provider Interface ────────────────────────────────────────────

class SignalProvider(ABC):
    """Base class for alpha signal providers.

    Each provider scores a market and returns a confidence value in [0, 1].
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def score(self, market: dict, context: dict) -> SignalResult:
        """Score a market.

        Args:
            market: market data dict (ticker, price, category, etc.)
            context: shared context (features, macro, model_prob, etc.)
        Returns:
            SignalResult with confidence in [0.0, 1.0]
        """
        ...


# ── AlphaCombinator ──────────────────────────────────────────────────────

class AlphaCombinator:
    """11-step alpha combination engine (Grinold-Kahn framework).

    Maintains a rolling window of signal observations. Below the window size,
    falls back to equal-weight averaging.
    """

    def __init__(self, window: int = 50):
        self.window = window
        self.history: list[dict[str, float]] = []

    @property
    def ready(self) -> bool:
        """True when enough history for full 11-step combination."""
        return len(self.history) >= self.window

    def combine(self, signals: dict[str, float]) -> CombinedResult:
        """Combine multiple signal confidence scores into one probability.

        Args:
            signals: dict of signal_name -> confidence (0.0-1.0)
        Returns:
            CombinedResult with combined probability, weights, IR estimate
        """
        if not signals:
            return CombinedResult(
                probability=0.5, weights={}, ir_estimate=0.0,
                diagnostics={"error": "no signals", "fallback": True},
            )

        # Steps 1-2: collect and buffer
        self.history.append(signals.copy())
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]

        if not self.ready:
            return self._equal_weight_combine(signals)

        return self._full_combine(signals)

    # ── Fallback ─────────────────────────────────────────────────────

    def _equal_weight_combine(self, signals: dict[str, float]) -> CombinedResult:
        """Equal-weight average when history is insufficient."""
        names = sorted(signals.keys())
        n = len(names)
        weights = {name: 1.0 / n for name in names}
        combined = sum(signals[name] * weights[name] for name in names)
        return CombinedResult(
            probability=float(np.clip(combined, 0.01, 0.99)),
            weights=weights,
            ir_estimate=0.0,
            diagnostics={
                "n_observations": len(self.history),
                "fallback": True,
                "reason": f"need {self.window} obs, have {len(self.history)}",
            },
        )

    # ── Full 11-step engine ──────────────────────────────────────────

    def _full_combine(self, signals: dict[str, float]) -> CombinedResult:
        names = sorted(signals.keys())
        n_signals = len(names)

        # Build history matrix: (window, n_signals)
        history_matrix = np.array([
            [obs.get(name, 0.5) for name in names]
            for obs in self.history
        ])
        current = np.array([signals[name] for name in names])

        # Step 3: demean (subtract rolling mean)
        means = history_matrix.mean(axis=0)
        demeaned = current - means

        # Step 4: normalize by std -> z-scores
        stds = history_matrix.std(axis=0)
        stds = np.where(stds < 1e-8, 1.0, stds)
        z_scores = demeaned / stds

        # Step 5: cross-sectional demean
        cs_mean = z_scores.mean()
        z_cs = z_scores - cs_mean

        # Step 6: correlation matrix
        demeaned_history = history_matrix - means
        z_history = demeaned_history / stds

        # Step 7: sequential regression (orthogonalize via Gram-Schmidt)
        residuals = np.zeros(n_signals)
        residual_stds = np.zeros(n_signals)

        for i in range(n_signals):
            if i == 0:
                residuals[i] = z_cs[i]
                residual_stds[i] = max(z_history[:, i].std(), 1e-8)
            else:
                X = z_history[:, :i]
                y = z_history[:, i]
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    predicted = z_cs[:i] @ beta
                    residuals[i] = z_cs[i] - predicted
                    hist_resid = y - X @ beta
                    residual_stds[i] = max(hist_resid.std(), 1e-8)
                except np.linalg.LinAlgError:
                    residuals[i] = z_cs[i]
                    residual_stds[i] = max(z_history[:, i].std(), 1e-8)

        # Step 8: residual stds already computed above

        # Step 9: precision-weighted residuals
        raw_weights = np.abs(residuals) / residual_stds

        # Step 10: normalize weights
        weight_sum = raw_weights.sum()
        if weight_sum < 1e-8:
            weights = np.ones(n_signals) / n_signals
        else:
            weights = raw_weights / weight_sum

        # Step 11: combined score via sigmoid
        combined_z = float(np.sum(weights * z_cs))
        combined_prob = 1.0 / (1.0 + np.exp(-combined_z))

        # IR estimate: IC x sqrt(N_effective)
        corr_matrix = np.corrcoef(z_history.T)
        if corr_matrix.ndim < 2:
            # Single signal: N_effective = 1 by definition
            n_effective = 1.0
        else:
            try:
                eigenvalues = np.linalg.eigvalsh(corr_matrix)
                eigenvalues = eigenvalues[eigenvalues > 0]
                n_effective = float(
                    (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
                ) if len(eigenvalues) > 0 else float(n_signals)
            except np.linalg.LinAlgError:
                n_effective = float(n_signals)
        mean_ic = float(np.mean(np.abs(residuals) / residual_stds))
        ir_estimate = mean_ic * np.sqrt(n_effective)

        weight_dict = {names[i]: float(weights[i]) for i in range(n_signals)}

        return CombinedResult(
            probability=float(np.clip(combined_prob, 0.01, 0.99)),
            weights=weight_dict,
            ir_estimate=float(ir_estimate),
            diagnostics={
                "n_observations": len(self.history),
                "n_effective": n_effective,
                "mean_ic": mean_ic,
                "combined_z": combined_z,
                "fallback": False,
            },
        )


# ── Concrete Signal Providers ────────────────────────────────────────────

class RFSignalProvider(SignalProvider):
    """Wraps sklearn RF model. predict_proba already returns [0,1]."""

    @property
    def name(self) -> str:
        return "rf_model"

    def __init__(self, model):
        self._model = model

    def score(self, market: dict, context: dict) -> SignalResult:
        features = context.get("features")
        if features is None or self._model is None:
            return SignalResult(name=self.name, confidence=0.5)
        try:
            prob = float(self._model.predict_proba(features)[0][1])
            return SignalResult(
                name=self.name,
                confidence=float(np.clip(prob, 0.0, 1.0)),
                metadata={"raw_prob": prob},
            )
        except Exception as e:
            return SignalResult(
                name=self.name, confidence=0.5, metadata={"error": str(e)},
            )


class HMMSignalProvider(SignalProvider):
    """Maps HMM regime to continuous confidence score.

    Uses HMMRegimeDetector.confidence_score() added in Task 3.
    Falls back to regime-to-confidence mapping if method unavailable.
    """

    REGIME_CONFIDENCE = {
        "MEAN_REVERTING": 0.7,
        "TRENDING": 0.5,
        "VOLATILE": 0.2,
    }

    @property
    def name(self) -> str:
        return "hmm_regime"

    def __init__(self, detector=None):
        self._detector = detector

    def score(self, market: dict, context: dict) -> SignalResult:
        if self._detector is None or not self._detector.fitted:
            return SignalResult(
                name=self.name, confidence=0.5,
                metadata={"regime": "UNKNOWN", "fitted": False},
            )
        returns = context.get("returns")
        if returns is None:
            return SignalResult(
                name=self.name, confidence=0.5,
                metadata={"regime": "UNKNOWN"},
            )
        if hasattr(self._detector, "confidence_score"):
            confidence = self._detector.confidence_score(returns)
            regime = self._detector.predict_current(returns)
        else:
            regime = self._detector.predict_current(returns)
            confidence = self.REGIME_CONFIDENCE.get(regime, 0.5)
        return SignalResult(
            name=self.name,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            metadata={
                "regime": regime,
                "kelly_mult": self._detector.kelly_multiplier(regime),
            },
        )


class CrossVenueSignalProvider(SignalProvider):
    """Cross-venue pricing: confidence from Kalshi vs Polymarket price agreement.

    High confidence when prices agree (low divergence).
    Returns 0.5 (neutral) when no cross-venue match found.
    Rejects matches below 90% word-level Jaccard similarity.
    """

    SIMILARITY_THRESHOLD = 0.90

    @property
    def name(self) -> str:
        return "cross_venue"

    def score(self, market: dict, context: dict) -> SignalResult:
        cross_prices = context.get("cross_venue_prices")
        if not cross_prices:
            return SignalResult(
                name=self.name, confidence=0.5, metadata={"matched": False},
            )

        market_title = (
            market.get("title", "") or market.get("question", "")
        )
        best_match = self._find_match(market_title, cross_prices)

        if best_match is None:
            return SignalResult(
                name=self.name, confidence=0.5, metadata={"matched": False},
            )

        other_price = best_match["price"]
        own_price = float(
            market.get("last_price", market.get("yes_price", 0.5))
        )
        divergence = abs(own_price - other_price)
        confidence = float(np.clip(1.0 - divergence / 0.10, 0.0, 1.0))

        return SignalResult(
            name=self.name,
            confidence=confidence,
            metadata={
                "matched": True,
                "other_price": other_price,
                "own_price": own_price,
                "divergence": divergence,
                "match_title": best_match.get("title", ""),
                "similarity": best_match.get("similarity", 0),
            },
        )

    def _find_match(
        self, title: str, cross_prices: list[dict]
    ) -> dict | None:
        if not title or not cross_prices:
            return None
        best = None
        best_sim = 0.0
        for cp in cross_prices:
            other_title = cp.get("title", "")
            sim = self._title_similarity(title, other_title)
            if sim > best_sim:
                best_sim = sim
                best = {**cp, "similarity": sim}
        if best and best_sim >= self.SIMILARITY_THRESHOLD:
            return best
        return None

    @staticmethod
    def _title_similarity(a: str, b: str) -> float:
        """Word-level Jaccard similarity (stop words removed)."""
        if not a or not b:
            return 0.0
        stop = {
            "the", "a", "an", "in", "on", "at", "to", "for",
            "of", "is", "will", "be", "by", "or", "and",
        }
        words_a = set(a.lower().split()) - stop
        words_b = set(b.lower().split()) - stop
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)


class MomentumSignalProvider(SignalProvider):
    """CLOB momentum: rate of price change from persisted history buffer.

    Positive momentum -> confidence > 0.5 (bullish).
    Negative momentum -> confidence < 0.5 (bearish).
    No history -> 0.5 (neutral).
    Delegates buffer management to MomentumTracker.
    """

    LOOKBACK = 12       # ~1 hour of 5-min ticks

    @property
    def name(self) -> str:
        return "momentum"

    def __init__(self, tracker=None, buffer_path: Path | None = None):
        if tracker is not None:
            self._tracker = tracker
        else:
            from data.momentum_tracker import MomentumTracker
            self._tracker = MomentumTracker(
                path=buffer_path or Path("data/momentum_history.json")
            )

    def score(self, market: dict, context: dict) -> SignalResult:
        market_id = market.get("ticker", market.get("condition_id", ""))
        price = float(
            market.get("last_price", market.get("yes_price", 0.5))
        )

        if not market_id:
            return SignalResult(name=self.name, confidence=0.5)

        self._tracker.record(market_id, price, time.time())

        hourly_slope = self._tracker.compute_momentum(
            market_id, lookback=self.LOOKBACK,
        )
        if hourly_slope is None:
            history = self._tracker.get_history(market_id)
            return SignalResult(
                name=self.name, confidence=0.5,
                metadata={"observations": len(history)},
            )

        # Sigmoid mapping: positive slope -> confidence > 0.5
        confidence = 1.0 / (1.0 + np.exp(-hourly_slope / 0.01))

        history = self._tracker.get_history(market_id)
        recent = history[-self.LOOKBACK:]
        prices = [h["price"] for h in recent]

        return SignalResult(
            name=self.name,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            metadata={
                "hourly_slope": hourly_slope,
                "observations": len(history),
                "price_range": float(max(prices) - min(prices)),
            },
        )


class CalibrationSignalProvider(SignalProvider):
    """Calibration signal: bias-corrected probability by category.

    Uses CalibrationAnalyzer.adjusted_prob() to correct model overconfidence.
    Returns calibration-adjusted probability as confidence score.
    """

    @property
    def name(self) -> str:
        return "calibration"

    def __init__(self, analyzer=None):
        self._analyzer = analyzer

    def score(self, market: dict, context: dict) -> SignalResult:
        if self._analyzer is None:
            return SignalResult(name=self.name, confidence=0.5)

        raw_prob = context.get("model_prob", 0.5)
        category = market.get("category", "")

        if hasattr(self._analyzer, "confidence_score"):
            adjusted = self._analyzer.confidence_score(raw_prob, category)
        else:
            adjusted = self._analyzer.adjusted_prob(raw_prob, category)

        return SignalResult(
            name=self.name,
            confidence=float(np.clip(adjusted, 0.0, 1.0)),
            metadata={
                "raw_prob": raw_prob,
                "adjusted_prob": adjusted,
                "category": category,
            },
        )


class OBISignalProvider(SignalProvider):
    """Order Book Imbalance signal: bid/ask volume ratio predicts short-term price.

    OBI > 0.35 → bullish (confidence > 0.5)
    OBI < -0.35 → bearish (confidence < 0.5)
    Uses live orderbook when available, mock proxy from volume otherwise.
    """

    @property
    def name(self) -> str:
        return "obi"

    def __init__(self, obi_engine=None):
        if obi_engine is not None:
            self._obi = obi_engine
        else:
            from alpha.obi import OrderBookImbalance
            self._obi = OrderBookImbalance()

    def score(self, market: dict, context: dict) -> SignalResult:
        ticker = market.get("ticker", market.get("condition_id", ""))
        if not ticker:
            return SignalResult(name=self.name, confidence=0.5)

        try:
            result = self._obi.fetch_and_compute(ticker)
            obi_val = result.get("obi", 0.0)
            # Map OBI [-1, 1] to confidence [0, 1] via linear transform
            confidence = (obi_val + 1.0) / 2.0
            return SignalResult(
                name=self.name,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                metadata={
                    "obi": obi_val,
                    "ir": result.get("ir", 0.5),
                    "signal": result.get("signal", "NEUTRAL"),
                    "source": result.get("source", "unknown"),
                },
            )
        except Exception as e:
            return SignalResult(
                name=self.name, confidence=0.5,
                metadata={"error": str(e)},
            )


class WhaleSignalProvider(SignalProvider):
    """Whale/Smart Money signal: conviction from whale cluster consensus.

    Loads latest whale scan data. Markets with 3+ whales same direction
    get strong directional signal. SMI > 60 → bullish, SMI < 40 → bearish.
    Returns 0.5 (neutral) when no whale data available for a market.
    """

    @property
    def name(self) -> str:
        return "whale_smi"

    def __init__(self, whale_data_dir: str = "data/whale"):
        self._whale_dir = Path(whale_data_dir)
        self._latest: dict = {}
        self._load_latest()

    def _load_latest(self) -> None:
        """Load the most recent whale scan snapshot."""
        latest_path = self._whale_dir / "whale_latest.json"
        if latest_path.exists():
            try:
                self._latest = json.loads(latest_path.read_text())
            except Exception:
                self._latest = {}

    def score(self, market: dict, context: dict) -> SignalResult:
        if not self._latest:
            return SignalResult(name=self.name, confidence=0.5,
                                metadata={"available": False})

        # Match market to whale signals by slug/title similarity
        market_title = (
            market.get("title", "") or market.get("question", "")
        ).lower()
        market_slug = (market.get("slug", "") or "").lower()

        best_signal = None
        best_match = 0.0
        for signal in self._latest.get("top_signals", []):
            sig_slug = signal.get("market_slug", "").lower()
            sig_title = signal.get("title", "").lower()
            # Check slug match or title word overlap
            if sig_slug and (sig_slug in market_slug or market_slug in sig_slug):
                best_signal = signal
                break
            if sig_title and market_title:
                words_a = set(market_title.split())
                words_b = set(sig_title.split())
                if words_a and words_b:
                    overlap = len(words_a & words_b) / len(words_a | words_b)
                    if overlap > best_match and overlap > 0.5:
                        best_match = overlap
                        best_signal = signal

        if best_signal is None:
            return SignalResult(name=self.name, confidence=0.5,
                                metadata={"matched": False})

        # Map SMI (0-100) to confidence (0-1)
        smi = best_signal.get("smart_money_idx", 50.0)
        confidence = smi / 100.0

        return SignalResult(
            name=self.name,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            metadata={
                "matched": True,
                "smi": smi,
                "whale_count": best_signal.get("whale_count", 0),
                "conviction": best_signal.get("conviction_score", 0),
                "direction": best_signal.get("direction", ""),
                "strength": best_signal.get("signal_strength", ""),
            },
        )
