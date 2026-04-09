"""
src/execution/paper_trader.py — Paper trading against Kalshi Prod API.

Polls Kalshi Prod API every 60s for active markets.
Scores each with the trained RF model. Applies entry/exit signals,
fee model, market impact, and circuit breaker. Logs everything.

LIVE_MODE = False: paper trades only (log + CSV), never places real orders.

Usage:
  python src/execution/paper_trader.py           # foreground
  nohup python src/execution/paper_trader.py &   # background

Output:
  data/paper_trades.csv
  logs/paper_trader.log
"""

import base64
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from risk.fees import kalshi_taker_fee
from risk.market_impact import market_impact, should_trade
from risk.circuit_breaker import CircuitBreaker
from alerts.telegram_bot import (
    alert_entry_signal, alert_exit_signal, alert_circuit_breaker,
    alert_daily_summary, set_paper_trader, start_command_listener,
    start_daily_summary,
)
from data.kalshi_feature_adapter import (
    build_features as adapter_build_features,
    get_category,
    market_passes_filter,
    diagnose_features,
    set_api_client as adapter_set_api_client,
    MODEL_FEATURES,
)
from alpha.combinator import (
    AlphaCombinator, RFSignalProvider, HMMSignalProvider,
    CrossVenueSignalProvider, MomentumSignalProvider,
    CalibrationSignalProvider,
)
from alpha.cross_platform_arb import CrossPlatformArb
from models.calibration import CalibrationAnalyzer
from risk.kelly import kelly_with_cv

# ── Config ───────────────────────────────────────────────────────────────────

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
LIVE_MODE       = False   # NEVER set True without explicit user approval
POLL_INTERVAL   = 300     # seconds (5 min — series-based fetch takes ~90s)
EDGE_MULT       = 0.70    # entry: price <= model_prob * 0.70
EXIT_MULT       = 0.90    # exit:  price >= model_prob * 0.90
EXPIRY_EXIT_H   = 24      # exit if expiry <= 24 hours
TARGET_CATS     = {"weather", "crypto", "politics", "entertainment", "financials",
                   "sports", "economics", "elections"}
MODEL_DIR       = Path("models")
TRADES_CSV      = Path("data/paper_trades.csv")
LOG_DIR         = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
INITIAL_BANKROLL = 1000.0   # paper bankroll for Kelly sizing ($)

FEE_MODEL       = "kalshi"

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "paper_trader.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("paper_trader")


def _build_features(market: dict, macro: dict,
                    all_markets: list[dict] | None = None) -> np.ndarray:
    """Build feature vector using the Kalshi adapter."""
    return adapter_build_features(market, macro, all_markets)


def _load_macro() -> dict:
    """Load latest macro features from cached files."""
    macro = {}
    raw = Path("data/raw")

    # Fear & Greed
    fg = raw / "fear_greed.csv"
    if fg.exists():
        df = pd.read_csv(fg)
        if not df.empty:
            macro["fear_greed_value"] = float(df.iloc[-1].get("fear_greed_value", 50))

    # Binance BTC
    bn = raw / "binance_prices.csv"
    if bn.exists():
        df = pd.read_csv(bn)
        btc = df[df["symbol"] == "BTCUSDT"].tail(1)
        if not btc.empty:
            macro["btc_ret_1d"] = float(btc.iloc[-1].get("ret_1d", 0))
            macro["btc_ret_7d"] = float(btc.iloc[-1].get("ret_7d", 0))
            macro["btc_vol_20d"] = float(btc.iloc[-1].get("vol_20d", 0))
            macro["btc_zscore_30"] = float(btc.iloc[-1].get("price_zscore_30", 0))

    # FRED macro (dxy_proxy, fed_funds_rate, yield_10y)
    fred = raw / "fred_macro.csv"
    if fred.exists():
        df = pd.read_csv(fred)
        if not df.empty:
            last = df.iloc[-1]
            macro["dxy_proxy"] = float(last.get("dxy_proxy", 0) or 0)
            macro["fed_funds_rate"] = float(last.get("fed_funds_rate", 0) or 0)
            macro["yield_10y"] = float(last.get("yield_10y", 0) or 0)

    # Coinalyze (btc_funding_rate, btc_open_interest)
    coin = raw / "coinalyze.csv"
    if coin.exists():
        df = pd.read_csv(coin)
        if not df.empty:
            last = df.iloc[-1]
            macro["btc_funding_rate"] = float(last.get("funding_rate", 0) or 0)
            macro["btc_open_interest"] = float(last.get("open_interest", 0) or 0)

    return macro


# ── Kalshi Prod API (RSA-PSS auth) ───────────────────────────────────────────

_KALSHI_API_KEY = os.getenv("KALSHI_API_KEY", "")
_KALSHI_PEM_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
_private_key = None
_lib_api_client = None


def _init_library_client():
    """Create kalshi-python-sync ApiClient with RSA auth for adapter use."""
    global _lib_api_client
    if _lib_api_client is not None:
        return _lib_api_client
    from kalshi_python_sync import Configuration, ApiClient, KalshiAuth
    config = Configuration()
    _lib_api_client = ApiClient(configuration=config)
    # set_kalshi_auth is buggy (missing KalshiAuth import), set directly
    pem_text = Path(_KALSHI_PEM_PATH).read_text()
    _lib_api_client.kalshi_auth = KalshiAuth(_KALSHI_API_KEY, pem_text)
    adapter_set_api_client(_lib_api_client)
    log.info("kalshi-python-sync ApiClient initialized for adapter")
    return _lib_api_client


def _load_private_key():
    """Load RSA private key once at startup."""
    global _private_key
    if _private_key is not None:
        return _private_key
    if not _KALSHI_PEM_PATH or not Path(_KALSHI_PEM_PATH).exists():
        raise EnvironmentError(f"KALSHI_PRIVATE_KEY_PATH not found: {_KALSHI_PEM_PATH}")
    pem = Path(_KALSHI_PEM_PATH).read_bytes()
    _private_key = serialization.load_pem_private_key(pem, password=None)
    return _private_key


def _kalshi_headers(method: str, path: str) -> dict:
    """Generate RSA-PSS auth headers for a Kalshi API request."""
    key = _load_private_key()
    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + path).encode()
    sig = key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": _KALSHI_API_KEY,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


def _kalshi_get(path: str, params: dict | None = None,
                retries: int = 3) -> dict | None:
    """Authenticated GET request to Kalshi prod API."""
    url = KALSHI_API_BASE + path
    delay = 2.0
    for attempt in range(retries):
        try:
            headers = _kalshi_headers("GET", path)
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            log.warning(f"Kalshi API {path} returned {r.status_code}: {r.text[:100]}")
            return None
        except Exception as e:
            log.warning(f"Kalshi API error on {path}: {e}")
            time.sleep(delay)
            delay *= 2
    return None


# Kalshi category names that match our TARGET_CATS tags
_KALSHI_TARGET_CATEGORIES = {
    "Climate and Weather", "Crypto", "Politics", "Entertainment",
    "Financials", "Sports", "Economics", "Elections",
    "World", "Companies", "Science and Technology", "Health",
}

# Cached event metadata: refreshed hourly
_event_cache: dict[str, dict] = {}
_event_cache_ts: float = 0
_EVENT_CACHE_TTL = 3600  # 1 hour


def _refresh_events() -> dict[str, dict]:
    """Fetch all open events. Cached for 1 hour."""
    global _event_cache, _event_cache_ts
    if _event_cache and (time.time() - _event_cache_ts < _EVENT_CACHE_TTL):
        return _event_cache
    events = {}
    cursor = ""
    for _ in range(100):
        params = {"status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = _kalshi_get("/events", params)
        if not data:
            break
        for e in data.get("events", []):
            events[e.get("event_ticker", "")] = e
        cursor = data.get("cursor", "")
        if not cursor:
            break
        time.sleep(0.05)
    _event_cache = events
    _event_cache_ts = time.time()
    log.info(f"Refreshed event cache: {len(events)} events")
    return events


def fetch_markets() -> list[dict]:
    """Fetch active markets from Kalshi Prod via series-based strategy.

    Kalshi prod has 40K+ MVE markets that bury traditional ones.
    Strategy: events → target series_tickers → fetch markets per series.
    """
    event_map = _refresh_events()

    # Build series → category lookup from target-category events
    series_meta: dict[str, dict] = {}  # series_ticker → {category, event_tickers}
    for et, e in event_map.items():
        cat = e.get("category", "")
        if cat not in _KALSHI_TARGET_CATEGORIES:
            continue
        st = e.get("series_ticker", "")
        if not st:
            continue
        if st not in series_meta:
            series_meta[st] = {"category": cat, "series_ticker": st}

    log.info(f"Target series: {len(series_meta)} (from {len(event_map)} events)")

    # Fetch markets per series — limit to 300 series per cycle
    all_markets = []
    for i, (st, meta) in enumerate(series_meta.items()):
        if i >= 300:
            break
        cursor = ""
        for _ in range(5):  # up to 1000 markets per series
            params = {"series_ticker": st, "status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = _kalshi_get("/markets", params)
            if not data:
                break
            batch = data.get("markets", [])
            # Enrich each market with series-level metadata
            for m in batch:
                m["category"] = meta["category"]
                m["series_ticker"] = st
            all_markets.extend(batch)
            cursor = data.get("cursor", "")
            if not cursor:
                break
            time.sleep(0.03)
        time.sleep(0.03)

    return all_markets


def _market_price(m: dict) -> float:
    """Get current YES price from a Kalshi market.

    Kalshi prices are string dollars in 0-1 scale (e.g. "0.55").
    Priority: last_price > yes_ask > yes_bid.
    """
    for field_name in ("last_price", "last_price_dollars", "yes_ask", "yes_ask_dollars",
                       "yes_bid", "yes_bid_dollars"):
        val = m.get(field_name)
        if val is not None:
            try:
                p = float(val)
                if 0 < p < 1:
                    return p
            except (ValueError, TypeError):
                pass
    return 0.0


# ── Trade logging ────────────────────────────────────────────────────────────

_CSV_COLS = [
    "timestamp", "market_id", "market_question", "side", "entry_price",
    "model_prob", "combined_prob", "signal_weights", "ir_estimate",
    "convexity_score", "regime_hmm", "kelly_fraction", "fee", "impact",
    "exit_price", "pnl_net", "cumulative_pnl",
]


def _log_trade(row: dict) -> None:
    """Append a trade row to paper_trades.csv."""
    df = pd.DataFrame([row])
    header = not TRADES_CSV.exists()
    df.to_csv(TRADES_CSV, mode="a", header=header, index=False)


# ── Open positions ───────────────────────────────────────────────────────────

@dataclass
class PaperPosition:
    market_id: str
    question: str
    side: str
    entry_price: float
    model_prob: float
    combined_prob: float
    signal_weights: dict
    fee: float
    impact: float
    kelly_fraction: float = 0.0
    entry_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── Main loop ────────────────────────────────────────────────────────────────

class PaperTrader:
    def __init__(self):
        self.positions: dict[str, PaperPosition] = {}
        self.cumulative_pnl = 0.0
        self.bankroll = INITIAL_BANKROLL
        self.model = self._load_model()
        self.macro = _load_macro()
        self.cb = CircuitBreaker()
        self._running = True
        # Init library client so adapter can make authenticated prod API calls
        _init_library_client()
        # Alpha combination framework
        self.arb = CrossPlatformArb()
        self.combinator = AlphaCombinator(window=200)
        self.rf_signal = RFSignalProvider(self.model)
        self.hmm_signal = HMMSignalProvider(self._load_hmm())
        self.cross_venue_signal = CrossVenueSignalProvider()
        self.momentum_signal = MomentumSignalProvider()
        self.calibration_signal = CalibrationSignalProvider(self._load_calibration())
        self.signal_providers = [
            self.rf_signal, self.hmm_signal, self.cross_venue_signal,
            self.momentum_signal, self.calibration_signal,
        ]

    def _load_model(self):
        """Load the best trained RF model."""
        path = MODEL_DIR / "rf_clean_general.joblib"
        if path.exists():
            log.info(f"Loaded model: {path}")
            return joblib.load(path)
        for p in MODEL_DIR.glob("rf_*.joblib"):
            log.info(f"Loaded model: {p}")
            return joblib.load(p)
        log.error("No trained RF model found in models/")
        return None

    def _load_hmm(self):
        """Load global HMM regime detector if available."""
        path = MODEL_DIR / "hmm_global.joblib"
        if path.exists():
            try:
                from models.hmm_regime import HMMRegimeDetector
                detector = HMMRegimeDetector.load(path)
                log.info(f"Loaded HMM detector: {path}")
                return detector
            except Exception as e:
                log.warning(f"Failed to load HMM: {e}")
        return None

    def _load_calibration(self):
        """Load calibration analyzer if available."""
        try:
            ca = CalibrationAnalyzer.load()
            if ca.global_brier_score is not None:
                log.info(f"Loaded calibration (Brier={ca.global_brier_score:.4f})")
                return ca
        except Exception as e:
            log.warning(f"Failed to load calibration: {e}")
        return None

    def _score_market(self, market: dict,
                      all_markets: list[dict] | None = None
                      ) -> tuple[float, np.ndarray] | None:
        """Score a market with the RF model. Returns (probability, features)."""
        if self.model is None:
            return None
        try:
            X = _build_features(market, self.macro, all_markets)
            if np.isnan(X).any():
                nan_feats = [MODEL_FEATURES[i] for i in range(X.shape[1])
                             if np.isnan(X[0, i])]
                log.warning(f"NaN features for {market.get('title', '?')[:40]}: "
                            f"{nan_feats}")
                X = np.nan_to_num(X, nan=0.0)
            prob = self.model.predict_proba(X)[0][1]
            return float(prob), X
        except Exception as e:
            log.warning(f"Scoring failed for {market.get('title', '?')[:40]}: {e}")
            return None

    def _compute_fee(self, price: float) -> float:
        """Compute Kalshi taker fee."""
        return kalshi_taker_fee(price)

    def _compute_impact(self, price: float, volume: float) -> float:
        """Estimate market impact."""
        vol_dollars = volume * max(price, 0.01)
        return market_impact(price, vol_dollars)

    def scan(self) -> None:
        """One scan cycle: fetch markets, score, generate signals."""
        markets = fetch_markets()
        if not markets:
            log.warning("No markets fetched")
            return

        # Refresh macro periodically
        self.macro = _load_macro()

        # Build returns series from circuit breaker trade history (for HMM signal)
        returns_series = (
            pd.Series([t["pnl"] for t in self.cb.trades])
            if self.cb.trades else None
        )

        # Fetch cross-venue prices from Kalshi demo API (for cross-venue signal)
        try:
            kalshi_df = self.arb.scan_live_kalshi(limit=200)
            cross_venue_prices = [
                {"title": row["title"], "price": float(row["yes_price"])}
                for _, row in kalshi_df.iterrows()
                if float(row.get("yes_price", 0)) > 0
            ]
        except Exception as e:
            log.warning(f"Cross-venue fetch failed: {e}")
            cross_venue_prices = []

        scored = 0
        skipped_cat = 0
        skipped_price = 0

        for m in markets:
            market_id = m.get("ticker") or ""
            question = (m.get("title") or "")[:80]

            # Filter by category
            if not market_passes_filter(m, TARGET_CATS):
                skipped_cat += 1
                continue

            price = _market_price(m)
            if price <= 0 or price >= 0.95:
                skipped_price += 1
                continue

            volume = float(m.get("volume") or m.get("volume_fp") or 0)

            # Check if we have an open position -> check exit
            if market_id in self.positions:
                pos = self.positions[market_id]
                exit_signal = price >= pos.combined_prob * EXIT_MULT
                # Check expiry
                close_time = m.get("close_time")
                if close_time:
                    try:
                        exp = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                        hours_left = (exp - datetime.now(timezone.utc)).total_seconds() / 3600
                        if hours_left <= EXPIRY_EXIT_H:
                            exit_signal = True
                    except Exception:
                        pass

                if exit_signal:
                    self._paper_sell(market_id, price, m)
                continue

            # Score with model (pass all markets for volume_rank)
            score_result = self._score_market(m, markets)
            if score_result is None:
                continue
            prob, X = score_result
            scored += 1

            # Collect signals from all providers
            context = {
                "features": X,
                "model_prob": prob,
                "macro": self.macro,
                "all_markets": markets,
                "returns": returns_series,
                "cross_venue_prices": cross_venue_prices,
            }
            signals = {}
            for provider in self.signal_providers:
                result = provider.score(m, context)
                signals[result.name] = result.confidence

            # Combine via AlphaCombinator
            combined = self.combinator.combine(signals)

            # Use combined_prob if combinator has enough history, else model_prob
            if combined.diagnostics.get("fallback"):
                effective_prob = prob
            else:
                effective_prob = combined.probability

            # Entry signal: price <= effective_prob * EDGE_MULT
            if price > effective_prob * EDGE_MULT:
                log.debug(f"No edge: {question[:40]} price={price:.3f} "
                          f"eff_prob={effective_prob:.3f} "
                          f"threshold={effective_prob*EDGE_MULT:.3f}")
                continue

            # Circuit breaker check
            cb_status = self.cb.check()
            if not cb_status["allowed"]:
                log.warning(f"CIRCUIT BREAKER: {cb_status['reason']}")
                alert_circuit_breaker(cb_status["reason"])
                continue

            # Volume/liquidity check
            vol_dollars = volume * max(price, 0.01)
            if not should_trade(price, vol_dollars):
                continue

            self._paper_buy(
                market_id, question, price, prob,
                effective_prob, combined.weights, combined.ir_estimate, m,
            )

        log.info(f"Scan: {len(markets)} fetched | {skipped_cat} filtered(cat) | "
                 f"{skipped_price} filtered(price) | {scored} scored")

    def _paper_buy(self, market_id: str, question: str, price: float,
                   model_prob: float, combined_prob: float,
                   signal_weights: dict, ir_estimate: float,
                   market: dict) -> None:
        """Log a paper BUY with alpha combination and Kelly sizing."""
        fee = self._compute_fee(price)
        volume = float(market.get("volume") or market.get("volume_fp") or 0)
        impact = self._compute_impact(price, volume)

        # Kelly sizing with CV adjustment
        f_emp, cv_edge = kelly_with_cv(combined_prob, price)
        trade_size = min(f_emp * self.bankroll, self.bankroll * 0.10)

        pos = PaperPosition(
            market_id=market_id,
            question=question,
            side="PAPER_BUY",
            entry_price=price,
            model_prob=model_prob,
            combined_prob=combined_prob,
            signal_weights=signal_weights,
            fee=fee,
            impact=impact,
            kelly_fraction=f_emp,
        )
        self.positions[market_id] = pos

        weights_str = json.dumps({k: round(v, 3) for k, v in signal_weights.items()})
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": market_id,
            "market_question": question,
            "side": "PAPER_BUY",
            "entry_price": round(price, 4),
            "model_prob": round(model_prob, 4),
            "combined_prob": round(combined_prob, 4),
            "signal_weights": weights_str,
            "ir_estimate": round(ir_estimate, 4),
            "convexity_score": round(combined_prob - price, 4),
            "regime_hmm": "",
            "kelly_fraction": round(f_emp, 6),
            "fee": round(fee, 6),
            "impact": round(impact, 6),
            "exit_price": "",
            "pnl_net": "",
            "cumulative_pnl": round(self.cumulative_pnl, 4),
        }
        _log_trade(row)
        log.info(
            f"PAPER_BUY  {question[:50]} | price={price:.3f} "
            f"model={model_prob:.3f} combined={combined_prob:.3f} "
            f"edge={combined_prob/price:.1f}x kelly={f_emp:.4f} "
            f"size=${trade_size:.2f}"
        )

        # Telegram alert
        cat = get_category(market)
        alert_entry_signal(
            market=question, category=cat,
            model_prob=combined_prob, market_price=price,
            kelly_size=trade_size, convexity=combined_prob - price,
        )

    def _paper_sell(self, market_id: str, exit_price: float, market: dict) -> None:
        """Log a paper SELL."""
        pos = self.positions.pop(market_id, None)
        if not pos:
            return

        fee_exit = self._compute_fee(exit_price)
        pnl_gross = exit_price - pos.entry_price
        pnl_net = pnl_gross - pos.fee - fee_exit - pos.impact
        self.cumulative_pnl += pnl_net
        self.bankroll += pnl_net

        # Record in circuit breaker
        self.cb.record_trade(pnl_net)

        weights_str = json.dumps(
            {k: round(v, 3) for k, v in pos.signal_weights.items()}
        )
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": market_id,
            "market_question": pos.question,
            "side": "PAPER_SELL",
            "entry_price": round(pos.entry_price, 4),
            "model_prob": round(pos.model_prob, 4),
            "combined_prob": round(pos.combined_prob, 4),
            "signal_weights": weights_str,
            "ir_estimate": "",
            "convexity_score": round(pos.combined_prob - pos.entry_price, 4),
            "regime_hmm": "",
            "kelly_fraction": round(pos.kelly_fraction, 6),
            "fee": round(pos.fee + fee_exit, 6),
            "impact": round(pos.impact, 6),
            "exit_price": round(exit_price, 4),
            "pnl_net": round(pnl_net, 4),
            "cumulative_pnl": round(self.cumulative_pnl, 4),
        }
        _log_trade(row)
        tag = "WIN" if pnl_net > 0 else "LOSS"
        log.info(f"PAPER_SELL {pos.question[:50]} | {tag} pnl={pnl_net:+.4f} "
                 f"cum={self.cumulative_pnl:+.4f}")

        # Telegram alert
        alert_exit_signal(
            market=pos.question, entry_price=pos.entry_price,
            exit_price=exit_price, pnl=pnl_net,
            reason="exit_signal" if exit_price >= pos.combined_prob * EXIT_MULT else "expiry",
        )

    def run(self) -> None:
        """Main polling loop."""
        # Start Telegram bot threads
        set_paper_trader(self)
        start_command_listener()
        start_daily_summary()
        log.info("Telegram bot started — commands: /status /trades /stop")

        log.info(f"Paper trader started | poll={POLL_INTERVAL}s | "
                 f"edge={EDGE_MULT} | fee_model={FEE_MODEL} | api=kalshi-prod | "
                 f"LIVE_MODE={LIVE_MODE}")
        while self._running:
            try:
                self.scan()
                log.info(f"Positions: {len(self.positions)} | "
                         f"Cum P&L: {self.cumulative_pnl:+.4f}")
            except KeyboardInterrupt:
                log.info("Stopped by user")
                break
            except Exception as e:
                log.error(f"Scan error: {e}\n{traceback.format_exc()}")
            time.sleep(POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False


if __name__ == "__main__":
    trader = PaperTrader()
    trader.run()
