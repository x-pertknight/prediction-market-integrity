"""
src/execution/polymarket_trader.py — Polymarket trading loop.

Polls Gamma API for active markets, scores with RF model, sizes with
quarter-Kelly, and routes orders through polymarket_executor.

When LIVE_MODE=false (default): logs signals + dry-run orders.
When LIVE_MODE=true: places real CLOB orders (requires POLY_PRIVATE_KEY).

Usage:
  python src/execution/polymarket_trader.py           # foreground
  nohup python src/execution/polymarket_trader.py &   # background

Output:
  data/polymarket_trades.csv
  logs/polymarket_trader.log
"""

import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

from risk.circuit_breaker import CircuitBreaker
from risk.position_manager import PositionManager
from risk.fees import polymarket_taker_fee
from risk.market_impact import market_impact, should_trade
from data.polymarket_feature_adapter import (
    build_features, infer_category, market_passes_filter, MODEL_FEATURES,
)
from alerts.telegram_bot import (
    alert_entry_signal, alert_exit_signal, alert_circuit_breaker,
)
from execution.polymarket_executor import (
    place_limit_order, get_positions as get_poly_positions,
    get_markets as get_clob_markets, LIVE_MODE, POLY_PRIVATE_KEY,
)
from alpha.combinator import (
    AlphaCombinator, RFSignalProvider, HMMSignalProvider,
    CrossVenueSignalProvider, MomentumSignalProvider,
    CalibrationSignalProvider, OBISignalProvider, WhaleSignalProvider,
)
from alpha.cross_platform_arb import CrossPlatformArb
from alpha.bayes_updater import bayesian_update_prob
from models.calibration import CalibrationAnalyzer
from risk.kelly import kelly_with_cv

# ── Config ────────────────────────────────────────────────────────────────────

POLL_INTERVAL = 300       # seconds between scans
EDGE_MULT     = 0.70      # entry: price <= model_prob * 0.70 (30% edge)
EXIT_MULT     = 0.90      # exit:  price >= model_prob * 0.90
EXPIRY_EXIT_H = 24        # exit if expiry <= 24 hours
MIN_LIQUIDITY = 1000      # minimum $1k liquidity
TARGET_CATS   = {
    "weather", "crypto", "politics", "entertainment",
    "financials", "sports", "economics", "elections",
}

MODEL_DIR  = Path("models")
TRADES_CSV = Path("data/polymarket_trades.csv")
LOG_DIR    = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)

GAMMA_API = "https://gamma-api.polymarket.com"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "polymarket_trader.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("polymarket_trader")


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Position:
    market_id: str
    question: str
    token_id: str       # CLOB YES token ID for order placement
    entry_price: float
    model_prob: float
    size_dollars: float
    fee: float
    timestamp: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_macro() -> dict:
    """Load latest macro features from cached CSVs."""
    macro = {}
    raw = Path("data/raw")

    fg = raw / "fear_greed.csv"
    if fg.exists():
        df = pd.read_csv(fg)
        if not df.empty:
            macro["fear_greed_value"] = float(df.iloc[-1].get("fear_greed_value", 50))

    bn = raw / "binance_prices.csv"
    if bn.exists():
        df = pd.read_csv(bn)
        btc = df[df["symbol"] == "BTCUSDT"].tail(1)
        if not btc.empty:
            macro["btc_ret_1d"] = float(btc.iloc[-1].get("ret_1d", 0))
            macro["btc_ret_7d"] = float(btc.iloc[-1].get("ret_7d", 0))
            macro["btc_vol_20d"] = float(btc.iloc[-1].get("vol_20d", 0))
            macro["btc_zscore_30"] = float(btc.iloc[-1].get("price_zscore_30", 0))

    fred = raw / "fred_macro.csv"
    if fred.exists():
        df = pd.read_csv(fred)
        if not df.empty:
            last = df.iloc[-1]
            macro["dxy_proxy"] = float(last.get("dxy_proxy", 0) or 0)
            macro["fed_funds_rate"] = float(last.get("fed_funds_rate", 0) or 0)
            macro["yield_10y"] = float(last.get("yield_10y", 0) or 0)

    coin = raw / "coinalyze.csv"
    if coin.exists():
        df = pd.read_csv(coin)
        if not df.empty:
            last = df.iloc[-1]
            macro["btc_funding_rate"] = float(last.get("funding_rate", 0) or 0)
            macro["btc_open_interest"] = float(last.get("open_interest", 0) or 0)

    return macro


def _fetch_gamma_markets(limit: int = 500) -> list[dict]:
    """Fetch active, open markets from Gamma API sorted by volume."""
    import requests as req
    all_markets = []
    offset = 0
    while len(all_markets) < limit:
        try:
            r = req.get(f"{GAMMA_API}/markets", params={
                "active": "true", "closed": "false",
                "limit": min(100, limit - len(all_markets)),
                "offset": offset,
                "order": "volume24hr", "ascending": "false",
            }, timeout=15)
            if r.status_code != 200:
                break
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break
            all_markets.extend(batch)
            if len(batch) < 100:
                break
            offset += len(batch)
            time.sleep(0.2)
        except Exception as e:
            log.warning(f"Gamma API fetch error: {e}")
            break
    return all_markets


def _market_price(market: dict) -> float:
    """Extract YES outcome price from Gamma API market."""
    prices = market.get("outcomePrices")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except (json.JSONDecodeError, TypeError):
            return 0.0
    if isinstance(prices, list) and prices:
        return float(prices[0] or 0)
    return 0.0


def _yes_token_id(market: dict) -> str:
    """Extract the YES token ID for CLOB order placement."""
    ids = market.get("clobTokenIds")
    if isinstance(ids, str):
        try:
            ids = json.loads(ids)
        except (json.JSONDecodeError, TypeError):
            return ""
    if isinstance(ids, list) and ids:
        return str(ids[0])
    return ""


def _log_trade(row: dict) -> None:
    """Append a trade row to polymarket_trades.csv."""
    df = pd.DataFrame([row])
    df.to_csv(TRADES_CSV, mode="a", header=not TRADES_CSV.exists(), index=False)


# Microstructure entry guard (2026-07-07 incident: 30/37 buys were sub-5c
# dust where the spread alone is ruinous). Flag-gated via env, measured in
# paper — set MIN_ENTRY_PRICE=0 to disable. NOT backtested: paper-only slice.
MIN_ENTRY_PRICE = float(os.getenv("MIN_ENTRY_PRICE", "0.02"))
MAX_SPREAD_FRAC = float(os.getenv("MAX_SPREAD_FRAC", "0.20"))


def entry_microstructure_ok(market: dict, price: float) -> tuple[bool, str]:
    """Reject entries the book cannot support: dust prices, no bid to
    exit into, or a spread that is a large fraction of the mid."""
    if price < MIN_ENTRY_PRICE:
        return False, "dust_price"
    bid = float(market.get("bestBid") or 0)
    ask = float(market.get("bestAsk") or 0)
    if bid <= 0:
        return False, "no_bid"
    if ask > bid > 0 and MAX_SPREAD_FRAC > 0:
        mid = (bid + ask) / 2
        if (ask - bid) / mid > MAX_SPREAD_FRAC:
            return False, "wide_spread"
    return True, "ok"


def hours_to_expiry(market: dict) -> float | None:
    """Hours until market end date; None if missing/unparseable.
    Negative when the market is already past its end date."""
    end_date = market.get("endDate") or market.get("endDateIso")
    if not end_date:
        return None
    try:
        exp = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
        return (exp - datetime.now(timezone.utc)).total_seconds() / 3600
    except Exception:
        return None


# ── Trader ────────────────────────────────────────────────────────────────────

class PolymarketTrader:
    def __init__(self):
        self.model = self._load_model()
        self.macro = _load_macro()
        self.pm = PositionManager()
        # CB drawdown % must be computed on the same base used for sizing —
        # the default $100 made a $105 loss read as "108% drawdown"
        self.cb = CircuitBreaker(initial_capital=self.pm.state.bankroll)
        self.positions: dict[str, Position] = {}
        self.cumulative_pnl = 0.0
        self._running = True
        self._cb_alerted = False  # one alert per halt transition, not per market
        # Alpha combination framework
        self.arb = CrossPlatformArb()
        self.combinator = AlphaCombinator(window=50)
        self.rf_signal = RFSignalProvider(self.model)
        self.hmm_signal = HMMSignalProvider(self._load_hmm())
        self.cross_venue_signal = CrossVenueSignalProvider()
        self.momentum_signal = MomentumSignalProvider()
        self.calibration_signal = CalibrationSignalProvider(self._load_calibration())
        self.obi_signal = OBISignalProvider()
        self.whale_signal = WhaleSignalProvider()
        self.signal_providers = [
            self.rf_signal, self.hmm_signal, self.cross_venue_signal,
            self.momentum_signal, self.calibration_signal,
            self.obi_signal, self.whale_signal,
        ]

    def _load_model(self):
        """Load trained RF model."""
        candidates = [
            MODEL_DIR / "rf_clean_general.joblib",
            MODEL_DIR / "rf_model.joblib",
        ]
        for p in candidates:
            if p.exists():
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
                      all_markets: list[dict]
                      ) -> tuple[float, np.ndarray] | None:
        """Score a market with the RF model. Returns (probability, features)."""
        if self.model is None:
            return None
        try:
            X = build_features(market, self.macro, all_markets)
            if np.isnan(X).any():
                X = np.nan_to_num(X, nan=0.0)
            prob = self.model.predict_proba(X)[0][1]
            return float(prob), X
        except Exception as e:
            log.warning(f"Scoring failed: {e}")
            return None

    def scan(self) -> None:
        """One scan cycle: fetch, score, signal, size, execute."""
        markets = _fetch_gamma_markets(limit=300)
        if not markets:
            log.warning("No markets fetched from Gamma API")
            return

        self.macro = _load_macro()

        # Build returns series from circuit breaker trade history (for HMM)
        returns_series = (
            pd.Series([t["pnl"] for t in self.cb.trades])
            if self.cb.trades else None
        )

        # Fetch cross-venue prices (for cross-venue signal)
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
        entries = 0
        exits = 0
        skipped_cat = 0
        skipped_price = 0
        skipped_liq = 0

        for m in markets:
            market_id = m.get("conditionId") or m.get("id") or ""
            question = (m.get("question") or "")[:80]
            token_id = _yes_token_id(m)

            # Filter: category
            if not market_passes_filter(m, TARGET_CATS):
                skipped_cat += 1
                continue

            price = _market_price(m)
            if price <= 0 or price >= 0.95:
                skipped_price += 1
                continue

            # Filter: liquidity
            liquidity = float(m.get("liquidity") or m.get("liquidityNum") or 0)
            if liquidity < MIN_LIQUIDITY:
                skipped_liq += 1
                continue

            # Check exit for open positions
            if market_id in self.positions:
                pos = self.positions[market_id]
                exit_signal = price >= pos.model_prob * EXIT_MULT

                hours_left = hours_to_expiry(m)
                if hours_left is not None and hours_left <= EXPIRY_EXIT_H:
                    exit_signal = True

                if exit_signal:
                    self._execute_exit(market_id, price, m)
                    exits += 1
                continue

            # Entry gate: never open what the expiry rule above would dump
            # on the next scan (2026-07-07: bought expiry-day dust at ask,
            # force-sold at bid minutes later, -10.5% bankroll in 10 min)
            hours_left = hours_to_expiry(m)
            if hours_left is not None and hours_left <= EXPIRY_EXIT_H:
                skipped_price += 1
                continue

            # Entry gate: book must support the trade (dust/spread/no-bid)
            micro_ok, _micro_reason = entry_microstructure_ok(m, price)
            if not micro_ok:
                skipped_price += 1
                continue

            # Score with RF model (returns (prob, features))
            score_result = self._score_market(m, markets)
            if score_result is None:
                continue
            prob, X = score_result
            scored += 1

            # Bayesian update: chain evidence from news, FRED, NOAA
            category = infer_category(m)
            prob, bayes_updates = bayesian_update_prob(
                prob, category, m, self.macro)
            if bayes_updates:
                log.debug(f"Bayes: {question[:30]} {bayes_updates} "
                          f"→ {prob:.3f}")

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
                continue

            # Circuit breaker check — alert once per halt transition, not
            # per market per scan (2026-07-07: 12,813 duplicate alerts)
            cb_status = self.cb.check()
            if not cb_status["allowed"]:
                if not self._cb_alerted:
                    log.warning(f"CIRCUIT BREAKER: {cb_status['reason']}")
                    alert_circuit_breaker(cb_status["reason"])
                    self._cb_alerted = True
                continue
            self._cb_alerted = False

            # Position sizing via Kelly CV with combined probability
            category = infer_category(m)
            f_emp, cv_edge = kelly_with_cv(effective_prob, price)
            size_dollars = min(f_emp * self.pm.state.bankroll,
                               self.pm.state.bankroll * 0.10)

            # Check risk limits
            _, meta = self.pm.size_order(
                market_id=market_id,
                category=category,
                win_prob=effective_prob,
                entry_price=price,
            )
            if meta.get("blocked"):
                log.debug(f"Blocked: {question[:40]} — {meta['blocked']}")
                continue

            if size_dollars <= 0:
                continue

            # Apply circuit breaker sizing multiplier
            size_dollars *= cb_status.get("sizing_mult", 1.0)
            if size_dollars < 1.0:
                continue

            # Execute entry
            self._execute_entry(
                market_id=market_id,
                question=question,
                token_id=token_id,
                price=price,
                model_prob=prob,
                effective_prob=effective_prob,
                signal_weights=combined.weights,
                ir_estimate=combined.ir_estimate,
                size_dollars=size_dollars,
                category=category,
                market=m,
                meta=meta,
            )
            entries += 1

        log.info(
            f"Scan: {len(markets)} markets | {skipped_cat} cat-filtered | "
            f"{skipped_price} price-filtered | {skipped_liq} low-liq | "
            f"{scored} scored | {entries} entries | {exits} exits"
        )

    def _execute_entry(self, market_id: str, question: str, token_id: str,
                       price: float, model_prob: float,
                       effective_prob: float, signal_weights: dict,
                       ir_estimate: float, size_dollars: float,
                       category: str, market: dict, meta: dict) -> None:
        """Place entry order (real or dry-run based on LIVE_MODE)."""
        # Estimate quantity in shares
        quantity = size_dollars / max(price, 0.01)

        # Place order through executor
        order_id = place_limit_order(
            token_id=token_id, side="BUY",
            price=price, quantity=quantity,
        )

        if order_id is None:
            log.warning(f"Order rejected: {question[:40]}")
            return

        # Track position
        pos = Position(
            market_id=market_id,
            question=question,
            token_id=token_id,
            entry_price=price,
            model_prob=effective_prob,
            size_dollars=size_dollars,
            fee=polymarket_taker_fee(price, category) * size_dollars,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.positions[market_id] = pos

        # Record in position manager
        self.pm.record_fill(market_id, category, size_dollars, price)

        weights_str = json.dumps({k: round(v, 3) for k, v in signal_weights.items()})
        # Log trade
        _log_trade({
            "timestamp": pos.timestamp,
            "market_id": market_id,
            "question": question,
            "token_id": token_id[:20],
            "side": "BUY",
            "price": round(price, 4),
            "model_prob": round(model_prob, 4),
            "combined_prob": round(effective_prob, 4),
            "signal_weights": weights_str,
            "ir_estimate": round(ir_estimate, 4),
            "size_dollars": round(size_dollars, 2),
            "kelly_size": meta.get("kelly_size", 0),
            "ev": meta.get("ev", 0),
            "order_id": order_id,
            "category": category,
            "live": LIVE_MODE,
        })

        mode = "LIVE" if LIVE_MODE else "DRY-RUN"
        log.info(
            f"[{mode}] BUY {question[:50]} | price={price:.3f} "
            f"model={model_prob:.3f} combined={effective_prob:.3f} "
            f"edge={effective_prob/price:.1f}x size=${size_dollars:.2f} "
            f"order={order_id}"
        )

        alert_entry_signal(
            market=question, category=category,
            model_prob=effective_prob, market_price=price,
            kelly_size=size_dollars,
            convexity=effective_prob - price,
        )

    def _execute_exit(self, market_id: str, exit_price: float,
                      market: dict) -> None:
        """Place exit order."""
        pos = self.positions.pop(market_id, None)
        if not pos:
            return

        pnl_gross = (exit_price - pos.entry_price) * (
            pos.size_dollars / max(pos.entry_price, 0.01))
        pnl_net = pnl_gross - pos.fee
        self.cumulative_pnl += pnl_net

        # Record in circuit breaker + position manager
        self.cb.record_trade(pnl_net)
        category = infer_category(market)
        self.pm.record_close(market_id, category, pnl_net)

        _log_trade({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": market_id,
            "question": pos.question,
            "token_id": pos.token_id[:20],
            "side": "SELL",
            "price": round(exit_price, 4),
            "model_prob": round(pos.model_prob, 4),
            "size_dollars": round(pos.size_dollars, 2),
            "kelly_size": 0,
            "ev": round(pnl_net, 4),
            "order_id": "",
            "category": category,
            "live": LIVE_MODE,
        })

        tag = "WIN" if pnl_net > 0 else "LOSS"
        mode = "LIVE" if LIVE_MODE else "DRY-RUN"
        log.info(
            f"[{mode}] SELL {pos.question[:50]} | {tag} "
            f"pnl={pnl_net:+.4f} cum={self.cumulative_pnl:+.4f}"
        )

        alert_exit_signal(
            market=pos.question, entry_price=pos.entry_price,
            exit_price=exit_price, pnl=pnl_net,
        )

    def run(self) -> None:
        """Main polling loop."""
        log.info("=" * 60)
        log.info("Polymarket Trader started")
        log.info(f"  LIVE_MODE:     {LIVE_MODE}")
        log.info(f"  PRIVATE_KEY:   {'SET' if POLY_PRIVATE_KEY else 'NOT SET (dry-run only)'}")
        log.info(f"  POLL_INTERVAL: {POLL_INTERVAL}s")
        log.info(f"  EDGE_MULT:     {EDGE_MULT}")
        log.info(f"  MIN_LIQUIDITY: ${MIN_LIQUIDITY}")
        log.info(f"  Risk: {json.dumps(self.pm.status())}")
        log.info("=" * 60)

        while self._running:
            try:
                self.scan()
                log.info(
                    f"Positions: {len(self.positions)} | "
                    f"Cum P&L: {self.cumulative_pnl:+.4f} | "
                    f"Risk: can_trade={self.pm.status()['can_trade']}"
                )
            except KeyboardInterrupt:
                log.info("Stopped by user")
                break
            except Exception as e:
                log.error(f"Scan error: {e}\n{traceback.format_exc()}")
            time.sleep(POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False


if __name__ == "__main__":
    trader = PolymarketTrader()
    trader.run()
