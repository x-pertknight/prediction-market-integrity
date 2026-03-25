"""
src/execution/paper_trader.py — Paper trading against Polymarket Gamma API.

Polls Polymarket every 60s for active markets in target categories.
Scores each with the trained RF model. Applies entry/exit signals,
fee model, market impact, and circuit breaker. Logs everything.

Usage:
  python src/execution/paper_trader.py           # foreground
  nohup python src/execution/paper_trader.py &   # background

Output:
  data/paper_trades.csv
  logs/paper_trader.log
"""

import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from risk.fees import kalshi_taker_fee, polymarket_taker_fee
from risk.market_impact import market_impact, should_trade, liquidity_tier
from risk.circuit_breaker import CircuitBreaker
from alerts.telegram_bot import (
    alert_entry_signal, alert_exit_signal, alert_circuit_breaker,
    alert_daily_summary, set_paper_trader, start_command_listener,
    start_daily_summary,
)
from data.polymarket_feature_adapter import (
    build_features as adapter_build_features,
    infer_category,
    market_passes_filter,
    diagnose_features,
    MODEL_FEATURES,
)

# ── Config ───────────────────────────────────────────────────────────────────

GAMMA_API       = "https://gamma-api.polymarket.com/markets"
POLL_INTERVAL   = 60      # seconds
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

FEE_MODEL       = os.getenv("PAPER_FEE_MODEL", "polymarket")  # most Poly markets = free

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

# ── Features (matching CLEAN_FEATURES from features_clean.py) ────────────────

CLEAN_FEATURES = [
    "days_in_market", "floor_strike_pct", "market_type_enc", "category_enc",
    "day_of_week", "month", "day_of_year", "doy_sin", "doy_cos",
    "hist_win_rate_series", "volume_rank",
    "fear_greed_value", "btc_ret_1d", "btc_ret_7d", "btc_vol_20d", "btc_zscore_30",
    "dxy_proxy", "fed_funds_rate", "yield_10y",
    "btc_funding_rate", "btc_open_interest",
    "news_count_crypto", "news_count_weather", "news_count_macro",
    "news_count_politics", "news_count_sports",
    "smart_money_index", "whale_cluster_detected",
]

# Category encoding (same as training)
_CAT_ENC = {
    "Climate and Weather": 0, "Companies": 1, "Crypto": 2, "Economics": 3,
    "Education": 4, "Elections": 5, "Entertainment": 6, "Exotics": 7,
    "Financials": 8, "Health": 9, "Mentions": 10, "Politics": 11,
    "Science and Technology": 12, "Social": 13, "Sports": 14,
    "Transportation": 15, "World": 16,
}
# Polymarket tag → Kalshi category mapping
_TAG_TO_CAT = {
    "weather": "Climate and Weather",
    "crypto": "Crypto",
    "politics": "Politics",
    "entertainment": "Entertainment",
    "financials": "Financials",
    "sports": "Sports",
    "economics": "Economics",
    "elections": "Elections",
}


def _build_features(market: dict, macro: dict,
                    all_markets: list[dict] | None = None) -> np.ndarray:
    """Build feature vector using the Polymarket adapter.

    Delegates to polymarket_feature_adapter.build_features() which handles:
    - Keyword-based category inference (Gamma API has no tags/category)
    - Training-set medians for unmappable features
    - Proper days_in_market from startDate
    - Cross-sectional volume_rank from all current markets
    """
    return adapter_build_features(market, macro, all_markets)


def _load_macro() -> dict:
    """Load latest macro features from cached files."""
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
    return macro


# ── Polymarket Gamma API ─────────────────────────────────────────────────────

def fetch_markets(retries: int = 3) -> list[dict]:
    """Fetch active markets from Polymarket Gamma API."""
    delay = 2.0
    for _ in range(retries):
        try:
            r = requests.get(GAMMA_API, params={"closed": "false", "limit": 200}, timeout=15)
            if r.status_code == 200:
                return r.json()
            log.warning(f"Gamma API returned {r.status_code}")
        except Exception as e:
            log.warning(f"Gamma API error: {e}")
        time.sleep(delay)
        delay *= 2
    return []


def _market_tags(m: dict) -> set[str]:
    """Extract lowercase tags from a Polymarket market."""
    tags = set()
    for t in (m.get("tags") or []):
        tags.add(t.lower())
    # Also check category field
    cat = (m.get("category") or "").lower()
    if cat:
        tags.add(cat)
    return tags


def _market_price(m: dict) -> float:
    """Get current YES price for a market."""
    # Gamma API: outcomePrices is a JSON string like "[0.55, 0.45]"
    prices = m.get("outcomePrices")
    if isinstance(prices, str):
        try:
            parsed = json.loads(prices)
            if parsed:
                return float(parsed[0])
        except (json.JSONDecodeError, IndexError):
            pass
    if isinstance(prices, list) and prices:
        return float(prices[0])
    return 0.0


# ── Trade logging ────────────────────────────────────────────────────────────

_CSV_COLS = [
    "timestamp", "market_id", "market_question", "side", "entry_price",
    "model_prob", "convexity_score", "regime_hmm", "fee", "impact",
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
    fee: float
    impact: float
    entry_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── Main loop ────────────────────────────────────────────────────────────────

class PaperTrader:
    def __init__(self):
        self.positions: dict[str, PaperPosition] = {}
        self.cumulative_pnl = 0.0
        self.model = self._load_model()
        self.macro = _load_macro()
        self.cb = CircuitBreaker()
        self._running = True

    def _load_model(self):
        """Load the best trained RF model."""
        path = MODEL_DIR / "rf_clean_general.joblib"
        if path.exists():
            log.info(f"Loaded model: {path}")
            return joblib.load(path)
        # Fallback
        for p in MODEL_DIR.glob("rf_*.joblib"):
            log.info(f"Loaded model: {p}")
            return joblib.load(p)
        log.error("No trained RF model found in models/")
        return None

    def _score_market(self, market: dict,
                      all_markets: list[dict] | None = None) -> float | None:
        """Score a market with the RF model. Returns predicted probability."""
        if self.model is None:
            return None
        try:
            X = _build_features(market, self.macro, all_markets)
            # Check for NaN in feature vector
            if np.isnan(X).any():
                nan_feats = [MODEL_FEATURES[i] for i in range(X.shape[1])
                             if np.isnan(X[0, i])]
                log.warning(f"NaN features for {market.get('question', '?')[:40]}: "
                            f"{nan_feats}")
                # Replace NaN with 0 to avoid model crash
                X = np.nan_to_num(X, nan=0.0)
            prob = self.model.predict_proba(X)[0][1]
            return float(prob)
        except Exception as e:
            log.warning(f"Scoring failed for {market.get('question', '?')[:40]}: {e}")
            return None

    def _compute_fee(self, price: float, market: dict) -> float:
        """Compute fee based on configured fee model."""
        if FEE_MODEL == "polymarket":
            tags = _market_tags(market)
            mtype = ""
            if tags & {"crypto"}:
                mtype = "crypto_daily"
            elif tags & {"sports", "ncaab"}:
                mtype = "sport_fee"
            return polymarket_taker_fee(price, mtype)
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

        scored = 0
        skipped_cat = 0
        skipped_price = 0

        for m in markets:
            market_id = m.get("id") or m.get("condition_id") or ""
            question = (m.get("question") or "")[:80]

            # Filter by inferred category (keyword-based, not tag-based)
            if not market_passes_filter(m, TARGET_CATS):
                skipped_cat += 1
                continue

            price = _market_price(m)
            if price <= 0 or price >= 0.95:
                skipped_price += 1
                continue

            volume = float(m.get("volume", 0) or 0)

            # Check if we have an open position → check exit
            if market_id in self.positions:
                pos = self.positions[market_id]
                # Exit signal: price >= model_prob * EXIT_MULT OR near expiry
                exit_signal = price >= pos.model_prob * EXIT_MULT
                # Check expiry
                end_date = m.get("endDate") or m.get("end_date_iso")
                if end_date:
                    try:
                        exp = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                        hours_left = (exp - datetime.now(timezone.utc)).total_seconds() / 3600
                        if hours_left <= EXPIRY_EXIT_H:
                            exit_signal = True
                    except Exception:
                        pass

                if exit_signal:
                    self._paper_sell(market_id, price, m)
                continue

            # Score with model (pass all markets for volume_rank)
            prob = self._score_market(m, markets)
            if prob is None:
                continue
            scored += 1

            # Entry signal: price <= model_prob * EDGE_MULT
            if price > prob * EDGE_MULT:
                log.debug(f"No edge: {question[:40]} price={price:.3f} "
                          f"prob={prob:.3f} threshold={prob*EDGE_MULT:.3f}")
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

            self._paper_buy(market_id, question, price, prob, m)

        log.info(f"Scan: {len(markets)} fetched | {skipped_cat} filtered(cat) | "
                 f"{skipped_price} filtered(price) | {scored} scored")

    def _paper_buy(self, market_id: str, question: str, price: float,
                   model_prob: float, market: dict) -> None:
        """Log a paper BUY."""
        fee = self._compute_fee(price, market)
        volume = float(market.get("volume", 0) or 0)
        impact = self._compute_impact(price, volume)

        pos = PaperPosition(
            market_id=market_id,
            question=question,
            side="PAPER_BUY",
            entry_price=price,
            model_prob=model_prob,
            fee=fee,
            impact=impact,
        )
        self.positions[market_id] = pos

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": market_id,
            "market_question": question,
            "side": "PAPER_BUY",
            "entry_price": round(price, 4),
            "model_prob": round(model_prob, 4),
            "convexity_score": round(model_prob - price, 4),
            "regime_hmm": "",
            "fee": round(fee, 6),
            "impact": round(impact, 6),
            "exit_price": "",
            "pnl_net": "",
            "cumulative_pnl": round(self.cumulative_pnl, 4),
        }
        _log_trade(row)
        log.info(f"PAPER_BUY  {question[:50]} | price={price:.3f} "
                 f"prob={model_prob:.3f} edge={model_prob/price:.1f}x fee={fee:.4f}")

        # Telegram alert
        tags = _market_tags(market)
        cat = next((t for t in tags if t in TARGET_CATS), "unknown")
        alert_entry_signal(
            market=question, category=cat,
            model_prob=model_prob, market_price=price,
            kelly_size=0, convexity=model_prob - price,
        )

    def _paper_sell(self, market_id: str, exit_price: float, market: dict) -> None:
        """Log a paper SELL."""
        pos = self.positions.pop(market_id, None)
        if not pos:
            return

        fee_exit = self._compute_fee(exit_price, market)
        pnl_gross = exit_price - pos.entry_price
        pnl_net = pnl_gross - pos.fee - fee_exit - pos.impact
        self.cumulative_pnl += pnl_net

        # Record in circuit breaker
        self.cb.record_trade(pnl_net)

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": market_id,
            "market_question": pos.question,
            "side": "PAPER_SELL",
            "entry_price": round(pos.entry_price, 4),
            "model_prob": round(pos.model_prob, 4),
            "convexity_score": round(pos.model_prob - pos.entry_price, 4),
            "regime_hmm": "",
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
            reason="exit_signal" if exit_price >= pos.model_prob * EXIT_MULT else "expiry",
        )

    def run(self) -> None:
        """Main polling loop."""
        # Start Telegram bot threads
        set_paper_trader(self)
        start_command_listener()
        start_daily_summary()
        log.info("Telegram bot started — commands: /status /trades /stop")

        log.info(f"Paper trader started | poll={POLL_INTERVAL}s | "
                 f"edge={EDGE_MULT} | fee_model={FEE_MODEL}")
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
