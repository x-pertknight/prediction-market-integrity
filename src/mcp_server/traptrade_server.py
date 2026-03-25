"""
src/mcp/traptrade_server.py — MCP server for TrapTrade.

Exposes TrapTrade capabilities as tools that any AI agent can call.
Uses FastMCP for the server framework.

Usage:
  python src/mcp/traptrade_server.py              # stdio (MCP client)
  python src/mcp/traptrade_server.py --transport sse --port 8100  # HTTP/SSE

Tools:
  get_market_probability  — Score a market with the RF model
  get_portfolio_status    — Current paper portfolio state
  get_market_scan         — Find markets with entry signals
  get_backtest_summary    — Backtest performance summary
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import joblib
import requests
from fastmcp import FastMCP

from data.polymarket_feature_adapter import (
    build_features, infer_category, market_passes_filter, MODEL_FEATURES,
)
from risk.circuit_breaker import CircuitBreaker

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "mcp_server.log"),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger("traptrade_mcp")

# ── Server setup ──────────────────────────────────────────────────────────────

mcp = FastMCP(
    "TrapTrade",
    instructions="Prediction market analysis tools. Score markets, scan for "
                 "opportunities, check portfolio status.",
)

# ── Shared state ──────────────────────────────────────────────────────────────

GAMMA_API = "https://gamma-api.polymarket.com/markets"
MODEL_DIR = PROJECT_ROOT / "models"
TRADES_CSV = PROJECT_ROOT / "data" / "paper_trades.csv"
WF_PARQUET = PROJECT_ROOT / "data" / "processed" / "wf_clean_general.parquet"

_model = None
_cb = CircuitBreaker()


def _get_model():
    global _model
    if _model is None:
        path = MODEL_DIR / "rf_clean_general.joblib"
        if path.exists():
            _model = joblib.load(path)
            log.info(f"Loaded model: {path}")
    return _model


def _load_macro() -> dict:
    """Load macro features from cached CSVs."""
    macro = {}
    raw = PROJECT_ROOT / "data" / "raw"
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
            row = df.iloc[-1]
            macro["dxy_proxy"] = float(row.get("dxy_proxy", 0))
            macro["fed_funds_rate"] = float(row.get("fed_funds_rate", 0))
            macro["yield_10y"] = float(row.get("yield_10y", 0))
    coin = raw / "coinalyze.csv"
    if coin.exists():
        df = pd.read_csv(coin)
        if not df.empty:
            row = df.iloc[-1]
            macro["btc_funding_rate"] = float(row.get("btc_funding_rate", 0))
            macro["btc_open_interest"] = float(row.get("btc_open_interest", 0))
    return macro


def _fetch_markets(limit: int = 200) -> list[dict]:
    """Fetch active markets from Polymarket Gamma API."""
    try:
        r = requests.get(GAMMA_API, params={"closed": "false", "limit": limit}, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.warning(f"Gamma API error: {e}")
    return []


def _market_price(m: dict) -> float:
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


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_market_probability(
    market_question: str = "",
    market_id: str = "",
) -> dict:
    """Score a Polymarket market with the RF model.

    Provide either market_question (partial match) or market_id.
    Returns model probability, market price, edge, and recommendation.
    """
    model = _get_model()
    if model is None:
        return {"error": "No trained model found"}

    markets = _fetch_markets(200)
    if not markets:
        return {"error": "Could not fetch markets from Gamma API"}

    macro = _load_macro()

    # Find matching market
    target = None
    query = market_question.lower()
    for m in markets:
        mid = m.get("id") or m.get("condition_id") or ""
        q = (m.get("question") or "").lower()
        if market_id and str(mid) == str(market_id):
            target = m
            break
        if query and query in q:
            target = m
            break

    if target is None:
        return {"error": f"Market not found: {market_question or market_id}"}

    price = _market_price(target)
    cat = infer_category(target)
    X = build_features(target, macro, markets)
    X = np.nan_to_num(X, nan=0.0)
    prob = float(model.predict_proba(X)[0][1])
    edge = prob - price
    threshold = prob * 0.70

    rec = "HOLD"
    if price <= threshold and 0 < price < 0.95:
        rec = "BUY"
    elif price >= prob * 0.90:
        rec = "SELL"

    return {
        "question": target.get("question", ""),
        "market_id": target.get("id", ""),
        "category": cat,
        "model_probability": round(prob, 4),
        "market_price": round(price, 4),
        "edge": round(edge, 4),
        "entry_threshold": round(threshold, 4),
        "recommendation": rec,
    }


@mcp.tool()
def get_portfolio_status() -> dict:
    """Get current paper trading portfolio status.

    Returns total P&L, open positions count, and circuit breaker status.
    """
    result = {
        "total_pnl": 0.0,
        "trade_count": 0,
        "open_positions": 0,
        "circuit_breaker_status": _cb.check(),
    }

    if TRADES_CSV.exists():
        df = pd.read_csv(TRADES_CSV)
        if not df.empty:
            result["trade_count"] = len(df)
            sells = df[df["side"] == "PAPER_SELL"]
            if not sells.empty and "pnl_net" in sells.columns:
                result["total_pnl"] = round(float(sells["pnl_net"].sum()), 4)
            buys = df[df["side"] == "PAPER_BUY"]
            sell_ids = set(sells["market_id"]) if not sells.empty else set()
            buy_ids = set(buys["market_id"]) if not buys.empty else set()
            result["open_positions"] = len(buy_ids - sell_ids)
            if "cumulative_pnl" in df.columns:
                result["cumulative_pnl"] = round(
                    float(df["cumulative_pnl"].iloc[-1]), 4)

    return result


@mcp.tool()
def get_market_scan(
    category: str = "",
    min_edge: float = 0.3,
) -> dict:
    """Scan Polymarket for markets with entry signals.

    Returns markets where price <= model_prob * 0.70 and edge >= min_edge.
    Optionally filter by category (e.g., "crypto", "sports", "politics").
    """
    model = _get_model()
    if model is None:
        return {"error": "No trained model found"}

    markets = _fetch_markets(200)
    if not markets:
        return {"error": "Could not fetch markets"}

    macro = _load_macro()
    signals = []

    for m in markets:
        cat = infer_category(m)
        if category and category.lower() not in cat.lower():
            continue

        price = _market_price(m)
        if price <= 0 or price >= 0.95:
            continue

        X = build_features(m, macro, markets)
        X = np.nan_to_num(X, nan=0.0)
        prob = float(model.predict_proba(X)[0][1])
        edge = prob - price

        if price <= prob * 0.70 and edge >= min_edge:
            signals.append({
                "question": (m.get("question") or "")[:80],
                "market_id": m.get("id", ""),
                "category": cat,
                "price": round(price, 4),
                "model_prob": round(prob, 4),
                "edge": round(edge, 4),
                "volume_24h": float(m.get("volume24hr") or 0),
            })

    # Sort by edge descending
    signals.sort(key=lambda x: -x["edge"])

    return {
        "markets_scanned": len(markets),
        "signals_found": len(signals),
        "min_edge": min_edge,
        "signals": signals[:20],
    }


@mcp.tool()
def get_backtest_summary() -> dict:
    """Get the backtest performance summary from walk-forward validation."""
    if not WF_PARQUET.exists():
        return {"error": "Backtest data not found"}

    df = pd.read_parquet(WF_PARQUET)
    trades = df[df["pred_prob_clean"].notna()].copy()

    if trades.empty:
        return {"error": "No trades in backtest data"}

    # Entry price proxy
    if "price_lag1" in trades.columns:
        trades["entry"] = trades["price_lag1"]
    else:
        trades["entry"] = trades["market_price"]

    # PnL per trade (outcome=1 means YES resolved)
    trades["pnl"] = trades["outcome"] - trades["entry"]
    pnl = trades["pnl"]

    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0
    win_rate = float((pnl > 0).mean())
    max_dd = float((pnl.cumsum() - pnl.cumsum().cummax()).min())

    return {
        "total_trades": len(trades),
        "win_rate": round(win_rate, 4),
        "sharpe_annualized": round(sharpe, 2),
        "pnl_total": round(float(pnl.sum()), 2),
        "pnl_mean": round(float(pnl.mean()), 4),
        "max_drawdown": round(max_dd, 4),
        "features_count": 20,
        "model_type": "RandomForest",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TrapTrade MCP Server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"])
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    log.info(f"Starting TrapTrade MCP server (transport={args.transport})")

    if args.transport == "sse":
        mcp.run(transport="sse", host="0.0.0.0", port=args.port)
    else:
        mcp.run()
