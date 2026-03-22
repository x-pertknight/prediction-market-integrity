"""
src/signal_generator.py
Generate live trading signals from trained models + current market data.
Connects model predictions to paper_trader and position_manager.

Usage:
  python src/signal_generator.py           # single scan, print signals
  python src/signal_generator.py --loop    # continuous 30-min loop
"""
import argparse
import json
import time
import sys
from pathlib import Path
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import joblib

PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
RAW_DIR   = Path("data/raw")

# Edge threshold: only signal when model_prob >= market_price * MIN_EDGE_RATIO
MIN_EDGE_RATIO     = 2.0   # 2× edge requirement (backtest standard)
MIN_PRED_PROB      = 0.10  # don't signal below 10% model prob
MAX_MARKET_PRICE   = 0.45  # don't buy if market already >45 cents
MIN_VOLUME         = 0     # volume threshold (0 = any volume)


def load_models() -> dict:
    """Load all trained RF models."""
    models = {}
    for jf in MODEL_DIR.glob("rf_*.joblib"):
        name = jf.stem.replace("rf_", "")
        feat_path = MODEL_DIR / f"features_{name}.json"
        if feat_path.exists():
            feat_info = json.loads(feat_path.read_text())
            models[name] = {
                "clf":      joblib.load(jf),
                "features": feat_info["features"],
                "target":   feat_info["target"],
            }
    return models


def load_active_markets() -> pd.DataFrame:
    """Load most recent Polymarket active markets snapshot."""
    f = RAW_DIR / "polymarket_active.csv"
    if f.exists():
        return pd.read_csv(f)
    return pd.DataFrame()


def load_macro() -> dict:
    """Load latest macro features."""
    macro = {}
    fg_path = RAW_DIR / "fear_greed.csv"
    if fg_path.exists():
        df = pd.read_csv(fg_path)
        if not df.empty:
            macro["fear_greed_value"] = float(df.iloc[-1]["fear_greed_value"])
    bn_path = RAW_DIR / "binance_prices.csv"
    if bn_path.exists():
        df = pd.read_csv(bn_path)
        btc = df[df["symbol"] == "BTCUSDT"].tail(1)
        if not btc.empty:
            macro["btc_ret_1d"]    = float(btc.iloc[-1].get("ret_1d", 0))
            macro["btc_ret_7d"]    = float(btc.iloc[-1].get("ret_7d", 0))
            macro["btc_vol_20d"]   = float(btc.iloc[-1].get("vol_20d", 0))
            macro["btc_zscore_30"] = float(btc.iloc[-1].get("price_zscore_30", 0))
    return macro


def build_signal_features(market: dict, macro: dict, feat_list: list[str]) -> np.ndarray | None:
    """
    Build feature vector for a single market.
    Uses available columns, fills missing with defaults.
    Returns None if critical features are unavailable.
    """
    today = date.today()
    doy   = today.timetuple().tm_yday

    defaults = {
        "market_price":       float(market.get("yes_bid", 0.5)),
        "price_lag1":         float(market.get("yes_bid", 0.5)),
        "volume_fp":          float(market.get("volume", 0)),
        "volume_rank":        0.5,
        "days_in_market":     1,
        "day_of_week":        today.weekday(),
        "month":              today.month,
        "doy_sin":            np.sin(2 * np.pi * doy / 365),
        "doy_cos":            np.cos(2 * np.pi * doy / 365),
        "floor_strike_pct":   1.0,
        "hist_win_rate_series": 0.5,
        "category_enc":       -1,
        "market_type_enc":    0,
        # Macro defaults
        "fear_greed_value":   50.0,
        "btc_ret_1d":         0.0,
        "btc_ret_7d":         0.0,
        "btc_vol_20d":        0.0,
        "btc_zscore_30":      0.0,
        # Whale defaults
        "smart_money_index":       50.0,
        "whale_cluster_detected":  0,
        "whale_count_total":       0,
        "whale_avg_position_size": 0,
    }
    # Override with macro
    defaults.update({k: v for k, v in macro.items() if v is not None})

    row = [defaults.get(f, 0) for f in feat_list]
    return np.array(row).reshape(1, -1)


def generate_signals(models: dict, top_n: int = 20) -> list[dict]:
    """
    Generate top-N trading signals from all models.
    Uses active market data for current prices.
    """
    macro   = load_macro()
    markets = load_active_markets()

    if markets.empty:
        print("[SignalGen] No active market data. Run src/data/polymarket_gamma.py first.")
        return []

    signals = []

    for model_name, m in models.items():
        clf   = m["clf"]
        feats = m["features"]

        for _, row in markets.iterrows():
            yes_bid = float(row.get("yes_bid", 0)) if "yes_bid" in row else float(row.get("yes_ask", 0.5))
            if yes_bid <= 0 or yes_bid > MAX_MARKET_PRICE:
                continue

            market_data = row.to_dict()
            X = build_signal_features(market_data, macro, feats)
            if X is None:
                continue

            try:
                prob = clf.predict_proba(X)[0][1]
            except Exception:
                continue

            if prob < MIN_PRED_PROB:
                continue

            edge = prob / yes_bid if yes_bid > 0 else 0
            if edge < MIN_EDGE_RATIO:
                continue

            signals.append({
                "model":         model_name,
                "market_id":     str(row.get("id", row.get("market_id", "unknown"))),
                "market_title":  str(row.get("question", row.get("title", "")))[:60],
                "category":      str(row.get("category", "unknown")),
                "yes_bid":       round(yes_bid, 4),
                "model_prob":    round(float(prob), 4),
                "edge_ratio":    round(edge, 2),
                "volume":        float(row.get("volume", 0)),
                "generated_at":  datetime.now(timezone.utc).isoformat(),
            })

    # Sort by edge ratio descending, dedup by market_id
    signals.sort(key=lambda x: x["edge_ratio"], reverse=True)
    seen = set()
    unique_signals = []
    for s in signals:
        if s["market_id"] not in seen:
            seen.add(s["market_id"])
            unique_signals.append(s)

    return unique_signals[:top_n]


def print_signals(signals: list[dict]) -> None:
    if not signals:
        print("[SignalGen] No signals meeting criteria.")
        return
    print(f"\n{'='*70}")
    print(f"  TrapTrade Signals — {date.today().isoformat()}")
    print(f"  {len(signals)} signal(s) | edge >= {MIN_EDGE_RATIO}× | price <= {MAX_MARKET_PRICE}")
    print(f"{'='*70}")
    for i, s in enumerate(signals, 1):
        print(f"  {i:02d}. [{s['model']:<12}] {s['market_title']}")
        print(f"       Category: {s['category']:<20} Price: {s['yes_bid']:.3f}  "
              f"Model: {s['model_prob']:.3f}  Edge: {s['edge_ratio']:.1f}×")
    print()


def save_signals(signals: list[dict]) -> None:
    out_dir = Path("data/signals")
    out_dir.mkdir(parents=True, exist_ok=True)
    today   = date.today().isoformat()
    out_f   = out_dir / f"signals_{today}.jsonl"
    with open(out_f, "a") as f:
        for s in signals:
            f.write(json.dumps(s) + "\n")
    print(f"[SignalGen] {len(signals)} signals → {out_f}")


def run_once() -> list[dict]:
    print("[SignalGen] Loading models...")
    models = load_models()
    if not models:
        print("[SignalGen] No models found. Run model_ensemble.py first.")
        return []
    print(f"[SignalGen] Loaded: {list(models.keys())}")

    signals = generate_signals(models)
    print_signals(signals)
    save_signals(signals)
    return signals


def run_loop(interval_seconds: int = 1800) -> None:
    """Continuous signal generation loop."""
    print(f"[SignalGen] Starting loop (every {interval_seconds//60} min)...")
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[SignalGen] Error: {e}")
        print(f"[SignalGen] Next scan in {interval_seconds//60} min...")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true", help="Run continuously every 30 min")
    p.add_argument("--interval", type=int, default=1800, help="Loop interval seconds")
    p.add_argument("--top-n",   type=int, default=20, help="Max signals to output")
    args = p.parse_args()

    if args.loop:
        run_loop(args.interval)
    else:
        run_once()
