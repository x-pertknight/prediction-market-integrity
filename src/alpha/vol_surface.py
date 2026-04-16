"""
src/alpha/vol_surface.py — Black-Scholes implied vol surface scanner.

Runs as a background thread every 30 minutes. For each event with
markets at multiple expiries (same series_ticker, different event_tickers):

1. Compute binary BS price for each market
2. Back out implied volatility via Brent's method
3. Flag inconsistencies where IV deviates >30% from group mean
4. Generate direct Kelly-sized trades bypassing combinator

Binary option BS price:
  P = exp(-rT) * N(d2)
  d2 = [ln(S/K) + (r - 0.5*sigma^2)*T] / (sigma*sqrt(T))

For prediction markets:
  S = current market price (YES probability proxy)
  K = 0.5 (fair-value strike)
  T = time to expiry in years
  r = 0 (no risk-free rate in prediction markets)
  sigma = implied volatility to solve for
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

log = logging.getLogger("paper_trader")

VOL_SURFACE_LOG = Path("data/vol_surface_log.csv")
VOL_SURFACE_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────

SCAN_INTERVAL = 1800  # 30 minutes
IV_DEVIATION_THRESHOLD = 0.30  # 30% deviation from group mean IV
MIN_MARKETS_PER_SERIES = 3  # need 3+ expiries for meaningful surface
MIN_TIME_TO_EXPIRY_DAYS = 1  # skip markets expiring in < 1 day
MAX_TIME_TO_EXPIRY_DAYS = 365  # skip markets too far out


# ── Black-Scholes for binary options ────────────────────────────────────

def binary_bs_price(S: float, K: float, T: float, r: float,
                    sigma: float) -> float:
    """Binary call option price under Black-Scholes.

    Args:
        S: spot price (current market YES probability)
        K: strike (0.5 = fair-value)
        T: time to expiry in years
        r: risk-free rate (0 for prediction markets)
        sigma: volatility

    Returns:
        Theoretical binary option price [0, 1]
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return S  # at expiry or no vol, price is spot
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-r * T) * norm.cdf(d2)


def implied_vol(market_price: float, S: float, K: float = 0.5,
                T: float = None, r: float = 0.0) -> float | None:
    """Back out implied vol from observed market price via Brent's method.

    Args:
        market_price: observed YES price (0-1)
        S: spot price proxy (use market_price if no better estimate)
        K: strike (0.5)
        T: time to expiry in years
        r: risk-free rate

    Returns:
        Implied volatility, or None if solver fails.
    """
    if T is None or T <= 0:
        return None
    if market_price <= 0.01 or market_price >= 0.99:
        return None  # extreme prices → no meaningful IV

    try:
        iv = brentq(
            lambda sigma: binary_bs_price(S, K, T, r, sigma) - market_price,
            0.01, 10.0,
            xtol=1e-6,
            maxiter=100,
        )
        return float(iv)
    except (ValueError, RuntimeError):
        return None


# ── Vol surface analysis ────────────────────────────────────────────────

def _time_to_expiry_years(close_time_str: str) -> float | None:
    """Parse close_time ISO string to years until expiry."""
    if not close_time_str:
        return None
    try:
        exp = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        delta = (exp - datetime.now(timezone.utc)).total_seconds()
        if delta <= 0:
            return None
        return delta / (365.25 * 86400)
    except Exception:
        return None


def analyze_vol_surface(markets: list[dict]) -> list[dict]:
    """Analyze implied vol surface across multi-expiry markets.

    Groups markets by series_ticker. For each series with 3+ expiries,
    computes IV per market and flags >30% deviations from group mean.

    Args:
        markets: list of market dicts with ticker, series_ticker,
                 last_price, close_time fields

    Returns:
        list of anomaly dicts with trade recommendations
    """
    # Group by series
    series_groups: dict[str, list[dict]] = {}
    for m in markets:
        st = m.get("series_ticker", "")
        if not st:
            continue
        series_groups.setdefault(st, []).append(m)

    anomalies = []

    for series_ticker, group in series_groups.items():
        if len(group) < MIN_MARKETS_PER_SERIES:
            continue

        # Compute IV for each market in the series
        iv_data = []
        for m in group:
            price = _extract_price(m)
            if price <= 0.01 or price >= 0.99:
                continue

            close_time = m.get("close_time", m.get("expiration_time", ""))
            T = _time_to_expiry_years(close_time)
            if T is None:
                continue
            T_days = T * 365.25
            if T_days < MIN_TIME_TO_EXPIRY_DAYS or T_days > MAX_TIME_TO_EXPIRY_DAYS:
                continue

            iv = implied_vol(price, S=price, K=0.5, T=T, r=0.0)
            if iv is not None and 0.01 < iv < 10.0:
                iv_data.append({
                    "ticker": m.get("ticker", ""),
                    "title": (m.get("title", "") or "")[:80],
                    "event_ticker": m.get("event_ticker", ""),
                    "series_ticker": series_ticker,
                    "price": price,
                    "T_years": T,
                    "T_days": T_days,
                    "iv": iv,
                    "close_time": close_time,
                    "category": m.get("category", ""),
                    "volume": float(m.get("volume", 0) or 0),
                })

        if len(iv_data) < MIN_MARKETS_PER_SERIES:
            continue

        # Compute group statistics
        ivs = [d["iv"] for d in iv_data]
        mean_iv = np.mean(ivs)
        if mean_iv <= 0:
            continue

        # Flag deviations
        for d in iv_data:
            deviation = (d["iv"] - mean_iv) / mean_iv
            if abs(deviation) > IV_DEVIATION_THRESHOLD:
                # IV too high → market overpriced vol → SELL
                # IV too low → market underpriced vol → BUY
                side = "SELL" if deviation > 0 else "BUY"
                anomalies.append({
                    **d,
                    "mean_iv": mean_iv,
                    "iv_deviation": deviation,
                    "side": side,
                    "n_in_series": len(iv_data),
                })

    return anomalies


def _extract_price(m: dict) -> float:
    """Extract YES price from market dict."""
    for field_name in ("last_price", "last_price_dollars", "yes_ask",
                       "yes_ask_dollars", "yes_bid", "yes_bid_dollars"):
        val = m.get(field_name)
        if val is not None:
            try:
                p = float(val)
                if 0 < p < 1:
                    return p
            except (ValueError, TypeError):
                pass
    return 0.0


# ── Trade logger ────────────────────────────────────────────────────────

def _log_vol_trade(row: dict) -> None:
    """Append a vol surface trade to vol_surface_log.csv."""
    df = pd.DataFrame([row])
    header = not VOL_SURFACE_LOG.exists()
    df.to_csv(VOL_SURFACE_LOG, mode="a", header=header, index=False)


# ── Background scanner thread ──────────────────────────────────────────

class VolSurfaceScanner:
    """Background thread that scans for IV surface anomalies every 30 min.

    When anomalies found, generates Kelly-sized trade recommendations
    that bypass the AlphaCombinator (direct execution path).
    """

    def __init__(self, fetch_markets_fn, kelly_fn, bankroll_fn):
        """
        Args:
            fetch_markets_fn: callable that returns list[dict] of markets
            kelly_fn: callable(win_prob, entry_price) -> (f_emp, cv_edge)
            bankroll_fn: callable() -> float (current bankroll)
        """
        self._fetch_markets = fetch_markets_fn
        self._kelly = kelly_fn
        self._bankroll = bankroll_fn
        self._thread = None
        self._running = False
        self.last_anomalies: list[dict] = []

    def start(self) -> None:
        """Start the background scanner thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="vol-surface-scanner"
        )
        self._thread.start()
        log.info("Vol surface scanner started (interval=%ds)", SCAN_INTERVAL)

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            try:
                self._scan_once()
            except Exception as e:
                log.error(f"Vol surface scan error: {e}")
            time.sleep(SCAN_INTERVAL)

    def _scan_once(self) -> None:
        """Run one vol surface scan cycle."""
        markets = self._fetch_markets()
        if not markets:
            return

        anomalies = analyze_vol_surface(markets)
        self.last_anomalies = anomalies

        if not anomalies:
            log.info("Vol surface: no anomalies found "
                     f"(scanned {len(markets)} markets)")
            return

        log.info(f"Vol surface: {len(anomalies)} anomalies found")

        bankroll = self._bankroll()
        for a in anomalies:
            # Kelly sizing for vol surface trade
            # For BUY: win_prob estimated from mean_iv reversion
            # For SELL: inverse
            if a["side"] == "BUY":
                # Underpriced vol → price should be higher
                fair_price = binary_bs_price(
                    a["price"], 0.5, a["T_years"], 0.0, a["mean_iv"])
                win_prob = min(fair_price / max(a["price"], 0.01), 0.95)
            else:
                # Overpriced vol → price should be lower
                fair_price = binary_bs_price(
                    a["price"], 0.5, a["T_years"], 0.0, a["mean_iv"])
                win_prob = min((1.0 - fair_price) / max(1.0 - a["price"], 0.01), 0.95)

            f_emp, cv_edge = self._kelly(win_prob, a["price"])
            size = min(f_emp * bankroll, bankroll * 0.05)  # 5% cap for vol trades

            if size < 1.0:
                continue

            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": a["ticker"],
                "title": a["title"],
                "series_ticker": a["series_ticker"],
                "side": a["side"],
                "price": round(a["price"], 4),
                "iv": round(a["iv"], 4),
                "mean_iv": round(a["mean_iv"], 4),
                "iv_deviation": round(a["iv_deviation"], 4),
                "T_days": round(a["T_days"], 1),
                "fair_price": round(fair_price, 4),
                "win_prob": round(win_prob, 4),
                "kelly_fraction": round(f_emp, 6),
                "size_dollars": round(size, 2),
                "n_in_series": a["n_in_series"],
            }
            _log_vol_trade(row)

            log.info(
                f"VOL_SURFACE {a['side']} {a['title'][:40]} | "
                f"price={a['price']:.3f} iv={a['iv']:.2f} "
                f"mean_iv={a['mean_iv']:.2f} dev={a['iv_deviation']:+.1%} "
                f"fair={fair_price:.3f} size=${size:.2f}"
            )
