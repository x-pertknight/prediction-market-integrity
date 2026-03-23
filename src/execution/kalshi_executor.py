"""
src/execution/kalshi_executor.py — Kalshi order execution via kalshi-python-sync.

Uses RSA-PSS auth from .env (KALSHI_API_KEY + KALSHI_PRIVATE_KEY_PATH).
LIVE_MODE env var controls real vs dry-run execution. Default: False.

Interface:
  place_limit_order(ticker, side, price, quantity) → order_id
  cancel_order(order_id) → bool
  get_positions() → list[dict]
  get_balance() → float
"""

import os
import sys
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from risk.circuit_breaker import CircuitBreaker

log = logging.getLogger("kalshi_executor")

LIVE_MODE = os.getenv("LIVE_MODE", "false").lower() in ("true", "1", "yes")
API_KEY = os.getenv("KALSHI_API_KEY", "")
PEM_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")

_client = None
_cb = CircuitBreaker()


def _get_client():
    """Lazy-init Kalshi API client with RSA auth."""
    global _client
    if _client is not None:
        return _client

    try:
        from kalshi_python_sync.api_client import KalshiAPIClient
    except ImportError:
        raise ImportError("pip install kalshi-python-sync")

    if not API_KEY:
        raise EnvironmentError("KALSHI_API_KEY not set in .env")
    if not PEM_PATH or not Path(PEM_PATH).exists():
        raise EnvironmentError(f"KALSHI_PRIVATE_KEY_PATH not found: {PEM_PATH}")

    pem = Path(PEM_PATH).read_text()
    _client = KalshiAPIClient(
        key_id=API_KEY,
        private_key=pem,
        host="https://api.elections.kalshi.com",
    )
    log.info("Kalshi client initialized")
    return _client


def place_limit_order(ticker: str, side: str, price: float,
                      quantity: int = 1) -> str | None:
    """
    Place a limit order on Kalshi.

    Args:
        ticker: market ticker (e.g. "KXBTC-25AUG0116-B120625")
        side: "yes" or "no"
        price: limit price in cents (1-99)
        quantity: number of contracts

    Returns order_id or None on failure.
    """
    # Circuit breaker check
    cb = _cb.check()
    if not cb["allowed"]:
        log.warning(f"BLOCKED by circuit breaker: {cb['reason']}")
        return None

    price_cents = int(round(price * 100)) if price < 1 else int(price)

    if not LIVE_MODE:
        log.info(f"WOULD PLACE: {side.upper()} {quantity}x {ticker} @ {price_cents}c")
        return f"dry-run-{ticker}-{side}"

    try:
        client = _get_client()
        resp = client.create_order(
            ticker=ticker,
            side=side,
            type="limit",
            yes_price=price_cents if side == "yes" else None,
            no_price=price_cents if side == "no" else None,
            count=quantity,
        )
        order_id = resp.get("order", {}).get("order_id", "unknown")
        log.info(f"ORDER PLACED: {side.upper()} {quantity}x {ticker} @ {price_cents}c "
                 f"→ {order_id}")
        return order_id
    except Exception as e:
        log.error(f"Order failed: {e}")
        return None


def cancel_order(order_id: str) -> bool:
    """Cancel an open order."""
    if not LIVE_MODE:
        log.info(f"WOULD CANCEL: {order_id}")
        return True
    try:
        client = _get_client()
        client.cancel_order(order_id=order_id)
        log.info(f"CANCELLED: {order_id}")
        return True
    except Exception as e:
        log.error(f"Cancel failed: {e}")
        return False


def get_positions() -> list[dict]:
    """Get all open positions."""
    if not LIVE_MODE:
        log.info("WOULD GET POSITIONS (dry-run: empty)")
        return []
    try:
        client = _get_client()
        resp = client.get_positions()
        positions = resp.get("market_positions", [])
        return positions
    except Exception as e:
        log.error(f"Get positions failed: {e}")
        return []


def get_balance() -> float:
    """Get account balance in dollars."""
    if not LIVE_MODE:
        log.info("WOULD GET BALANCE (dry-run: $0)")
        return 0.0
    try:
        client = _get_client()
        resp = client.get_balance()
        balance = resp.get("balance", 0) / 100.0  # Kalshi returns cents
        return balance
    except Exception as e:
        log.error(f"Get balance failed: {e}")
        return 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"LIVE_MODE: {LIVE_MODE}")
    print(f"API_KEY: {'SET' if API_KEY else 'NOT SET'}")
    print(f"PEM_PATH: {PEM_PATH or 'NOT SET'}")
    # Dry-run test
    oid = place_limit_order("TEST-TICKER", "yes", 0.05, 1)
    print(f"Order: {oid}")
    print(f"Balance: ${get_balance():.2f}")
    print(f"Positions: {get_positions()}")
