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
_orders_api = None
_portfolio_api = None
_cb = CircuitBreaker()


def _get_client():
    """Lazy-init Kalshi API client with RSA auth."""
    global _client, _orders_api, _portfolio_api
    if _client is not None:
        return _client

    try:
        from kalshi_python_sync import Configuration, ApiClient, KalshiAuth, OrdersApi, PortfolioApi
    except ImportError:
        raise ImportError("pip install kalshi-python-sync")

    if not API_KEY:
        raise EnvironmentError("KALSHI_API_KEY not set in .env")
    if not PEM_PATH or not Path(PEM_PATH).exists():
        raise EnvironmentError(f"KALSHI_PRIVATE_KEY_PATH not found: {PEM_PATH}")

    config = Configuration()
    _client = ApiClient(configuration=config)
    pem_text = Path(PEM_PATH).read_text()
    _client.kalshi_auth = KalshiAuth(API_KEY, pem_text)
    _orders_api = OrdersApi(_client)
    _portfolio_api = PortfolioApi(_client)
    log.info("Kalshi client initialized (kalshi-python-sync)")
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
        _get_client()
        resp = _orders_api.create_order(
            ticker=ticker,
            side=side,
            action="buy",
            yes_price=price_cents if side == "yes" else None,
            no_price=price_cents if side == "no" else None,
            count=quantity,
        )
        order = resp.order
        order_id = order.order_id if order else "unknown"
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
        _get_client()
        _orders_api.cancel_order(order_id=order_id)
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
        _get_client()
        resp = _portfolio_api.get_positions()
        positions = [p.to_dict() for p in (resp.market_positions or [])]
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
        _get_client()
        resp = _portfolio_api.get_balance()
        balance = (resp.balance or 0) / 100.0  # Kalshi returns cents
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
