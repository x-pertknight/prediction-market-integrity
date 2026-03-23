"""
src/execution/polymarket_executor.py — Polymarket order execution placeholder.

Requires Polygon wallet with MATIC + USDC for live trading.
Uses py-clob-client for CLOB API access.
LIVE_MODE env var controls real vs dry-run. Default: False.

Same interface as kalshi_executor:
  place_limit_order(token_id, side, price, quantity) → order_id
  cancel_order(order_id) → bool
  get_positions() → list[dict]
  get_balance() → float
"""

import os
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from risk.circuit_breaker import CircuitBreaker

log = logging.getLogger("polymarket_executor")

LIVE_MODE = os.getenv("LIVE_MODE", "false").lower() in ("true", "1", "yes")
POLY_API_KEY = os.getenv("POLY_API_KEY", "")
POLY_SECRET = os.getenv("POLY_SECRET", "")
POLY_PASSPHRASE = os.getenv("POLY_PASSPHRASE", "")
POLY_PRIVATE_KEY = os.getenv("POLY_PRIVATE_KEY", "")  # Ethereum private key

_client = None
_cb = CircuitBreaker()


def _get_client():
    """Lazy-init Polymarket CLOB client."""
    global _client
    if _client is not None:
        return _client

    if not POLY_API_KEY or not POLY_SECRET or not POLY_PRIVATE_KEY:
        raise EnvironmentError(
            "Polymarket CLOB requires: POLY_API_KEY, POLY_SECRET, "
            "POLY_PASSPHRASE, POLY_PRIVATE_KEY in .env\n"
            "Get these from https://polymarket.com/settings/api"
        )

    try:
        from py_clob_client.client import ClobClient
        _client = ClobClient(
            host="https://clob.polymarket.com",
            key=POLY_API_KEY,
            chain_id=137,  # Polygon mainnet
            funder=POLY_PRIVATE_KEY,
            signature_type=2,  # POLY_GNOSIS_SAFE
            api_creds={
                "api_key": POLY_API_KEY,
                "api_secret": POLY_SECRET,
                "api_passphrase": POLY_PASSPHRASE,
            },
        )
        log.info("Polymarket CLOB client initialized")
        return _client
    except ImportError:
        raise ImportError("pip install py-clob-client")


def place_limit_order(token_id: str, side: str, price: float,
                      quantity: float = 10.0) -> str | None:
    """
    Place a limit order on Polymarket CLOB.

    Args:
        token_id: condition token ID (from Gamma API)
        side: "BUY" or "SELL"
        price: limit price (0-1)
        quantity: size in USDC

    Returns order_id or None.
    """
    cb = _cb.check()
    if not cb["allowed"]:
        log.warning(f"BLOCKED by circuit breaker: {cb['reason']}")
        return None

    if not LIVE_MODE:
        log.info(f"WOULD PLACE: {side} ${quantity:.2f} of {token_id[:16]}... @ {price:.3f}")
        return f"dry-run-{token_id[:8]}-{side}"

    try:
        client = _get_client()
        from py_clob_client.order_builder.constants import BUY, SELL
        order_side = BUY if side.upper() == "BUY" else SELL
        order = client.create_and_post_order(
            token_id=token_id,
            price=price,
            size=quantity,
            side=order_side,
        )
        order_id = order.get("orderID") or order.get("id", "unknown")
        log.info(f"ORDER PLACED: {side} ${quantity:.2f} @ {price:.3f} → {order_id}")
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
        client.cancel(order_id=order_id)
        log.info(f"CANCELLED: {order_id}")
        return True
    except Exception as e:
        log.error(f"Cancel failed: {e}")
        return False


def get_positions() -> list[dict]:
    """Get open positions (placeholder — requires authenticated wallet)."""
    if not LIVE_MODE:
        log.info("WOULD GET POSITIONS (dry-run: empty)")
        return []
    try:
        client = _get_client()
        # py-clob-client doesn't have a direct positions endpoint;
        # positions are tracked via on-chain token balances
        log.warning("Polymarket positions require on-chain query (not implemented)")
        return []
    except Exception as e:
        log.error(f"Get positions failed: {e}")
        return []


def get_balance() -> float:
    """Get USDC balance (placeholder)."""
    if not LIVE_MODE:
        log.info("WOULD GET BALANCE (dry-run: $0)")
        return 0.0
    log.warning("Polymarket balance requires on-chain USDC query (not implemented)")
    return 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"LIVE_MODE: {LIVE_MODE}")
    print(f"POLY_API_KEY: {'SET' if POLY_API_KEY else 'NOT SET'}")
    # Dry-run test
    oid = place_limit_order("0xabc123", "BUY", 0.05, 10.0)
    print(f"Order: {oid}")
