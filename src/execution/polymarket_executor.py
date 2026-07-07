"""
src/execution/polymarket_executor.py — Polymarket CLOB order execution.

Authentication tiers (per Polymarket docs):
  PUBLIC:  No auth — markets, orderbooks, prices, Data API positions
  L2:      API key + secret + passphrase — cancel orders, balance, post pre-signed orders
  L1:      Wallet private key — derive L2 creds, sign orders (EIP-712)

IMPORTANT: Order placement ALWAYS requires the wallet private key for EIP-712
signing. There is no API-key-only order flow. The private key signs the order
payload locally; it is never sent to Polymarket.

Credentials in .env:
  POLY_WALLET_ADDRESS  — Wallet address (for public Data API queries, no auth)
  POLY_API_KEY         — CLOB API key    (for L2: balance, cancel, post signed orders)
  POLY_SECRET          — CLOB API secret (for L2)
  POLY_PASSPHRASE      — CLOB passphrase (for L2)
  POLY_PRIVATE_KEY     — Wallet private key (for L1: key derivation + order signing)
  LIVE_MODE            — "true" for real orders, "false" for dry-run (default: false)

Public interface:
  test_connection()     → dict  (read-only, no auth needed)
  get_markets()         → list  (public, no auth)
  get_positions()       → list  (public Data API, wallet address only)
  get_balance()         → float (L2 auth required)
  get_open_orders()     → list  (L2 auth required)
  place_limit_order()   → str   (L1+L2: private key required for signing)
  cancel_order()        → bool  (L2 auth required)
"""

import os
import logging
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from risk.circuit_breaker import CircuitBreaker

log = logging.getLogger("polymarket_executor")

# ── Config ────────────────────────────────────────────────────────────────────

LIVE_MODE = os.getenv("LIVE_MODE", "false").lower() in ("true", "1", "yes")

# Public: only needs wallet address
POLY_WALLET_ADDR = (os.getenv("POLY_WALLET_ADDRESS", "")
                    or os.getenv("POLY_WALLET_ADDR", ""))

# L2: API credentials (for balance, cancel, order posting)
POLY_API_KEY    = os.getenv("POLY_API_KEY", "")
POLY_SECRET     = os.getenv("POLY_SECRET", "")
POLY_PASSPHRASE = os.getenv("POLY_PASSPHRASE", "")

# L1: Private key (for order signing + credential derivation)
POLY_PRIVATE_KEY = os.getenv("POLY_PRIVATE_KEY", "")

CLOB_HOST      = "https://clob.polymarket.com"
GAMMA_API_HOST = "https://gamma-api.polymarket.com"
DATA_API_HOST  = "https://data-api.polymarket.com"
CHAIN_ID       = 137  # Polygon mainnet

MAX_RETRIES   = 3
RETRY_BACKOFF = [1, 3, 10]

_public_client = None
_l2_client = None
_cb = CircuitBreaker()


# ── Retry helper ──────────────────────────────────────────────────────────────

def _retry(fn, *args, **kwargs):
    """Retry a callable with backoff on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (ConnectionError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            log.warning(f"Connection error (attempt {attempt+1}/{MAX_RETRIES}): {e}, "
                        f"retrying in {wait}s")
            time.sleep(wait)
        except Exception:
            raise
    raise ConnectionError(f"Failed after {MAX_RETRIES} retries")


# ── Public client (no auth) ──────────────────────────────────────────────────

def _get_public_client():
    """Public CLOB client — no credentials needed."""
    global _public_client
    if _public_client is not None:
        return _public_client

    from py_clob_client.client import ClobClient
    _public_client = ClobClient(host=CLOB_HOST, chain_id=CHAIN_ID)
    log.info("Polymarket public client initialized (no auth)")
    return _public_client


# ── L2 client (API key auth) ─────────────────────────────────────────────────

def _get_l2_client():
    """L2 CLOB client — requires API key + secret + passphrase.
    If private key is available, also enables order signing."""
    global _l2_client
    if _l2_client is not None:
        return _l2_client

    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds

    if POLY_PRIVATE_KEY:
        # Full L1+L2: can sign orders AND use API creds
        client = ClobClient(
            host=CLOB_HOST,
            key=POLY_PRIVATE_KEY,
            chain_id=CHAIN_ID,
        )
        if POLY_API_KEY and POLY_SECRET and POLY_PASSPHRASE:
            creds = ApiCreds(
                api_key=POLY_API_KEY,
                api_secret=POLY_SECRET,
                api_passphrase=POLY_PASSPHRASE,
            )
            log.info("Using pre-set L2 credentials from .env")
        else:
            log.info("Deriving L2 credentials from wallet...")
            creds = client.create_or_derive_api_creds()
            log.info("L2 credentials derived successfully")
        client.set_api_creds(creds)
        _l2_client = client
        log.info("Polymarket L2 client initialized (private key + API creds)")
        return _l2_client

    if POLY_API_KEY and POLY_SECRET and POLY_PASSPHRASE:
        # L2-only: can check balance, cancel orders, but NOT sign new orders
        client = ClobClient(host=CLOB_HOST, chain_id=CHAIN_ID)
        creds = ApiCreds(
            api_key=POLY_API_KEY,
            api_secret=POLY_SECRET,
            api_passphrase=POLY_PASSPHRASE,
        )
        client.set_api_creds(creds)
        _l2_client = client
        log.info("Polymarket L2 client initialized (API key only, no order signing)")
        return _l2_client

    raise EnvironmentError(
        "L2 auth requires either:\n"
        "  a) POLY_PRIVATE_KEY (derives creds automatically), or\n"
        "  b) POLY_API_KEY + POLY_SECRET + POLY_PASSPHRASE"
    )


# ── Public read-only operations (NO auth) ────────────────────────────────────

def get_markets(limit: int = 20) -> list[dict]:
    """Fetch active markets from CLOB (public, no auth)."""
    client = _get_public_client()
    result = _retry(client.get_markets, next_cursor="MA==")
    if isinstance(result, list):
        return result[:limit]
    return (result.get("data", []) or [])[:limit]


def get_positions() -> list[dict]:
    """Get positions via public Data API (no auth, just wallet address)."""
    addr = POLY_WALLET_ADDR
    if not addr:
        log.warning("POLY_WALLET_ADDRESS not set — cannot query positions")
        return []

    try:
        r = _retry(requests.get,
                    f"{DATA_API_HOST}/positions",
                    params={"user": addr.lower()},
                    timeout=15)
        if r.status_code != 200:
            log.warning(f"Data API positions: {r.status_code} {r.text[:200]}")
            return []
        data = r.json()
        positions = data if isinstance(data, list) else data.get("positions", [])
        log.info(f"Fetched {len(positions)} positions from Data API")
        return positions
    except Exception as e:
        log.error(f"Get positions failed: {e}")
        return []


def get_orderbook(token_id: str) -> dict:
    """Get orderbook for a token (public, no auth)."""
    client = _get_public_client()
    return _retry(client.get_order_book, token_id)


# ── L2 authenticated operations ──────────────────────────────────────────────

def get_balance() -> float:
    """Get USDC balance (requires L2 auth)."""
    try:
        client = _get_l2_client()
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        result = _retry(client.get_balance_allowance, params)
        balance = float(result.get("balance", 0)) / 1e6  # USDC has 6 decimals
        return balance
    except Exception as e:
        log.error(f"Get balance failed: {e}")
        return 0.0


def get_open_orders() -> list[dict]:
    """Get open orders (requires L2 auth)."""
    try:
        client = _get_l2_client()
        result = _retry(client.get_orders)
        orders = result if isinstance(result, list) else result.get("data", [])
        return orders
    except Exception as e:
        log.error(f"Get orders failed: {e}")
        return []


def cancel_order(order_id: str) -> bool:
    """Cancel an open order (requires L2 auth)."""
    if not LIVE_MODE:
        log.info(f"[DRY-RUN] CANCEL: {order_id}")
        return True
    try:
        client = _get_l2_client()
        _retry(client.cancel, order_id=order_id)
        log.info(f"CANCELLED: {order_id}")
        return True
    except Exception as e:
        log.error(f"Cancel failed: {e}")
        return False


# ── Order execution (requires L1 private key for signing) ────────────────────

def place_limit_order(token_id: str, side: str, price: float,
                      quantity: float = 10.0) -> str | None:
    """
    Place a limit order on Polymarket CLOB.

    REQUIRES POLY_PRIVATE_KEY — orders must be EIP-712 signed by the wallet.
    The private key signs locally; it is never sent to Polymarket.

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
        log.info(f"[DRY-RUN] {side} ${quantity:.2f} of {token_id[:16]}... @ {price:.3f}")
        return f"dry-run-{token_id[:8]}-{side}"

    if not POLY_PRIVATE_KEY:
        log.error("Cannot place orders: POLY_PRIVATE_KEY required for EIP-712 signing")
        return None

    try:
        client = _get_l2_client()
        from py_clob_client.order_builder.constants import BUY, SELL
        order_side = BUY if side.upper() == "BUY" else SELL
        order = _retry(client.create_and_post_order,
                       token_id=token_id, price=price,
                       size=quantity, side=order_side)
        order_id = order.get("orderID") or order.get("id", "unknown")
        log.info(f"ORDER PLACED: {side} ${quantity:.2f} @ {price:.3f} -> {order_id}")
        return order_id
    except Exception as e:
        log.error(f"Order failed: {e}")
        return None


# ── Connection test ───────────────────────────────────────────────────────────

def test_connection() -> dict:
    """Test connection at each auth tier. Returns status dict."""
    results = {"public": {}, "data_api": {}, "l2": {}, "errors": []}

    # Tier 1: Public CLOB (no auth)
    try:
        client = _get_public_client()
        ok = _retry(client.get_ok)
        results["public"]["server"] = "OK" if ok else "DOWN"
    except Exception as e:
        results["public"]["server"] = f"FAIL: {e}"
        results["errors"].append(f"Public server: {e}")

    try:
        markets = _retry(client.get_markets, next_cursor="MA==")
        data = markets if isinstance(markets, list) else markets.get("data", [])
        results["public"]["markets"] = f"{len(data)} fetched"
        if data:
            m = data[0]
            results["public"]["sample"] = (
                f"{m.get('question', m.get('condition_id', '?'))[:60]}"
            )
    except Exception as e:
        results["public"]["markets"] = f"FAIL: {e}"
        results["errors"].append(f"Public markets: {e}")

    # Tier 2: Data API positions (no auth, just wallet address)
    if POLY_WALLET_ADDR:
        try:
            positions = get_positions()
            results["data_api"]["positions"] = f"{len(positions)} found"
            if positions:
                # Show summary
                total_value = sum(
                    float(p.get("currentValue", 0) or 0)
                    for p in positions
                )
                results["data_api"]["total_value"] = f"${total_value:.2f}"
        except Exception as e:
            results["data_api"]["positions"] = f"FAIL: {e}"
            results["errors"].append(f"Data API positions: {e}")
    else:
        results["data_api"]["positions"] = "SKIPPED (no wallet address)"

    # Tier 3: L2 auth (balance, orders)
    has_l2 = bool(
        POLY_PRIVATE_KEY
        or (POLY_API_KEY and POLY_SECRET and POLY_PASSPHRASE)
    )
    if has_l2:
        try:
            bal = get_balance()
            results["l2"]["balance"] = f"${bal:.2f} USDC"
        except Exception as e:
            results["l2"]["balance"] = f"FAIL: {e}"
            results["errors"].append(f"L2 balance: {e}")
        try:
            orders = get_open_orders()
            results["l2"]["open_orders"] = f"{len(orders)}"
        except Exception as e:
            results["l2"]["open_orders"] = f"FAIL: {e}"
            results["errors"].append(f"L2 orders: {e}")
    else:
        results["l2"]["status"] = "SKIPPED (no L2 credentials)"

    # Order signing capability
    results["can_sign_orders"] = bool(POLY_PRIVATE_KEY)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    print("=" * 60)
    print("Polymarket Executor — Connection Test")
    print("=" * 60)
    print(f"LIVE_MODE:          {LIVE_MODE}")
    print(f"POLY_WALLET_ADDR:   {'SET' if POLY_WALLET_ADDR else 'NOT SET'}")
    print(f"POLY_API_KEY:       {'SET' if POLY_API_KEY else 'NOT SET'}")
    print(f"POLY_SECRET:        {'SET' if POLY_SECRET else 'NOT SET'}")
    print(f"POLY_PASSPHRASE:    {'SET' if POLY_PASSPHRASE else 'NOT SET'}")
    print(f"POLY_PRIVATE_KEY:   {'SET' if POLY_PRIVATE_KEY else 'NOT SET'}")
    print()

    print("Running connection test (read-only)...\n")
    result = test_connection()

    print("--- PUBLIC (no auth) ---")
    for k, v in result["public"].items():
        print(f"  {k:15s}: {v}")

    print("\n--- DATA API (wallet address) ---")
    for k, v in result["data_api"].items():
        print(f"  {k:15s}: {v}")

    print("\n--- L2 AUTH (API key) ---")
    for k, v in result["l2"].items():
        print(f"  {k:15s}: {v}")

    print(f"\n--- ORDER SIGNING ---")
    print(f"  can_sign:       {result['can_sign_orders']}")
    if not result["can_sign_orders"]:
        print("  (POLY_PRIVATE_KEY required for EIP-712 order signing)")

    if result["errors"]:
        print(f"\nErrors ({len(result['errors'])}):")
        for e in result["errors"]:
            print(f"  x {e}")
    else:
        print("\nAll checks passed.")

    # Show positions detail if any
    if POLY_WALLET_ADDR:
        positions = get_positions()
        if positions:
            print(f"\n--- POSITIONS ({len(positions)}) ---")
            for p in positions[:10]:
                q = (p.get("title") or p.get("question")
                     or p.get("asset", {}).get("condition_id", "?"))
                size = p.get("size", p.get("amount", "?"))
                val = p.get("currentValue", "?")
                print(f"  {str(q)[:50]:50s}  size={size}  val={val}")
            if len(positions) > 10:
                print(f"  ... and {len(positions) - 10} more")
