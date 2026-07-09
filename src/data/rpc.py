"""
src/data/rpc.py — Minimal, dependency-light JSON-RPC client for public endpoints.

Built for one job: pulling large `eth_getLogs` ranges off free/public RPCs that
each impose a different, undocumented ceiling (block span, result count, or
payload size). Rather than guessing a safe window, the client starts wide and
halves on the specific errors providers use to signal "too much", then creeps
back up. Endpoints are rotated on transport failure so one flaky provider does
not abort a multi-hour backfill.

No API keys are embedded. Override the endpoint list with ETH_RPC_URL /
POLYGON_RPC_URL in .env (comma-separated for several).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import requests

# Public endpoints verified 2026-07-09 to serve eth_getLogs with wide windows.
DEFAULT_ETH_RPCS = [
    "https://mainnet.gateway.tenderly.co",
    "https://rpc.mevblocker.io",
    "https://eth.blockscout.com/api/eth-rpc",
    "https://eth.drpc.org",
]
# Ordered by verified ability to serve HISTORICAL receipts and archive state, and
# to accept wide eth_getLogs windows (2026-07-09). Two traps live here:
#   * several otherwise-healthy public nodes answer eth_getTransactionReceipt with
#     a null result for old transactions instead of erroring -- see call_nonnull.
#   * drpc's free tier caps eth_getLogs at 10k blocks and reports it as HTTP 400
#     with a JSON error body, so tenderly (which takes ~200k-block windows) leads.
DEFAULT_POLYGON_RPCS = [
    "https://polygon.gateway.tenderly.co",
    "https://polygon.drpc.org",
]

_HEADERS = {"User-Agent": "traptrade/uma-backfill", "Content-Type": "application/json"}

# Substrings providers use to say "your getLogs range/result set is too big".
# Each provider phrases this differently and there is no error code convention;
# a phrase missing from this list makes the scan abort instead of shrinking.
_TOO_MUCH = (
    "more than", "too large", "too many", "query timeout", "exceed",
    "limit exceeded", "response size", "range is too large", "block range",
    "are not supported",   # drpc: "ranges over 10000 blocks are not supported on freetier"
    "freetier", "too big", "query returned more than", "log response size",
)


# Provider errors that mean "try again / try elsewhere" rather than "your request is
# wrong". Treating these as fatal aborts a multi-thousand-call backfill on one blip.
_TRANSIENT = ("request timeout", "rate limit", "too many requests", "capacity",
              "temporarily unavailable", "server busy", "execution aborted")


class RpcError(RuntimeError):
    pass


def _endpoints(env_var: str, defaults: list[str]) -> list[str]:
    raw = os.environ.get(env_var, "").strip()
    return [u.strip() for u in raw.split(",") if u.strip()] or list(defaults)


@dataclass
class JsonRpc:
    """Round-robin JSON-RPC client with adaptive `get_logs` windowing."""
    urls: list[str]
    timeout: int = 120
    max_retries: int = 4
    _idx: int = field(default=0, init=False)

    @classmethod
    def ethereum(cls, **kw) -> "JsonRpc":
        return cls(_endpoints("ETH_RPC_URL", DEFAULT_ETH_RPCS), **kw)

    @classmethod
    def polygon(cls, **kw) -> "JsonRpc":
        return cls(_endpoints("POLYGON_RPC_URL", DEFAULT_POLYGON_RPCS), **kw)

    @property
    def url(self) -> str:
        return self.urls[self._idx % len(self.urls)]

    def _rotate(self) -> None:
        self._idx += 1

    def call(self, method: str, params: list) -> object:
        """One RPC call, rotating endpoints on transport/5xx failure.

        Provider-level JSON errors (bad params, result cap) are raised as
        RpcError immediately: rotating would not help and the caller needs to
        see them to shrink its window.
        """
        last: Exception | None = None
        for _ in range(self.max_retries):
            try:
                resp = requests.post(
                    self.url, headers=_HEADERS, timeout=self.timeout,
                    json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                )
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise requests.RequestException(f"HTTP {resp.status_code}")
                # A provider may report a legitimate JSON-RPC error (notably "your
                # getLogs range is too wide") with a 4xx status. Surfacing that as a
                # transport failure would rotate endpoints forever instead of letting
                # the caller shrink its window, so read the body before raising.
                try:
                    payload = resp.json()
                except ValueError:
                    payload = None
                if payload and "error" in payload:
                    err = str(payload["error"])
                    # A transient provider error must rotate+retry, not abort. If every
                    # attempt fails we still surface the original text, so get_logs_chunked
                    # can decide to shrink its window.
                    if any(s in err.lower() for s in _TRANSIENT):
                        raise requests.RequestException(err)
                    raise RpcError(err)
                resp.raise_for_status()
                return payload["result"]
            except RpcError:
                raise
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                last = exc
                self._rotate()
                time.sleep(0.8)
        raise RpcError(f"all endpoints failed for {method}: {last}")

    def call_nonnull(self, method: str, params: list) -> object:
        """Like `call`, but treats a null result as an endpoint failure.

        Pruned/light public nodes answer eth_getTransactionReceipt for old
        transactions with `"result": null` rather than an error. Accepting that
        as "no such transaction" silently turns a data-availability problem into
        a wrong dataset, so rotate and retry instead. Raises when every endpoint
        returns null.
        """
        for _ in range(len(self.urls) * 2):
            result = self.call(method, params)
            if result is not None:
                return result
            self._rotate()
        raise RpcError(f"{method}{params} returned null on every endpoint")

    # ── conveniences ─────────────────────────────────────────────────────────

    def block_number(self) -> int:
        return int(self.call("eth_blockNumber", []), 16)

    def block_timestamp(self, block: int) -> int:
        blk = self.call("eth_getBlockByNumber", [hex(block), False])
        return int(blk["timestamp"], 16)

    def block_at_timestamp(self, target_ts: int, lo: int, hi: int) -> int:
        """First block whose timestamp >= target_ts (binary search)."""
        while lo < hi:
            mid = (lo + hi) // 2
            if self.block_timestamp(mid) < target_ts:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def get_logs_chunked(self, address: str | list[str], topics: list, start: int, end: int,
                         window: int = 20_000, min_window: int = 250, progress=None):
        """Yield logs from `start`..`end`, adapting the window to the provider.

        Halves on a "too much" error, and doubles back up (capped at the initial
        window) after a clean pass, so a single dense stretch does not pin the
        whole scan to a tiny window.
        """
        initial = window
        frm = start
        while frm <= end:
            to = min(frm + window - 1, end)
            params = {"fromBlock": hex(frm), "toBlock": hex(to), "topics": topics}
            if address:
                params["address"] = address
            try:
                logs = self.call("eth_getLogs", [params])
            except RpcError as exc:
                msg = str(exc).lower()
                # "all endpoints failed" usually means every provider timed out on a
                # window this wide, which is the same signal as an explicit range
                # error -- shrink rather than abort. Only give up at min_window.
                too_much = any(s in msg for s in _TOO_MUCH) or "all endpoints failed" in msg
                if too_much and window > min_window:
                    window = max(window // 2, min_window)
                    continue
                raise
            yield from logs
            if progress:
                progress(to, len(logs))
            frm = to + 1
            if window < initial:
                window = min(window * 2, initial)
