"""
scripts/chain/uma_ids.py: identifiers that link a Polygon dispute to the
mainnet DVM votes cast on it.

A Polygon OO dispute reaches the Ethereum DVM through two rewrites of the
ancillary data, which is why a raw text match finds nothing:

  1. OptimisticOracleV2._stampAncillaryData appends ",ooRequester:<adapter>"
     (address as lowercase utf8 hex, no 0x).
  2. OracleChildTunnel then COMPRESSES that, so every VoteRevealed carries
     "ancillaryDataHash:<keccak of the stamped bytes>,...,childChainId:137".

So the join key is keccak(ancillaryData || ",ooRequester:" || requester). It is
not sufficient alone: a question disputed, reset and disputed again reuses the
same ancillary data, so the DVM request time is the second key. That time is
the PROPOSAL time, not the OO request timestamp; it is read from the dispute's
own transaction (see oo_bridge_logs.py).
"""
from __future__ import annotations

from eth_utils import keccak


def question_id(ancillary_data: bytes) -> str:
    """Polymarket's questionID = keccak(ancillaryData). Stable across resets."""
    return keccak(ancillary_data).hex()


def stamped_ancillary_hash(ancillary_data: bytes, requester: str) -> str:
    """The `ancillaryDataHash` the mainnet DVM sees for a Polygon OO request."""
    addr = requester[2:] if requester.startswith("0x") else requester
    if len(addr) != 40:
        raise ValueError(f"requester must be a 20 byte address, got {requester!r}")
    stamped = ancillary_data + b",ooRequester:" + addr.lower().encode()
    return keccak(stamped).hex()
