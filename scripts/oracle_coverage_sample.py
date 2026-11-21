#!/usr/bin/env python3
"""Random sample of CTF ConditionResolution events on Polygon, counted by the
oracle that resolved them (replays the oracle coverage audit in the article).
Public Polygon RPC only; the seed is fixed so the same block chunks are drawn.

    python3 -I scripts/oracle_coverage_sample.py
"""
import json, time, urllib.request, random
from collections import Counter

from eth_utils import keccak

TOPIC = '0x' + keccak(text="ConditionResolution(bytes32,address,bytes32,uint256,uint256[])").hex()
CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
RPCS = ["https://polygon.drpc.org","https://polygon-bor-rpc.publicnode.com","https://polygon-rpc.com","https://1rpc.io/matic","https://polygon.gateway.tenderly.co"]
rpc_i = 0
def call(method, params, tries=18):
    global rpc_i
    for t in range(tries):
        url = RPCS[rpc_i % len(RPCS)]
        try:
            req = urllib.request.Request(url, json.dumps({"jsonrpc":"2.0","id":1,"method":method,"params":params}).encode(), {"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=25) as r:
                out = json.loads(r.read())
            if 'error' in out: raise RuntimeError(str(out['error']))
            return out['result']
        except Exception:
            rpc_i += 1; time.sleep(1.2)
    raise RuntimeError("rpc fail")

random.seed(42)  # fixed seed: same chunks as the published audit
latest = int(call("eth_blockNumber", []), 16)
starts = sorted(random.sample(range(88_400_000, min(latest, 90_561_975)-10_000, 1), 60))
hist = Counter(); total = 0
for s in starts:
    logs = call("eth_getLogs", [{"fromBlock": hex(s), "toBlock": hex(s+9_999), "address": CTF, "topics": [TOPIC]}])
    for lg in logs:
        hist['0x'+lg['topics'][2][-40:]] += 1; total += 1
print(f"sampled {total} ConditionResolution events over {len(starts)} 10k-block chunks")
for o, c in hist.most_common(12):
    print(f"{o}  {c}  {c/total:.1%}")
# Audit run on 2026-07-20 (latest=90561975): 135,126 events; managed adapter 0x6507 58.2%,
# automated resolver 0x58e1 21.0%, NegRiskAdapter 0xd91e 12.0%, externally owned account 0x29ae 6.8%
