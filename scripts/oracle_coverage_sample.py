#!/usr/bin/env python3
"""Random-sample oracle histogram on CTF ConditionResolution events (replays the coverage audit).
Usage: python3 oracle_coverage_sample.py  (public Polygon RPC only; seed fixed for replay)"""
import json, time, urllib.request, random
from collections import Counter
try:
    from Crypto.Hash import keccak
    k = keccak.new(digest_bits=256); k.update(b"ConditionResolution(bytes32,address,bytes32,uint256,uint256[])")
    TOPIC = '0x' + k.hexdigest()
except ImportError:
    TOPIC = None  # pip install pycryptodome
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
# Published audit (2026-07-20, latest=90561975): 135,126 events; 0x6507 58.2%, 0x58e1 21.0%, 0xd91e 12.0%, 0x29ae 6.8%
