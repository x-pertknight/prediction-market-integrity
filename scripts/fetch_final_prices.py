#!/usr/bin/env python3
"""Fetch final QuestionResolved settledPrice (topic2) for too-early disputed questions."""
import json, time, urllib.request, random
import pandas as pd

RPCS = [
    "https://polygon.drpc.org",
    "https://polygon-bor-rpc.publicnode.com",
    "https://polygon-rpc.com",
    "https://polygon.gateway.tenderly.co",
    "https://1rpc.io/matic",
]
TOPIC_RESOLVED = "0x566c3fbdd12dd86bb341787f6d531f79fd7ad4ce7e3ae2d15ac0ca1b601af9df"
ADAPTERS = ["0x6a9d222616c90fca5754cd1333cfd9b7fb6a4f74", "0x2f5e3684cb1f318ec51b00edba38d79ac2c0aa9d"]
START, END = 66158921, 90400000

rpc_i = 0
def call(method, params, tries=10):
    global rpc_i
    for t in range(tries):
        url = RPCS[rpc_i % len(RPCS)]
        try:
            req = urllib.request.Request(url, json.dumps({"jsonrpc":"2.0","id":1,"method":method,"params":params}).encode(), {"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read())
            if 'error' in out:
                raise RuntimeError(out['error'].get('message','err'))
            return out['result']
        except Exception as e:
            rpc_i += 1
            time.sleep(1 + t*0.5)
    raise RuntimeError("all rpcs failed")

def get_logs(frm, to, qids):
    """Adaptive: on failure, halve range."""
    try:
        return call("eth_getLogs", [{"fromBlock": hex(frm), "toBlock": hex(to),
            "address": ADAPTERS, "topics": [TOPIC_RESOLVED, qids]}])
    except RuntimeError:
        if to - frm < 50000:
            raise
        mid = (frm + to)//2
        return get_logs(frm, mid, qids) + get_logs(mid+1, to, qids)

df = pd.read_csv('disputes_with_qid.csv')
INT_MIN = -(2**255)
te = df[(df.dvm_confirmed_proposal == False) & (df.settled_price.apply(lambda p: int(p)==INT_MIN))]
qids = sorted(set(te.questionID))
print(f"{len(qids)} too-early questionIDs")

results = {}
BATCH = 80
for bi in range(0, len(qids), BATCH):
    batch = qids[bi:bi+BATCH]
    logs = []
    step = 3000000
    frm = START
    while frm <= END:
        to = min(frm + step - 1, END)
        logs += get_logs(frm, to, batch)
        frm = to + 1
    for lg in logs:
        qid = lg['topics'][1]
        price = int(lg['topics'][2], 16)
        if price >= 2**255: price -= 2**256
        blk = int(lg['blockNumber'], 16)
        # keep the LAST (highest block) resolution per qid
        if qid not in results or blk > results[qid][1]:
            results[qid] = (price, blk)
    print(f"batch {bi//BATCH+1}/{(len(qids)+BATCH-1)//BATCH}: cumulative {len(results)} resolved", flush=True)

out = pd.DataFrame([(q, p, b) for q, (p, b) in results.items()], columns=['questionID','final_price','final_block'])
out.to_csv('too_early_final_prices.csv', index=False)
print("saved", len(out))
