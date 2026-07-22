import json, time, urllib.request, csv, sys
from eth_utils import keccak

RPCS = ["https://polygon.drpc.org", "https://polygon-rpc.com",
        "https://polygon.gateway.tenderly.co", "https://polygon-bor-rpc.publicnode.com"]
ADAPTERS = ["0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74",  # v2
            "0x2F5e3684cb1F318ec51b00Edba38d79Ac2c0aA9d"]  # v3 negrisk
TOPIC_RESOLVED = "0x" + keccak(text="QuestionResolved(bytes32,int256,uint256[])").hex()
TOPIC_RESET    = "0x" + keccak(text="QuestionReset(bytes32)").hex()

ri = 0
def rpc(method, params, tries=8):
    global ri
    for t in range(tries):
        url = RPCS[ri % len(RPCS)]
        try:
            req = urllib.request.Request(url, data=json.dumps(
                {"jsonrpc":"2.0","id":1,"method":method,"params":params}).encode(),
                headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.load(r)
            if "result" in out: return out["result"]
            raise RuntimeError(str(out.get("error"))[:100])
        except Exception as e:
            ri += 1; time.sleep(1 + t)
    raise RuntimeError("all rpcs failed")

def block_ts(n):
    b = rpc("eth_getBlockByNumber", [hex(n), False])
    return int(b["timestamp"], 16)

latest = int(rpc("eth_blockNumber", []), 16)
# binary search block at 2025-01-01
target = 1735689600
lo, hi = 40_000_000, latest
while lo < hi - 1000:
    mid = (lo+hi)//2
    if block_ts(mid) < target: lo = mid
    else: hi = mid
start = lo
print(f"latest={latest} start_block={start} start_ts={block_ts(start)}", flush=True)

rows = []
CHUNK = 40_000
b = start
while b <= latest:
    e = min(b + CHUNK - 1, latest)
    ok = False
    size = CHUNK
    while not ok:
        try:
            logs = rpc("eth_getLogs", [{"fromBlock": hex(b), "toBlock": hex(min(b+size-1, latest)),
                                        "address": ADAPTERS,
                                        "topics": [[TOPIC_RESOLVED, TOPIC_RESET]]}])
            ok = True
        except Exception as ex:
            size = size // 2
            if size < 1000: raise
    e = min(b+size-1, latest)
    for lg in logs:
        rows.append((int(lg["blockNumber"],16), lg["topics"][0], lg["address"].lower(), lg["topics"][1]))
    if len(rows) % 500 < 50 or True:
        print(f"block {e}/{latest} logs_total={len(rows)}", flush=True)
    with open("data/resolutions_raw.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["block","topic0","adapter","questionID"]); w.writerows(rows)
    b = e + 1
print("DONE", len(rows), flush=True)
