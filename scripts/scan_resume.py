import json, time, urllib.request, csv, sys
from eth_utils import keccak
RPCS = ["https://polygon-bor-rpc.publicnode.com", "https://polygon.drpc.org",
        "https://polygon-rpc.com", "https://polygon.gateway.tenderly.co"]
ADAPTERS = ["0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74","0x2F5e3684cb1F318ec51b00Edba38d79Ac2c0aA9d"]
T1 = "0x" + keccak(text="QuestionResolved(bytes32,int256,uint256[])").hex()
T2 = "0x" + keccak(text="QuestionReset(bytes32)").hex()
ri = 0
def rpc(method, params, tries=16):
    global ri
    last = None
    for t in range(tries):
        url = RPCS[ri % len(RPCS)]
        try:
            req = urllib.request.Request(url, data=json.dumps({"jsonrpc":"2.0","id":1,"method":method,"params":params}).encode(), headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=30) as r: out = json.load(r)
            if "result" in out: return out["result"]
            last = str(out.get("error"))[:120]; raise RuntimeError(last)
        except Exception as e:
            last = str(e)[:120]; ri += 1; time.sleep(min(2**min(t,5), 30))
    raise RuntimeError("all rpcs failed: " + str(last))
start = int(sys.argv[1]); latest = int(rpc("eth_blockNumber", []), 16)
print(f"resume from {start} to {latest}", flush=True)
f = open("data/resolutions_raw.csv","a",newline="")
w = csv.writer(f)
b = start; n = 0
while b <= latest:
    size = 40_000
    while True:
        try:
            logs = rpc("eth_getLogs", [{"fromBlock": hex(b), "toBlock": hex(min(b+size-1, latest)),
                                        "address": ADAPTERS, "topics": [[T1, T2]]}])
            break
        except Exception:
            size //= 2
            if size < 1000: time.sleep(60); size = 40_000
    e = min(b+size-1, latest)
    for lg in logs:
        w.writerow([int(lg["blockNumber"],16), lg["topics"][0], lg["address"].lower(), lg["topics"][1]]); n += 1
    f.flush()
    print(f"block {e}/{latest} new={n}", flush=True)
    b = e + 1
print("DONE", n, flush=True)
