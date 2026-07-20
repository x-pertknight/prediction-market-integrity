import json, time, urllib.request, csv, sys
from eth_utils import keccak
RPCS = ["https://polygon.gateway.tenderly.co","https://polygon.drpc.org","https://polygon-bor-rpc.publicnode.com","https://polygon-rpc.com","https://1rpc.io/matic"]
A = ["0x65070be91477460d8a7aeeb94ef92fe056c2f2a7"]
T1 = "0x" + keccak(text="QuestionResolved(bytes32,int256,uint256[])").hex()
T2 = "0x" + keccak(text="QuestionReset(bytes32)").hex()
ri = 0
def rpc(m, p, tries=30):
    global ri
    last=None
    for t in range(tries):
        try:
            req = urllib.request.Request(RPCS[ri%len(RPCS)], data=json.dumps({"jsonrpc":"2.0","id":1,"method":m,"params":p}).encode(), headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=30) as r: out = json.load(r)
            if "result" in out: return out["result"]
            last=str(out.get("error"))[:100]; raise RuntimeError(last)
        except Exception as e:
            last=str(e)[:100]; ri += 1; time.sleep(min(1.5**min(t,8), 20))
    raise RuntimeError("fail "+str(last))
latest = int(rpc("eth_blockNumber", []), 16)
b = 67238545  # ~2025-01-01
f = open("data/v4_resolutions.csv","w",newline=""); w = csv.writer(f)
w.writerow(["block","topic0"])
n=0
while b <= latest:
    size = 25_000
    while True:
        try:
            logs = rpc("eth_getLogs", [{"fromBlock": hex(b), "toBlock": hex(min(b+size-1, latest)), "address": A, "topics": [[T1,T2]]}])
            break
        except Exception:
            size //= 2
            if size < 1000: time.sleep(90); size = 25_000
    for lg in logs: w.writerow([int(lg["blockNumber"],16), lg["topics"][0]]); n+=1
    f.flush()
    print(f"{min(b+size-1,latest)}/{latest} n={n}", flush=True)
    b = min(b+size-1, latest) + 1
    time.sleep(0.15)
print("DONE", n, flush=True)
