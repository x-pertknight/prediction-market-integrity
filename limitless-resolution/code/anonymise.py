"""Run after pull.py, before publishing: replaces account addresses with per-file ids and drops tx hashes.
Usage: python3 anonymise.py runs/*.jsonl
Note: timestamps, sizes and prices are kept, so trades can still be matched to public on-chain fills."""
import json, sys
for f in sys.argv[1:]:
    ids, out = {}, []
    for l in open(f):
        r = json.loads(l)
        for e in r.get("events", []):
            e["acct"] = ids.setdefault(e["acct"], f"acct{len(ids)+1:04d}")
            e.pop("txHash", None)
        r.pop("safeTx", None)
        out.append(json.dumps(r))
    open(f, "w").write("\n".join(out) + "\n")
