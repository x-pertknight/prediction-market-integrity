"""Pull settled Limitless recurring up/down markets and their finalized trades.
Usage: python3 pull.py SYMBOL SUBFREQ PREFIX STEP_SEC N_SLOTS OUTFILE
Example: python3 pull.py BTC minutes_5 btc-up-or-down-5-min 300 576 btc5.jsonl
Slugs follow <prefix>-<start_unix>; slots are walked back from the last closed slot.
"""
import json, sys, time, urllib.request, concurrent.futures as cf

API = "https://api.limitless.exchange"
prefix, step, n, out = sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]


def get(path, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(API + path, headers={"User-Agent": "research"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                return {"_error": str(e)}
            time.sleep(1.5 * (i + 1))


def one(start):
    slug = f"{prefix}-{start}"
    m = get(f"/markets/{slug}?include=all")
    mk = m.get("market", m)
    if "_error" in m or not mk.get("slug"):
        return {"slug": slug, "error": m.get("_error", "missing")}
    md = mk.get("metadata") or {}
    res = md.get("chainlinkTwapResolution") or {}
    ev, page = [], 1
    while True:
        e = get(f"/markets/{slug}/events?limit=100&page={page}")
        ev += e.get("events", [])
        if page >= e.get("totalPages", 1) or "_error" in e:
            break
        page += 1
    return {
        "slug": slug, "start": start, "end": start + step,
        "status": mk.get("status"), "volume": mk.get("volume"),
        "winning": mk.get("winningOutcomeIndex"),
        "yes": (mk.get("tokens") or {}).get("yes"), "no": (mk.get("tokens") or {}).get("no"),
        "updatedAt": mk.get("updatedAt"),
        "externalProvider": md.get("externalProvider"), "isPolyArbitrage": md.get("isPolyArbitrage"),
        "takerDelayMs": md.get("takerDelayMs"), "openPrice": md.get("openPrice"),
        "resolvePrice": md.get("resolvePrice"),
        "openCheck": md.get("polymarketOpenCheck"),
        "res_source": res.get("source"), "res_outcome": res.get("outcome"),
        "res_open_offset": (res.get("open") or {}).get("offsetSeconds"),
        "res_close_offset": (res.get("close") or {}).get("offsetSeconds"),
        "res_close_selectedAt": (res.get("close") or {}).get("selectedAt"),
        "poly_payout": (res.get("polymarket") or {}).get("payoutNumerators"),
        "res_attemptedAt": md.get("chainlinkTwapResolutionAttemptedAt"),
        "safeTx": res.get("safeTxHash"),
        "events": [{k: x.get(k) for k in ("createdAt", "price", "side", "matchedSize", "tokenId", "txHash")}
                   | {"acct": (x.get("profile") or {}).get("account")} for x in ev],
    }


now = int(time.time())
last_closed = (now // step) * step - step - step  # leave margin for settlement
starts = [last_closed - i * step for i in range(n)]
with cf.ThreadPoolExecutor(6) as ex, open(out, "w") as f:
    for i, r in enumerate(ex.map(one, starts)):
        f.write(json.dumps(r) + "\n")
        if i % 100 == 0:
            print(i, r.get("slug"), r.get("status"), len(r.get("events", [])), flush=True)
