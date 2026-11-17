"""Charts for the report. Usage: python3 charts.py OUTDIR set1.jsonl:step:label ..."""
import json, sys
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
outdir, sets = sys.argv[1], [a.split(":") for a in sys.argv[2:]]
data = [(lab, int(st), [r for r in (json.loads(l) for l in open(f)) if "error" not in r]) for f, st, lab in sets]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1. resolution lag
fig, ax = plt.subplots(figsize=(7, 3.6))
for lab, step, rows in data:
    lag = [ts(r["res_attemptedAt"]) - r["end"] for r in rows if r.get("res_attemptedAt")]
    ax.hist(lag, bins=range(60, 200, 2), histtype="step", linewidth=1.4, label=lab)
ax.set_xlabel("seconds from market close to resolution attempt"); ax.set_ylabel("markets"); ax.set_yscale("log")
ax.legend(frameon=False); ax.set_title("Resolution latency")
fig.tight_layout(); fig.savefig(f"{outdir}/resolution_lag.png", dpi=160)

# 2. share of trades ending long the winner, by position in window
bins = [(0, .5), (.5, .8), (.8, .95), (.95, 1.0)]
fig, ax = plt.subplots(figsize=(7, 3.6))
w = 0.8 / len(data)
for i, (lab, step, rows) in enumerate(data):
    vals, ns = [], []
    for lo, hi in bins:
        sel = []
        for r in rows:
            for e in r["events"]:
                rel = (ts(e["createdAt"]) - r["start"]) / step
                oi = 0 if e["tokenId"] == r["yes"] else (1 if e["tokenId"] == r["no"] else None)
                if lo <= rel < hi and oi is not None:
                    won = (e["side"] == 0 and oi == r["winning"]) or (e["side"] == 1 and oi != r["winning"])
                    paid = e["price"] if e["side"] == 0 else 1 - e["price"]
                    sel.append(100 * (won - paid))
        vals.append(sum(sel) / len(sel) if sel else 0); ns.append(len(sel))
    xs = [j + i * w for j in range(len(bins))]
    bars = ax.bar(xs, vals, width=w, label=lab)
    for x, v, n in zip(xs, vals, ns):
        ax.text(x, (v if v >= 0 else 0) + 0.6, f"{n}", ha="center", fontsize=6, rotation=90, va="bottom")
ax.set_xticks([j + w * (len(data) - 1) / 2 for j in range(len(bins))])
ax.set_xticklabels(["0-50%", "50-80%", "80-95%", "last 5%"])
ax.set_xlabel("position of trade in market window"); ax.set_ylabel("realised minus implied win rate, pp")
ax.axhline(0, color="grey", linewidth=0.8); ax.legend(frameon=False, fontsize=8); ax.set_title("Late trades vs price paid (bar labels = trade count; low power)")
fig.tight_layout(); fig.savefig(f"{outdir}/late_flow.png", dpi=160)

# 3. close vs price to beat gap
fig, ax = plt.subplots(figsize=(7, 3.6))
for lab, step, rows in data:
    g = sorted(abs(float(r["resolvePrice"]) / float(r["openPrice"]) - 1) * 1e4 for r in rows)
    ax.plot(g, [k / len(g) for k in range(len(g))], label=lab)
ax.axvline(1, color="grey", linestyle=":"); ax.set_xscale("log")
ax.set_xlabel("|close / price to beat - 1| in basis points (log)"); ax.set_ylabel("cumulative share of markets")
ax.legend(frameon=False); ax.set_title("How many markets are decided inside 1 bp")
fig.tight_layout(); fig.savefig(f"{outdir}/near_ties.png", dpi=160)
print("charts ok")
