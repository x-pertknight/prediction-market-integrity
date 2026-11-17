"""Compute aggregate resolution and trading metrics for one pulled set.

Usage: python3 metrics.py RAW.jsonl STEP_SEC LABEL  -> prints JSON

Denomination: trade value in USDC = matched shares (6 decimals) * trade price.
A trade "ended long the winner" if it bought the winning token or sold the losing one.
"""
import json, sys, statistics as st
from collections import Counter
from datetime import datetime


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


def long_winner(side, outcome_idx, winning):
    # side 0 = buy, 1 = sell
    return (side == 0 and outcome_idx == winning) or (side == 1 and outcome_idx != winning)


def compute(rows, step):
    ok = [r for r in rows if "error" not in r]
    out = {"markets": len(rows), "errors": len(rows) - len(ok)}
    out["resolved"] = sum(r["status"] == "RESOLVED" for r in ok)
    out["source_polymarket_settlement"] = sum(r["res_source"] == "polymarket-settlement" for r in ok)
    out["open_check_match"] = sum(bool((r["openCheck"] or {}).get("matches")) for r in ok)
    out["taker_delay_ms"] = dict(Counter(str(r["takerDelayMs"]) for r in ok))

    rule_bad = poly_bad = checked = 0
    for r in ok:
        w = r["winning"]
        try:
            o, c = float(r["openPrice"]), float(r["resolvePrice"])
        except (TypeError, ValueError):
            continue
        checked += 1
        if (0 if c >= o else 1) != w:
            rule_bad += 1
        p = r["poly_payout"]
        if not p or "1" not in p or p.index("1") != w:
            poly_bad += 1
    out.update(rule_checked=checked, rule_disagree=rule_bad, poly_payout_disagree=poly_bad)
    gaps = [abs(float(r["resolvePrice"]) / float(r["openPrice"]) - 1) for r in ok
            if r["openPrice"] and r["resolvePrice"]]
    out["within_1bp"] = sum(g < 1e-4 for g in gaps)
    out["within_1bp_share"] = round(out["within_1bp"] / len(gaps), 4)

    lag = [ts(r["res_attemptedAt"]) - r["end"] for r in ok if r["res_attemptedAt"]]
    out["lag_median_s"] = round(st.median(lag), 1)
    out["lag_p95_s"] = round(st.quantiles(lag, n=20)[18], 1)
    out["lag_max_s"] = round(max(lag), 1)
    out["second_attempt"] = sum(x > 150 for x in lag)
    srt = sorted(ok, key=lambda r: r["start"])
    pairs = [(x, y) for x, y in zip(srt, srt[1:]) if y["start"] == x["end"]]
    out["continuity_pairs"] = len(pairs)
    out["open_eq_prev_close"] = sum(abs(float(y["openPrice"]) - float(x["resolvePrice"])) <= 1e-9 * float(x["resolvePrice"]) for x, y in pairs)

    nt = [len(r["events"]) for r in ok]
    out["zero_trade_markets"] = sum(x == 0 for x in nt)
    out["zero_trade_share"] = round(out["zero_trade_markets"] / len(ok), 4)
    out["trades"] = sum(nt)

    acct_usd, usd_total = Counter(), 0.0
    late, alltr = [], []
    for r in ok:
        w = r["winning"]
        for e in r["events"]:
            v = float(e["matchedSize"]) / 1e6 * float(e["price"])
            usd_total += v
            acct_usd[e["acct"]] += v
            oi = 0 if e["tokenId"] == r["yes"] else (1 if e["tokenId"] == r["no"] else None)
            if oi is None or w is None:
                continue
            rel = (ts(e["createdAt"]) - r["start"]) / step
            lw = long_winner(e["side"], oi, w)
            alltr.append(lw)
            if rel >= 0.95:
                paid = float(e["price"]) if e["side"] == 0 else 1 - float(e["price"])
                late.append((lw, paid))
    out["usdc_traded"] = round(usd_total, 2)
    out["accounts"] = len(acct_usd)
    top = [v for _, v in acct_usd.most_common(5)]
    out["top1_share"] = round(top[0] / usd_total, 4) if top else None
    out["top5_share"] = round(sum(top) / usd_total, 4) if top else None
    out["late_n"] = len(late)
    out["late_long_winner_share"] = round(sum(x[0] for x in late) / len(late), 4) if late else None
    # implied win rate = average price paid for the side held; z from a sum-of-Bernoulli null
    if late:
        exp = sum(x[1] for x in late)
        var = sum(x[1] * (1 - x[1]) for x in late) or 1e-9
        out["late_implied_win"] = round(exp / len(late), 4)
        out["late_excess_pp"] = round(100 * (sum(x[0] for x in late) - exp) / len(late), 1)
        out["late_z"] = round((sum(x[0] for x in late) - exp) / var ** 0.5, 2)
    out["all_long_winner_share"] = round(sum(alltr) / len(alltr), 4) if alltr else None
    out["window_start"] = datetime.utcfromtimestamp(min(r["start"] for r in ok)).isoformat() + "Z"
    out["window_end"] = datetime.utcfromtimestamp(max(r["end"] for r in ok)).isoformat() + "Z"
    return out


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1])]
    m = compute(rows, int(sys.argv[2]))
    m["label"] = sys.argv[3]
    print(json.dumps(m))
