"""Blocking gate: every metric-derived number quoted in README and report must match final/metrics.json.
Run from the study root: python3 code/check_numbers.py"""
import json, re, sys

m = json.load(open("final/metrics.json"))
S = ["btc5_7d", "eth5_7d", "btc15_14d", "eth15_14d"]
pct = lambda k, s: f"{100*m[s][k]:.1f}%"
expected = set()
for s in S:
    for k in ("zero_trade_share", "top5_share", "within_1bp_share"):
        expected.add(pct(k, s))
    for k in ("lag_median_s", "lag_max_s", "late_excess_pp"):
        expected.add(f"{m[s][k]}")
    expected.add(f"{m[s]['within_1bp']} ({pct('within_1bp_share', s)})")
    expected.add(f"{round(m[s]['usdc_traded']):,}")
    expected.add(str(m[s]["accounts"])); expected.add(str(m[s]["late_n"]))
    expected.add(str(m[s]["markets"] - m[s]["errors"]))
    expected.add(str(m[s]["second_attempt"])); expected.add(str(m[s]["open_eq_prev_close"]))
total = sum(m[s]["markets"] - m[s]["errors"] for s in S)
expected.add(f"{total:,}")

fails = []
# hard asserts on headline claims
for s in S:
    if m[s]["rule_disagree"] or m[s]["poly_payout_disagree"]:
        fails.append(f"{s}: disagreements present")
    if m[s]["source_polymarket_settlement"] != m[s]["markets"] - m[s]["errors"]:
        fails.append(f"{s}: source not 100%")
sec = sum(m[s]["second_attempt"] for s in S)
if sec != 58 or f"{100*sec/total:.1f}%" != "0.9%":
    fails.append(f"second attempts {sec}")
if any(m[s]["open_eq_prev_close"] != m[s]["continuity_pairs"] for s in S):
    fails.append("continuity not complete but text says every pair matches")
# rounded descriptives used in prose tables: "about 91 s", "about 180 s", "0.9%"
if all(90.5 <= m[s]["lag_median_s"] < 91.5 for s in S): expected.add("91")
if all(175 <= m[s]["lag_max_s"] <= 190 for s in S): expected.add("180")
expected.add(f"{100*sec/total:.1f}%")
if total != 6719:
    fails.append(f"total markets {total} != 6719")
shares = sorted(m[s]["within_1bp_share"] for s in S)
claims = {
    "4.4%": pct("within_1bp_share", "eth15_14d"), "6.6%": pct("within_1bp_share", "btc5_7d"),
    "64.7%": pct("top5_share", "eth5_7d"), "69.2%": pct("top5_share", "btc5_7d"),
    "90.9%": pct("zero_trade_share", "eth5_7d"), "-0.2": f"{m['btc15_14d']['late_excess_pp']}",
    "-14.8": f"{m['eth15_14d']['late_excess_pp']}", "n=104": f"n={m['btc15_14d']['late_n']}",
    "n=66": f"n={m['eth15_14d']['late_n']}", "91.1": f"{m['eth5_7d']['lag_median_s']}",
}
for want, got in claims.items():
    if want != got:
        fails.append(f"claim {want} but metrics give {got}")
if not (claims["4.4%"] == f"{100*shares[0]:.1f}%" and claims["6.6%"] == f"{100*shares[-1]:.1f}%"):
    fails.append("1 bp range endpoints are not the min and max")
for f in ("README.md", "final/report.md"):
    txt = open(f).read()
    for want in ("6,719", "4.4%", "6.6%", "-0.2 pp", "-14.8 pp", "n=104", "n=66"):
        if want not in txt:
            fails.append(f"{f}: headline value {want} missing")
    # every percentage in the results table must be an expected value
    table = [l for l in txt.splitlines() if l.startswith("|")]
    for l in table:
        for tok in re.findall(r"-?\d[\d,]*\.?\d*%?(?: \(\d+\.\d%\))?", l):
            if tok in ("5", "15", "1", "0", "2016", "1344", "2015", "100%", "250", "500", "150") or tok.startswith("5%"):
                continue
            if tok not in expected and tok.replace(",", "") not in expected:
                fails.append(f"{f}: table value {tok} not in metrics")
print("\n".join(fails) if fails else "numbers gate: PASS")
sys.exit(1 if fails else 0)
