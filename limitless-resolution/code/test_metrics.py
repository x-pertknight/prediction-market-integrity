"""Smoke tests on synthetic markets. Run: python3 test_metrics.py"""
from metrics import compute, long_winner

assert long_winner(0, 1, 1) and long_winner(1, 0, 1)
assert not long_winner(0, 0, 1) and not long_winner(1, 1, 1)


def mk(start, o, c, win, events, poly=None):
    return {"slug": f"t-{start}", "start": start, "end": start + 300, "status": "RESOLVED",
            "winning": win, "yes": "Y", "no": "N", "openPrice": str(o), "resolvePrice": str(c),
            "res_source": "polymarket-settlement", "openCheck": {"matches": True}, "takerDelayMs": 250,
            "poly_payout": poly or (["1", "0"] if win == 0 else ["0", "1"]),
            "res_attemptedAt": "1970-01-01T00:%02d:%02dZ" % divmod(start + 300 + 90, 60) if start + 390 < 3600 else None,
            "events": events}

ev = lambda t, side, tok, size, price: {"createdAt": "1970-01-01T00:%02d:%02dZ" % divmod(t, 60), "side": side,
                                         "tokenId": tok, "matchedSize": str(size), "price": price, "acct": "a"}
rows = [
    mk(0, 100, 101, 0, [ev(10, 0, "Y", 2_000_000, 0.5), ev(290, 0, "N", 1_000_000, 0.2)]),
    mk(300, 100, 99, 1, []),
    mk(600, 100, 100, 0, [ev(895, 1, "N", 1_000_000, 0.1)]),  # tie resolves Up; late sell of loser
]
m = compute(rows, 300)
assert m["rule_disagree"] == 0 and m["poly_payout_disagree"] == 0, m
assert m["zero_trade_markets"] == 1
assert abs(m["usdc_traded"] - (1.0 + 0.2 + 0.1)) < 1e-9, m["usdc_traded"]
assert m["late_n"] == 2 and m["late_long_winner_share"] == 0.5, m  # t=290 rel .967 loses, t=895 rel .983 wins
assert m["lag_median_s"] == 90
assert m["late_implied_win"] == 0.55 and m["late_excess_pp"] == -5.0, m
bad = mk(900, 100, 99, 0, [])  # Down market marked Up
assert compute([bad], 300)["rule_disagree"] == 1
print("smoke ok")
