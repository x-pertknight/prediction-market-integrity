# Research log

## 2026-11-17

**Docs first.** Read Market Resolution, Smart Contracts, Security, Fees, Maker Rebates, LP Rewards, User-Generated Markets, Custom Markets, Refund Policy, Terms of Service, Analytics, Venue System. Documented:
- 5 and 15 minute crypto up/down markets resolve automatically from a Chainlink TWAP, Up if close TWAP >= price to beat.
- Misresolution refunds cost basis only; wrong-side winners keep payouts.
- Audits cover the upstream Polymarket CTF Exchange and NegRisk code, not Limitless changes.
- Terms prohibit wash trading, spoofing, layering, coordinated trading and multiple accounts; insider trading is not addressed.

To measure: actual resolution source, rule consistency, latency, cases decided inside oracle noise, late flow.

**Window 1 (exploratory, BTC).** Pulled BTC 5m (7 days) and BTC 15m (14 days) through the public API before writing predictions. Saw: every market sourced from "polymarket-settlement", zero rule disagreements, resolution about 91 s after close.

**Predictions written** (predictions.md) before pulling ETH.

**Window 2 (pre-registered, ETH).** ETH 5m (7 days, one market 404, 2015 usable) and ETH 15m (14 days).

| Prediction | Result | Verdict |
|---|---|---|
| P1 rule and payout consistency 100% | 0 disagreements in 2015 + 1344 | Held |
| P2 source polymarket-settlement >= 95% | 100% in both | Held |
| P3 median lag 80 to 110 s, max < 300 s | median 91.1 and 91.3 s, max 183.5 s | Held |
| P4 ETH 5m zero-trade share > BTC 54% | 90.9% | Held |
| P5 late long-winner share within 45% to 70% | ETH 5m n=5, inconclusive; ETH 15m n=66, 10.6% | **Failed** |
| P6 top 5 accounts > 50% of USDC | 64.7% and 67.9% | Held |

**P5 failure and belief update.** The prediction was badly specified: it compared the raw share of late trades ending long the winner with a fixed band, ignoring the price paid. Late ETH 15m trades were mostly cheap buys (average implied win 25.4%), so a low raw share is expected. Replaced with realised minus implied win rate. On that metric ETH 15m late trades run 14.8 pp below the price they paid (z = -3.19 under independence). This is the opposite of informed flow. Two caveats: 66 trades sit in 30 markets so trades are not independent and the z overstates significance, and one account made 47% of those trades. Read: late flow on Limitless looks like lottery buying, not sniping. BTC 15m late trades match their price almost exactly (-0.2 pp, n=104).

**Read by hand (BTC 5m).** Largest single trades are 195 USDC at 0.97 and several 70.28 USDC buys at 0.93 to 0.99, mid to late window, all on the side that won; repeated identical sizes suggest one automated participant buying near-certain outcomes. Slowest resolutions (about 183 s) are spread across days with no pattern found. No exact ties between close and price to beat.

**Second demolition pass (separate reviewer, cold, hostile).** Verdict: survives with repairs, every number reproduced independently. Repairs made: the "docs vs data" contrast restated as Chainlink TWAP price plus Polymarket parity, matching the docs' external venue parity category; "clean resolution" restated as record consistency; Safe transaction claim removed because the published data does not carry it; late-flow claim restated as a low-power test; near-tie wording softened; unverified Dune dashboard notes removed; anonymiser added; dead code removed from the number gate. Added from the review: retry pattern in resolution lag (58 markets at about 180 s) and open equals previous close continuity check.
