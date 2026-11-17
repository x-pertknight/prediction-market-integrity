# Pre-registered predictions (window 2, ETH)

Written 2026-11-17 before any ETH data was pulled. Not to be edited after the run.
Window 1 (BTC 5m 7d, BTC 15m 14d) was exploratory and informed these numbers.

Window 2: ETH up/down 5 minute markets, last 7 days, and ETH 15 minute markets, last 14 days.

P1. Resolution consistency. 100% of settled markets show winning index equal to the close >= open rule and equal to the mirrored Polymarket payout. Falsified by any single disagreement.
P2. Resolution source. 100% of markets report source "polymarket-settlement", not a direct Chainlink resolution. Falsified below 95%.
P3. Resolution lag. Median time from market end to resolution attempt between 80 and 110 seconds, max under 300 seconds.
P4. Thin activity. Share of ETH 5 minute markets with zero trades is higher than BTC's 54%.
P5. Late flow. In the last 5% of the market window, the share of trades that ended long the winning side is within 45% to 70%, i.e. no detectable sniping edge. Only evaluated if n >= 50; otherwise reported as inconclusive.
P6. Concentration. Top 5 accounts carry more than 50% of USDC traded in each ETH set.
