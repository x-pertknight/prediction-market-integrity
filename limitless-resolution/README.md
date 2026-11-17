# Limitless resolution study

Every settled BTC and ETH up/down market on Limitless (Base), 5 minute cadence over 7 days and 15 minute cadence over 14 days: 6,719 markets and all their finalized trades, pulled from the public API.

- 100% of markets are flagged as mirrored from Polymarket, with resolution source "polymarket-settlement" inside the chainlinkTwapResolution metadata: Chainlink TWAP price, outcome taken from Polymarket's settlement (the docs' external venue parity category). The docs' 5 and 15 minute section does not mention the parity step.
- Median resolution 91.1 seconds after close, with 58 markets (0.9%) resolved on a second attempt at about 180 seconds.
- Zero disagreements between outcome, price rule and stored Polymarket payout. This shows Limitless records agree with each other, not that they agree with Chainlink or Polymarket.
- 4.4% to 6.6% of markets close within 1 bp of the price to beat.
- No detectable late edge, in samples too small and clustered to rule one out: -0.2 pp on BTC 15m (n=104), -14.8 pp on ETH 15m (n=66).

Full write-up: [final/report.md](final/report.md).

## Layout

- `brief.md` question, audience, acceptance criteria
- `predictions.md` pre-registered predictions for the ETH window
- `research_log.md` what was done, in order, including the prediction that failed
- `adversarial.md` two demolition passes and the repairs
- `code/` pull, anonymise, metrics, charts, smoke tests, number check
- `runs/` raw JSONL per set and per-set metrics
- `final/` report, charts, metrics.json

## Reproduce

```
python3 code/pull.py BTC minutes_5 btc-up-or-down-5-min 300 2016 runs/btc5_7d.jsonl
python3 code/anonymise.py runs/btc5_7d.jsonl
python3 code/metrics.py runs/btc5_7d.jsonl 300 btc5_7d
python3 code/test_metrics.py
python3 code/check_numbers.py
```

`pull.py` writes account addresses and tx hashes; `anonymise.py` replaces addresses with per-file ids and drops tx hashes. Timestamps, sizes and prices are kept, so trades remain matchable to public on-chain fills. Findings are reported in aggregate only.
