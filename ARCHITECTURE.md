# Architecture

## Overview

Research infrastructure for Polymarket, Kalshi and UMA. It started as a prediction market model (random forest plus a fat-tail risk framework, trained on Kalshi history and run in paper on Polymarket) and became the tooling for the integrity studies in the README. Everything runs in paper, `LIVE_MODE=False`. The strategies that were tested and killed are recorded in POSTMORTEM.md.

## Quick start

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/ -q          # test suite
python -I scripts/flip_test_uma.py       # UMA counterfactual flip test, offline
python src/execution/paper_trader.py     # paper trading (Polymarket)
python src/data/fetch_all.py             # refresh macro data
```

## Layout

```
src/
  data/          API fetchers, public RPC client, DVM logs, CTF positions, polymarket_feature_adapter.py
  models/        RF training, HMM regime, calibration, retrain
  risk/          fees.py, market_impact.py, circuit_breaker.py, fat-tail tooling
  execution/     paper_trader.py, kalshi_executor.py, polymarket_executor.py
  alerts/        Telegram alerts
  alpha/         UMA dispute and whale-edge tests, favorite-longshot, cross-platform rules matcher, negRisk scanner, OBI, wash filter
  whale/         whale tracker, SMI, conviction scoring
  mcp_server/    MCP server exposing the model and scans as tools
scripts/         UMA fetch, backfill, flip test and report scripts
data/            committed UMA dispute, vote and position datasets
docs/            preregistrations, audits, method notes
articles/        The Missing Greek, settlement risk on Polymarket and UMA
tests/           pytest suite
```

`.env` and `keys/` hold API credentials and are never committed.

## Feature pipeline

```
features_generic.py -> features_clean.py -> model_clean.py -> backtest_clean.py
                                                           -> paper_trader.py
```

## Model details

- RF on 20 features (8 constant features excluded by a nunique > 1 filter).
- Training data: Kalshi bulk history, 2021 to 2026.
- Entry when price <= model_prob * 0.70 (30 percent edge requirement).
- Exit when price >= model_prob * 0.90 or expiry is within 24h.

## Polymarket adapter

The Gamma API has no tags or category fields. The adapter infers the category by regex on the slug and question text, uses training-set medians for features it cannot map (hist_win_rate_series = 0.0149), computes days_in_market from startDate and ranks volume cross-sectionally.

## Circuit breaker

- 15 percent peak equity drawdown halts trading.
- 8 percent daily drawdown halves size.
- Rolling Sharpe window of 100 trades.
- Drawdown triggers only, no loss-count triggers.

## negRisk set arbitrage

- `src/alpha/negrisk_scanner.py` sums prices across multi-outcome events and nets fees per leg, in paper mode.
- `src/alpha/negrisk_paper.py` simulates buying the set at the CLOB touch and realizes it at resolution from the actual outcomes.
- Exhaustiveness whitelist in `data/negrisk_whitelist.json`, protocol in `docs/negrisk-whitelist-dossier.md`. Long side gated, short side free.
- Polymarket fees are charged per category (only geopolitics is free). Use `polymarket_fee_from_schedule()` when the market dict is available.

## Method rules

- All fees use the Kalshi parabolic formula or the Polymarket schedule, never flat estimates.
- Check the Hill tail index before reporting Sharpe. If alpha < 2, report gain-to-pain instead.
- No new feature without a permutation importance test.
- No entry threshold change without rerunning the backtest.

## Known limitations

- hist_win_rate_series uses the training median (0.0149), since Polymarket has no series history.
- Macro CSVs need a daily refresh via fetch_all.py.
- The model is trained on Kalshi and applied to Polymarket, so domain shift is expected.
- The entertainment Sharpe of 10.73 is likely not tradable (low liquidity).
- Edge decay measured at -2.84 Sharpe per quarter.
