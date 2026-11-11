# TrapTrade — Architecture & Development Notes

## Overview
Polymarket/Kalshi prediction market bot. RF model + Taleb risk framework.
Trained on Kalshi historical data, paper trading on Polymarket via Gamma API.

## Quick Start
```bash
source .venv/bin/activate
python src/execution/paper_trader.py     # paper trading (Polymarket)
pytest tests/ -v                         # tests
python src/data/fetch_all.py             # refresh macro data
```

## Architecture
```
src/
  data/          — API fetchers + polymarket_feature_adapter.py
  models/        — RF training, HMM regime, calibration, retrain
  risk/          — fees.py, market_impact.py, circuit_breaker.py (Taleb framework)
  execution/     — paper_trader.py, kalshi_executor.py, polymarket_executor.py
  alerts/        — Telegram bot (alerts bot)
  alpha/         — favorite-longshot, cross-platform arb, OBI, wash filter
  whale/         — whale tracker, SMI, conviction scoring
data/
  raw/           — API CSVs, OHLCV cache (market_history/)
  processed/     — wf_clean_general.parquet, feature matrices
models/          — rf_clean_general.joblib (trained RF, 20 features)
tests/           — pytest, 119 tests
keys/            — RSA keys (NEVER commit)
```

## Key Files
- `.env` — API keys (NEVER commit, NEVER print contents)
- `src/data/polymarket_feature_adapter.py` — maps Gamma API → RF features
- `src/features_clean.py` — CLEAN_FEATURES definition (28 total, 20 used by model)
- `src/model_clean.py` — walk-forward RF/XGB/LR ensemble training
- `data/processed/wf_clean_general.parquet` — 499K rows, WF predictions

## Feature Pipeline
```
features_generic.py → features_clean.py → model_clean.py → backtest_clean.py
                                                          → paper_trader.py (live)
```

## Model Details
- RF trained on 20 features (8 constant features excluded by nunique>1 filter)
- Training data: Kalshi bulk historical (2021-2026)
- Entry: price <= model_prob * 0.70 (30% edge requirement)
- Exit: price >= model_prob * 0.90 OR expiry <= 24h

## Polymarket Adapter (CRITICAL)
Gamma API has NO tags/category fields. The adapter:
- Infers category via regex on slug/question text
- Uses training-set medians for unmappable features (hist_win_rate_series=0.0149)
- Computes days_in_market from startDate, volume_rank cross-sectionally

## Circuit Breaker
- 15% peak equity drawdown → HALT
- 8% daily drawdown → reduce 50%
- Rolling Sharpe window = 100 trades
- Drawdown-based ONLY, no loss-count triggers

## NegRisk Set Arbitrage (2026-07-07 — see docs/handoff-2026-07-07.md)
- `src/alpha/negrisk_scanner.py` — margini sum-of-prices su eventi multi-outcome,
  gira sotto launchd `com.traptrade.negrisk-scanner` con `--paper`
- `src/alpha/negrisk_paper.py` — executor simulato, realizza a risoluzione
- Whitelist esaustività: `data/negrisk_whitelist.json` + protocollo in
  `docs/negrisk-whitelist-dossier.md`. Long gated, short libero.
- Evidenze: `data/negrisk_log.csv`, `data/negrisk_paper_trades.csv`
- Fee Polymarket: per categoria (solo geopolitics gratis) — usare
  `polymarket_fee_from_schedule()` quando c'è il market dict

## Session Protocol
1. Read this file + `docs/handoff-2026-07-07.md` + `git log --oneline -10`
2. Check: `ps aux | grep -E "paper_trader|polymarket_trader|negrisk"`
3. Run `pytest tests/ -v` after every code change
4. Commit with descriptive message after every completed task

## Rules
- NEVER print or log API keys, private keys, or .env contents
- NEVER commit keys/ or .env
- NEVER override LIVE_MODE without explicit user approval
- All fees: Kalshi parabolic formula or Polymarket conditional — no flat estimates
- Sharpe: check Hill alpha first — if alpha < 2, use gain-to-pain ratio
- Circuit breaker: ONLY drawdown, not loss count

## MCP Server
```bash
python src/mcp_server/traptrade_server.py                           # stdio (MCP client)
python src/mcp_server/traptrade_server.py --transport sse --port 8100  # HTTP/SSE
```
Tools: `get_market_probability`, `get_portfolio_status`, `get_market_scan`, `get_backtest_summary`

## Known Limitations
- hist_win_rate_series uses training median (0.0149) — no Polymarket series history
- Macro CSVs need daily refresh via fetch_all.py
- Model trained on Kalshi, applied to Polymarket — domain shift expected
- Entertainment Sharpe 10.73 likely non-tradable (low liquidity)
- Edge decay: -2.84 Sharpe/quarter — validate fast

## Do NOT
- Add features without running permutation importance test
- Change entry threshold without re-running backtest
- Deploy to live without 7 days paper trading data
- Use Sharpe as primary metric if Hill alpha < 2
