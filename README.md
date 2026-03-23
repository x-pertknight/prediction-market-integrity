# TrapTrade

Quantitative prediction market trading system for Kalshi and Polymarket. Uses Random Forest models with lookahead-free features, Taleb risk framework, and mechanical circuit breakers.

## Architecture

```
src/
├── features_clean.py        # Lookahead-free feature engineering (27 features)
├── model_clean.py           # RF model training (general + weather)
├── backtest_clean.py        # Walk-forward backtest with fees + impact
├── run_backtest_signal_only.py  # Cache-only backtest runner
│
├── risk/
│   ├── fees.py              # Kalshi parabolic + Polymarket conditional fees
│   ├── market_impact.py     # Almgren-Chriss impact model, liquidity tiers
│   ├── circuit_breaker.py   # 5 mechanical halt rules
│   ├── taleb.py             # Jensen convexity, Hill estimator, ergodicity
│   ├── kelly.py             # Kelly criterion sizing
│   └── position_manager.py  # Portfolio position tracking
│
├── execution/
│   ├── paper_trader.py      # Live paper trading via Polymarket Gamma API
│   ├── kalshi_executor.py   # Kalshi order execution (RSA-PSS auth)
│   └── polymarket_executor.py  # Polymarket CLOB execution
│
├── models/
│   ├── block_bootstrap.py   # Series-grouped bootstrap CI
│   ├── hmm_regime.py        # Hidden Markov Model regime detection
│   ├── calibration.py       # Probability calibration
│   └── cross_market.py      # Cross-market correlation
│
├── alerts/
│   └── telegram_bot.py      # Telegram commands + auto-alerts
│
├── data/                    # Data fetchers (FRED, NewsAPI, Coinalyze, etc.)
├── whale/                   # Whale tracker + smart money index
└── alpha/                   # Alpha strategies (arb, OBI, favorite-longshot)
```

## Performance (Walk-Forward Backtest)

| Metric | Value |
|--------|-------|
| Sharpe (gross) | 5.34 |
| Sharpe (net, after fees + impact) | 5.05 |
| Total P&L (gross) | $11,882 |
| Total fees (Kalshi parabolic) | $340 |
| Total market impact | $16 |
| Trades | 100,168 |
| Win rate | 19% |
| Edge multiplier | 0.70 (enter when price <= model_prob * 0.70) |

## Setup

```bash
# Clone and create venv
git clone <repo-url> && cd traptrade
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
#   KALSHI_API_KEY, KALSHI_PRIVATE_KEY_PATH
#   POLY_API_KEY, POLY_SECRET, POLY_PASSPHRASE, POLY_PRIVATE_KEY
#   TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
```

## Paper Trading

```bash
# Start paper trader (polls Polymarket Gamma API every 60s)
python src/execution/paper_trader.py

# Or with Docker
docker compose up paper-trader
```

The paper trader:
1. Polls active markets from Polymarket Gamma API
2. Scores each market with the RF model
3. Enters when `price <= model_prob * 0.70` (30% edge minimum)
4. Exits when `price >= model_prob * 0.90` or expiry <= 24h
5. Checks circuit breaker before every trade
6. Logs trades to `data/paper_trades.csv`
7. Sends Telegram alerts for entries, exits, whale clusters

## Live Trading

```bash
# Kalshi (requires RSA key pair)
LIVE_MODE=true python src/execution/kalshi_executor.py

# Polymarket (requires Polygon wallet)
LIVE_MODE=true python src/execution/polymarket_executor.py
```

**LIVE_MODE is False by default.** All orders are dry-run logged unless explicitly enabled.

## Running Backtest

```bash
# Full walk-forward backtest with Kalshi fees
python src/run_backtest_signal_only.py

# Run tests
pytest tests/ -v --cov=src
```

## Fee Models

- **Kalshi**: Parabolic taker fee = `0.07 * P * (1-P)`. Max 1.75c at P=0.50. Settlement is free.
- **Polymarket**: Most markets FREE. Crypto/NCAAB/sports: `P*(1-P)*0.02`.

## Risk Framework

- **Circuit Breaker**: 5 mechanical rules — rolling Sharpe(30) < 0.5, daily DD > 5%, 3 consecutive losses, weekly Sharpe < 1.0 (reduce 50%), monthly Sharpe < 0.5 (halt).
- **Market Impact**: Almgren-Chriss model. Liquidity tiers: ILLIQUID (<$100/day, skip), HIGH ($100-500, quarter-Kelly), MODERATE ($500-5K, half-Kelly), LOW (>$5K, full).
- **Taleb Framework**: Jensen convexity scoring, Hill fat-tail detection, Peters-Gell-Mann ergodicity check, 1/sqrt(N) diversification.

## Disclaimer

This software is for educational and research purposes. Trading prediction markets involves significant risk of loss. Past backtest performance does not guarantee future results. The authors are not responsible for any financial losses. Always paper trade before going live.
