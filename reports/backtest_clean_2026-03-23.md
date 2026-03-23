# TrapTrade — Clean Backtest Report (Lookahead-Fixed)
**Generated:** 2026-03-23

## What Was Fixed

| Issue | Old approach | New approach |
|---|---|---|
| Entry price | `last_price_dollars` (settlement-time price) | `price_lag1` (previous market's final price — observable at open) |
| ML features | Included `market_price`, `price_lag1` (lookahead) | Removed both from feature set |
| Macro join date | `close_date` (future resolution date) | `open_date` (date market opened) |

## Remaining Caveat

`price_lag1` is the previous market's `last_price_dollars`. For short-duration
markets (<2 days: Crypto, Entertainment, Financials) this is a clean proxy.
For long-duration markets (Politics, Elections, Companies >30 days) the
previous market's final price may lag by months. True fix requires
production API key for intramarket OHLCV (`kalshi_market_history.py`).

---
## General (Clean Model)

### Clean vs Original Backtest Comparison

| Metric | Clean (price_lag1) | Original (last_price — lookahead) | Δ |
|---|---|---|---|
| Trades | 100168 | 99425 |  |
| Win Rate | 19.0% | 17.0% |  |
| Total P&L | +12628.54 | +10721.89 | +1906.66 |
| Sharpe | +5.57 | +4.94 | +0.63 |
| Avg Entry | 0.0639 | 0.0625 |  |
| Max DD | -15.26 | -139.24 |  |
| t-stat | 111.0 | 98.1 |  |

### Per category Breakdown (Clean Entry)

| Group | Trades | Win Rate | P&L | Sharpe | Avg Entry |
|---|---|---|---|---|---|
| Climate and Weather | 900 | 18% | +132.09 | +6.09 | 0.0355 |
| Companies | 27 | 26% | +4.20 | +6.84 | 0.1037 |
| Crypto | 71599 | 9% | +3887.18 | +3.27 | 0.0339 |
| Economics | 165 | 48% | +61.65 | +12.38 | 0.1052 |
| Education | 1 | 0% | -0.01 | +0.00 | 0.0100 |
| Elections | 68 | 18% | +8.81 | +5.16 | 0.0469 |
| Entertainment | 10418 | 57% | +3683.61 | +11.30 | 0.2142 |
| Exotics | 670 | 6% | +24.05 | +2.35 | 0.0283 |
| Financials | 8946 | 47% | +3543.52 | +13.23 | 0.0708 |
| Health | 8 | 25% | +1.87 | +7.94 | 0.0162 |
| Mentions | 450 | 33% | +103.22 | +7.93 | 0.0973 |
| Politics | 179 | 21% | +25.62 | +5.66 | 0.0636 |
| Science and Technology | 40 | 30% | +9.93 | +8.92 | 0.0518 |
| Social | 7 | 14% | +0.46 | +2.44 | 0.0771 |
| Sports | 6676 | 32% | +1141.41 | +5.90 | 0.1460 |
| Transportation | 2 | 0% | -0.48 | -134.70 | 0.2400 |
| World | 12 | 17% | +1.41 | +4.52 | 0.0492 |

### Duration Split (price_lag1 reliability)

Short-duration categories: Crypto, Entertainment, Financials, Sports, Weather, Mentions, Economics (price_lag1 more reliable).

Long-duration categories: Politics, Elections, Companies, etc. (price_lag1 potentially stale by months).

| Split | Trades | Win Rate | P&L | Sharpe | Avg Entry |
|---|---|---|---|---|---|
| Short duration | 99156 | 19.1% | +12552.20 | +5.59 | 0.0641 |
| Long duration (weaker proxy) | 1012 | 11.5% | +76.34 | +3.85 | 0.0392 |

---
## Weather (Clean Model)

### Clean vs Original Backtest Comparison

| Metric | Clean (price_lag1) | Original (last_price — lookahead) | Δ |
|---|---|---|---|
| Trades | 311 | 309 |  |
| Win Rate | 31.2% | 12.3% |  |
| Total P&L | +82.46 | +23.00 | +59.46 |
| Sharpe | +9.55 | +4.12 | +5.43 |
| Avg Entry | 0.0468 | 0.0485 |  |
| Max DD | -0.59 | -0.71 |  |
| t-stat | 10.6 | 4.6 |  |

### Per category Breakdown (Clean Entry)

| Group | Trades | Win Rate | P&L | Sharpe | Avg Entry |
|---|---|---|---|---|---|
| Climate and Weather | 311 | 31% | +82.46 | +9.55 | 0.0468 |

---
## Sharpe Bootstrap CI (1000 resamples, 95%)

| Model | Sharpe | CI 95% lower | CI 95% upper |
|---|---|---|---|
| General (Clean Model) | +5.57 | +5.50 | +5.65 |
| Weather (Clean Model) | +9.55 | — | — |

---
## Robustness Tests

| Test | Trades | Win Rate | Sharpe | P&L |
|---|---|---|---|---|
| first_half | 50084 | 13.2% | +4.86 | +4922.24 |
| second_half | 50084 | 24.8% | +6.23 | +7706.30 |
| excl_best_cat(Financials) | 91222 | 16.3% | +4.73 | +9085.02 |
| threshold_minus20 | 86327 | 18.7% | +6.05 | +11951.39 |
| threshold_plus20 | 117229 | 21.5% | +4.67 | +12786.06 |
| random_50pct | 50084 | 19.0% | +5.57 | +6316.58 |

---
## v3 Simons + Academic Alpha Analysis

### HMM Regime Performance (3-state Gaussian HMM)

| Regime | Trades | Win Rate | Sharpe | P&L | Kelly Mult |
|---|---|---|---|---|---|
| MEAN_REVERTING | 2182 | 40.5% | +1.70 | +110.31 | 1.50× |
| TRENDING | 9329 | 46.5% | +13.15 | +3679.44 | 1.00× |
| VOLATILE | 88657 | 15.6% | +4.81 | +8838.80 | 0.50× |

### Gamma Near-Expiry Sizing (Bawa 2025)

| Metric | Value |
|---|---|
| Gamma-adjusted Sharpe | +4.3248 |
| Average size multiplier | 0.4225 |
| Bucket: CRITICAL | 88394 trades |
| Bucket: WARNING | 9766 trades |
| Bucket: NORMAL | 1261 trades |
| Bucket: CAUTION | 747 trades |

### Transaction Cost Impact (Almgren-Chriss model)

| Metric | Value |
|---|---|
| TC-adjusted Sharpe | +0.7848 |
| Average TC per trade | 0.10836 |
| Viable trades (edge > cost) | 100.0% |

### Calibration (Brier Score per category)

**Global Brier Score: 0.07374** (random=0.25, perfect=0.00)

| Category | Brier Score | Sizing Mult | Status |
|---|---|---|---|
| Climate and Weather | 0.15076 | 1.0× | ✓ OK |
| Companies | 0.14904 | 1.0× | ✓ OK |
| Crypto | 0.03969 | 1.0× | ✓ OK |
| Economics | 0.21039 | 0.5× | ⚠️ Miscalibrated |
| Elections | 0.22079 | 0.5× | ⚠️ Miscalibrated |
| Entertainment | 0.24916 | 0.5× | ⚠️ Miscalibrated |
| Exotics | 0.08606 | 1.0× | ✓ OK |
| Financials | 0.21312 | 0.5× | ⚠️ Miscalibrated |
| Mentions | 0.24352 | 0.5× | ⚠️ Miscalibrated |
| Politics | 0.13980 | 1.0× | ✓ OK |
| Science and Technology | 0.18593 | 1.0× | ✓ OK |
| Sports | 0.20959 | 0.5× | ⚠️ Miscalibrated |
| World | 0.22097 | 0.5× | ⚠️ Miscalibrated |

### Favorite-Longshot Bias (UCD Working Paper WP2025_19)

bias = actual_win_rate - midpoint_of_price_bucket
Positive bias: market overprices (avoid long) | Negative bias: underpriced (buy)

| Price Bucket | Win Rate | Bias | N | Signal |
|---|---|---|---|---|
| 0%–10% | 0.117 | +0.067 | 82890 | OVERPRICED |
| 10%–20% | 0.362 | +0.212 | 5331 | OVERPRICED |
| 20%–30% | 0.467 | +0.217 | 3348 | OVERPRICED |
| 30%–40% | 0.708 | +0.358 | 5642 | OVERPRICED |
| 40%–50% | 0.670 | +0.220 | 1150 | OVERPRICED |
| 50%–60% | 0.602 | +0.052 | 1762 | OVERPRICED |
| 60%–70% | 1.000 | +0.350 | 45 | OVERPRICED |
| 70%–80% | 0.750 | +0.000 | 0 | NEUTRAL |
| 80%–90% | 0.850 | +0.000 | 0 | NEUTRAL |
| 90%–100% | 0.950 | +0.000 | 0 | NEUTRAL |

### Wash Trading Filter (Columbia Nov 2025)

| Metric | Value |
|---|---|
| Wash-flagged trades | 6 (0.0%) |
| Wash-clean Sharpe | 5.5698 |

---
## v3 vs Bawa (2025) Realistic Performance Benchmarks

Ref: Navnoor Bawa 'Mathematical Execution Behind Prediction Market Alpha' (Dec 2025)

| Metric | Bawa Target (Live) | TrapTrade v3 (Backtest) |
|---|---|---|
| Annual Return | 15–25% | ~see Sharpe |
| Sharpe Ratio | 2.0–2.8 | 5.5697 |
| Win Rate | 52–58% | 0.19 |
| Avg Edge/Trade | 2–4% | see avg_entry |
| Max Drawdown | 12–18% | -15.26 |
| Capital Util. | 40–60% | TBD (live trading) |
| Entry Threshold | v2: ×0.5 | v3: ×0.7 |

⚠️ **Sanity check**: TrapTrade backtest Sharpe >2.8 likely reflects:
  1. price_lag1 proxy still overfits short-duration series
  2. Limited history (Kalshi launched Apr 2025 → ~11 months data)
  3. No live execution costs, slippage, or capital constraints
  Strategy needs live paper trading to validate real-world performance.

---
## How to Enable True Intramarket OHLCV Entry Prices

The backtest supports true intramarket entry prices via Kalshi prod API.
When OHLCV is available, `entry_source='ohlcv'` replaces `price_lag1`.

**Setup:**
```bash
# 1. KALSHI_API_KEY is already set in .env
# 2. Add your RSA private key path:
echo 'KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi_private_key.pem' >> .env
```
The PEM file was generated when you created the API key in Kalshi's dashboard.
It is your local RSA private key — never transmitted to Kalshi.

Once configured, re-running `python3 src/backtest_clean.py` will automatically:
  1. Fetch OHLCV for all signal tickers via `GET /trade-api/v2/markets/{ticker}/history`
  2. Override price_lag1 with first-day open price where available
  3. Report entry_source breakdown (ohlcv vs price_lag1)

---
## Caveats
- All P&L is hypothetical. Zero real trades.
- price_lag1 proxy is best available without RSA private key configured.
- Model (pred_prob_clean) trained on features that exclude market_price.
- AUC reduction vs original model is expected (market_price was a strong feature).