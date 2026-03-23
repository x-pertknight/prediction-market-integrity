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

| Metric | Clean (net) | Original (net) | Δ |
|---|---|---|---|
| Trades | 100168 | 99425 |  |
| Win Rate | 19.0% | 17.0% |  |
| P&L Gross | +11883.44 | +10721.89 |  |
| Total Fees | 339.79 | 325.44 |  |
| **P&L Net** | **+11543.65** | **+10396.45** | +1147.21 |
| Sharpe Gross | +5.34 | +4.94 |  |
| **Sharpe Net** | **+5.20** | **+4.80** | +0.40 |
| Avg Entry | 0.0713 | 0.0625 |  |
| Avg Fee/Trade | $0.0034 | $0.0033 |  |
| Fee % of P&L | 2.9% | 3.0% |  |
| Max DD | -26.81 | -175.80 |  |
| t-stat | 103.7 | 95.3 |  |

### Per category Breakdown (Clean Entry)

| Group | Trades | Win Rate | P&L | Sharpe | Avg Entry |
|---|---|---|---|---|---|
| Climate and Weather | 900 | 18% | +118.91 | +5.44 | 0.0473 |
| Companies | 27 | 26% | +4.00 | +6.62 | 0.1063 |
| Crypto | 71599 | 9% | +3465.76 | +2.98 | 0.0377 |
| Economics | 165 | 48% | +56.18 | +11.56 | 0.1333 |
| Education | 1 | 0% | -0.01 | +0.00 | 0.0100 |
| Elections | 68 | 18% | +8.96 | +5.31 | 0.0422 |
| Entertainment | 10418 | 57% | +3576.50 | +10.95 | 0.2147 |
| Exotics | 670 | 6% | +22.81 | +2.23 | 0.0283 |
| Financials | 8946 | 47% | +3073.74 | +11.77 | 0.1197 |
| Health | 8 | 25% | +1.86 | +7.90 | 0.0162 |
| Mentions | 450 | 33% | +97.58 | +7.54 | 0.1043 |
| Politics | 179 | 21% | +24.17 | +5.26 | 0.0679 |
| Science and Technology | 40 | 30% | +8.78 | +8.08 | 0.0765 |
| Social | 7 | 14% | -0.09 | -0.69 | 0.1486 |
| Sports | 6676 | 32% | +1084.03 | +5.60 | 0.1468 |
| Transportation | 2 | 0% | -0.51 | -136.87 | 0.2400 |
| World | 12 | 17% | +0.97 | +3.69 | 0.0817 |

### Duration Split (price_lag1 reliability)

Short-duration categories: Crypto, Entertainment, Financials, Sports, Weather, Mentions, Economics (price_lag1 more reliable).

Long-duration categories: Politics, Elections, Companies, etc. (price_lag1 potentially stale by months).

| Split | Trades | Win Rate | P&L | Sharpe | Avg Entry |
|---|---|---|---|---|---|
| Short duration | 99156 | 19.1% | +11472.20 | +5.22 | 0.0716 |
| Long duration (weaker proxy) | 1012 | 11.5% | +71.45 | +3.63 | 0.0416 |

---
## Sharpe Bootstrap CI (1000 resamples, 95%)

| Model | Sharpe | CI 95% lower | CI 95% upper |
|---|---|---|---|
| General (Clean Model) | +5.20 | +5.13 | +5.28 |

---
## Robustness Tests

| Test | Trades | Win Rate | Sharpe | P&L |
|---|---|---|---|---|
| first_half | 50084 | 13.2% | +4.26 | +4073.26 |
| second_half | 50084 | 24.8% | +6.05 | +7470.39 |
| excl_best_cat(Financials) | 91222 | 16.3% | +4.46 | +8469.91 |
| threshold_minus20 | 86327 | 18.7% | +6.05 | +11951.39 |
| threshold_plus20 | 117229 | 21.5% | +4.67 | +12786.06 |
| random_50pct | 50084 | 19.0% | +5.20 | +5765.20 |

---
## v3 Simons + Academic Alpha Analysis

### HMM Regime Performance (3-state Gaussian HMM)

| Regime | Trades | Win Rate | Sharpe | P&L | Kelly Mult |
|---|---|---|---|---|---|
| MEAN_REVERTING | 2182 | 40.5% | +1.24 | +80.55 | 1.50× |
| TRENDING | 9329 | 46.5% | +11.72 | +3201.23 | 1.00× |
| VOLATILE | 88657 | 15.6% | +4.56 | +8261.88 | 0.50× |

### Gamma Near-Expiry Sizing (Bawa 2025)

| Metric | Value |
|---|---|
| Gamma-adjusted Sharpe | +4.0034 |
| Average size multiplier | 0.4225 |
| Bucket: CRITICAL | 88394 trades |
| Bucket: WARNING | 9766 trades |
| Bucket: NORMAL | 1261 trades |
| Bucket: CAUTION | 747 trades |

### Transaction Cost Impact (Almgren-Chriss model)

| Metric | Value |
|---|---|
| TC-adjusted Sharpe | -0.2511 |
| Average TC per trade | 0.12143 |
| Viable trades (edge > cost) | 98.9% |

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
| 0%–10% | 0.107 | +0.057 | 81704 | OVERPRICED |
| 10%–20% | 0.380 | +0.230 | 5613 | OVERPRICED |
| 20%–30% | 0.465 | +0.215 | 3485 | OVERPRICED |
| 30%–40% | 0.704 | +0.354 | 5646 | OVERPRICED |
| 40%–50% | 0.662 | +0.212 | 1164 | OVERPRICED |
| 50%–60% | 0.600 | +0.050 | 1777 | NEUTRAL |
| 60%–70% | 0.768 | +0.118 | 69 | OVERPRICED |
| 70%–80% | 0.776 | +0.026 | 85 | NEUTRAL |
| 80%–90% | 0.762 | -0.088 | 21 | UNDERPRICED |
| 90%–100% | 0.975 | +0.025 | 604 | NEUTRAL |

### Wash Trading Filter (Columbia Nov 2025)

| Metric | Value |
|---|---|
| Wash-flagged trades | 6 (0.0%) |
| Wash-clean Sharpe | 5.2019 |

---
## v3 vs Bawa (2025) Realistic Performance Benchmarks

Ref: Navnoor Bawa 'Mathematical Execution Behind Prediction Market Alpha' (Dec 2025)

| Metric | Bawa Target (Live) | TrapTrade v3 (Backtest) |
|---|---|---|
| Annual Return | 15–25% | ~see Sharpe |
| Sharpe Ratio | 2.0–2.8 | 5.2019 |
| Win Rate | 52–58% | 0.19 |
| Avg Edge/Trade | 2–4% | see avg_entry |
| Max Drawdown | 12–18% | -26.8133 |
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