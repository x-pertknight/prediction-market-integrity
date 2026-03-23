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
| P&L Gross | +11443.27 | +10721.89 |  |
| Total Fees | 344.00 | 325.44 |  |
| **P&L Net** | **+11083.51** | **+10396.45** | +687.07 |
| Sharpe Gross | +5.20 | +4.94 |  |
| **Sharpe Net** | **+5.05** | **+4.80** | +0.25 |
| Avg Entry | 0.0757 | 0.0625 |  |
| Avg Fee/Trade | $0.0034 | $0.0033 |  |
| Fee % of P&L | 3.0% | 3.0% |  |
| Max DD | -26.84 | -175.80 |  |
| t-stat | 100.7 | 95.3 |  |

### Per category Breakdown (Clean Entry)

| Group | Trades | Win Rate | P&L | Sharpe | Avg Entry |
|---|---|---|---|---|---|
| Climate and Weather | 900 | 18% | +105.92 | +4.86 | 0.0609 |
| Companies | 27 | 26% | +3.99 | +6.61 | 0.1063 |
| Crypto | 71599 | 9% | +3273.41 | +2.86 | 0.0403 |
| Economics | 165 | 48% | +50.52 | +10.67 | 0.1664 |
| Education | 1 | 0% | -0.01 | +0.00 | 0.0100 |
| Elections | 68 | 18% | +8.95 | +5.30 | 0.0422 |
| Entertainment | 10418 | 57% | +3563.73 | +10.92 | 0.2159 |
| Exotics | 670 | 6% | +22.81 | +2.23 | 0.0283 |
| Financials | 8946 | 47% | +2853.70 | +11.05 | 0.1438 |
| Health | 8 | 25% | +1.86 | +7.90 | 0.0162 |
| Mentions | 450 | 33% | +93.91 | +7.28 | 0.1121 |
| Politics | 179 | 21% | +22.81 | +4.95 | 0.0751 |
| Science and Technology | 40 | 30% | +8.41 | +7.86 | 0.0858 |
| Social | 7 | 14% | -0.16 | -1.25 | 0.1571 |
| Sports | 6676 | 32% | +1074.27 | +5.55 | 0.1479 |
| Transportation | 2 | 0% | -0.51 | -136.36 | 0.2400 |
| World | 12 | 17% | -0.09 | -0.53 | 0.1675 |

### Duration Split (price_lag1 reliability)

Short-duration categories: Crypto, Entertainment, Financials, Sports, Weather, Mentions, Economics (price_lag1 more reliable).

Long-duration categories: Politics, Elections, Companies, etc. (price_lag1 potentially stale by months).

| Split | Trades | Win Rate | P&L | Sharpe | Avg Entry |
|---|---|---|---|---|---|
| Short duration | 99156 | 19.1% | +11014.95 | +5.07 | 0.0760 |
| Long duration (weaker proxy) | 1012 | 11.5% | +68.57 | +3.50 | 0.0443 |

---
## Sharpe Bootstrap CI (1000 resamples, 95%)

| Model | Sharpe | CI 95% lower | CI 95% upper |
|---|---|---|---|
| General (Clean Model) | +5.05 | +4.98 | +5.13 |

---
## Robustness Tests

| Test | Trades | Win Rate | Sharpe | P&L |
|---|---|---|---|---|
| first_half | 50084 | 13.2% | +3.92 | +3620.59 |
| second_half | 50084 | 24.8% | +6.05 | +7462.92 |
| excl_best_cat(Financials) | 91222 | 16.3% | +4.37 | +8229.82 |
| threshold_minus20 | 86327 | 18.7% | +6.05 | +11951.39 |
| threshold_plus20 | 117229 | 21.5% | +4.67 | +12786.06 |
| random_50pct | 50084 | 19.0% | +5.05 | +5542.08 |

---
## v3 Simons + Academic Alpha Analysis

### HMM Regime Performance (3-state Gaussian HMM)

| Regime | Trades | Win Rate | Sharpe | P&L | Kelly Mult |
|---|---|---|---|---|---|
| MEAN_REVERTING | 2182 | 40.5% | +1.22 | +79.58 | 1.50× |
| TRENDING | 9329 | 46.5% | +11.01 | +2973.87 | 1.00× |
| VOLATILE | 88657 | 15.6% | +4.46 | +8030.07 | 0.50× |

### Gamma Near-Expiry Sizing (Bawa 2025)

| Metric | Value |
|---|---|
| Gamma-adjusted Sharpe | +3.8394 |
| Average size multiplier | 0.4225 |
| Bucket: CRITICAL | 88394 trades |
| Bucket: WARNING | 9766 trades |
| Bucket: NORMAL | 1261 trades |
| Bucket: CAUTION | 747 trades |

### Transaction Cost Impact (Almgren-Chriss model)

| Metric | Value |
|---|---|
| TC-adjusted Sharpe | -0.7005 |
| Average TC per trade | 0.12864 |
| Viable trades (edge > cost) | 98.3% |

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
| 0%–10% | 0.102 | +0.051 | 81008 | OVERPRICED |
| 10%–20% | 0.391 | +0.241 | 5771 | OVERPRICED |
| 20%–30% | 0.457 | +0.207 | 3572 | OVERPRICED |
| 30%–40% | 0.701 | +0.351 | 5649 | OVERPRICED |
| 40%–50% | 0.650 | +0.200 | 1169 | OVERPRICED |
| 50%–60% | 0.597 | +0.046 | 1792 | NEUTRAL |
| 60%–70% | 0.733 | +0.083 | 86 | OVERPRICED |
| 70%–80% | 0.790 | +0.040 | 124 | NEUTRAL |
| 80%–90% | 0.800 | -0.050 | 60 | NEUTRAL |
| 90%–100% | 0.978 | +0.028 | 937 | NEUTRAL |

### Wash Trading Filter (Columbia Nov 2025)

| Metric | Value |
|---|---|
| Wash-flagged trades | 6 (0.0%) |
| Wash-clean Sharpe | 5.0515 |

---
## v3 vs Bawa (2025) Realistic Performance Benchmarks

Ref: Navnoor Bawa 'Mathematical Execution Behind Prediction Market Alpha' (Dec 2025)

| Metric | Bawa Target (Live) | TrapTrade v3 (Backtest) |
|---|---|---|
| Annual Return | 15–25% | ~see Sharpe |
| Sharpe Ratio | 2.0–2.8 | 5.0514 |
| Win Rate | 52–58% | 0.19 |
| Avg Edge/Trade | 2–4% | see avg_entry |
| Max Drawdown | 12–18% | -26.8368 |
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