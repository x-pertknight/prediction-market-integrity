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
| Trades | 98825 | 98643 |  |
| Win Rate | 14.2% | 12.8% |  |
| Total P&L | +10383.12 | +9033.86 | +1349.26 |
| Sharpe | +4.98 | +4.52 | +0.46 |
| Avg Entry | 0.0368 | 0.0368 |  |
| Max DD | -5.91 | -39.79 |  |
| t-stat | 98.7 | 89.4 |  |

### Per category Breakdown (Clean Entry)

| Group | Trades | Win Rate | P&L | Sharpe | Avg Entry |
|---|---|---|---|---|---|
| Climate and Weather | 904 | 18% | +131.34 | +6.06 | 0.0328 |
| Companies | 25 | 12% | +2.02 | +3.90 | 0.0392 |
| Crypto | 76137 | 7% | +3432.30 | +2.89 | 0.0258 |
| Economics | 144 | 43% | +55.11 | +12.30 | 0.0478 |
| Education | 1 | 0% | -0.01 | +0.00 | 0.0100 |
| Elections | 65 | 18% | +9.83 | +6.10 | 0.0334 |
| Entertainment | 6497 | 46% | +2288.31 | +11.21 | 0.1126 |
| Exotics | 704 | 6% | +22.57 | +2.13 | 0.0290 |
| Financials | 8430 | 44% | +3363.86 | +13.12 | 0.0447 |
| Health | 8 | 25% | +1.87 | +7.94 | 0.0162 |
| Mentions | 405 | 29% | +91.37 | +7.97 | 0.0682 |
| Politics | 167 | 19% | +25.47 | +6.20 | 0.0391 |
| Science and Technology | 36 | 25% | +8.06 | +8.09 | 0.0261 |
| Social | 6 | 17% | +0.86 | +5.55 | 0.0233 |
| Sports | 5284 | 27% | +948.82 | +6.45 | 0.0884 |
| Transportation | 1 | 0% | -0.26 | +0.00 | 0.2600 |
| World | 11 | 18% | +1.60 | +5.49 | 0.0364 |

### Duration Split (price_lag1 reliability)

Short-duration categories: Crypto, Entertainment, Financials, Sports, Weather, Mentions, Economics (price_lag1 more reliable).

Long-duration categories: Politics, Elections, Companies, etc. (price_lag1 potentially stale by months).

| Split | Trades | Win Rate | P&L | Sharpe | Avg Entry |
|---|---|---|---|---|---|
| Short duration | 97802 | 14.2% | +10310.85 | +5.00 | 0.0368 |
| Long duration (weaker proxy) | 1023 | 10.2% | +72.27 | +3.73 | 0.0310 |

---
## Weather (Clean Model)

### Clean vs Original Backtest Comparison

| Metric | Clean (price_lag1) | Original (last_price — lookahead) | Δ |
|---|---|---|---|
| Trades | 300 | 282 |  |
| Win Rate | 28.7% | 8.2% |  |
| Total P&L | +76.01 | +13.88 | +62.13 |
| Sharpe | +9.16 | +2.98 | +6.18 |
| Avg Entry | 0.0333 | 0.0323 |  |
| Max DD | -0.41 | -1.52 |  |
| t-stat | 10.0 | 3.1 |  |

### Per category Breakdown (Clean Entry)

| Group | Trades | Win Rate | P&L | Sharpe | Avg Entry |
|---|---|---|---|---|---|
| Climate and Weather | 300 | 29% | +76.01 | +9.16 | 0.0333 |

---
## Sharpe Bootstrap CI (1000 resamples, 95%)

| Model | Sharpe | CI 95% lower | CI 95% upper |
|---|---|---|---|
| General (Clean Model) | +4.98 | +4.91 | +5.05 |
| Weather (Clean Model) | +9.16 | — | — |

---
## Robustness Tests

| Test | Trades | Win Rate | Sharpe | P&L |
|---|---|---|---|---|
| first_half | 49412 | 12.1% | +4.70 | +4646.13 |
| second_half | 49413 | 16.3% | +5.26 | +5736.99 |
| excl_best_cat(Financials) | 90395 | 11.4% | +4.07 | +7019.26 |
| threshold_minus20 | 87908 | 13.6% | +5.13 | +9506.54 |
| threshold_plus20 | 106782 | 14.4% | +4.81 | +10787.61 |
| random_50pct | 49412 | 14.0% | +4.93 | +5114.31 |

---
## How to Get True Intramarket Prices

Set `KALSHI_API_KEY` and run:
```bash
python3 src/kalshi_market_history.py <ticker>
```
This uses `GET /trade-api/v2/markets/{ticker}/history` (prod API, requires auth).
Once OHLCV data is available, re-run this backtest — entry_source will show 'ohlcv'.

---
## Caveats
- All P&L is hypothetical. Zero real trades.
- price_lag1 proxy is best available without production API auth.
- Model (pred_prob_clean) trained on features that exclude market_price.
- AUC reduction vs original model is expected (market_price was a strong feature).