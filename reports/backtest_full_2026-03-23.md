# TrapTrade — Full Backtest Report
**Generated:** 2026-03-23
**Mode:** Backtest only — hypothetical P&L, zero real trades

## Summary

| Model | Trades | Win Rate | Total P&L | Sharpe | MaxDD | Avg MAE | Avg MFE |
|---|---|---|---|---|---|---|---|
| Weather v2 | 284 | 6.7% | +11.96 | +2.74 | -1.44 | 0.0210 | 0.0631 |
| Crypto v1 | 72047 | 13.8% | +7855.52 | +5.23 | -12.40 | 0.0190 | 0.1280 |
| General v1 | 26936 | 27.8% | +5363.92 | +7.35 | -49.87 | 0.0450 | 0.2441 |
| Ensemble | 3688 | 27.0% | +813.77 | +7.93 | -2.38 | 0.0333 | 0.2540 |

---
## Weather v2

### Overall Metrics

| Metric | Value |
|---|---|
| n_trades | 284 |
| win_rate | 0.0669 |
| total_pnl | 11.96 |
| mean_pnl | 0.0421 |
| sharpe_abs | 2.7414 |
| mean_roi | 3.3533 |
| sharpe_roi | 2.7141 |
| max_drawdown | -1.44 |
| avg_entry | 0.0248 |
| mean_mae | 0.021 |
| mean_mfe | 0.0631 |

### Breakdown by Category

| Group | Trades | Win Rate | P&L | Sharpe | MaxDD |
|---|---|---|---|---|---|
| Atlanta | 5 | 0% | -0.13 | -18.84 | -0.08 |
| Chicago | 76 | 7% | +3.17 | +2.63 | -0.42 |
| Dallas | 2 | 50% | +0.98 | +11.00 | +0.00 |
| Miami | 76 | 4% | +1.73 | +1.83 | -0.38 |
| NYC | 100 | 3% | +0.22 | +0.21 | -0.99 |
| Seattle | 25 | 28% | +5.99 | +8.87 | -0.99 |

---
## Crypto v1

### Overall Metrics

| Metric | Value |
|---|---|
| n_trades | 72047 |
| win_rate | 0.1377 |
| total_pnl | 7855.52 |
| mean_pnl | 0.109 |
| sharpe_abs | 5.2256 |
| mean_roi | 6.9243 |
| sharpe_roi | 4.3433 |
| max_drawdown | -12.4 |
| avg_entry | 0.0286 |
| mean_mae | 0.019 |
| mean_mfe | 0.128 |

### Breakdown by Category

| Group | Trades | Win Rate | P&L | Sharpe | MaxDD |
|---|---|---|---|---|---|
| Companies | 33 | 27% | +7.43 | +8.25 | -0.99 |
| Crypto | 62707 | 9% | +4024.37 | +3.69 | -156.18 |
| Financials | 9307 | 46% | +3823.72 | +13.56 | -405.39 |

---
## General v1

### Overall Metrics

| Metric | Value |
|---|---|
| n_trades | 26936 |
| win_rate | 0.2777 |
| total_pnl | 5363.917 |
| mean_pnl | 0.1991 |
| sharpe_abs | 7.3457 |
| mean_roi | 9.2383 |
| sharpe_roi | 5.5292 |
| max_drawdown | -49.87 |
| avg_entry | 0.0786 |
| mean_mae | 0.045 |
| mean_mfe | 0.2441 |

### Breakdown by Category

| Group | Trades | Win Rate | P&L | Sharpe | MaxDD |
|---|---|---|---|---|---|
| Climate and Weather | 1791 | 11% | +142.84 | +4.04 | -8.68 |
| Economics | 313 | 36% | +94.06 | +10.10 | -6.50 |
| Education | 2 | 0% | -0.12 | -13.47 | -0.01 |
| Elections | 174 | 17% | +24.65 | +5.91 | -8.72 |
| Entertainment | 12521 | 49% | +4659.00 | +12.04 | -345.90 |
| Exotics | 1684 | 3% | +16.21 | +0.85 | -28.89 |
| Health | 21 | 14% | +2.62 | +5.72 | -1.86 |
| Mentions | 890 | 12% | +63.99 | +3.43 | -12.30 |
| Politics | 369 | 14% | +38.68 | +4.96 | -8.82 |
| Science and Technology | 66 | 5% | +1.54 | +1.76 | -1.11 |
| Social | 3 | 0% | -0.08 | -14.66 | -0.07 |
| Sports | 9085 | 9% | +318.00 | +1.91 | -136.07 |
| World | 17 | 18% | +2.53 | +5.92 | -0.99 |

---
## Ensemble

### Overall Metrics

| Metric | Value |
|---|---|
| n_trades | 3688 |
| win_rate | 0.2698 |
| total_pnl | 813.768 |
| mean_pnl | 0.2207 |
| sharpe_abs | 7.9291 |
| mean_roi | 14.1934 |
| sharpe_roi | 6.8368 |
| max_drawdown | -2.38 |
| avg_entry | 0.0491 |
| mean_mae | 0.0333 |
| mean_mfe | 0.254 |

### Breakdown by Category

| Group | Trades | Win Rate | P&L | Sharpe | MaxDD |
|---|---|---|---|---|---|
| Climate and Weather | 581 | 31% | +160.45 | +9.55 | -7.19 |
| Companies | 20 | 45% | +7.82 | +12.61 | -0.20 |
| Crypto | 759 | 25% | +158.19 | +7.78 | -7.74 |
| Economics | 89 | 34% | +25.75 | +9.50 | -1.89 |
| Education | 1 | 0% | -0.01 | +0.00 | +0.00 |
| Elections | 75 | 36% | +25.25 | +10.85 | -9.86 |
| Entertainment | 342 | 35% | +102.37 | +10.12 | -8.67 |
| Exotics | 16 | 31% | +4.27 | +8.79 | -0.30 |
| Financials | 283 | 39% | +99.18 | +11.78 | -5.72 |
| Health | 11 | 9% | +0.84 | +4.00 | -0.10 |
| Mentions | 142 | 23% | +24.69 | +6.52 | -1.93 |
| Politics | 128 | 20% | +20.00 | +6.26 | -2.65 |
| Science and Technology | 27 | 11% | +2.49 | +4.59 | -0.43 |
| Social | 2 | 0% | -0.07 | -15.71 | -0.06 |
| Sports | 1097 | 23% | +168.77 | +5.81 | -19.64 |
| World | 8 | 12% | +0.56 | +2.94 | -0.18 |

---
## Caveats
- All P&L is hypothetical. No real money involved.
- Kalshi demo API prices may differ from live bid/ask at entry time.
- No fee model (Kalshi charges ~$0.07/contract round-trip).
- Walk-forward has limited folds due to data starting 2025-04.
- Generic model trained on all categories — predictive power varies by category.