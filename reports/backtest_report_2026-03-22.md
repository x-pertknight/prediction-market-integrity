# TrapTrade Backtest Report
**Generated:** 2026-03-22  
**Mode:** Backtest only — no real trades, no wallet

## Data Sources
- **Weather actuals:** NOAA GHCND (6 cities, 2025-01-01 → 2026-03-17)
- **Market prices:** Kalshi daily high-temp markets (demo API, no auth)
- **Cities:** Atlanta, Chicago, Dallas, Miami, NYC, Seattle

## Model
- **Algorithm:** RandomForestClassifier (n_estimators=200, max_depth=10)
- **Validation:** Walk-forward (train=8mo, test=1mo)
- **Features:** 23 (bracket distance, rolling NOAA stats, calendar, market price, city encoding)
- **WF AUC:** 0.8682  |  **WF LogLoss:** 0.3384
- **Data limitation:** Only 1 WF fold due to 8-month history (NYC/Chicago/Miami from 2025-04)

## Entry / Exit Rules
- **Entry:** `market_price ≤ model_probability × 0.5`
- **Exit:** `market_price ≥ model_probability × 0.9` OR `days_to_expiry ≤ 1`
- **Note:** All Kalshi daily markets expire in 1 day → exit always at expiry

## Backtest Period
- **Test period:** 2026-03-01 → 2026-03-17 (15 days)
- **WF observations:** 342
- **Entry signals:** 141 (41.2% of observations)

## Overall Results

| Metric | Value |
|--------|-------|
| n_trades | 141 |
| win_rate | +0.0638 |
| total_pnl_dollars | +7.0700 |
| mean_pnl_dollars | +0.0501 |
| std_pnl_dollars | +0.2403 |
| sharpe_abs | +3.3126 |
| mean_roi | +3.7335 |
| sharpe_roi | +2.8751 |
| total_log_return | -1179.8943 |
| mean_log_return | -8.3680 |
| max_drawdown | -0.3200 |
| avg_entry_price | +0.0137 |
| mean_mae | +0.0115 |
| mean_mfe | +0.0616 |

## Per-City Breakdown

| City | Trades | Win Rate | Total P&L | Sharpe (abs) | Max DD |
|------|--------|----------|-----------|--------------|--------|
| Atlanta | 5 | 0% | -0.13 | -18.84 | -0.12 |
| Chicago | 35 | 3% | +0.52 | +1.39 | -0.27 |
| Dallas | 2 | 50% | +0.98 | +11.00 | +0.00 |
| Miami | 46 | 2% | +0.50 | +1.17 | -0.29 |
| NYC | 33 | 3% | +0.55 | +1.69 | -0.25 |
| Seattle | 20 | 25% | +4.65 | +8.48 | -0.06 |

## Interpretation

- Strategy enters on markets where Kalshi price ≤ 50% of model probability.
  This targets **mispriced cheap options** (average entry: 1.37%).
- Win rate of 6.4% is consistent with buying low-probability brackets.
  Expected value is positive when entry price < true probability.
- **Sharpe (abs):** +3.31 — high due to low variance of small-ticket trades.
- **Sharpe (ROI):** +2.88 — return on invested capital per trade.

## Caveats

1. Only 1 WF fold (17 days of test data). Results need more history to be conclusive.
2. `market_price` from Kalshi demo API may not reflect live bid/ask spread at time of entry.
3. Win rate of 6.4% is low but consistent with the low-price bracket strategy.
4. Atlanta/Dallas/Seattle have very few trades due to shorter market history.
5. No transaction costs modelled (Kalshi charges fees per trade).

## Files
- Equity curve: `equity_curve_2026-03-22.csv`
- WF predictions: `data/processed/wf_predictions.parquet`
- Trade log: `data/processed/backtest_trades.parquet`
- Model: `models/rf_traptrade_v1.joblib`