# Regime Stationarity Analysis

## Per-Quarter Sharpe

| Quarter | N trades | Sharpe Net | P&L Net | Win Rate |
|---------|----------|-----------|---------|----------|
| 2025-Q2 | 11 | +7.80 | +2.2 | 36.4% |
| 2025-Q3 | 51,867 | +3.35 | +3,040.6 | 13.4% |
| 2025-Q4 | 35,600 | +4.24 | +3,374.0 | 21.5% |
| 2026-Q1 | 12,690 | +1.33 | +485.6 | 34.7% |

- Q2 2025 has only 11 trades — not statistically meaningful.
- Q3 and Q4 2025 are the bulk of the data.
- **Q1 2026 shows significant Sharpe decline: +1.33 vs +3.35/+4.24 in prior quarters.**

## Trend Analysis

Linear regression (Sharpe ~ quarter index, excluding Q2 with n=11):

| Metric | Value |
|--------|-------|
| Slope | **-1.01 Sharpe/quarter** |
| Direction | Declining |

**The edge is decaying at approximately 1 Sharpe point per quarter.**

## Interpretation

Several factors likely explain the Q1 2026 decline:

1. **Market efficiency increasing**: As prediction markets grow (Kalshi volume up ~3x
   in 2025), pricing becomes more efficient. The easy mispricings our model exploits
   are being arbitraged away by other participants.

2. **Model staleness**: The walk-forward trains on 6 months of history. As market
   dynamics shift, the model's learned patterns become less predictive.

3. **Regime change**: Q1 2026 may represent a different market regime (e.g., different
   political/economic environment, new market types, different participant mix).

4. **Sample size**: Q1 2026 has only 12,690 trades (vs 51,867 in Q3). The lower
   Sharpe may partly reflect higher variance from fewer observations.

## Extrapolation Warning

If the trend continues:
- Q2 2026 projected Sharpe: ~+0.3 (barely positive)
- Q3 2026 projected Sharpe: ~-0.7 (negative — strategy stops working)

**This is the most important finding in the entire audit.** A strategy with
declining Sharpe is a wasting asset. The model needs continuous retraining and
adaptation, or the edge will vanish entirely.

## Recommendations

1. **Monitor monthly Sharpe in paper trading**: If rolling 30-day Sharpe drops below
   +0.5, reassess the model
2. **Increase retraining frequency**: Consider 3-month train window instead of 6-month
   to adapt faster to regime changes
3. **Do NOT assume the backtest Sharpe will persist**: The Q1 2026 number (+1.33) is
   the most representative of future expected performance
4. **Conservative forward-looking Sharpe estimate: +1.0 to +1.5**
