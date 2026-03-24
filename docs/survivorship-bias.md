# Survivorship Bias Analysis

## Dataset Composition

Raw Kalshi bulk data: **11,606,631** markets total.

| Result | Count | % |
|--------|-------|---|
| no | 9,707,462 | 83.6% |
| yes | 1,894,431 | 16.3% |
| scalar | 4,738 | 0.04% |

All markets have `status = finalized`. The Kalshi bulk API returns **only finalized
(settled) markets**. Cancelled, delisted, voided, or still-open markets are not
included in the bulk export.

## What We're Missing

The Kalshi API `GET /markets` with `status` filter supports:
- `active` — currently open
- `closed` — trading closed, awaiting settlement
- `settled` — resolved with result

There is **no way to retrieve cancelled/voided markets** from the public bulk export.
The API documentation does not expose a `cancelled` or `delisted` status.

**This means our dataset has survivorship bias**: we only see markets that reached
settlement. Markets that were cancelled mid-life (due to rule changes, ambiguous
resolution criteria, or low interest) are invisible.

## Impact Estimation

### Lower Bound (optimistic)
Prediction markets rarely cancel. Kalshi's regulatory status (CFTC-regulated) means
markets are well-defined before listing. Cancellation rate is likely **<1%** of total
markets. Impact on Sharpe: negligible.

### Upper Bound (pessimistic)
If 5% of markets were cancelled and those markets were systematically different
(e.g., low-liquidity exotic markets where our model performs poorly), the bias
could inflate Sharpe by **0.2-0.5 points**.

The `scalar` result type (4,738 markets, 0.04%) represents non-binary markets.
These are excluded by `clean_bulk()` which filters `result.isin(["yes", "no"])`.
This is correct — our model only handles binary markets.

### Base Rate Asymmetry
The base rate is **16.3% YES**, not 50/50. This extreme skew means:
- Most markets resolve NO
- A model that always predicts NO gets 83.6% accuracy
- The model's 19% win rate on SELECTED trades (where it bets YES) must be evaluated
  against this base rate, not against 50%

## Conservative Adjustment

**Survivorship haircut: -0.30 Sharpe points**

Rationale:
- We cannot verify what percentage of markets were cancelled
- The dataset may over-represent well-functioning, liquid markets
- Edge in cancelled/voided markets is unknowable
- -0.30 is conservative: assumes ~3% cancellation rate with moderately negative
  impact on our strategy

## Adjusted Sharpe

| Metric | Value |
|--------|-------|
| Sharpe Net (OHLCV) | +3.27 |
| Survivorship haircut | -0.30 |
| **Adjusted Sharpe** | **+2.97** |

## Recommendation

1. Query Kalshi API for `status != settled` to quantify cancellation rate
2. If cancellation rate > 3%: increase haircut to -0.50
3. In live trading: track cancelled markets separately as a data quality metric
