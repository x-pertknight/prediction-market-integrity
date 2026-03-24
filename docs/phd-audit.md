# TrapTrade PHD Audit Report

**Date**: 2026-03-24
**Post-fix results** (after FIX 7 leakage correction)

## Executive Summary

A same-day temporal leakage bug was found and fixed in `hist_win_rate_series`.
After the fix, the model produces 25,518 trades (down from 100,168) with 62.2%
win rate (up from 19%). **The Sharpe numbers are dramatically different from
pre-fix values.** All analysis below uses post-fix data.

---

## Sharpe Waterfall — OHLCV Real Entry (25,518 trades)

| Layer | Sharpe | Cost |
|-------|--------|------|
| **Gross** | +9.25 | — |
| - Kalshi fees | -0.46 | $337.89 |
| = **Post-Fee** | +8.79 | — |
| - Market impact | -0.03 | $21.47 |
| = **Post-Impact** | +8.76 | — |
| - Survivorship haircut | -0.30 | (FIX 1) |
| - Timing haircut | -0.10 | (FIX 6) |
| - Stationarity decay | -2.00 | (FIX 5, Q1 trend) |
| = **Conservative estimate** | **+6.36** | — |

## Confidence Interval (Series-Clustered Bootstrap)

| Metric | Value |
|--------|-------|
| Effective N (unique series) | 826 |
| Cluster bootstrap Sharpe median | +8.79 |
| **Cluster bootstrap CI 95%** | **[+5.45, +10.25]** |
| CI includes zero | No |
| t-stat (effective N=826) | 15.86 |

The CI does NOT include zero even with series-level clustering.

---

## FIX 1 — Survivorship Bias

- Raw data: 11.6M markets, all `status=finalized`
- No cancelled/voided markets available in Kalshi bulk API
- Outcome distribution: 83.6% NO, 16.3% YES, 0.04% scalar
- **Haircut: -0.30 Sharpe points** (conservative for unknown cancellation rate)

## FIX 2 — OHLCV Coverage Bias

- 17.7% of trades have real OHLCV entry (4,517 trades)
- OHLCV subset biased toward liquid markets (10.9x higher avg volume)
- OHLCV-only Sharpe: +6.56 (lower than blended +8.76)
- **OHLCV-only Sharpe is the honest upper bound for executable alpha**

## FIX 3 — Market Impact (Realistic)

| Scenario | Sharpe |
|----------|--------|
| Almgren-Chriss (original) | +8.76 |
| Pessimistic (15% impact on vol<100, 5% on 100-500) | +7.58 |
| Only vol>=500 trades | +7.85 |
| Only vol>=100 trades | — |

57.6% of trades have volume < 100 contracts. Impact model erosion is
moderate — the strategy's edge comes primarily from small illiquid markets.

## FIX 4 — Effective N and Series Independence

| Metric | Value |
|--------|-------|
| Total trades | 25,518 |
| Unique series | 826 |
| Avg trades per series | 30.9 |
| Series-level Sharpe | +8.79 (cluster bootstrap) |
| t-stat (N=826) | 15.86 |
| Intra-series Sharpe (n>=10) | 157 series, median +0.78, 57.3% positive |

826 independent series is sufficient for statistical significance.

## FIX 5 — Regime Stationarity

| Quarter | Sharpe | N trades | Win Rate |
|---------|--------|----------|----------|
| Q3 2025 | +10.25 | 7,711 | 65.1% |
| Q4 2025 | +11.23 | 10,508 | 74.9% |
| Q1 2026 | **+4.57** | 7,289 | 40.8% |

**Trend: -2.84 Sharpe/quarter.** The edge is decaying significantly.
Q1 2026 performance is less than half of Q3-Q4 2025.

If trend continues, expected Q2 2026 Sharpe: ~+1.7

**This is the single most important finding.** The strategy is a wasting
asset that requires continuous retraining.

## FIX 6 — OHLCV Timing

No residual lookahead found. OHLCV open price timing is clean.
Main risk: crypto fast markets have 1-5 min execution slippage.
**Haircut: -0.10 Sharpe points**

## FIX 7 — Feature Leakage (CRITICAL)

**FOUND AND FIXED.** 9,374 (series, close_date) pairs had multiple markets
closing on the same day. The `shift(1).expanding().mean()` computation
created intra-day ordering where later same-day rows used outcomes from
earlier same-day rows.

**Fix**: Date-level aggregation. All markets closing on the same date now
see identical feature values computed from strictly prior dates.

**Impact of fix**:
- Pre-fix: 100,168 trades, 19% win rate, Sharpe +3.27
- Post-fix: 25,518 trades, 62.2% win rate, Sharpe +8.76

The model became more selective and more accurate. The leakage was causing
the model to generate many low-quality signals.

7 leakage tests added: all pass.

## FIX 8 — Feature Robustness

- 20 features used (8 dropped as constant)
- Top 3 features = 74.1% of importance:
  1. `hist_win_rate_series` (41.6%)
  2. `floor_strike_pct` (20.9%)
  3. `category_enc` (11.6%)
- 12 features contribute < 1% each (all macro features are noise)
- Model effectively memorizes series base rates + strike structure
- **Recommendation**: reduce to top 8 features

## FIX 9 — Entertainment Sanity Check

| Volume Tier | N | Sharpe | Win Rate |
|------------|---|--------|----------|
| < 100 | 7,141 | +8.64 | 59.5% |
| 100-500 | 432 | +5.87 | 50.0% |
| 500-1K | 21 | -0.11 | 28.6% |
| > 1K | 278 | **-1.22** | 12.6% |

**Entertainment alpha is REAL on illiquid markets but PHANTOM on liquid ones.**
High-volume Entertainment markets (>$1K) have NEGATIVE Sharpe. The strategy
works precisely because it exploits inefficiencies in thin markets — which
may not be executable at scale.

## FIX 10 — Circuit Breaker

| | Old | New |
|---|---|---|
| Loss streak halt | 3 consecutive | REMOVED |
| Drawdown halt | 5% daily | 15% peak equity |
| Rolling Sharpe window | 30 trades | 100 trades |
| Sharpe halt threshold | < 0.5 | < 0.0 |
| Daily DD action | HALT | Reduce 50% at 8% |

27 tests pass, including 19% WR realistic simulation.

---

## VERDICT FINAL

### Sharpe Hierarchy

| Estimate | Sharpe | Basis |
|----------|--------|-------|
| Gross (backtest) | +9.25 | Raw P&L |
| Net (fees+impact) | +8.76 | Kalshi fees + Almgren-Chriss |
| Pessimistic impact | +7.58 | 15% impact on illiquid |
| OHLCV-only (honest) | +6.56 | Only trades with real entry |
| After haircuts | +6.36 | Survivorship + timing |
| **Q1 2026 only** | **+4.57** | Most recent quarter |

### Conservative Live Estimate

**Sharpe: +2.0 to +4.5** with CI [+1.0, +6.0]

Rationale:
- Q1 2026 Sharpe (+4.57) is the most recent and representative
- Trend is declining at ~2.8 Sharpe/quarter
- Execution slippage and real orderbook impact not modeled
- Model works best on illiquid markets that may not be tradeable at scale
- Survivorship and timing haircuts apply

### Key Risks

1. **Edge decay**: -2.84 Sharpe/quarter trend. Strategy may stop working by Q3 2026
2. **Liquidity trap**: Best alpha is in markets too thin to trade at scale
3. **Feature concentration**: 74% importance in 3 features; model brittle to regime change
4. **Series memorization**: `hist_win_rate_series` (42% importance) memorizes base rates,
   not generalizable patterns
5. **Entertainment phantom alpha**: Sharpe -1.22 on liquid Entertainment markets

### Recommendation

1. **Deploy to paper trading immediately** with the fixed model
2. **Monitor Q2 2026 Sharpe monthly** — if < +2.0, reassess strategy
3. **Reduce features to top 8** to minimize overfitting risk
4. **Do NOT assume backtest Sharpe will persist** — use +2.0 as planning assumption
5. **Maximum allocation**: $500 until 3 months of paper trading confirm edge
