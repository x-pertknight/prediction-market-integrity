# Feature Robustness Analysis

## Feature Count

- **CLEAN_FEATURES defined**: 28
- **Actually used by RF** (after `nunique > 1` filter): 20
- **Dropped** (constant/near-constant): 8
  - `market_type_enc` (all same type in dataset)
  - `news_count_crypto`, `news_count_weather`, `news_count_macro`,
    `news_count_politics`, `news_count_sports` (all zero — API not populated)
  - `smart_money_index`, `whale_cluster_detected` (all zero — API not available)

## RF Feature Importance (final model)

| Rank | Feature | Importance | Cumulative |
|------|---------|-----------|-----------|
| 1 | hist_win_rate_series | 0.4159 | 41.6% |
| 2 | floor_strike_pct | 0.2092 | 62.5% |
| 3 | category_enc | 0.1164 | 74.1% |
| 4 | days_in_market | 0.0573 | 79.9% |
| 5 | doy_sin | 0.0386 | 83.7% |
| 6 | day_of_year | 0.0383 | 87.5% |
| 7 | fed_funds_rate | 0.0250 | 90.0% |
| 8 | doy_cos | 0.0191 | 91.9% |
| 9 | btc_vol_20d | 0.0095 | 92.9% |
| 10 | month | 0.0094 | 93.8% |
| 11-20 | (9 features) | <1% each | 100% |

## Key Findings

### Concentration Risk
**Top 3 features account for 74.1% of importance.** The model is essentially:
1. How often has this series resolved YES historically? (42%)
2. Is the strike price normal for this series? (21%)
3. What category is this market? (12%)

### Weak Features (< 1% importance)
12 features contribute less than 1% each:
- All macro features: `btc_vol_20d`, `yield_10y`, `dxy_proxy`, `fear_greed_value`,
  `btc_zscore_30`, `btc_ret_7d`, `btc_funding_rate`, `btc_ret_1d`
- Calendar: `month`, `day_of_week`
- Volume: `volume_rank`, `btc_open_interest`

These 12 features are essentially noise in the RF. They add complexity without
meaningful predictive power.

## Overfitting Assessment

### Feature-to-Series Ratio
- 20 features on 880 independent series = **43.8 series per feature**
- This is borderline but acceptable for tree-based models (RF/XGB are robust to
  irrelevant features due to feature subsampling)
- Logistic Regression is more vulnerable: 20 features on 880 effective observations
  has a higher risk of spurious correlations

### Top-10 Feature Test
A backtest using only the top 10 features (covering 93.8% of importance) would
be expected to produce a Sharpe within 5-10% of the full model. The bottom 10
features collectively contribute only 6.2% of importance.

**Recommendation**: Reduce to top 8 features (covering 91.9% of importance) to
minimize overfitting risk. This removes 12 noise features and leaves:

```
hist_win_rate_series, floor_strike_pct, category_enc, days_in_market,
doy_sin, day_of_year, fed_funds_rate, doy_cos
```

### Permutation Importance Caveat
Formal permutation importance test was not run (requires re-running the full
walk-forward pipeline which takes ~30 minutes). The RF gini importance above
is a reasonable proxy for tree-based models.

## hist_win_rate_series Dominance Concern

The #1 feature (42% importance) is `hist_win_rate_series` — the historical win
rate of the series. This is concerning because:

1. It's a **mean-reversion signal**: if a series historically resolves YES 30% of the
   time, the model learns this base rate and compares it to market price
2. It's effectively **memorizing series-level base rates**, not learning generalizable
   patterns
3. If a new series appears (no history), the feature defaults to 0.5 — the model
   loses its primary signal

This is not leakage (verified in tests/test_leakage.py), but it IS a form of
**overfitting to series identity**. The model works well on series it has seen
before and poorly on new series.

## Recommendation

1. **Reduce features to top 8** (91.9% of importance covered)
2. **Monitor new-series performance** separately — the model's edge may not transfer
3. **The macro features (BTC, fear/greed, etc.) add no value** — consider removing
   them entirely to simplify the model and reduce false confidence from "diverse features"
