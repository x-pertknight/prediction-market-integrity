# Missing Values Policy
**Date:** 2026-03-23

## Guiding Principle
NEVER backfill. Forward-fill only. Backfilling introduces lookahead bias.

## Rules by Source

### FRED (DXY, Fed Funds, Treasury 10Y, CPI)
- Published M–F; weekends and US federal holidays have no update
- **Rule:** Forward-fill from last known value for gaps ≤3 calendar days
- **Rule:** If gap >3 days (e.g., market closure, data publication delay): leave as NaN
- **NaN treatment in RF:** RandomForest fills NaN with `fillna(0)` at training/inference time. This is acceptable for macro features (0 ≈ "no change") but flagged as a limitation.
- **Implementation:** `ffill(limit=3)` on daily-indexed DataFrame

### Coinalyze (OI, Funding Rate, CVD)
- Published continuously for crypto; gaps indicate exchange downtime
- **Rule:** Forward-fill for gaps ≤4 hours (intraday outages)
- **Rule:** Gap >4 hours: NaN
- **Special case:** Funding rate reset every 8h — do NOT ffill across the 8h boundary; use 0 as neutral fill
- **Implementation:** `ffill(limit=2)` on hourly-indexed data (2×1h = 2h max ffill)

### NewsAPI (news volume, sentiment)
- Daily aggregates; no data on days with no relevant news
- **Rule:** No news = 0 articles, sentiment = 0 (neutral). This is semantically correct (absence of news is information)
- **Rule:** NO forward-fill for news. Each day's count is independent
- **Implementation:** `fillna(0)` for news_count, `fillna(0.0)` for sentiment_score

### NOAA (temperature)
- Station-level observations; occasional missing days due to sensor failure
- **Rule:** Forward-fill ≤2 days (e.g., weekend gaps at manual stations)
- **Rule:** Gap >2 days: NaN (no imputation — signal is ambiguous)
- **Implementation:** Already handled in noaa_fetch.py via `ffill(limit=2)`

### Fear & Greed Index
- Published daily; weekends sometimes missing
- **Rule:** Forward-fill ≤2 days
- **Implementation:** `ffill(limit=2)` on daily index

### Binance (BTC/ETH prices)
- 24/7 data; no expected gaps. If gap detected:
- **Rule:** Forward-fill ≤1 day (likely API downtime)
- **Rule:** Gap >1 day: NaN — DO NOT interpolate crypto prices

## Implementation Pattern

```python
def apply_missing_value_policy(df: pd.DataFrame, source: str) -> pd.DataFrame:
    limits = {"fred": 3, "noaa": 2, "binance": 1, "fear_greed": 2, "coinalyze": 2}
    limit = limits.get(source, 3)
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].ffill(limit=limit)
        # Never bfill
    return df
```

## Audit Trail
All missing value imputation is logged to `logs/missing_values_{date}.log` with:
- source, column, date, original_value (NaN), fill_value, fill_method
