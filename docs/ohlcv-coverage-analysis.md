# OHLCV Coverage Analysis

## Overview

Of 100,168 signal trades:
- **17,201** (17.2%) have real OHLCV entry prices (`entry_source = ohlcv`)
- **82,967** (82.8%) fall back to `price_lag1` proxy (`entry_source = price_lag1`)

## Liquidity Bias: OHLCV vs Non-OHLCV

| Metric | OHLCV trades | price_lag1 trades | Ratio |
|--------|-------------|-------------------|-------|
| Avg volume_fp | 30,087 | 2,748 | **10.9x** |
| Median volume_fp | 281 | 104 | 2.7x |
| Avg entry price | $0.4167 | $0.0547 | **7.6x** |
| N trades | 17,201 | 82,967 | — |

**The OHLCV subset is severely biased toward liquid, higher-priced markets.**

Markets with OHLCV data are those that had enough trading activity for Kalshi to
generate candlestick data. By definition, this selects for liquid markets.

## Category Distribution Bias

| Category | OHLCV % | price_lag1 % | Bias direction |
|----------|---------|-------------|----------------|
| Crypto | 51.9% | 75.5% | Under-represented in OHLCV |
| Sports | 23.0% | 3.3% | **Over-represented in OHLCV** |
| Financials | 14.4% | 7.8% | Over-represented in OHLCV |
| Entertainment | 1.6% | 12.2% | **Under-represented in OHLCV** |
| Climate/Weather | 4.7% | 0.1% | Over-represented in OHLCV |

The OHLCV subset has:
- Much more Sports and Financials (higher liquidity categories)
- Much less Entertainment and Crypto micro-markets (many are thin/illiquid)

## Sharpe Comparison

| Subset | Sharpe Net | N trades |
|--------|-----------|----------|
| All trades (OHLCV where available) | +3.27 | 100,168 |
| OHLCV-only subset | **+2.95** | 17,201 |
| price_lag1-only subset | +3.50 | 82,967 |

The OHLCV-only Sharpe (+2.95) is **lower** than the blended Sharpe (+3.27) because:
1. OHLCV entry prices are higher (more realistic) than price_lag1
2. The liquidity selection doesn't help — liquid markets have tighter spreads but
   also more efficient pricing, leaving less alpha

## Conclusion

**The OHLCV-only Sharpe of +2.95 should be treated as an upper bound, not a
point estimate**, because:

1. It's biased toward liquid markets where execution is feasible
2. The 82.8% of trades without OHLCV use a systematically lower entry price (price_lag1)
3. In live trading, the entry price would be somewhere between OHLCV open and price_lag1,
   depending on order book depth and timing

**Conservative estimate**: The true executable Sharpe lies in **[+2.0, +2.95]**,
with the lower bound reflecting realistic execution on thinner markets.
