# Kalshi vs Polymarket — Platform Mapping
**Date:** 2026-03-23

## Fee Structure

| Feature | Kalshi | Polymarket |
|---|---|---|
| Fee model | $0.07/contract round-trip (taker) | Maker: 0 bps, Taker: ~2% of notional |
| Contract unit | $1 notional YES/NO | $0.01–$1 USDC per share |
| Settlement | USD (regulated) | USDC on-chain (Polygon) |
| Auth required | Yes (prod), No (demo) | API key + wallet signing |
| Min trade | ~$0.01 | ~$0.01 |

**Impact on backtest:** Current backtest uses Kalshi demo data, no fee model. Add `fee_model` parameter to backtest_clean.py:
- `"kalshi"`: $0.07/contract round-trip (7% on $1 notional at avg entry 5¢)
- `"polymarket"`: taker 2% of notional per side → 4% round-trip at $1 notional

## Category Overlap Analysis

| Category | Kalshi | Polymarket | Overlap |
|---|---|---|---|
| US Politics | ✅ | ✅ | HIGH — both have elections, political events |
| Crypto prices | ✅ | ✅ | HIGH — BTC/ETH daily/weekly ranges |
| Sports (NFL/NBA/MLB) | ✅ | ✅ | MEDIUM — Kalshi daily, Polymarket event-based |
| Economics (CPI/NFP) | ✅ | ✅ | HIGH — macro data releases |
| Climate & Weather | ✅ | ❌ | KALSHI ONLY — no weather markets on Polymarket |
| Entertainment (Oscars, etc.) | ✅ | ✅ | MEDIUM |
| Company earnings/M&A | ✅ | Partial | LOW — Polymarket less active here |
| On-chain/DeFi | ❌ | ✅ | POLYMARKET ONLY |

## Model Applicability

The RF model was trained exclusively on Kalshi settled market data (2025-04 onward).

**Reliable categories for Polymarket deployment:**
- US Politics (high overlap, similar contract structure)
- Crypto prices (high overlap)
- Economics data releases (high overlap)
- Entertainment (medium overlap)

**NOT reliable for Polymarket:**
- Weather/Climate: no Polymarket equivalent → model will not generate signals
- On-chain/DeFi: no Kalshi training data

## Fee Model Implementation

```python
FEE_MODELS = {
    "none":       0.0,    # backtest with zero fees
    "kalshi":     0.07,   # $0.07 round-trip per $1 notional
    "polymarket": 0.04,   # 4% round-trip (2% maker/taker each side)
}
```

Add `fee_model: str = "kalshi"` parameter to `backtest_clean.py`.
When switching to live Polymarket: change to `fee_model="polymarket"`.
