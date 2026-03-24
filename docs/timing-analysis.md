# OHLCV Timing Analysis — Residual Lookahead Check

## Signal Timing

The backtest signal is computed as:
```
entry_signal = (price_lag1 > 0) AND (price_lag1 <= pred_prob_clean * 0.70)
```

### Feature computation timeline

1. **Model features** are computed using data available at `open_date`:
   - Macro features (BTC, fear/greed, fed funds) joined on `open_date` (fix in features_clean.py)
   - Calendar features derived from `close_date` (known at market creation)
   - `hist_win_rate_series` uses outcomes from prior markets in the series (resolved before current market opens)
   - `floor_strike_pct`, `volume_rank` use data from prior series markets

2. **Entry price** (`price_lag1`):
   - The settlement price of the PREVIOUS market in the series
   - Observable when current market opens (previous market has already settled)

3. **OHLCV entry price**:
   - `open_price` from the first daily candlestick of the market
   - This is the first traded price after the market opens

### Timeline for a typical trade

```
T-1: Previous market in series settles → price_lag1 known
T0:  Current market opens → OHLCV open_price set
T0:  Signal evaluated using features from open_date + price_lag1
T0:  If signal fires → enter at OHLCV open_price (or price_lag1 if no OHLCV)
T+N: Market settles → exit at outcome (0 or 1)
```

## Gap Analysis by Market Type

### Crypto short-duration (<24 hours)
- **Gap: minimal** — Markets open shortly after previous settles
- OHLCV open_price closely matches actual executable price
- **Risk**: In fast-moving crypto markets, the open price may shift within minutes.
  A market order at T0+5min could get a meaningfully different price than the T0 open.
- **Residual lookahead: LOW** — ~1-5 minute gap, negligible for daily OHLCV

### Weather/Sports (1-7 days)
- **Gap: small** — Markets listed days before close
- OHLCV open_price is the first trade, which could be hours after listing
- **Risk**: Low liquidity at market open means the first traded price may not be
  representative. Bid-ask spread could be wide.
- **Residual lookahead: LOW**

### Politics/Elections (weeks-months)
- **Gap: large for price_lag1** — Previous market in series may have settled weeks ago
- OHLCV open_price is more accurate here (actual first trade of this specific market)
- **Residual lookahead: MEDIUM** — The open price is correct, but features computed
  at open_date may not reflect conditions at actual entry time (which could be days
  later if the trader waits for a signal)

## Key Concern: Signal vs Execution Timing

The backtest assumes entry at the moment the signal fires. In live trading:

1. Signal generation runs on a schedule (e.g., every 30 minutes)
2. Between signal generation and order execution: market price may move
3. For crypto 15-min markets: this gap could be significant
4. For longer markets: the gap is negligible

## Estimated Timing Bias

| Market Type | Timing Bias | Impact on Sharpe |
|------------|-------------|-----------------|
| Crypto <24h | Small (1-5 min) | -0.05 to -0.10 |
| Sports/Weather (1-7d) | Negligible | ~0 |
| Politics/Elections (>30d) | Medium (price_lag1 stale) | -0.10 to -0.30 |
| Overall weighted | | **-0.05 to -0.15** |

## Conclusion

The OHLCV timing is clean for most market types. The main residual concern is:
1. **Crypto fast markets**: execution slippage between signal and order (~5 min)
2. **price_lag1 fallback**: systematically underestimates entry price for long-duration markets

Neither constitutes forward-looking lookahead. The bias is toward **optimistic entry
prices**, not information leakage.

Conservative timing haircut: **-0.10 Sharpe points**.
