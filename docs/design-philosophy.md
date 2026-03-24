# TrapTrade Design Philosophy

Simons + Taleb architectural constraints as code rules.

## SIMONS PRINCIPLES

### 1. Market Clustering

> "Il modello e' primitivo se tratta mercati diversi con le stesse features."

**RULE**: Markets must be clustered by resolution speed. A crypto 15-min market
and a "Will Beyonce win a Grammy" market cannot share the same model.

Clusters:
- **Fast** (<24h): Crypto short-term, some financials
- **Medium** (1-7 days): Weather, sports, entertainment
- **Slow** (>7 days): Politics, elections, economics

**STATUS**: VIOLATED. Current model uses one RF for all categories.
`category_enc` partially captures this (12% feature importance), but a
dedicated model per cluster would be more principled.

**FIX**: Future — implement `src/models/cluster_models.py` with separate
RF per cluster. Each cluster gets its own feature subset.

### 2. Transaction Cost Model Priority

> "Il transaction cost model e' il vero vantaggio competitivo, non il segnale."

**RULE**: Market impact model must be the most tested module. Every improvement
to cost model is worth 10x a signal improvement.

**STATUS**: PARTIALLY MET. Almgren-Chriss model implemented in
`src/risk/market_impact.py` with category volume proxies. 4 liquidity tiers.
But uses theoretical model, not real orderbook data.

**FIX**: Future — replace theoretical impact with empirical slippage from
paper trading execution logs.

### 3. Never Override the Computer

> "Never override the computer."

**RULE**: In LIVE_MODE, no human intervention can modify a trade in execution.
Telegram confirmation is ONLY for paper trading. In live, the bot executes
without asking. Human can only stop the bot, not modify trades.

**STATUS**: NOT YET APPLICABLE. Paper trading only.

## TALEB PRINCIPLES

### 4. Tail Exponent Check

> "Se alpha < 2, il Sharpe e' UNDEFINED."

**RULE**: For each category, compute Hill estimator. If alpha < 2: do NOT
report Sharpe. Use gain-to-pain ratio instead.

**STATUS**: NOT IMPLEMENTED. Current code reports Sharpe for all categories
without checking tail exponents.

**FIX**: Future — add Hill estimator to `src/risk/tail_analysis.py`. Flag
categories with alpha < 2 in reports.

### 5. Convexity as Architecture

> "La convessita' non e' un segnale — e' l'architettura del portfolio."

**RULE**: `convexity_bias` is the PRIMARY selection criterion, not an RF feature.
The bot ranks trade candidates by convexity FIRST, then applies the model to
filter among convex trades.

**STATUS**: PARTIALLY MET. The entry condition `price <= pred_prob * 0.70`
creates inherent convexity (buy cheap, payoff at $1), but convexity is not
explicitly computed as a separate criterion.

### 6. Complexity is Fragility

> "La complessita' e' fragilita'."

**RULE**: Every component must pass: "if this module breaks at 3 AM, does the
bot continue to function?" If no, it's a single point of failure and needs
a fallback.

**Component Failure Modes:**

| Module | If it fails | Fallback |
|--------|-----------|----------|
| RF model | No predictions | Use base rate (hist_win_rate_series) |
| XGBoost | Ensemble degrades | RF + LR still vote |
| Kalshi API | No market data | Retry with exponential backoff |
| OHLCV fetch | No entry prices | Fall back to price_lag1 |
| HMM regime | No regime signal | Use full Kelly (regime_mult=1.0) |
| Circuit breaker | No risk limits | HALT all trading (fail safe) |
| Telegram | No notifications | Log locally, continue trading |

### 7. Accept Long Loss Streaks

> "Con payoff convesso, devi essere disposto a perdere 50 volte di fila."

**RULE**: Circuit breaker NEVER uses loss count. With 62% win rate (post-fix),
loss streaks of 3-5 are still common. The circuit breaker uses ONLY:
- Drawdown in dollars (>15% of peak equity)
- Rolling Sharpe (100-trade window < 0)

**STATUS**: FIXED in FIX 10. Old circuit breaker halted at 3 consecutive
losses. New one uses drawdown-based rules only.

### 8. Gain-to-Pain over Sharpe

> "Guarda il gain-to-pain ratio, non il Sharpe."

**RULE**: Report shows GPR before Sharpe. GPR = sum(wins) / abs(sum(losses)).
GPR > 1.5 = good. GPR > 2.0 = excellent.

**STATUS**: NOT IMPLEMENTED in current reports. Sharpe is primary metric.

**FIX**: Add GPR to `backtest_ohlcv_definitive.py` report output.

## UNIFYING PRINCIPLE

### 9. Fallback Mode

> "Il bot deve funzionare con UN solo modulo: compra contratti cheap con payoff
> convesso e aspetta."

**RULE**: If ALL advanced modules fail, the bot switches to:
- Buy any contract with price < $0.10
- AND convexity_bias > 5x (payoff >= 10x if wins)
- AND volume > $500
- No model needed. Pure convexity.

This is the floor. The bot never drops below this level of sophistication.
Everything else is an upgrade above fallback.

**STATUS**: NOT IMPLEMENTED.

**FIX**: Future — implement `src/execution/fallback.py`.

## COMPLIANCE SUMMARY

| Rule | Status | Priority |
|------|--------|----------|
| 1. Market clustering | VIOLATED | HIGH |
| 2. Cost model priority | PARTIAL | MEDIUM |
| 3. Never override | N/A (paper) | — |
| 4. Tail check | NOT IMPL | HIGH |
| 5. Convexity arch | PARTIAL | MEDIUM |
| 6. Failure modes | DOCUMENTED | LOW |
| 7. Loss streaks | FIXED | DONE |
| 8. GPR over Sharpe | NOT IMPL | MEDIUM |
| 9. Fallback mode | NOT IMPL | HIGH |
