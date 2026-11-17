# Brief

## Question
Do Limitless Exchange's recurring crypto up/down markets on Base resolve correctly and consistently, and is there on-chain evidence of informed or late flow around resolution?

## Audience
Limitless team (BD, risk, surveillance), then a public GitHub repo as portfolio material. No external assignment exists; no brief to map against.

## Scope
- Recurring BTC and ETH up/down markets, 5 and 15 minute cadence, settled only.
- Data: Limitless public REST API (market detail + finalized mined CLOB trades). No authentication.
- Documentation read first: Market Resolution, Smart Contracts, Security, Fees, Maker Rebates, LP Rewards, User-Generated and Custom Markets, Refund Policy, Terms of Service, Analytics, Venue System.

## Acceptance (written before window 2)
1. No number in the report contradicts the run outputs (programmatic diff of README and report numbers against `final/metrics.json`).
2. Every claim about resolution or late flow is backed by two windows: window 1 (BTC, already run, exploratory) and window 2 (ETH, pre-registered, out of sample).
3. Denomination stated: trade value is approximated as matched shares times trade price, in USDC. Share counts and USDC are never mixed.
4. Limitations section with a production path per gap.
5. Findings reported in aggregate only. No addresses, no usernames.
6. Window 1 was run before predictions existed. This is stated, and only window 2 counts as a test.
