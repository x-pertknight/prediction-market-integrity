# Whale Tracker API Status
**Date:** 2026-03-23

## Polymarket Leaderboard API

### Status: PARTIALLY ACCESSIBLE (rate-limited, limited depth)

Attempted endpoints:
- `https://data-api.polymarket.com/positions?user={address}&sortBy=CASHPNL&limit=200` — requires individual wallet address
- `https://data-api.polymarket.com/leaderboard?limit=200` — endpoint not documented, returns 404

### Current Implementation
`src/whale/tracker.py` uses a list of known high-activity Polymarket wallet addresses (manually curated or from public leaderboard scraping). The full 200-wallet scan is not possible without authenticated access to a leaderboard endpoint.

### Fallback: Mock Data
`src/whale/tracker.py` generates synthetic whale signals when real data is unavailable:
- Randomly samples 0–5 "whales" per market
- SMI score = random ∈ [40, 80]
- Used for pipeline testing only. Flag: `using_mock_data = True`

### Next Steps to Enable Real Whale Data
1. **Manual wallet curation:** Identify top 50 addresses from Polymarket leaderboard UI, hardcode in `tracker.py`
2. **On-chain scan:** Use Polygon RPC to scan conditional token transfers > $1000 threshold
3. **Polymarket Data API (authenticated):** Apply for API access at polymarket.com/careers or via partnership

### Impact on Current Model
- SMI features in RF model: `smart_money_index`, `whale_count`, `whale_cluster_detected`
- With mock data: these features add noise, not signal. RF will learn to ignore them (low SHAP importance)
- With real data: expected SHAP importance boost of 5–15% for crypto/politics categories
