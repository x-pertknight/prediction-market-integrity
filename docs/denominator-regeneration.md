# Regenerating the resolution denominators

data/dispute_rate_full.csv is the monthly aggregation behind the classic-book rates
(1.48% clean window, 0.61% post-migration, 0.81% full window) and the 912,987 total.

The underlying raw scans (130,561 events on the classic adapters, 935,944 on the managed
adapter) are too large for the repo. Regenerate them from public Polygon RPC with:
- scripts/scan_resolutions.py (classic adapters v2 0x6A9D2226..., v3 0x2F5e3684...)
- scripts/scan_v4.py (managed adapter 0x65070be9...)
- scripts/fetch_final_prices.py (the too-early final-outcome join)

Each script is resumable and uses only free public RPCs. Full rescan takes a few hours.
Raw CSVs available on request if you want to diff instead of rescan.
