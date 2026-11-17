# Adversarial pass (v1 -> v2)

## Pass 1

Reviewer: same author, separate pass on final/report.md v1. Warm context, so weaker than a cold review; stated here as a coverage limit.

Ordered by lethality.

## Fatal

F1. **"The docs do not mention Polymarket's settlement as a dependency" is false as written.** The Market Resolution page lists "External venue parity: used for markets mirrored from another venue". These markets are flagged as mirrored from Polymarket in metadata. The real gap is narrower: the docs place 5 and 15 minute crypto markets under Chainlink TWAP, while the metadata classifies them under Polymarket settlement. Cheapest repair: restate as a classification mismatch, not an omission.

F2. **"Resolution is clean" overstates what was tested.** The rule check uses open and close prices stored by Limitless, and the Polymarket payout is the copy stored in Limitless metadata. Both are internal consistency checks, not independent verification against Chainlink or Polymarket. Cheapest repair: say "internally consistent" in the summary and keep independent re-verification as the named production path.

## Repairable

R1. "Polymarket's settlement is a live dependency" is an inference from a metadata label. Whether resolution would proceed if Polymarket did not settle is not observable here. Repair: label as inference.

R2. "Taker delay not described in user docs" was not checked against every page (CLOB overview, market orders not read). Repair: "not found in the pages read".

R3. "Looks like lottery buying" is an interpretation of a cheap-price, clustered, one-account-heavy sample. Repair: mark as interpretation and keep the caveats next to the number.

R4. Concentration figures rely on the one profile the API attaches to each trade. Already in limitations; add a pointer in the summary line.

## Cosmetic

C1. Late-flow chart labels overlap slightly. Left as is.
C2. Research log: "largest trades all on the winning side" is true only for BTC 5m, where it was checked. Repair: scope it.

## Checks performed

- Every number in report.md and README.md diffed programmatically against final/metrics.json (check_numbers.py). Exhaustive for numbers that map to a metric; free-text numbers that come from the docs (e.g. 250 ms) checked by hand.
- Trade completeness: sum of matched shares equals reported market volume for all 6,719 markets.
- Smoke tests on synthetic markets pass (code/test_metrics.py).
- Not recomputed: oracle prices from Chainlink, Polymarket payouts from Polymarket, on-chain fills.

## Pass 2 (cold, hostile, independent recomputation)

Verdict: survives with repairs. All headline numbers reproduced with independent code. No addresses or tx hashes in files or git history.

- FATAL for outreach, repairable for the repo: "docs say Chainlink TWAP, data say Polymarket settlement" is likely a false contradiction. The source label sits inside the Chainlink TWAP metadata object, and the docs' external venue parity category covers mirrored markets. Repaired as TWAP price plus Polymarket parity.
- "Internally consistent" only shows the records agree with each other. Repaired wording and limitation.
- `polymarketOpenCheck` is Limitless's own flag, not a check by the author. Repaired.
- Safe transaction claim not supported by the published data. Removed.
- Late flow is absence of evidence. Restated as a low-power test.
- Near-tie wording overstated. Softened.
- Unverified claims about the official Dune dashboard. Removed from the public log.
- Number gate had no-op loops; anonymiser missing from the repo. Fixed.
- Added from the review: retry at about 180 s on 58 markets; open equals previous close on every adjacent pair.
