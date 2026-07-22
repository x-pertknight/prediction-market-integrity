# Audit note: negRisk condition-to-question ratio (2026-07-20)

Finding from external review round 2: ConditionResolution events with oracle = NegRiskAdapter
(0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296) run ~12% of Jun-Jul 2026 CTF resolutions (~16k
implied), while QuestionResolved events on the classic adapters (v2+v3) total ~2,600 in the
same months. Gap ~6x.

Empirical trace (60 sampled negRisk ConditionResolution txs, seed 7):
- 60/60 contain NO QuestionResolved event in the same tx
- Bulk resolution confirmed: single txs resolve 1 to 15 conditions (mode 1, heavy tail 10-15)

Interpretation: the UMA question resolves at the market level through the v3 adapter; sibling
outcome conditions are then resolved in separate bulk transactions by the negRisk operator with
no per-condition UMA event. The UMA dispute right on negRisk markets is per-question, not
per-condition. Any coverage share computed on ConditionResolution counts overweights negRisk
relative to UMA-question counts by this ratio.

Open item: exact per-month condition/question ratio table.
