# Prediction market integrity research

Research infrastructure for **prediction-market integrity, oracle risk, and microstructure** — across Polymarket, Kalshi, and UMA. Everything runs in paper (`LIVE_MODE=False`); the value here is the method and the on-chain investigation, not a live edge.

> **Start with [`POSTMORTEM.md`](./POSTMORTEM.md)** — the honest record: five hypotheses tested under preregistered gates, five killed, zero capital lost, eight self-deceptions caught before they shipped.

## Showcase — UMA dispute / whale-conflict investigation

An oracle-integrity question on Polymarket's UMA resolution layer: *do whales who hold a position vote their own book?* Reconstructed end-to-end from **public RPC alone — zero paid data providers**.

- Linked **939/973** on-chain disputes to their DVM votes. The join validates independently: the stake-weighted vote reproduces the on-chain settled price **939/939** — a misjoin cannot do that.
- Reconstructed each voter's **historical** Polymarket position *at the dispute block*: Safe proxies derived offline via CREATE2, CTF balances replayed from ERC-1155 transfers, negRisk `questionId → conditionId` mapped to **100% coverage**.
- Finding: positioned voters **do** vote their own book — **63.2%** vs 50% random (p=1.5×10⁻⁵). But positioned voters held at most 1.30% of final-round DVM stake (median ≈0%). Counterfactual flip test: reassigning every positioned voter's stake to their own book across all 939 disputes changes zero outcomes (0/939) — structurally incapable of moving settlement. (The flip test reassigns observed voters' stake only; it does not model positioned holders who don't vote, or deliberate stake accumulation.) corr(tilt, outcome) = −0.02, Fisher p = 1.00. **Documentable conflict, not a predictive edge** — and a mildly reassuring result for oracle integrity.

Full method, every number, and the eight self-deceptions caught → [`POSTMORTEM.md`](./POSTMORTEM.md). Predictions written *before* each run → [`docs/experiments/`](./docs/experiments).

## The Missing Greek: settlement risk on Polymarket and UMA

How much does the settlement mechanism itself cost a holder, after the event is over? Reconstructed from public Polygon RPC on the classic resolution adapters.

- Conditional on a dispute, the final outcome differs from the original proposal 14.0 percent of the time (136 of 973).
- The repricing component of the settlement adjustment is about 8.5 bps of notional on the current regime (0.61 percent dispute rate) and about 21 bps on the pre-migration regime (1.48 percent). The delay component is about 0.07 bps.
- An earlier draft led with "the disputer wins 74.7 percent of the time" and put the cost at 105 to 110 bps by treating every DVM overturn as a terminal loss. Both were retracted. The preregistered guess was 1 to 5 bps.

Article: [`articles/settlement-risk-article.md`](./articles/settlement-risk-article.md). Preregistration: [`docs/settlement-premium-prereg.md`](./docs/settlement-premium-prereg.md).

## What's inside

- **UMA on-chain pipeline** — `src/alpha/uma_dispute.py`, `src/alpha/uma_whale_edge.py`, `scripts/fetch_votes_web3.py`, `scripts/fetch_positions_web3.py`, `src/data/` (public-RPC client, DVM logs, CTF positions, cross-chain bridge decoding).
- **Cross-platform rules matcher** — `src/alpha/cross_platform_arb.py`: two-stage Polymarket↔Kalshi matcher (fuzzy candidates → structured rule diff → `EXACT` / `DIVERGENT` / `UNKNOWN`), real fees. First live scan: 0 EXACT of 2,500 — it refuses to fabricate arbs the way naive text-similarity would.
- **negRisk scanner** — `src/alpha/negrisk_scanner.py`: read-only set-price exhaustiveness scanner (ref. arXiv 2508.03474).
- **Risk engine** — `src/risk/`: Kalshi parabolic + Polymarket schedule fee models, mechanical circuit breakers, position manager, Taleb/Hill fat-tail tooling.
- **Tests** — 439 passing (14 skipped).

## Method

Preregister the prediction before running. Gate on both temporal halves. Kill a strategy the moment it fails a gate — never re-tune to a pass. Pull on-chain data from public RPC and read it defensively (null-RPC rotation, archive-node control balances, keys verified against source contracts). **The negative results are the deliverable.**

## Disclaimer

Research and educational use. Paper-only by design; no tradable edge is claimed — the opposite is documented. Nothing here is financial advice.
