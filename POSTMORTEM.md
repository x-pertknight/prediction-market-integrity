# TrapTrade — Postmortem & Negative-Results Record

*Last updated: 2026-07-09*

This file is the honest record of what was tried, what died, and why. It is not
a README that sells the project — it is the documentary proof that the system
was built to **say no**, and that it did so, repeatedly, without burning real
capital. Every strategy below ran in paper (`LIVE_MODE=False`); zero real
dollars were exposed at any point.

If you came here looking for a money printer: there isn't one. What there is is
a cost-aware research infrastructure and a method that caught eight
self-deceptions in two nights. That is the haul.

---

## The verdict in one line

Five hypotheses tested with preregistered gates. **Five dead or dying. Zero
tradable edge confirmed. Zero capital lost.**

| Strategy | Outcome | Why |
|---|---|---|
| UMA whale-conflict | **REFUTED** (clean kill) | Whales do vote their own book (true, p=1.5e-05) but their stake is negligible against the total DVM: they do not move the outcome. corr(tilt, outcome) = −0.02, Fisher p = 1.00. |
| Cross-platform arb | **DEAD** | 0 EXACT pairs out of 2500 candidates. No real rule equivalence across the scanned universe. |
| Void-arb (disputes) | **DEAD in one night** | The "67% voids" was a mislabel: the sentinel is a RESET (it goes back to a vote), not a refund. True voids: 5 out of 973. |
| RF trader (Polymarket) | **headed for the archive** | Longshot bias: edge ~+30% almost uniform on every entry, sub-5c markets. Same signature as the dojo-dante detectors. Formal verdict 2026-07-20. |
| negRisk set-arb | **verdict 2026-07-20** | 345 scans, 0 windows with ≥2c net margin on the verified whitelist. Post-2026-03-30 sports fees erode the margins. Trend: fail. |

---

## The two halves of the system

### Old TrapTrade (running in paper on the M4)

negRisk scanner under launchd (auto-restart), two nohup traders (RF Polymarket +
paper Kalshi), daily Telegram digest at 09:00. 328 tests.

- **negRisk:** 38,608 log rows over 345 scans (2026-07-07 → 07-08). Windows with
  a net margin ≥2c on the verified whitelist: **0**. The only positive windows
  sit on `catchall` (exhaustiveness inferred, not verified, 85 cases) or on
  non-exhaustive `GATED` events (traps by construction: the sum is below 1
  because an outcome is missing, not because there is edge). 3 sets open in
  paper.
- **RF trader:** 80 trades, 76% bought below 5c. Incident 2026-07-07: 34
  positions opened in 10 minutes, −$105.68, circuit breaker fired correctly. The
  digest shows edge ~+30% uniform = systematic bias, not alpha.
- **Paper Kalshi:** 1,633 trades, +$3.19 lifetime (noise).

Risk-management bugs found and fixed in this cycle:
- Daily halt that did not auto-reset across midnight with the process alive
  (fix: `halt_scope` daily/sticky, reset inside `can_trade()`).
- Per-process `cumulative_pnl` masquerading as lifetime (fix: seed from the CSV
  at startup).
- Non-atomic position-state writes; a malformed event that could crash-loop the
  scanner; per-share fee multiplied by the notional.

### Branch `feat/uma-dispute-crossarb` (research, M1)

Two additive modules, no wiring into the trading loops.

**UMA dispute module.** WSJ hypothesis: UMA whales holding Polymarket positions
on the disputed market vote their own book. Pipeline entirely on public RPC,
zero paid Dune.

- Base rate "the DVM confirms the proposal": **77.6% (246/317)**. Predicted
  85% / 80-85%. Close, both high.
- Dispute↔DVM vote join: **939/973**, verified (the stake-weighted vote
  reproduces the on-chain `settled_price` 939/939 — a misjoin cannot do that).
- Historical position reconstruction: Safe proxies derived offline via CREATE2,
  CTF balances read with `balanceOfBatch` pinned to the dispute block.
- negRisk questionId→conditionId mapping: **830/830 questions, 100% coverage.**
- **Directional own-book rate: 63.2% (172/272) against 50% random,
  p=1.5×10⁻⁵.** The phenomenon is real.
- **Edge backtest: NOT TRADABLE.** N=76, sign accuracy 6/76=0.079, net PnL
  −$2,314.91 (fees $317.04), temporal gate failed (+0.005 then −0.614).
  Rule-independent test: corr(tilt, outcome) = −0.020, Fisher p = 1.00.

**Cross-platform arb module (rules-first).** Kalshi PROD, two-stage matcher
(fuzzy → structured diff of the rules → EXACT/DIVERGENT/UNKNOWN), real fees.
First real scan: **0 EXACT, 1775 DIVERGENT, 725 UNKNOWN.** Zero signals. The old
version (3c divergence on textual similarity) would have found dozens of fake
arbs; this one tells the truth.

---

## The whale chain, resolved

```
whale holds a directional position       → TRUE
  → votes their own book                  → TRUE (63.2%, p=1.5e-05)
    → that vote moves the DVM outcome      → FALSE (negligible stake vs total)
      → the market had not priced it        → irrelevant, link 3 is broken
```

The conflict of interest is **documentable but not predictive**. It is an
oracle-integrity result, not a strategy. The preregistration had named link 3 as
the likely killer: so it was.

Methodological note: the preregistered metric (share of whales with
conflict_rate > 0.8) came out at **0.176**, almost identical to the predicted
18%. **It is an artifact**, not a confirmation: a whale with a single linked
vote scores 1.00 or 0.00 by construction. Raising the floor to ≥3 linked votes
collapses it to 4.2%. The prediction was "confirmed" by n=1 noise. Retracted.

---

## The eight self-deceptions avoided (the real value)

1. **Sentinel mislabel** — the "67% voids" was a RESET, not a void. Caught by
   reading the `UmaCtfAdapter` source, not by trusting the number.
2. **Silent null RPCs** — some public RPCs answer `null` instead of an error on
   old txs: read naively it gave 941/973 false negatives in an apparently
   healthy dataset. `call_nonnull()` rotates on null.
3. **Dune cartesian product** — joining on the hash alone merged the reset
   rounds (577k rows). Clean key: `(stamped_hash, dvm_request_time)`.
4. **Sharpe on 6 winners** — the Hill estimator returns `inf` on few tail
   observations; read as α≥2 it would have printed a Sharpe on 6 winning trades
   out of 76. Withdrawn, fallback to gain-to-pain.
5. **The n=1 artifact** in conflict_rate >0.8 (see above).
6. **Net vs gross exposure** — an equal YES+NO is a complete set that pays $1
   either way (hedged inventory, not conflict). Counting it would have inflated
   the conflict_rate.
7. **Pruned node = fake archive** — a pruned node answers 0 to every historical
   query: same class as the nulls. Asserted a nonzero control balance before
   trusting the archive.
8. **Self-diagnosed specification error** — the forecast could not disagree with
   extreme prices (flip in 1 trade out of 76); recognized, and followed by a
   rule-independent test that killed the hypothesis anyway.

---

## Prediction report card

| Metric | Prediction | Observed | Outcome |
|---|---|---|---|
| Base rate, proposal confirmed | 85% / 80-85% | 77.6% | close, both high |
| Whale conflict_rate >0.8 | 18% / <30% | 0.176 (n=1 artifact) | bogus "bullseye" |
| Backtest edge passes the gate | 25-30% | failed decisively | right direction, wrong mechanism |
| Sign accuracy | 0.58 | 0.079 (inverted) | wrong |

Recurring lesson: **getting the direction of a conclusion right is not
understanding the mechanism.** Both advisor and executor called the final
verdict correctly, but neither had modelled the price distribution of disputes
(concentrated at the extremes), which is what determined the outcome.

---

## The pattern, no discounts

Counting the previous project too (dojo-dante): simple momentum, daily TSM, and
now UMA whale-conflict — **three disciplined, preregistered hypotheses, three
dead**, plus cross-arb and void-arb. This is not bad luck. It is that the
classes of edge a single operator can generate on these markets lie in crowded
or efficient corners, where the inefficiency is already compressed or never
existed.

The process is first-rate — very few build a system that says no five times
without losing a dollar. But the process does not manufacture edge where there
is none. The open question is not "which variant to try next", it is whether the
next attempt should come from a structurally different advantage (informational,
or get-paid type such as liquidity/funding) rather than from another
micro-inefficiency.

---

## What remains, and it is real

- **Reusable infrastructure:** cost-aware backtester, free on-chain
  dispute↔vote join, historical position reconstruction via CREATE2 + Transfer
  replay, negRisk mapping, gate harness with bootstrap CI, scanner, digest,
  hardened risk manager, 328+ tests.
- **One real (non-tradable) result:** on Polymarket/UMA, positioned voters vote
  their own book (63.2%, p=1.5e-05), but their stake is negligible against the
  DVM — documentable conflict, oracle substantially sound on the outcome.
- **A demonstrated method:** preregistration, gates on both temporal halves,
  kill criteria, and the refusal to print numbers below the floor or a Sharpe on
  nonexistent tails.

Capital was never touched. The project closes these threads with the
infrastructure as the haul and zero real losses — which was the entire purpose
of the gates.
