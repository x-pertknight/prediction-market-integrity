# NegRisk Whitelist Dossier — 2026-07-06

Exhaustiveness analysis for `data/negrisk_whitelist.json`. The long side of
the negRisk set arbitrage is only safe on outcome sets that are provably
exhaustive; every id below was verified against the event's full resolution
rules and live leg structure via the Gamma API on 2026-07-06.

**Mode: PAPER ONLY.** No real dollars. A whitelist error here costs simulated
PnL and teaches us something; the process below is the one we'd repeat with
real capital gates later.

## Whitelisted events (13)

### World Cup "Stage of Elimination" — 12 events (all surviving teams)

| Event id | Team | Live legs |
|---|---|---|
| 563639 | Argentina | 5 (R16→Champion) |
| 569753 | Belgium | 5 |
| 569758 | Colombia | 5 |
| 569764 | Egypt | 5 |
| 563653 | England | 4 (QF→Champion) |
| 563652 | France | 4 |
| 569774 | Morocco | 4 |
| 569784 | Norway | 4 |
| 563654 | Portugal | 5 |
| 563651 | Spain | 5 |
| 569794 | Switzerland | 5 |
| 569798 | USA | 5 |

**Why exhaustive:** a team still in the tournament must either be eliminated
at exactly one remaining stage or win it — the live legs partition all
outcomes. Earlier-stage legs (Group Stage, R32, R16 where passed) are already
closed and resolved NO; they are excluded by the scanner's `live_legs()`
filter and do not affect the sums.

**Resolution clauses verified on all 12** (identical template, fetched
per-event):
1. *"disqualified, withdraws, removed, or tournament partially completed"* →
   resolves to the furthest completed round — still exactly one live leg
   pays. Set stays exhaustive.
2. *"cancelled in full, postponed after August 2, 2026"* → resolves to
   **'Other'** — see residual risk below.

### 98349 — Which continent will win the World Cup?

Live legs: Africa (CAF), Europe (UEFA), North America (CONCACAF), South
America (CONMEBOL). Asia (AFC) and Oceania (OFC) legs are already closed
(all their teams eliminated).

**Why exhaustive:** all 12 surviving teams in the Winner event (Spain,
England, France, Argentina, Portugal, USA, Belgium, Colombia, Norway,
Morocco, Egypt, Switzerland) map to the four live continents per the
market's stated source (World Population Review). The condition is now
locked in: teams can only leave the tournament, not enter it.

## Residual risk accepted (documented, not eliminable)

Every event carries a hidden **'Other'** leg (`active=false`, empty book,
bid 0 / ask 1) that fires only if the tournament is cancelled in full or
postponed past 2026-08-02 — the final is scheduled 2026-07-19. The leg
cannot be bought (no asks below $1), so a long set cannot cover it. If it
fired, all long sets pay $0. Probability treated as negligible over the
13-day residual window; with real capital this would justify a position cap,
in paper it is a known, logged tail.

## Rejected / not whitelisted

- **Nobel Peace Prize Winner 2026** (+38c apparent margin): 20 listed
  candidates out of hundreds of possible laureates, no catch-all leg.
  Textbook phantom margin — the reason the gate exists.
- **Next UK Chancellor / PM 2026**: open candidate sets, no catch-all.
- Everything else defaults to non-exhaustive until reviewed here.

## Related observation (future work, not this slice)

Winner event vs Continent event are logically linked (sum of European teams'
winner prices must equal the Europe continent price). This is the
combinatorial-arbitrage relationship-graph idea (arXiv 2508.03474) — a
separate pre-registered hypothesis if the set-arb slice earns its keep.

## Review protocol for future additions

1. Fetch the event via `GET /events/{id}`; read the FULL description.
2. Confirm every non-listed outcome is impossible, or a catch-all leg exists.
3. Confirm cancellation/void clauses resolve to exactly one leg (or document
   the tail as above).
4. Add the id to `data/negrisk_whitelist.json` with a row in this dossier.
