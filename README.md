# Prediction market integrity

Two studies of how Polymarket settles disputed markets through UMA, rebuilt from public Polygon and Ethereum RPC. The data and code are in this repo. One command, offline and with no keys, prints every number on this page. The few that need voter rows are labelled where they appear:

```
python3 -I scripts/reproduce.py
```

The command needs the packages in `requirements.txt`; on a stock python3 without pandas it stops with `ModuleNotFoundError`. Install them first:

```
python3 -m pip install -r requirements.txt
```

## 1. The Missing Greek: settlement risk on the classic UMA book

Scope: 973 Polymarket disputes that reached the UMA DVM on the classic adapters, disputed between 1 April 2025 and 1 July 2026.

* Conditional on a dispute, the final outcome differs from the original proposal 14.0% of the time (136 of 973): 67 direct DVM flips plus 69 flips after a too early reset.
* Retraction. An earlier draft of this study led with "the disputer wins 74.7% of the time" (727 of 973). That count treats 656 too early resets as losses. A reset is a delay: of the 648 reset questions that have resolved since, 577 ended on the outcome first proposed. The 74.7% headline is retracted and replaced by the 14.0% above.
* Cost to a holder of the proposed side, repricing component only (DVM dispute rate times 14.0%): 8.5 bps of notional in the current regime (DVM dispute rate 0.61%, 553 of 90,566 resolutions, September 2025 to June 2026) and 20.7 bps before the August 2025 move to whitelisted proposers (1.48%, 406 of 27,375, April to August 2025).
* Median time from dispute to DVM settlement: 3.25 days. The median too early question reached its final outcome 6.3 hours after the dispute.

Full write up, including the Kalshi comparison and the limits: [`articles/settlement-risk-article.md`](articles/settlement-risk-article.md).

## 2. Do DVM voters vote their own Polymarket book?

* 939 of the 973 disputes were linked to their DVM votes: 463,555 votes from 4,371 voters (the voter count is computed from the raw tables before aggregation, not recomputable from the shipped data). As a check on the join, the stake weighted final round reproduces the settled price on 939 of 939.
* Each voter's Polymarket position was rebuilt at the block of the dispute (proxy wallet derived offline, outcome token balances read at that block). 214 disputes had at least one voter holding a position.
* On directional votes, voters with a position voted their own book 63.2% of the time (172 of 272), against 50% by chance (binomial p = 1.5e-05).
* Their stake was too small to matter: at most 1.30% of the final round stake in any dispute, median 0% (computed from the raw tables before aggregation, not recomputable from the shipped data). The smallest winning margin in any final round is 51.5% of the stake, more than twice the published 2% bound on positioned stake, so no reassignment of that stake, including every positioned voter voting their own side, changes any of the 939 outcomes (`scripts/flip_test_uma.py`). The exact voter level replay, run on the raw tables, also gives 0 of 939.

How to read it: the conflict is real and measurable, and it moved no historical outcome. The flip test covers the voters observed. It does not cover holders who do not vote, positions held outside the voter's Polymarket proxy, or someone building stake and position together on purpose.

## Data

| File | What it holds |
|---|---|
| `data/uma_disputes.csv` | 973 disputes: adapter, dispute and settle date, latency in days, proposed and settled price |
| `data/dvm_round_tally.csv` | per dispute, round and price: number of votes and total stake |
| `data/dvm_disputes.csv` | per dispute: settled price and the bound on positioned stake (0, or 2% when any voter held a position) |
| `data/own_book_votes.csv` | one row per vote by a voter holding a position: vote side and position side only |
| `data/summary_raw.json` | the few figures that need voter level rows, computed from the raw tables before aggregation |
| `data/too_early_outcomes.csv` | final outcome of each too early question, by Polymarket questionID and dispute date. The `flipped` column sums to 70. The article counts 69 because it leaves out questionID `0x29949d1f5cfe4891acea05c9eff5f2ae2abeb89058d6e47306c3ace5ec9a0b47`: its proposal was a 0.5 tie, not a yes or no answer, so a final outcome of 0 is not a reversal of an answer |
| `data/dispute_rate_full.csv` | monthly resolutions and DVM disputes, January 2025 to July 2026 |

Wallets are reported in aggregate only. The published tables carry no addresses, no transaction hashes and no voter rows: votes appear only as totals per dispute, round and price, and positioned votes only as a vote side and a position side, under a random key that does not join to any other table. Disputes themselves are public chain events, so a dispute can still be found on chain from its date and prices; what the tables do not give is who voted how while holding which side.

## Rebuilding the data from chain

The fetch scripts write to `data/raw/`, which is not committed because the raw tables name wallets. The fetch scripts need the network and were not rerun live for this release. The offline reproduce line above is what was run.

1. `scripts/fetch_disputes.py`: the dispute list, from a saved Dune query (needs `DUNE_API_KEY` and `DUNE_DISPUTES_QUERY_ID`, the id of your own copy of the query; mine is not published).
2. `scripts/fetch_votes_web3.py`: links each dispute to its DVM votes, public RPC only.
3. `scripts/fetch_positions_web3.py`: voter positions at the dispute block, public RPC only.
4. `scripts/build_tables.py`: turns the raw tables into the published tables and `summary_raw.json`.

For the Missing Greek: `scripts/fetch_final_prices.py` (final outcome of too early questions), `scripts/scan_resolutions.py` and `scripts/scan_v4.py` (resolution counts), `scripts/oracle_coverage_sample.py` (which oracle resolves what). The last steps that turn the scans into `dispute_rate_full.csv` and the final prices into `too_early_outcomes.csv` are simple joins and are not scripted here; both tables are shipped as built.

## Limits

Rates are per question on the classic UMA adapters only. The newer managed adapter and its disputes are not reconstructed here, so none of these rates apply to it. The repricing figure assumes a flipped position is worth zero, an upper bound. The voter study measures positions in each voter's Polymarket proxy wallet only.

Code is MIT licensed. Text and data are CC BY 4.0.

Not investment advice.
