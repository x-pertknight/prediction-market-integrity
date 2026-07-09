# negRisk questionId → conditionId — SOLVED (2026-07-09)

**Status:** implemented and verified on chain. Superseded the "plan" version of this
document. Live in `scripts/fetch_positions_web3.py` + `src/data/ctf_positions.py`.

## The problem

A CTF `positionId` requires a `conditionId`. For the **V2 adapter**
(`0x6A9D2226…`) the adapter is itself the CTF oracle, so
`conditionId = keccak(adapter ‖ questionID ‖ 2)` — verified, 157/157 conditions
located.

For the **V3 adapter** (`0x2F5e3684…`) that formula resolves to
`getOutcomeSlotCount == 0` under every plausible oracle (the adapter itself, the
NegRiskAdapter, the NegRiskOperator). 735 of 939 disputes are V3, so the
whale-conflict hypothesis was being measured on 22% of the sample — and on the
*binary* markets, not the grouped multi-outcome events where a conflicted whale
has the most to gain.

## The mapping (read from verified source, not guessed)

`UmaCtfAdapter` V3's constructor takes `_ctf = NegRiskOperator`. Its
`prepareCondition` is an **explicit no-op**:

```solidity
/// @notice Allows the Oracle to treat the Operator like the CTF, i.e., to call prepareCondition
function prepareCondition(address, bytes32, uint256) external {
    // no-op
}
```

The real condition is prepared by `NegRiskAdapter.prepareQuestion`, under a
different questionId. The two ids are linked by `NegRiskOperator`:

```solidity
mapping(bytes32 _requestId => bytes32) public questionIds;
function reportPayouts(bytes32 _requestId, uint256[] calldata _payouts) external onlyOracle {
    bytes32 questionId = questionIds[_requestId];
```

The oracle here is the UmaCtfAdapter, and `_resolve()` calls
`ctf.reportPayouts(questionID, payouts)` with **its own** `questionID`. So:

```
requestId  ==  UMA questionID  ==  keccak(ancillaryData)
```

and the full chain is:

```
ancillaryData ──keccak──► UMA questionID (= requestId)
        │
        │  NegRiskOperator.questionIds(requestId)      [public getter, eth_call]
        ▼
   negRisk questionId
        │
        │  conditionId = keccak(NegRiskAdapter ‖ negRiskQuestionId ‖ 2)
        ▼
   conditionId ──► NegRiskAdapter.getPositionId(questionId, outcome) ──► positionId
```

### Two traps that would have failed silently

1. **The collateral is not USDC.** `NegRiskAdapter.getPositionId` prices positions
   in the adapter's `WrappedCollateral` (`wcol()` = `0x3a3bd7bb…`), not USDC.
   Deriving a negRisk positionId against USDC returns a perfectly well-formed id
   that simply never has a holder — every voter reads as holding nothing.

2. **A wrong conditionId is indistinguishable from an unheld position.** Both give
   zeros. Every condition is therefore positively controlled with
   `getOutcomeSlotCount(conditionId) == 2` before use, and a failure is recorded as
   `condition_unresolved`, never as `no_voter_position`.

### NegRiskIdLib

`questionId` shares its first 31 bytes with its `marketId`; the last byte is the
question index within the multi-outcome event. Verified on a real pair: indices 6
and 3 of market `0x4789db09…a400`.

## Verification performed before applying

On three known negRisk disputes:

| check | result |
|---|---|
| `NegRiskOperator.questionIds(requestId)` nonzero | 3/3 |
| local `keccak(NegRiskAdapter ‖ qid ‖ 2)` == `NegRiskAdapter.getConditionId(qid)` | 3/3 |
| `getOutcomeSlotCount(conditionId) == 2` | 3/3 |
| positionIds derived, distinct YES/NO | 3/3 |

Then a 12-dispute smoke run (2 V2 + 10 negRisk) resolved 12/12 conditions and
returned real nonzero voter positions.

## Cost

All free public RPC. Per unique question: 1 `questionIds` + 1 `getOutcomeSlotCount`
+ 2 `getPositionId` ≈ 4 `eth_call`. Per dispute: 1 `balanceOfBatch` pinned to the
dispute block (halving retry on provider timeout). The full 939-dispute run
completes in minutes, not the ~2 hours the abandoned log-replay scan needed.

## Remaining caveats

- **Proxy coverage.** Only Polymarket's Safe proxy factory (`0xaacFeEa0…`) is
  derived. Users onboarded through the older `ProxyWalletFactory` (`0xaB45c5A4…`,
  magic/email login) hold positions at a different address, so their positions are
  missed. 62.6% of DVM voters have a deployed Safe proxy; the residual is a floor
  on measured conflict, not a ceiling.
- **Sub-block granularity.** Balances are read at the dispute block. A voter who
  opened and closed within that block reads as flat.
- **Trap inventory** (do not re-pay): silent `null` receipts on public Polygon RPCs
  (`JsonRpc.call_nonnull`); pruned state returning 0 for every historical
  `balanceOf` (`verify_archive_state`); drpc's 10k-block `eth_getLogs` cap reported
  as HTTP 400 with a JSON body; `ProxyCreation(proxy, owner)` argument order being
  the reverse of what the Dune column names imply.
