# Preregistration — UMA dispute whale-conflict hypothesis (2026-07-08)

Ipotesi da testare: le whale UMA che hanno posizioni Polymarket sul mercato in
disputa votano il proprio book (fonte: inchiesta WSJ 05/2026, ~1 disputa su 5
ha un votante in conflitto).

Dataset: dispute Polymarket via UMA Optimistic Oracle V2, filtrate a
POST 2025-04-01 (post-MOOV2/UMIP-189 — il regime precedente non è comparabile,
non viene usato). Pipeline: `scripts/backfill_uma_disputes.py` (base rate, Dune)
→ `scripts/fetch_votes_web3.py` (join dispute↔voto, RPC pubblico)
→ `scripts/fetch_positions_web3.py` (posizioni storiche, RPC pubblico)
→ `src/alpha/uma_dispute.py`.

## Predizioni (scritte PRIMA di eseguire il backfill)

| Metrica | Mia stima | Advisor (a verbale) |
|---|---|---|
| Base rate: il DVM conferma la proposta iniziale | 85% | 80-85% |
| % whale (≥3 voti storici, link di posizione ad alta confidenza) con conflict_rate > 0.8 | 18% | <30% |

**Motivazione base rate:** la maggior parte delle dispute UMA OO storicamente
documentate sono tentativi frivoli o di grief contro un prezzo proposto
corretto; il DVM tende a confermare il proposer nella maggioranza dei casi.

**Motivazione conflict_rate:** soglia 0.8 è severa — richiede autoselezione
sistematica, non un singolo voto isolato. Col filtro ≥3 voti storici il pool
di whale utili è piccolo; mi aspetto un tasso sotto al 30% ma non
trascurabile, data l'evidenza WSJ di 1 disputa su 5 con conflitto rilevato.

## Soglia di testabilità

Se il dataset post-filtro contiene **meno di 20 dispute contese** (proposta
disputata E risolta dal DVM, con almeno un voto rivelato), il verdetto è
**NON TESTABILE**. Non va riportato come "promettente" in nessun caso — solo
il conteggio N e perché il campione è insufficiente.

---

## Correzione (2026-07-09): il sentinel NON è un annullamento

La versione precedente di questo documento riportava che il 67.4% delle dispute
"risolve al sentinel *ignore price* → il mercato viene **annullato/rimborsato**".
**Era sbagliato**, ed era l'errore più grave del documento. Verificato leggendo
il sorgente del `UmaCtfAdapter` deployato (Polygon
`0x2F5e3684cb1F318ec51b00Edba38d79Ac2c0aA9d`):

```solidity
// _resolve()
if (price == _ignorePrice()) return _reset(address(this), questionID, true, questionData);

// _constructPayouts()  — payouts sono [YES, NO]
if (price != 0 && price != 0.5 ether && price != 1 ether) revert InvalidOOPrice();
if (price == 0)          payouts = [0, 1];   // NO
else if (price == 0.5e18) payouts = [1, 1];  // 50/50  <- l'UNICO vero void
else                      payouts = [1, 0];  // YES
```

Il sentinel (`type(int256).min`) fa **ripartire** la price request: il mercato
non viene cancellato, nessuno viene rimborsato, la domanda torna all'oracolo.
L'unico annullamento reale è `0.5 ether` → payout `[1,1]`, ogni share paga $0.50.

### Breakdown corretto (973 dispute risolte)

| Esito DVM | N | % | Significato |
|---|---:|---:|---|
| RESET (sentinel) | 656 | 67.4% | Nessun verdetto, richiesta rifatta |
| NO (`0`) | 166 | 17.1% | payout `[0,1]` |
| YES (`1e18`) | 146 | 15.0% | payout `[1,0]` |
| VOID 50/50 (`0.5e18`) | 5 | 0.5% | payout `[1,1]` — **il solo vero void** |

- Dispute genuinamente decise (RESET escluso): **317**
- **Base rate osservata (DVM conferma la proposta): 77.6% (246/317)**

La correzione non muove il base rate — il denominatore era già giusto — ma
cambia la lettura. La vera scoperta non è "il 67% dei mercati viene annullato"
(falso), è: **una disputa compra ritardo, non un ribaltamento.** Nel 67.4% dei
casi il DVM non decide affatto e rimanda la domanda all'oracolo. Chi tradava
"disputa ⇒ mercato annullato" stava prezzando un evento che accade nello 0.5%
dei casi.

Codificato in `src/alpha/uma_dispute.py`: `classify_outcome()`, `expected_payouts()`
(che **solleva** su sentinel invece di restituire un 50/50 silenzioso),
`summarize_disputes()`. Il base rate resta ~5-8 punti sotto entrambe le
predizioni (85% mia, 80-85% advisor).

---

## Il join dispute ↔ voto DVM: risolto (2026-07-09)

La sessione precedente riportava **0/973 voti collegati**, e concludeva
correttamente "gap tecnico, non assenza di segnale". Il gap era doppio.

**1. L'ancillary data viene riscritta due volte.** Una disputa su Polygon
raggiunge il DVM su Ethereum dopo che:
  - `OptimisticOracleV2` appende `,ooRequester:<adapter>` (indirizzo in hex utf8
    minuscolo, senza `0x`);
  - `OracleChildTunnel` **comprime** il risultato, per cui il DVM (e ogni evento
    `VoteRevealed`) porta solo
    `ancillaryDataHash:<keccak dei byte stampati>,childBlockNumber:...,childChainId:137`.

  Cercare il testo grezzo trova 0. Cercare l'hash porta il match a **939/973**.

**2. Il timestamp giusto non è `request_timestamp`.** Le richieste Polymarket
sono *event-based*: alla disputa OOv2 escala al DVM usando il tempo della
**proposta**, non il timestamp della request. Verificato on-chain sulla disputa
`0xaba67ff5...`, dove `DisputePrice.timestamp = 1750183833` mentre il
`PriceRequestAdded` emesso da `OracleChildTunnel` **nella stessa transazione**
porta `time = 1750780863` — ed è quest'ultimo che ogni `VoteRevealed` ripete.
Scarto mediano misurato: **125 ore**, fino a 7925. Chiavizzare su
`request_timestamp` matcha zero.

**L'hash da solo non è una chiave.** Una question disputata → resettata →
ridisputata riusa ancillary data byte-identica: stesso `stamped_hash`, due
distinte price request DVM. Sui nostri dati: 973 dispute → 857 question
distinte, di cui 109 con più round. Un join sul solo hash le fonde (era il
difetto della query Dune `7920094`: 577.752 righe, prodotto cartesiano).

**Join pulito** = `(stamped_hash, dvm_request_time)`, dove entrambi i campi
sono **letti dalla transazione di disputa stessa** (evento `PriceRequestAdded`
di `OracleChildTunnel`), non inferiti. Risultato stage 1:

- **939/973** dispute hanno una price request DVM (34 non sono mai escalate)
- **939/939** hash derivati offline coincidono con quelli emessi on-chain
- i round di reset restano separati

Tutto via RPC pubblico, zero Dune, zero API key.

### Trappola incontrata (documentata perché ricorrente)

Diversi RPC pubblici Polygon rispondono a `eth_getTransactionReceipt` per
transazioni vecchie con `"result": null` **invece di un errore**. Trattare quel
null come "la disputa non è mai escalata" produceva 941/973 falsi negativi
silenziosi — un dataset sbagliato che sembrava sano. `JsonRpc.call_nonnull()`
ruota endpoint sul null e solleva se tutti lo restituiscono.

---

## Risultati

### Base rate — **TESTABILE**

- N dispute Polymarket (post 2025-04-01, requester `0x2f5e3684...aa9d` +
  `0x6a9d2226...4f74`): **973**, tutte risolte.
- Dispute decise (RESET escluso): **317**.
- **Base rate: 77.6% (246/317)** — vs 85% (mia) / 80-85% (advisor). Vicina ma
  sotto entrambe.
- Scoperta collaterale non prevista da nessuno: **67.4% reset**, non
  annullamenti (vedi correzione sopra).

### conflict_rate whale — **ANCORA NON CALCOLABILE** (motivo cambiato)

Il collo di bottiglia **non è più il join**, che ora funziona. È la
ricostruzione delle posizioni storiche:

- Il voter DVM ha un EOA su mainnet; la posizione Polymarket sta in un Safe
  proxy su Polygon. Il ponte è deterministico (CREATE2,
  `salt = keccak(abi.encode(owner))`), risolto via `computeProxyAddress(owner)`
  — verificato contro eventi `ProxyCreation` reali (2/2).
- `data-api.polymarket.com/positions` restituisce solo il **presente**: inutile
  per una disputa chiusa mesi fa. Non viene usata.
- Il saldo al blocco della disputa va **rigiocato** dagli eventi ERC-1155
  `TransferSingle`/`TransferBatch` del ConditionalTokens (un `balanceOf` a un
  blocco storico richiederebbe un nodo archive, che gli RPC pubblici non danno).
- **Limite reale e non aggirabile con questa derivazione:** il `positionId`
  richiede il `conditionId`. Per l'adapter V2 (`0x6A9D...`) l'adapter è esso
  stesso l'oracle CTF, quindi `conditionId = keccak(adapter, questionID, 2)`.
  Ma l'adapter V3 (`0x2F5e...`) è deployato contro il **NegRiskOperator**
  (verificato: constructor `_ctf = 0x71523d0f...` = NegRiskOperator), e le sue
  condizioni sono preparate dal NegRiskAdapter sotto un questionId diverso.
  **742/973 dispute vengono da V3.** Quelle vengono marcate
  `negrisk_unsupported` invece di essere collegate a un positionId sbagliato.

Quindi il conflict_rate è calcolabile oggi solo sul ramo V2 (231 dispute), e va
riportato come tale. **Non riportare "nessuna conferma dell'ipotesi WSJ"**: è
un "non ho ancora gli strumenti per misurarla su tutto il campione".

## Verdetto

- **Base rate: TESTABILE**, N=317, 77.6% — vicino alle predizioni ma ~5-8 punti
  sotto entrambe, e con la scoperta del 67.4% di reset che nessuno aveva
  previsto (e che il documento precedente etichettava erroneamente come
  annullamenti).
- **Ipotesi whale-conflict: PARZIALMENTE ABILITATA.** Il join dispute↔voto è
  risolto (939/973, esatto anche sui round di reset). Resta da coprire il ramo
  negRisk per il questionId → conditionId, senza il quale il conflict_rate è
  misurabile solo su 231/973 dispute. Prossimo passo: mappare
  `NegRiskOperator`/`NegRiskAdapter` questionId → conditionId.
