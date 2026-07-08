# Preregistration — UMA dispute whale-conflict hypothesis (2026-07-08)

Ipotesi da testare: le whale UMA che hanno posizioni Polymarket sul mercato in
disputa votano il proprio book (fonte: inchiesta WSJ 05/2026, ~1 disputa su 5
ha un votante in conflitto).

Dataset: dispute Polymarket via UMA Optimistic Oracle V2, filtrate a
POST 2025-04-01 (post-MOOV2/UMIP-189 — il regime precedente non è comparabile,
non viene usato). Pipeline: `scripts/backfill_uma_disputes.py` →
`src/alpha/uma_dispute.py`.

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

## Risultati (compilato dopo il backfill, eseguito 2026-07-08)

- N dispute Polymarket nel dataset (post 2025-04-01, requester noti
  `0x2f5e3684...aa9d` + `0x6a9d2226...4f74`): **973**, tutte risolte (973/973
  hanno un evento `settle` corrispondente).
- Scoperta non prevista: **656/973 (67.4%)** delle dispute risolvono al
  sentinel "ignore price" di UMA (`-2^255`, `int256.min`) — il mercato viene
  **annullato/rimborsato**, non deciso YES/NO. Non è un caso di "DVM
  respinge la proposta"; escluso dal calcolo del base rate.
- Dispute genuinamente decise (YES/NO/unknown, sentinel escluso): **317**.
- **Base rate osservata (DVM conferma la proposta iniziale): 77.6% (246/317)**
  — vs predizione 85% (mia) / 80-85% (advisor). Vicina ma sotto entrambe;
  il 67.4% di annullamenti è il vero elemento sorprendente, non catturato
  da nessuna delle due predizioni.
- N voti DVM collegati a una disputa specifica: **0/973**. Investigato
  live (2026-07-08): tre strategie di join testate su dispute reali e
  interamente risolte (match diretto identifier+timestamp, correlazione
  same-tx via bridge Polygon→mainnet, match per contenuto su tutto il
  dataset da 973 dispute) — tutte a zero risultati. Vedi
  `scripts/backfill_uma_disputes.py` (docstring) per il dettaglio tecnico.
  Confermato invece, separatamente: il ponte EOA-voter(mainnet)→proxy-wallet
  Polymarket(Polygon) esiste ed è risolvibile al 61.1% via
  `polymarket_polygon.safeproxyfactory_evt_proxycreation` — non è quello il
  collo di bottiglia.
- N wallet con ≥3 voti storici e posizione Polymarket collegabile: **0**
  (dipende dal punto precedente, non calcolabile).
- % di questi con conflict_rate > 0.8: **non calcolabile** — vs predizione
  18% (mia) / <30% (advisor).

## Verdetto

- **Base rate: TESTABILE**, N=317, risultato 77.6% — vicino alle
  predizioni ma non "confermato" nel senso stretto (~5-8 punti sotto
  entrambe le stime), e con la scoperta collaterale del 67.4% di
  annullamenti che nessuno aveva previsto.
- **Ipotesi whale-conflict (voter vota il proprio book): NON TESTABILE**
  con il metodo di linking attuale — 0 voti collegati a dispute
  specifiche, non per assenza di segnale ma per un gap tecnico
  identificato e documentato (vedi sopra), non un'assenza di dati grezzi.
  Non riportare questo come "nessuna conferma dell'ipotesi WSJ" — è un
  "non ho ancora gli strumenti per misurarlo", condizione diversa.
