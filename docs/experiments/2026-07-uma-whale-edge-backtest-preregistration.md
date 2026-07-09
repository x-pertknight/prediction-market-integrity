# Preregistration — Whale-conflict trading edge (2026-07-09)

**Written BEFORE running the backtest. Committed before the first result was
computed.** Nothing below was edited after seeing an outcome; results go in a
separate "Risultati" section appended afterwards.

## Ipotesi

Sappiamo (misurato, non ipotizzato) che sui 272 voti direzionali collegati a una
posizione, il 63.2% vota il proprio book (null 50%, binomiale p = 1.5e-05).

L'ipotesi **tradabile** è più forte e più fragile: *se* le whale posizionate
votano il proprio libro, e *se* il loro peso sposta l'esito del DVM, allora la
direzione netta delle posizioni whale al momento della disputa contiene
informazione sull'esito **non ancora prezzata** dal mercato Polymarket.

Va detto subito perché potrebbe non funzionare: (a) la posizione è endogena — la
whale la detiene perché crede già in quell'esito, quindi non è informazione
*aggiuntiva*; (b) lo stake delle whale posizionate è minuscolo rispetto allo
stake totale del DVM, quindi il voto del proprio book può non spostare l'esito;
(c) il prezzo alla disputa incorpora già la proposta contestata.

## Regola di trading — parametri FISSATI A PRIORI

| Parametro | Valore | Perché fissato ora |
|---|---|---|
| Floor whale | voter con **≥3 voti collegati a posizione** su tutto il dataset | è l'unico insieme difendibile: con 1 voto collegato il conflict_rate è 1.00 o 0.00 per costruzione |
| Disputa eleggibile | ≥1 whale (come sopra) con esposizione **netta direzionale** ≠ 0 al blocco della disputa | net = yes_shares − no_shares; set completo = coperto, non conflitto |
| `tilt` | `Σ stake_i · sign(net_i) / Σ stake_i` ∈ [−1, 1] | stake = UMA staked del voter su quel voto |
| `base_yes` | **0.4606** = P(YES \| DVM ha deciso) = 146/317 | misurato, RESET esclusi |
| `K` | **0.25** | scelto ora, non ottimizzato |
| Forecast | `f = clip(base_yes + K · tilt, 0.02, 0.98)` | |
| Prezzo `p` | VWAP dei trade YES in `[dispute_time − 24h, dispute_time)` | serve ≥3 trade, altrimenti la disputa è scartata (no price) |
| Entry | `|f − p| ≥ 0.10` → compra YES se `f > p`, altrimenti compra NO | soglia fissata ora |
| Size | $100 notional piatto per disputa | niente Kelly, niente sizing ottimizzato |
| Exit | alla **risoluzione terminale della question** (YES=1, NO=0, VOID 50/50=0.5) | RESET non è un esito: si segue la question fino al round deciso |
| Question mai decisa | scartata | |
| Fee | `polymarket_taker_fee(p, category)` per share, **solo in entrata** | la redemption a risoluzione non paga taker fee. Categoria ignota → `other` (0.05), conservativo |
| Slippage | non modellato a $100 notional | dichiarato come limite, non nascosto |

Split temporale: ordina per `dispute_time`, taglia alla **mediana**.

## Gate di accettazione (tutti necessari)

1. **N ≥ 20 dispute tradabili**, altrimenti **NON TESTABILE** — si riporta solo N.
2. **Avg net edge > 0 su ENTRAMBE le metà temporali.**
3. **Concentration check:** se le prime 3 dispute per PnL fanno ≥50% del PnL netto,
   va detto esplicitamente e il risultato è declassato a "guidato dalla coda".
4. Hill alpha sui PnL: se **α < 2**, si riporta **gain-to-pain**, non lo Sharpe
   (regola di progetto).

## Predizioni (mie, scritte PRIMA di eseguire)

| # | Metrica | Predizione | Confidenza |
|---|---|---|---|
| 1 | N dispute tradabili | **18** (range plausibile 8–35) | mi aspetto di sbattere contro il floor: **P(NON TESTABILE) ≈ 0.5** |
| 2 | Sign accuracy (direzione forecast vs esito) | **0.58** (null 0.50) | bassa |
| 3 | Avg net edge per trade | **+0.02** per $1 di notional | **P(edge > 0) ≈ 0.45** — cioè non ci credo più che a una moneta |
| 4 | Gate "positivo su entrambe le metà" | **FALLISCE** | P(fallisce) ≈ 0.6 |
| 5 | Concentrazione: quota PnL netto delle prime 3 dispute | **≥ 60%** | alta — code grasse, N piccolo |
| 6 | Hill α sui PnL | **< 2** | alta — payoff binari, code grasse per costruzione |
| 7 | **Verdetto complessivo** | **NON TRADABILE / sottodimensionato** | ≈ 0.65 |

**Motivazione della predizione 3.** Il 63.2% di own-book vote è un fatto sul
*comportamento del votante*, non sul *mispricing del mercato*. Perché ci sia edge
servono tre anelli: whale posizionata → vota il suo book (misurato, vero) → il
suo voto sposta l'esito DVM (non misurato, dubbio: stake minuscolo sul totale) →
il mercato non lo prezzava (non misurato, dubbio). Il secondo e il terzo anello
sono dove mi aspetto che l'edge muoia.

**Motivazione della predizione 1.** 214 dispute hanno ≥1 votante posizionato, ma
il floor ≥3 voti collegati lascia 71 whale su 4371 votanti. Le dispute che ne
contengono almeno una, *e* hanno una question che arriva a risolversi, *e* hanno
≥3 trade nella finestra, saranno molte meno. Se N < 20 il verdetto è NON
TESTABILE e va riportato come tale, senza scorciatoie.

## Cosa NON farò

- Non cambierò `K`, la soglia di entry (0.10), o il floor (≥3) dopo aver visto i
  risultati. Se voglio provarne altri, vanno riportati **tutti**, etichettati come
  post-hoc, e il numero preregistrato resta quello sopra.
- Non riporterò lo Sharpe se α < 2.
- Non definirò "promettente" un risultato che passa i gate con N < 20.
- Non escluderò dispute dopo aver visto il loro PnL.
