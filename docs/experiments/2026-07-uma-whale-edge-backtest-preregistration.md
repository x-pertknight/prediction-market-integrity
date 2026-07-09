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

---

# Risultati (eseguito 2026-07-09, dopo il commit `fb66f4a`)

Campione: 76 dispute tradabili, 2025-04-20 → 2026-07-01, $7.600 di notional
totale ($100 × 76). Prezzo YES = VWAP dei trade nelle 24h precedenti la disputa
(mediana 245 trade in finestra), da `data-api.polymarket.com/trades`.

## Gate

| # | Gate | Esito |
|---|---|---|
| 1 | N ≥ 20 | **PASS** — N = 76 |
| 2 | Avg net edge > 0 su entrambe le metà | **FAIL** — 1ª metà +0.0047, 2ª metà **−0.6139** |
| 3 | Concentrazione top-3 | non calcolabile: PnL netto totale **negativo** |
| 4 | Hill α | **non stimabile** (troppo poche osservazioni di coda a q=0.95). Non potendo certificare α ≥ 2, lo Sharpe è **soppresso**. Gain-to-pain = **0.683** |

**Verdetto preregistrato: NOT TRADABLE.**

- Sign accuracy: **6/76 = 0.079** (null 0.50; binomiale p = 6.3e-15)
- Avg net edge: **−0.3046** per $1 di notional
- PnL netto: **−$2.314,91** (vincitori +$4.983 su 6 trade, perdenti −$7.298 su 70)
- Fee pagate: $317,04

## Scoring delle mie predizioni

| # | Predizione | Realtà | Esito |
|---|---|---|---|
| 1 | N = 18, P(NON TESTABILE) ≈ 0.5 | N = **76** | **SBAGLIATA** — ho sottostimato di 4x |
| 2 | Sign accuracy 0.58 | **0.079** | **SBAGLIATA**, e nella direzione opposta |
| 3 | Avg net edge +0.02, P(>0) ≈ 0.45 | **−0.3046** | direzione **giusta** (dubitavo), magnitudo no |
| 4 | Gate metà temporali FALLISCE | **FALLISCE** | **GIUSTA** |
| 5 | Top-3 ≥ 60% del PnL | PnL totale negativo | non applicabile |
| 6 | α < 2 | non stimabile | non applicabile |
| 7 | Verdetto NON TRADABILE | **NOT TRADABLE** | **GIUSTA** |

Ho azzeccato la conclusione e sbagliato quasi tutti i numeri. Vale la pena dirlo:
avevo predetto il fallimento per il motivo giusto (anelli 2 e 3 della catena),
ma non avevo previsto *quanto male* avrebbe perso, perché non avevo previsto la
distribuzione dei prezzi.

## Errore di specifica: la regola non ha testato l'ipotesi

Il forecast è `base_yes ± K·tilt` = `0.4606 ± 0.25`, quindi può dissentire dal
prezzo solo se il prezzo sta **dentro [0.21, 0.71]**. Ma i mercati in disputa
stanno agli estremi: p25 = 0.070, mediana = **0.866**, p75 = 0.967.

Conseguenza misurata: **il tilt delle whale ha cambiato il lato del trade in 1
trade su 76.** Negli altri 75 il lato era deciso interamente dal prezzo. La regola
che ho preregistrato era, di fatto, *"vendi ogni mercato in disputa verso 0.46"* —
non un test del segnale whale.

E il mercato ha ragione: `corr(prezzo, esito) = +0.849`; prezzo medio 0.861 quando
risolve YES, 0.133 quando risolve NO. Fadeare un mercato ben calibrato perde in
modo affidabile. **Il −30% per trade misura la mia specifica sbagliata, non
l'ipotesi.**

## Il test vero dell'ipotesi (post-hoc, indipendente dalla regola)

La domanda è: *il tilt delle whale prevede l'esito del DVM?*

| | n | P(YES resolution) |
|---|---:|---:|
| whale nette YES (tilt > 0) | 40 | 0.625 |
| whale nette NO (tilt < 0) | 36 | 0.639 |
| incondizionata | 76 | 0.632 |

- `corr(tilt, esito) = −0.020`
- Fisher exact sulla 2×2: **p = 1.00**

**Zero segnale.** Non "debole": zero, e con il segno leggermente sbagliato.

## Conclusione

La catena era:

1. whale detiene una posizione direzionale → **vero**
2. whale vota il proprio book → **vero e misurato** (63.2%, p = 1.5e-05)
3. quel voto sposta l'esito del DVM → **FALSO** (corr = −0.02, Fisher p = 1.00)
4. il mercato non l'aveva prezzato → **irrilevante**, l'anello 3 è già rotto

L'anello 2 è reale e resta un fatto interessante sulla *governance* di UMA: i
votanti posizionati votano il proprio libro. Ma non è un *edge*: lo stake delle
whale posizionate è irrilevante rispetto allo stake totale del DVM, quindi il loro
conflitto d'interesse è documentabile ma non predittivo.

Il conflict_rate resta un **risultato di integrità dell'oracolo**, non un segnale
di trading. Chi volesse tradarlo dovrebbe prima dimostrare l'anello 3 — e su
questo campione l'anello 3 è confutato, non "non testato".

### Limiti di questo risultato

- N = 76 sulle dispute con esito deciso e prezzo disponibile; il campione è
  vincolato dal floor ≥3 voti collegati (l'unico difendibile).
- Il prezzo è il VWAP 24h pre-disputa, non il book al blocco esatto: uno snapshot
  del mid al blocco potrebbe differire, ma non di 40 punti di sign accuracy.
- Slippage non modellato (a $100 di notional è trascurabile rispetto a −30%).
- L'anello 3 è confutato *su questo campione e con questa definizione di tilt*.
  Un tilt pesato per dimensione della posizione (invece che per solo segno) non è
  stato testato: la preregistrazione lo escludeva a priori e non lo aggiungo ora.
