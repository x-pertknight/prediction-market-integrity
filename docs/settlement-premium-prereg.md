# Prereg — Settlement Risk Premium study (2026-07-17, prima di toccare i dati)
Predizioni (advisor priors, Marco invitato ad aggiungere le proprie prima dei risultati):
- Dispute rate sui mercati Polymarket risolti: 0.5-2% (point: ~1%)
- Latenza mediana dispute->settle: 2-5 giorni (point: ~3g)
- Premio implicito (freq x costo capitale bloccato + tail flip risk): 1-5 bps sul nozionale per mercato medio; 20-80 bps condizionato all'entrare in dispute
- Kill criterion: se il denominatore (totale risoluzioni) non e' ricostruibile con copertura >90%, il claim si ridimensiona a "conditional on dispute" e NIENTE numero headline di premio unconditional.

## Esito vs predizioni (2026-07-18 ~01:00, dati raccolti via RPC pubblico)
- Dispute rate DVM apr25-giu26: 0.813% (959/117,940) -> DENTRO la banda predetta 0.5-2%, vicino al point ~1%
- Latenza mediana dispute->settle: 3.25g -> DENTRO 2-5g, point ~3g
- Nuovo dato non predetto: conditional on dispute, la proposta originale viene RIBALTATA il 74.7% delle volte (727/973). Il rischio dominante e' il ribaltamento, non il ritardo.
- Reset (dispute 1° livello) 0.76% vs DVM 0.813%: quasi tutte le contestazioni arrivano al DVM. Da verificare la meccanica v2 vs v3.
## RED FLAG copertura da risolvere prima di pubblicare
Risoluzioni mensili crollano da 19.280 (feb26) a ~2.200 (apr-giu26) e 471 in mezzo luglio: sospetto NUOVO adapter (v4?) non incluso nello scan. Se confermato, il denominatore recente e' sottostimato e il dispute rate recente e' gonfiato. Azione: cercare adapter aggiuntivi (factory/deployer, docs Polymarket) e ri-scannare.

## Round 2 — correzione post-contrarian esterno (2026-07-20)
- Overturn rate 74.7% DECOMPOSTO: 656/727 sono "too early" (int256.min), non flip. Join too-early -> esito finale on-chain (648/656 risolti): 577 stesso esito, 69 flip post-reset, 1 half, 8 pending.
- P(flip vero | dispute) = 136/973 = 14.0% (67 DVM diretti + 69 post-reset). Il contrarian stimava ~7.3% contando solo i diretti: il join era necessario, sottostimava di 2x.
- S repricing corretto: 8.5 bps (regime corrente 0.61%) - 21 bps (regime clean 1.48%). Il draft diceva 105-110 bps; la prereg diceva 1-5 bps. Il vero sta in mezzo, piu' vicino alla prereg.
- Copertura ri-auditata (campione 135,126 ConditionResolution giu-lug 2026): UMA 70.2%, Chainlink resolver 21.0%, EOA diretto 6.8%. Il claim "venue" era rotto; rescoped a "UMA books". NB il contrarian contava il NegRiskAdapter (12%) come non-UMA: errore suo, e' routing UMA v3.
- Classic rate post-migrazione: 0.61% (553/90,566), non "~1%". Range onesto 0.6-1.5% con regime dichiarato (meccanismo: UMIP-189 proposer whitelist, ago 2025).
- Kalshi: Cardi B $4.4M (il $47M era il complesso halftime), settle via 6.3(c). Il "within 24 hours" VERIFICATO testuale in Rule 7.1(b) del rulebook CFTC v1.12.
