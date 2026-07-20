# Audit note: managed-oracle dispute path (2026-07-20)

Trigger: Clavicular ($16M, clavicular-pregnancy-in-2026) and US-Iran ceasefire extension
($200M+, us-x-iran-ceasefire-extended-by-april-22-2026) disputes are NOT in the 973
reconstructed disputes. Verified by conditionId derivation against all 973 questionIDs
and by ancillaryData keyword search: zero matches.

On-chain trace (tx 0xa1112a1aa9a5f8535f8ddeea4af8b2d1d57da1ee6c3c0d0dae800aff624e7555):
- Clavicular conditionId 0x6747bae2... prepared with oracle = 0x6507...f2a7 (the newer
  managed adapter), questionID 0x24b6bade...
- Its price request goes to 0x2c0367a9db231ddebd88a94b4f6461a6e47c58b1 (managed
  optimistic oracle, MOOv2 per UMIP-189).

Quantification (eth_getLogs, DisputePrice on 0x2c03..., blocks 76.0M-90.6M, zero failed chunks):
- 1,889 DisputePrice events since Sep 2025
- Requesters: 0x6507...f2a7 (main managed adapter) 1,517; 0x69c4...5f24 372 (unmapped, likely
  negRisk/sports variant of the managed adapter; identify)
- ~0.16% of the managed adapter's ~936k resolutions

Implications applied to the article:
- The managed book hosts new human-judged markets, not only automated price questions
- "DVM escalation near zero" for that book was a scope artifact and is retired
- 14% flip rate and 8-20 bps S are scoped to the classic adapters only

Open items:
1. Round-by-round reconstruction of the 1,889 managed-path disputes (votes, outcomes,
   overturn decomposition) analogous to the classic 973
2. CLOSED (review round 4): 0x69c47de9... is Polygonscan-labeled "Polymarket: Neg Risk UMA CTF Adapter V4", the negRisk sibling of the managed adapter
3. How managed-path disputes reach the DVM vote (both 2026 cases were decided by UMA
   token voters per WSJ) and whether receipts land on the same voting contract
