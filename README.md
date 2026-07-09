# Liénard Registry — Falsification-First Market Research
**The only signal platform that publishes its own failures, hash-chained and Bitcoin-anchored.**

## What this is
A pre-registered research program on cycle-positioning anchors (on-chain MVRV,
Liénard oscillator phase), with every claim tied to a published falsification test.
Per Harvey, Liu & Zhu (RFS 2016), 27–53% of published trading anomalies are likely
false. Our answer: frozen specs, negative results published, prospective registry.

## Track record (prospective, tamper-evident)
`registry/` — hash-chained entries (each embeds SHA-256 of predecessor), OpenTimestamps
automation anchors every entry in Bitcoin. Current (entry 003): ETH, LTC, ADA **LONG**;
BTC, XRP, DOGE **FLAT**. Evaluation due 2026-11-27. Rule frozen: z(MVRV) < −0.5 → LONG.

## Falsification ledger (R1–R11)
| Test | Verdict | Finding |
|---|---|---|
| R1 frozen spec on real data | ✗ | v1.0 Sharpe 2.7–3.8 → 0.4/−0.4/0.1/−0.7 — interpolation artifact |
| R3-F funding rate | ~ | Contrarian, weak: +3.19%/wk after P10 decile |
| R3-C on-chain MVRV | ✓ | MVRV<1 → +66.6% fwd 52w, 100% positive (n=85 wk) |
| R6 event study | ~ | 8/9 tops preceded by reverting phase; low specificity |
| R7 out-of-asset | ~ | Accumulation zone generalizes; thresholds must be z-scored |
| R8 traditional assets | ~ | Anchor holds on S&P (1871–), inverts on gold |
| R9 sign-switching anchor | ~ | Rescues gold (+8.9pp), hurts S&P — novel, asymmetric |
| R10 Cointime anchor | ✗ | Investor Price ≪ MVRV (spread 10.7 vs 64.8pp) |
| R11 overlap stats | ⚠ | Cold-zone excess = 2 independent episodes; p=0.81 |

## Verify
`ots verify registry/<entry>.json.ots` — no trust in us required.

## Intégrité
- 2026-07-09 : les preuves .ots initiales se sont révélées malformées lors d'un audit interne ; régénérées et re-soumises publiquement (voir commit 629e1d1). L'incident est conservé dans l'historique.
- Blockchain-proven existence: 2026-07-09; internal creation dates are self-reported.

## Paper
`paper/lienard_paper_v1_3.pdf` — living document, immutable changelog v1.0→v1.3.
