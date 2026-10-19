# Liénard Registry — Falsification-First Market Research
**A crypto signal registry that publishes its own failures — hash-chained and Bitcoin-anchored.**

**[→ Live dashboard](site/index.html)**

## What this is
A research program on cycle-positioning anchors (on-chain MVRV,
Liénard oscillator phase), with every claim tied to a published falsification test.
Per Harvey, Liu & Zhu (RFS 2016), 27–53% of published trading anomalies are likely
false. Our answer: frozen specs, negative results published, and a prospective registry
whose entries are timestamped in Bitcoin as they are made.

## Track record (prospective, tamper-evident)
`registry/` — hash-chained entries (each embeds SHA-256 of predecessor), OpenTimestamps
automation anchors every entry in Bitcoin. Current (entry 016, 2026-10-16): ADA, DOGE **LONG**; BTC, ETH, LTC, XRP **FLAT**.
Evaluation due 2027-04-16. Rule frozen: z(MVRV) < −0.5 → LONG.

## Verdicts (26-week)
<!-- VERDICTS:START -->
No verdict yet. Each one is computed automatically at its due date, published whatever the outcome, and anchored in Bitcoin.

Next due: 2026-11-27 (entries 001, 002, 003); 2027-01-15 (entry 004); 2027-01-29 (entry 005).

Method fixed by protocol amendment v2.3 (`registry/protocol_amendment_v2_3.json`), declared before the first verdict: published entry price → Coin Metrics price on the due date, equal-weight baskets, gross returns, no success threshold. Entries 001 and 002 are subsets of 003 (same observation).
<!-- VERDICTS:END -->

## Falsification ledger (R1–R11)
| Test | Verdict | Finding |
|---|---|---|
| R1 frozen spec on real data | ✗ | v1.0 Sharpe 2.7–3.8 → 0.4/−0.4/0.1/−0.7 — interpolation artifact |
| R3-F funding rate | ~ | Contrarian, weak: +3.19%/wk after P10 decile |
| R3-C on-chain MVRV | ✓ | MVRV<1 → +66.6% fwd 52w, 100% positive (n=85 overlapping wk — ≈2 independent episodes, see R11) |
| R6 event study | ~ | 8/9 tops preceded by reverting phase; low specificity |
| R7 out-of-asset | ~ | Accumulation zone generalizes; thresholds must be z-scored |
| R8 traditional assets | ~ | Anchor holds on S&P (1871–), inverts on gold |
| R9 sign-switching anchor | ~ | Rescues gold (+8.9pp), hurts S&P — novel, asymmetric |
| R10 Cointime anchor | ✗ | Investor Price ≪ MVRV (spread 10.7 vs 64.8pp) |
| R11 overlap stats | ⚠ | Cold-zone excess = 2 independent episodes; p=0.81 |

## Known limitations
Published so nobody has to find them for us:
- **XRP can never be LONG under the frozen rule.** The z-score uses each asset's full history; a single 2014 MVRV of 131.7 inflates XRP's standard deviation to 7.58, so z < −0.5 would require MVRV < −0.84 — impossible. Its lowest reachable z is −0.39. XRP is FLAT by construction, not by valuation.
- **The full-history z-score is sensitive to the window.** Early-history extremes (BTC MVRV 23.2 in 2010) shift every threshold; a rolling or log-scale normalization would give different signals. Changing it would break the frozen rule, so it is not changed silently.
- **Six assets, highly correlated.** Weekly return correlations since 2022 range from 0.57 to 0.84 (mean 0.68), i.e. roughly 1.5–2 independent bets per week, and consecutive weekly entries overlap almost entirely over 26 weeks. Proving an edge statistically would take many years of verdicts.
- **Entries 001–003 are not prospective.** They were recorded on 2026-07-07/08 using closes of 2026-05-23 (labelled 2026-05-29) and first anchored on 2026-07-09; entries 001 and 002 are subsets of 003. Only entries from 004 onward were recorded at the time of their data.
- **Week labels vs. data dates.** Each entry is labelled with the week-ending Friday, but uses the latest daily close available when the pipeline runs (usually Thursday's). From entry 017 on, each asset row also records `data_through`, the actual date of the data used.

## Verify
`ots verify registry/<entry>.json.ots` — no trust in us required (entry 004 needs one extra step, see Integrity).
From entry 017 on, each signal can also be recomputed from the archived inputs named in the entry (`inputs_file`, whose `inputs_sha256` is part of the anchored entry): for each asset, z = (last − mean) / population std of its `CapMVRVCur` column.

## Integrity
- 2026-07-09: the initial .ots proofs were found to be malformed during an internal audit; regenerated and re-submitted publicly (see commit 629e1d1). The incident is preserved in the history.
- Blockchain-proven existence: 2026-07-09; internal creation dates are self-reported.
- 2026-07-09: the Coin Metrics community feed has been frozen since 2026-05-23; entry 004 was not created (the protocol forbids entries without a new observation). A freshness guard was added to the update pipeline (`registry/update_registry.py`) to refuse duplicate entries automatically. Entries 001-003 and their 2026-11-27 evaluation are unaffected — only the cadence of new entries is paused until the feed resumes or an alternative source is adopted (to be declared here in advance, never after the fact). Resolved 2026-07-18: protocol amendment v2.2 (declared and stamped before use) switched to the Coin Metrics community REST API; entry 004 resumed the cadence.
- 2026-07-24: the first live scheduled run of the weekly pipeline failed. `update_registry.py` generated entry 005 correctly, but the workflow's stamp-and-push step crashed on a path bug (a repo-root-relative path from `git status` was resolved against a subdirectory working directory), so the entry was never committed. Consistent with the 2026-07-09 precedent, we do not reconstruct entries after the fact: no entry exists for the week of 2026-07-24, the gap is left on the record, and the cadence resumes at entry 005 (week of 2026-07-31). The workflow was fixed the same day, and a separate future-dating guard was added to the pipeline so an off-schedule run can no longer emit an entry labelled with an upcoming Friday.
- 2026-10-19: an external audit found that the OpenTimestamps proof of entry 004 commits to the file with Windows (CRLF) line endings — the entry was generated and stamped on Windows on 2026-07-18 — while git stores it with LF, so `ots verify` on a Linux/macOS clone fails for that one entry. The content is identical (only line endings differ), and the proof still shows it existed by Bitcoin blocks 958543/958544. To verify: `python3 -c "import sys;open('e004.json','wb').write(open(sys.argv[1],'rb').read().replace(b'\n',b'\r\n'))" registry/r5_log_entry004.json && ots verify -f e004.json registry/r5_log_entry004.json.ots`. The proof is kept as stamped, not regenerated. All other proofs match the stored bytes. `.gitattributes` now forbids line-ending conversion of registry files, and both pipeline scripts write LF on every OS.

## Data & license
Market and on-chain data: [Coin Metrics Community Data](https://docs.coinmetrics.io/packages/coin-metrics-community-data), licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use only). This registry is published free of charge, for non-commercial research. From entry 017 on, `registry/inputs/` redistributes, under the same license, the exact weekly series each entry was computed from.

## Paper
`paper/lienard_paper_v1_3.pdf` — living document, immutable changelog v1.0→v1.3.
