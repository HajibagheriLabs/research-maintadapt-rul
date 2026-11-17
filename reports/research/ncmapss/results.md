# N-CMAPSS: E1-E3 (secondary, descriptive)

**SECONDARY S8: N-CMAPSS replication of E1-E3 (PREREGISTRATION.md section 8), descriptive; not confirmatory; affected by the deviations in DEVIATIONS.md.**

99 units pooled from the readable sub-datasets (`reports/research/ncmapss/inspection.md`), cap 50, 113 configurations. Primary setting: K = 10, L = 10 flights, rule R1, validation-tuned thresholds. Leaderboard RMSE is the RMSE over every flight of a fit's outer_test units against the capped truth, averaged over fits; jitter uses the same units. 2,000 bootstrap replicates draw the units once for D and the accuracy metrics alike; percentile 95% intervals (DECISIONS.md, N-CMAPSS).

Sources: `runs/research/derived/archive_ncmapss/NCMAPSS/` (make research-ncmapss-derived); written by make research-ncmapss-report at commit 867f2e4a6a051ce293e5d2e7f77bac818b4438c8+dirty.

| endpoint | estimate [95% interval] |
|---|---|
| E1, tau-b(leaderboard RMSE, D) | 0.677 [0.374, 0.724] (moderate) |
| E2, selection loss | -0.281 [-0.660, 0.057] |
| S chosen by validation cost | -0.450 [-1.354, 0.146] |
| S chosen by validation RMSE | -0.169 [-1.030, 0.402] |
| E3, delta-tau | -0.579 [-0.702, -0.302] |
| tau-b(jitter, D) | 0.098 [-0.109, 0.182] |

## Section 6's fleet-level conditions, read for this fleet

The verdicts on H1-H3 count C-MAPSS fleets and are in `reports/research/confirmatory/verdicts.md`; N-CMAPSS does not enter them. Rules quoted from PREREGISTRATION.md section 6:

- "E1 bands by point estimate: strong >= 0.7; moderate 0.4-0.7; weak < 0.4. H1 is supported if at least three fleets are weak with interval upper bounds below 0.6; rejected if at least three fleets are strong with interval lower bounds above 0.5; otherwise mixed and reported per fleet."
- "E2: H2 is supported if the selection loss is at least 0.05 with an interval excluding 0 on at least two fleets; rejected if the interval lies within [-0.02, 0.02] on at least three fleets; otherwise inconclusive."
- "E3: H3 is supported if delta-tau > 0 with an interval excluding 0 on at least three fleets; rejected if delta-tau <= 0 on at least three fleets; otherwise inconclusive."

| condition | holds here |
|---|---|
| E1 weak with interval upper bound below 0.6 (counts towards H1 supported) | no |
| E1 strong with interval lower bound above 0.5 (counts towards H1 rejected) | no |
| E2 at least 0.05 with an interval excluding 0 (counts towards H2 supported) | no |
| E2 interval within [-0.02, 0.02] (counts towards H2 rejected) | no |
| E3 above 0 with an interval excluding 0 (counts towards H3 supported) | no |
| E3 at most 0 (counts towards H3 rejected) | yes |

Excluded configurations: none.
