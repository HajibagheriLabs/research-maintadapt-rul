# Smoke run of the confirmatory analysis

**SMOKE: pipeline check on the smoke archive (FD001, repeat 0, 3 training epochs for the networks); not confirmatory and says nothing about the hypotheses.**

Primary setting: K = 10, L = 10, rule R1, validation-tuned thresholds. 2,000 engine bootstrap replicates and percentile 95% intervals: training engines are resampled once per fleet and shared by every configuration and repeat, official test engines independently; models, thresholds and selections are held fixed (PREREGISTRATION.md section 6).

Sources: `runs/research/derived/smoke/<fleet>/` (make research-outcomes, make research-metrics). This file, e1.csv, e2.csv, e3.csv, configurations.csv and selections.csv were written by make research-confirmatory at commit c0cbf965a6205d71d4a8030212a273b24adecf7e+dirty.

## H1: mixed

H1: across a realistic pool of RUL models, ranking by leaderboard RMSE agrees only weakly with ranking by maintenance value.

Rule: "E1 bands by point estimate: strong >= 0.7; moderate 0.4-0.7; weak < 0.4. H1 is supported if at least three fleets are weak with interval upper bounds below 0.6; rejected if at least three fleets are strong with interval lower bounds above 0.5; otherwise mixed and reported per fleet."

| fleet | configurations | tau-b (E1) | 95% interval | band |
|---|---|---|---|---|
| FD001 | 10 | 0.022 | [-0.333, 0.422] | weak |

## H2: inconclusive

H2: selecting a model by validation RMSE yields materially lower maintenance value than selecting by validation maintenance cost.

Rule: "E2: H2 is supported if the selection loss is at least 0.05 with an interval excluding 0 on at least two fleets; rejected if the interval lies within [-0.02, 0.02] on at least three fleets; otherwise inconclusive."

| fleet | selection loss (E2) | 95% interval | S by validation cost | S by validation RMSE |
|---|---|---|---|---|
| FD001 | -0.364 | [-0.950, 0.009] | 0.348 [-0.395, 0.836] | 0.712 [0.325, 0.967] |

## H3: inconclusive

H3: the temporal stability of predictions (jitter) is more strongly associated with maintenance value than leaderboard RMSE is.

Rule: "E3: H3 is supported if delta-tau > 0 with an interval excluding 0 on at least three fleets; rejected if delta-tau <= 0 on at least three fleets; otherwise inconclusive."

| fleet | delta-tau (E3) | 95% interval | tau-b(jitter, D) | tau-b(RMSE, D) |
|---|---|---|---|---|
| FD001 | 0.267 | [-0.311, 0.667] | 0.289 [-0.067, 0.556] | 0.022 |

## Excluded configurations

- FD001: none
