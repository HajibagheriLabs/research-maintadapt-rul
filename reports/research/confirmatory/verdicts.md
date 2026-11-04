# Confirmatory verdicts

**CONFIRMATORY: the pre-registered analysis of the confirmatory archive.**

Primary setting: K = 10, L = 10, rule R1, validation-tuned thresholds. 2,000 engine bootstrap replicates and percentile 95% intervals: training engines are resampled once per fleet and shared by every configuration and repeat, official test engines independently; models, thresholds and selections are held fixed (PREREGISTRATION.md section 6).

Sources: `runs/research/derived/archive/<fleet>/` (make research-outcomes, make research-metrics). This file, e1.csv, e2.csv, e3.csv, configurations.csv and selections.csv were written by make research-confirmatory at commit 01a7fd1e90b6f5b8819f07b081d1b32837067fef.

## H1: supported

H1: across a realistic pool of RUL models, ranking by leaderboard RMSE agrees only weakly with ranking by maintenance value.

Rule: "E1 bands by point estimate: strong >= 0.7; moderate 0.4-0.7; weak < 0.4. H1 is supported if at least three fleets are weak with interval upper bounds below 0.6; rejected if at least three fleets are strong with interval lower bounds above 0.5; otherwise mixed and reported per fleet."

| fleet | configurations | tau-b (E1) | 95% interval | band |
|---|---|---|---|---|
| FD001 | 113 | 0.286 | [-0.182, 0.408] | weak |
| FD002 | 113 | 0.041 | [-0.281, 0.355] | weak |
| FD003 | 113 | -0.037 | [-0.295, 0.281] | weak |
| FD004 | 113 | -0.062 | [-0.285, 0.193] | weak |

## H2: inconclusive

H2: selecting a model by validation RMSE yields materially lower maintenance value than selecting by validation maintenance cost.

Rule: "E2: H2 is supported if the selection loss is at least 0.05 with an interval excluding 0 on at least two fleets; rejected if the interval lies within [-0.02, 0.02] on at least three fleets; otherwise inconclusive."

| fleet | selection loss (E2) | 95% interval | S by validation cost | S by validation RMSE |
|---|---|---|---|---|
| FD001 | -0.882 | [-1.710, -0.363] | -0.304 [-1.219, 0.283] | 0.578 [0.189, 0.839] |
| FD002 | -0.621 | [-0.967, -0.324] | 0.018 [-0.359, 0.347] | 0.639 [0.454, 0.784] |
| FD003 | -1.136 | [-1.933, -0.537] | -0.653 [-1.573, 0.015] | 0.483 [0.060, 0.776] |
| FD004 | -0.129 | [-0.297, 0.019] | 0.601 [0.417, 0.742] | 0.729 [0.575, 0.839] |

## H3: rejected

H3: the temporal stability of predictions (jitter) is more strongly associated with maintenance value than leaderboard RMSE is.

Rule: "E3: H3 is supported if delta-tau > 0 with an interval excluding 0 on at least three fleets; rejected if delta-tau <= 0 on at least three fleets; otherwise inconclusive."

| fleet | delta-tau (E3) | 95% interval | tau-b(jitter, D) | tau-b(RMSE, D) |
|---|---|---|---|---|
| FD001 | -0.068 | [-0.208, 0.192] | 0.219 [-0.104, 0.306] | 0.286 |
| FD002 | -0.050 | [-0.209, 0.117] | -0.008 [-0.247, 0.222] | 0.041 |
| FD003 | -0.220 | [-0.396, 0.126] | -0.257 [-0.331, 0.039] | -0.037 |
| FD004 | 0.027 | [-0.154, 0.217] | -0.035 [-0.235, 0.199] | -0.062 |

## Excluded configurations

- FD001: none
- FD002: none
- FD003: none
- FD004: none
