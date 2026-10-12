# Errata for the engineering project

These corrections apply to the engineering project at tag `fork-point`: its README, kept as
[README_engineering.md](README_engineering.md), the legacy part of [DECISIONS.md](../../DECISIONS.md)
and the committed files in `reports/`. The legacy code and reports are unchanged. Every number below
was recomputed for this page from the committed file or raw data file named next to it, and agrees
with the values the research playbook expected.

## a. The XGBoost interval was not calibrated

The engineering README calls the interval behind its interval-aware alarm "calibrated" and reports a
"coverage after conformal calibration" for every model. For XGBoost quantile regression the
split-conformal margin added to each side of the interval was:

| subset | 50% | 80% | 90% | 95% |
|---|---|---|---|---|
| FD001 | 0 | 0 | 0 | 0 |
| FD002 | 0 | 0 | 0.03 to 0.11 | 1.18 to 1.69 |
| FD003 | 0 | 0 | 0 | 0 |
| FD004 | 0 | 0 | 0 | 0 |

Margins in cycles, range over seeds 0 to 4 (`reports/uncertainty_per_seed.csv`, column `margin`,
rows with `model` = xgboost).

On FD001, FD003 and FD004 calibration changed nothing, and on FD002 it widened the 90% interval by at
most 0.11 cycles on each side. The "calibrated" interval is therefore the raw quantile interval. Its
raw 90% coverage on the official test engines, mean over the five seeds, is 86%, 93%, 83% and 86% on
FD001 to FD004 (`reports/uncertainty_per_seed.csv`, column `coverage`, `level` = 0.9; the same values
appear as `coverage_mean` in `reports/uncertainty.csv`), below the nominal level on three of four
subsets.

Cause. The conformal scores were computed on every cycle of the validation engines, and 42%, 37%,
50% and 51% of those cycles (FD001 to FD004) have a capped truth of exactly 125, where the upper
quantile, clipped at the cap, also sits, so their score is zero and they count as covered. With that
many trivially covered points the 90th-percentile score is zero, while the points the coverage is
reported on, the last observed cycles of the test engines, sit at the cap in only 11%, 22%, 15% and
27% of cases.

Sources for the cause: validation shares from `data/raw/CMAPSS/train_FD00x.txt` restricted to the
`val` units of `configs/splits.json`, labelled with `maintadapt.data.load_subset` (4,314, 10,357,
5,014 and 12,651 validation cycles); test shares from `data/raw/CMAPSS/RUL_FD00x.txt` and
`test_FD00x.txt` (11, 57, 15 and 67 engines, also `test_rul_above_cap` in `reports/dataset_stats.csv`).

## b. The decision analysis on test engines rests on an extrapolation

The official test engines stop before failure, and most stop well before it. Share of test engines
whose true RUL at the last observed cycle is at most 15 or at most 25 cycles:

| subset | test engines | RUL ≤ 15 | RUL ≤ 25 |
|---|---|---|---|
| FD001 | 100 | 10 (10%) | 19 (19%) |
| FD002 | 259 | 46 (18%) | 58 (22%) |
| FD003 | 100 | 10 (10%) | 16 (16%) |
| FD004 | 248 | 36 (15%) | 50 (20%) |

Source: `data/raw/CMAPSS/RUL_FD00x.txt`.

For an engine that has not alarmed by its last observed cycle, `decision.replay_alarms` (with
`run_to_failure=False`) keeps its last booking: maintenance falls due when the last prediction,
counted down one cycle per cycle, reaches the threshold, and the engine fails if its true RUL runs out
first. For the large majority of test engines, whether and when the alarm fires is never observed;
the cost is a function of one prediction extrapolated linearly, not of the policy an operator would
have run. The test cost curve and the savings in the engineering README inherit that extrapolation.

The threshold was tuned on validation engines under a different protocol (every validation history
cut at every cycle, `decision.replay_snapshots`), and the two protocols disagree about where the
optimum lies. At a failure cost of 100 cycles, means over seeds 0 to 4:

| subset | policy | validation-tuned threshold | test-optimal threshold | difference | cost, validation-tuned | cost, test-optimal | excess cost |
|---|---|---|---|---|---|---|---|
| FD001 | alarm on predicted RUL | 11.8 | 20.2 | -8.4 | 27.32 | 24.02 | 13.7% |
| FD002 | alarm on predicted RUL | 21.0 | 14.4 | +6.6 | 36.59 | 34.96 | 4.7% |
| FD003 | alarm on predicted RUL | 10.6 | 21.2 | -10.6 | 32.32 | 27.67 | 16.8% |
| FD004 | alarm on predicted RUL | 13.8 | 20.4 | -6.6 | 39.40 | 38.15 | 3.3% |
| FD001 | alarm on lower interval bound | 2.0 | 4.8 | -2.8 | 26.05 | 22.77 | 14.4% |
| FD002 | alarm on lower interval bound | 3.8 | 1.6 | +2.2 | 34.80 | 33.64 | 3.5% |
| FD003 | alarm on lower interval bound | 2.8 | 7.2 | -4.4 | 24.59 | 22.12 | 11.2% |
| FD004 | alarm on lower interval bound | 2.0 | 4.6 | -2.6 | 36.12 | 34.45 | 4.9% |

Source: `reports/decision_per_seed.csv`, rows with `cost_ratio` = 100, columns `setting`,
`test_optimal_setting`, `expected_cost` and `test_optimal_cost`. Thresholds are in cycles of predicted
RUL, costs per test engine; excess cost is mean `expected_cost` / mean `test_optimal_cost` - 1.

The point alarm misses the test optimum by 6.6 to 10.6 cycles and pays 3.3% to 16.8% more than it
would at that optimum; the lower-bound alarm misses by 2.2 to 4.4 cycles and pays 3.5% to 14.4% more.
The research protocol drops this extrapolation: decision metrics are computed only on complete
run-to-failure trajectories predicted out of fold, and the truncated test engines are used only for
leaderboard metrics.

## c. The ± values are seed spread, not uncertainty over engines

Every ± in the engineering README and the legacy decision log is the sample standard deviation over
five training seeds (0 to 4) on one fixed train/validation split (`configs/splits.json`) and one set
of official test engines (for example `rmse_std` in `reports/baselines.csv`). It measures training
randomness only. It leaves out sampling uncertainty over engines, that is, which engines happened to
land in validation and in the test files, which is why ridge, deterministic given the split, shows
± 0.00 (`reports/baselines.csv`). The Welch t-tests (`reports/deep_vs_xgboost.csv`, and the transfer
p-values in `reports/results.md`) treat the five seeds as the samples, so a small p-value says the
seed-to-seed difference is consistent on these particular engines, not that the result generalises to
other engines from the same fleet. The research protocol resamples engines instead (paired engine
bootstrap) and averages over seeds and fold assignments.

## d. Context, not an error

Savings of this size against a time-based policy were reported before on the same four subsets:
Mitici, de Pater, Barros and Zeng (2023), "Dynamic predictive maintenance for multiple components
using data-driven probabilistic RUL prognostics: The case of turbofan engines", Reliability
Engineering & System Safety 234, 109199, doi:10.1016/j.ress.2023.109199. Using engines from FD001 to
FD004 and CNN prognostics with Monte Carlo dropout, they report a 53% lower maintenance cost than a
time-based strategy. The engineering README's 44% to 60% saving against the best fixed schedule
(`saving_vs_fixed_schedule_mean` for the point alarm at `cost_ratio` = 100 in
`reports/decision_analysis.csv`) is in the same range and should be read as a replication of a known
effect, under the caveat in b, not as a new finding.
