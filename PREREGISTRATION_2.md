# Pre-registration (DRAFT): Can calibrated predictive distributions replace threshold tuning in RUL-based maintenance?

Status: **draft, not registered.** Branch `paper-2`. Nothing in this study reads or changes the
confirmatory archive or outputs of the first study (PREREGISTRATION.md, tag `prereg-v1`).
Author: Hadi Hajibagheri, Independent Researcher, ORCID 0009-0002-5404-911X
Format: adapted from the template for preregistration of secondary data analysis
(Van den Akker et al., 2021, Meta-Psychology), as for the first study.

## 1. Prior knowledge of the data
- The first study's results are known: on C-MAPSS FD001-FD004, leaderboard RMSE agrees only
  weakly with maintenance value (H1 supported), selection by validation cost did not beat
  selection by validation RMSE (H2 inconclusive, E2 negative on all four fleets), and jitter was
  not more strongly associated with maintenance value than RMSE (H3 rejected)
  (`reports/research/confirmatory/verdicts.md`).
- The engineering project's XGBoost interval, called calibrated, was not: its conformal margin
  was zero on three subsets because most validation cycles sit at the cap
  (`docs/legacy/ERRATA.md`, a).
- Before this draft, one smoke run fitted the three probabilistic families of section 4 on
  FD001, repeat 0, fold 0 with three training epochs for the networks
  (`make paper2-smoke`), to check the pipeline. Its predictions say nothing about the
  hypotheses and are not reused.

## 2. Research questions and hypotheses
RQ4 (cost shift). When the cost ratio K changes after deployment, does a plug-in rule on
calibrated predictive distributions keep more maintenance value than a threshold tuned at the
old K?
H4: rule R4 on calibrated distributions, applied at K_test, reaches a higher savings share S
than rule R1 tuned at K_train and applied at K_test.
RQ5 (scarce failures). When only a few failures are available to tune on, does the plug-in rule
keep more maintenance value than a tuned threshold?
H5: with thresholds tuned on 5 run-to-failure engines, R4 on calibrated distributions reaches
a higher S than R1 tuned on the same 5 engines.
RQ6 (calibration). Do truncation-matched conformal methods calibrate predictive intervals where
decisions are made?
H6: with matched conformal calibration, the nominal 90% interval covers between 85% and 95% of
outer_test points with true RUL <= 50, while the uncalibrated interval does not.

## 3. Data
C-MAPSS FD001-FD004, the files and integrity manifest of the first study
(`docs/research/data_manifest.json`). N-CMAPSS is not used.

## 4. Protocol
4.1 Folds, preparation and seeds: those of the first study (`configs/research_folds.json`, R = 3,
preparation fitted on inner_fit engines only, seeds from fleet, repeat, fold and configuration).
4.2 Probabilistic families, each fitted in every (fleet, repeat, outer fold)
(`src/maintadapt/research/probabilistic.py`):
- `xgboost_quantile`: quantile XGBoost on the settings of xgboost-00, one booster for all levels;
- `mc_dropout`: cnn-00, 100 forward passes with dropout on;
- `ensemble`: five fits of cnn-00 with distinct seeds, read as a normal distribution with the
  members' mean and standard deviation.
Every fit stores, for every cycle of its inner_val, outer_test and official test engines, the
point prediction (median, or ensemble mean) and 26 quantiles at levels 0.001, 0.0025, 0.005,
0.01, 0.025, 0.05 to 0.95 in steps of 0.05, 0.975 and 0.99, clipped to [0, 125] and sorted.
4.3 Calibration, fitted on each fold's inner_val engines only
(`src/maintadapt/uncertainty.py`): calibration points are 20 cycles per engine drawn uniformly
from 1..T (fixed seed per engine), every engine weighted equally, with the finite-sample
correction counting engines. Variants: none (raw quantiles); matched CQR per level
(`quantile_shifts`, the primary calibration); Mondrian margins of the 90% interval by
predicted-RUL bin {[0, 25), [25, 50), [50, 75), [75, 100), [100, 125]}, at least 5 engines per
bin.

## 5. Measures
5.1 Decision model, costs, cost rate, age replacement, S and D: as in sections 5.2 and 5.6 of
PREREGISTRATION.md.
5.2 Rules. R1: the point prediction with a tuned threshold (the first study's rule). R3: the
first crossing of the 0.1 quantile, threshold tuned the same way (0.05 and 0.2 secondary). R4:
alarm at the first cycle with P(RUL_t <= L) >= p*, P read from the (calibrated) quantiles by
linear interpolation, p* = (L + 1) / ((K - 1) x mean outer-train life), capped at 1, with no
tuning (derivation: DECISIONS.md, "Paper 2: the plug-in probability of rule R4"). A p* below
the lowest stored level acts as 0.001 and is reported.
5.3 Threshold tuning for R1 and R3: section 5.4 of PREREGISTRATION.md, on the engines each
setting names.
5.4 Coverage: share of points inside the interval, by true-RUL bin and for true RUL <= 50, at
20 uniform cut points per outer_test engine, every engine weighted equally.

## 6. Primary settings and endpoints
Lead time L = 10 cycles throughout. Every endpoint is computed per family and fleet, with
outcomes pooled over the folds of a repeat and averaged over repeats, and intervals from 2,000
engine-bootstrap replicates (training engines resampled once per fleet; models, calibration,
thresholds and p* held fixed).
- **Setting A, cost-ratio shift.** R1 and R3 are tuned on inner_val engines at K_train = 10 and
  evaluated on outer_test engines at K_test in {2, 50}; R4 uses p* at K_test; S at K_test uses
  age replacement tuned at K_test. Endpoint E4 = S(R4, calibrated) - S(R1 tuned at K_train),
  at each K_test.
- **Setting B, scarce failures.** At K = 10, R1 is tuned and R4 is calibrated on the same n
  inner_val engines, n in {5, 10, 20}, with 20 draws of the n engines per fold
  (`scarce_tuning_sets`, fixed seeds), outcomes averaged over draws. Endpoint E5 =
  S(R4, calibrated on n) - S(R1 tuned on n) at n = 5 (10 and 20 secondary).
- **Calibration.** Endpoint E6 = coverage of the nominal 90% interval at true RUL <= 50 on
  outer_test cut points, raw and with matched CQR.

Interpretation rules, fixed before any fit:
- H4 is supported if E4 >= 0.05 with an interval excluding 0 on at least three fleets at either
  K_test, for at least two of the three families; rejected if E4 <= 0 on at least three fleets
  at both K_test for at least two families; otherwise inconclusive.
- H5: the same rule with E5 at n = 5.
- H6 is supported if, for at least two families on at least three fleets, matched CQR coverage
  lies in [0.85, 0.95] and raw coverage lies outside it; rejected if matched CQR coverage lies
  outside [0.85, 0.95] for at least two families on at least three fleets; otherwise mixed.

## 7. Secondary analyses
- Every K_train and K_test of the first study's grid {1.5, 2, 3, 5, 10, 20, 50}, and L in
  {0, 20}.
- R3 against R1 and R4 in both settings; the oracle reference R1 retuned at K_test.
- R4 on raw quantiles, and with Mondrian calibration.
- R4 with the age-replacement cost rate in place of CR* in p*.
- Coverage of every calibration variant at levels 0.5, 0.8, 0.9 and 0.95, by true-RUL bin.
- The realised failure rate and waste per maintenance of every rule.

## 8. Failures, exclusions and stopping
As section 9 of PREREGISTRATION.md: one retry with seed + 1000, exclusion of a family from a
fleet if a fit still fails, and no decision metric or coverage computed from the outer_test or
official test predictions before the analysis code of this study is tagged.

## 9. Deviations
Every change after registration goes to a DEVIATIONS_2.md, as for the first study.

## 10. Software, hardware and compute
Python 3.11 and the pinned dependencies; one RTX 3090 and an Intel Core i7-4790K (4 cores,
8 threads). The fits run with `make paper2-prob`, resumable; the hours are projected from a dry
run before registration.
