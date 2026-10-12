# Pre-registration: Do remaining-useful-life leaderboards predict maintenance value?

Status: binding from the commit tagged `prereg-v1` ([[AUTHOR: date]]).
Author: [[AUTHOR: full name]], Independent Researcher, ORCID [[AUTHOR]]
Repository: github.com/HajibagheriLabs/research-maintadapt-rul (fork of maintadapt-rul, tag fork-point)
Format: adapted from the template for preregistration of secondary data analysis
(Van den Akker et al., 2021, Meta-Psychology).

## 1. Prior knowledge of the data
- C-MAPSS (NASA PCoE data set 6) and N-CMAPSS (data set 17) are public. The author built
  maintadapt-rul on C-MAPSS. The official-test RMSE and NASA score of ridge, XGBoost, TCN and LSTM,
  and a decision analysis on truncated test engines, are known (docs/legacy/).
- Pilot, 2026-09-30: eight untuned models, one seed, one fold assignment, thresholds chosen on the
  other folds' out-of-fold predictions, zero lead time. Kendall's tau between official-test RMSE
  and per-engine maintenance regret (failure cost 100 cycles) was 0.43, 0.07, 0.14 and 0.07 on
  FD001-FD004. A random forest had higher RMSE but lower maintenance cost than XGBoost on all four
  fleets. The pilot shaped this design; none of its fits are reused.
- No result from the frozen pool, folds and seeds below has been computed when this document is
  tagged.

## 2. Research questions and hypotheses
RQ1 (agreement). H1: across a realistic pool of RUL models, ranking by leaderboard RMSE agrees only
weakly with ranking by maintenance value.
RQ2 (consequence). H2: selecting a model by validation RMSE yields materially lower maintenance
value than selecting by validation maintenance cost.
RQ3 (mechanism). H3: the temporal stability of predictions (jitter) is more strongly associated
with maintenance value than leaderboard RMSE is.
Not studied here: probabilistic decision rules, calibration, transfer learning.

## 3. Data
- C-MAPSS FD001-FD004: run-to-failure training engines 100/260/100/249; official test engines
  100/259/100/248, truncated, with the true RUL at truncation.
- N-CMAPSS: secondary analysis (section 8).
- Integrity: docs/research/data_manifest.json, SHA-256 fc553f8742325ab650db8b0d33d62d692bcda7087256154367d0b6366b601dda.

## 4. Protocol
4.1 Folds. For each fleet and repeat: five outer folds over all training engines, stratified by
lifetime; within each outer-train set, 25% inner-validation engines (stratified the same way) and
75% inner-fit engines. File configs/research_folds.json, SHA-256 157754641e81d05d86cb8b275b4f3e9744fdb078ca797a39a2a5ec2729baa021.
Sizes per fleet (engines; per fold outer_test / inner_val / inner_fit): FD001 100; 20 / 20 / 60.
FD002 260; 52 / 52 / 156. FD003 100; 20 / 20 / 60. FD004 249; 50 / 50 / 149 in 9 of the 15
(repeat, fold) pairs, 50 / 49 / 150 in 3 and 49 / 50 / 150 in 3.
4.2 Preparation, fitted on inner-fit engines only: operating-regime clustering, sensor screening
(a sensor is dropped if one value covers at least 95% of its readings in any regime), z-score
normalisation within regime. Tabular inputs: current readings, cycle, and trailing mean, standard
deviation and slope over 5, 10, 20 and 30 cycles. Sequence inputs: windows of normalised readings,
left-padded with the engine's first reading. Target: RUL capped at 125.
4.3 Model pool: 10 families, 113 configurations; configuration 0 of each family is its reference
configuration. File configs/research_pool.yaml, SHA-256 c6d87e4bd469ac96fce5e3cd6730134159d8620de96d9ea5518b263b313f3809.
| family | inputs | n | reference configuration (config 0) | space |
|---|---|---|---|---|
| ridge | rolling features, scaled | 5 | alpha 1 | alpha {0.1, 1, 10, 100, 1000} |
| knn | rolling features, scaled | 12 | k 30, uniform weights | k {5, 10, 20, 30, 50, 100} x weights {uniform, distance} |
| forest | rolling features | 12 | max_features 0.3, min_samples_leaf 5, max_depth None | max_features {0.1, 0.2, 0.3, 0.5}; min_samples_leaf {1, 5, 10, 20}; max_depth {None, 12, 20}; all: 200 trees, max_samples 0.5 |
| xgboost | rolling features | 12 | legacy settings: max_depth 5, learning_rate 0.03, min_child_weight 5, subsample 0.8, colsample_bytree 0.6, reg_lambda 1 (XGBoost default), 2000 trees | max_depth {3, 4, 5, 6, 8}; learning_rate log-uniform [0.01, 0.1]; min_child_weight {1, 3, 5, 10, 20}; subsample uniform [0.6, 1]; colsample_bytree uniform [0.3, 1]; reg_lambda log-uniform [0.1, 10]; other configurations 3000 trees; all: early stopping after 100 rounds |
| xgboost_raw | cycle and current readings | 12 | as xgboost | as xgboost |
| mlp | rolling features, scaled | 12 | hidden (128, 64), dropout 0.1, lr 1e-3, weight_decay 1e-4 | hidden {(64,), (128, 64), (256, 128), (256, 128, 64)}; dropout {0, 0.1, 0.2}; lr log-uniform [3e-4, 3e-3]; weight_decay {0, 1e-5, 1e-4}; all: batch 256 |
| lstm | windows | 12 | legacy: hidden 64, 2 layers, dropout 0.2, lr 1e-3, window 30 | hidden {32, 64, 128}; layers {1, 2, 3}; dropout {0.1, 0.2, 0.3, 0.5}; lr log-uniform [3e-4, 3e-3]; window {20, 30, 40}; all: batch 256, weight_decay 1e-4 |
| tcn | windows | 12 | legacy: 64 channels, kernel 3, dilations (1, 2, 4, 8), dropout 0.2, lr 1e-3, window 30 | channels {32, 64, 128}; kernel {2, 3, 5}; dilations {(1, 2, 4), (1, 2, 4, 8), (1, 2, 4, 8, 16)}; dropout {0.1, 0.2, 0.3}; lr log-uniform [3e-4, 3e-3]; window {20, 30, 40}; all: batch 256, weight_decay 1e-4 |
| cnn | windows | 12 | Li et al. (2018): 4 conv layers of 10 filters, kernel 10, then 1 filter of kernel 3; tanh; dropout 0.5; FC 100; lr 1e-3; window 30 | filters {10, 16, 32}; kernel {5, 10, 15}; conv layers {3, 4, 5}; fc {50, 100, 200}; dropout {0.2, 0.5}; lr log-uniform [3e-4, 3e-3]; window {20, 30, 40}; all: batch 512, weight_decay 1e-4 |
| transformer | windows | 12 | d_model 64, 4 heads, 2 layers, ff 128, dropout 0.1, lr 1e-3, window 30 | d_model {32, 64, 128}; heads {2, 4}; layers {1, 2, 3}; ff {64, 128, 256}; dropout {0.1, 0.2}; lr log-uniform [3e-4, 3e-3]; window {20, 30, 40}; all: batch 256, weight_decay 1e-4 |

"all:" settings apply to every configuration of the family, the reference included. Configurations
1 onwards of ridge and knn are the remaining grid points in grid order; those of the other families
are drawn with numpy default_rng([2026, family index]) in the order of the table (ridge = 0), without
duplicates, continuous values rounded to three significant figures.
Neural families: Adam, MSE on RUL/125, gradient clipping at 1.0, at most 100 epochs, early stopping
after 15 epochs without improvement in inner-validation RMSE. XGBoost: early stopping after 100
rounds without improvement in inner-validation RMSE. Every configuration is fitted in every
(fleet, repeat, outer fold). Seeds derive from (fleet, repeat, fold, configuration id).
4.4 Repeats. R = 3 if a dry run (after tagging) projects at most 72 hours of wall time for each of
the CPU and GPU streams run concurrently; otherwise R = 2; if R = 2 still exceeds 120 hours,
configurations 8-11 of every family with twelve configurations are dropped. The outcome is recorded
in DECISIONS.md.
4.5 Stored predictions: every cycle of inner-validation, outer-test and official test engines.

## 5. Measures
5.1 Leaderboard metrics, per fit, on official test engines, averaged over all fits of a
configuration: RMSE at the last observed cycle against truth capped at 125 (primary); NASA score
summed over engines; RMSE over all observed cycles.
5.2 Decision model. Engine life T is the last recorded cycle; RUL_t = T - t. The alarm fires at the
first cycle a with score_a <= theta, and maintenance happens at m = a + L. If m <= T it is
preventive (cost 1, uses m cycles, wastes T - m); otherwise, or without an alarm, the engine fails
(cost K, uses T cycles). Cost rate CR = sum of costs / sum of cycles used (renewal reward).
References: perfect prognostics CR* = N / sum of T; age replacement at A*, tuned on the outer-train
lifetimes. Savings share S = (CR_age - CR) / (CR_age - CR*); decision loss D = 1 - S.
5.3 Rules. R1: the score is the point prediction. R2: the score is a causal exponential moving
average of the predictions with alpha tuned from {1, 0.5, 0.3, 0.2, 0.1}.
5.4 Policy tuning (primary): on the fold's inner-validation engines. Candidate thresholds are the
inverted-CDF quantiles at 401 equally spaced levels of the pooled validation scores, plus -inf and
+inf; the candidate with the lowest validation cost rate is chosen, ties broken by the median
minimiser.
5.5 Settings: K in {1.5, 2, 3, 5, 10, 20, 50}; L in {0, 10, 20}. Primary: K = 10, L = 10, R1,
validation tuning.
5.6 Maintenance value of a configuration: S computed from outcomes pooled over the five outer folds
of a repeat, averaged over repeats.
5.7 Candidate proxies, on official test engines over all observed cycles: jitter (mean over engines
of the mean |y_t - y_(t-1) + 1| over cycles whose previous true RUL is below 125; engines with at
least five such cycles), and late-phase RMSE and bias (true RUL <= 50).
5.8 Validation metrics used for selection: validation RMSE over all inner-validation cycles against
capped truth; validation cost = cost rate on inner-validation engines with the threshold tuned on
them.

## 6. Primary endpoints, analysis and interpretation
Every metric is oriented so that lower is better (decision loss D stands for maintenance value).
Uncertainty: 2,000 bootstrap replicates and percentile 95% intervals. Training engines are resampled
with replacement within a fleet (one draw shared by all configurations and repeats); official test
engines are resampled independently; fitted models, thresholds and selections are held fixed.
- E1 (RQ1): per fleet, Kendall's tau-b between leaderboard RMSE and D across all configurations.
- E2 (RQ2): per fleet, selection loss = S(choose by validation cost) - S(choose by validation RMSE).
  The choice is made independently in each (repeat, outer fold) among all configurations; outcomes
  of the chosen models are pooled per repeat and averaged over repeats. Positive values mean that
  choosing by RMSE loses maintenance value.
- E3 (RQ3): per fleet, delta-tau = tau-b(jitter, D) - tau-b(leaderboard RMSE, D).
Interpretation rules, fixed now:
- E1 bands by point estimate: strong >= 0.7; moderate 0.4-0.7; weak < 0.4. H1 is supported if at
  least three fleets are weak with interval upper bounds below 0.6; rejected if at least three
  fleets are strong with interval lower bounds above 0.5; otherwise mixed and reported per fleet.
- E2: H2 is supported if the selection loss is at least 0.05 with an interval excluding 0 on at
  least two fleets; rejected if the interval lies within [-0.02, 0.02] on at least three fleets;
  otherwise inconclusive.
- E3: H3 is supported if delta-tau > 0 with an interval excluding 0 on at least three fleets;
  rejected if delta-tau <= 0 on at least three fleets; otherwise inconclusive.
Multiplicity: the three endpoints are judged only by these rules. Secondary tests use
Benjamini-Hochberg at 0.05 within each analysis. Exploratory analyses are labelled and uncorrected.

## 7. Secondary analyses
S1 E1 with the NASA score and with all-cycle RMSE.
S2 E1-E3 over the whole settings grid and for rule R2.
S3 Family level: in each fold, the family's configuration with the lowest validation RMSE (the
family representative); ranks by leaderboard RMSE and by S; pairwise inversions where both
differences have intervals excluding 0 in opposite directions (Benjamini-Hochberg).
S4 Threshold variants: crossfit (tuned on the other folds' outer-test engines), oracle (tuned on all
outer-test engines of the repeat), nested (4-fold inner cross-fitting on outer-train engines;
FD001 and FD003, family representatives, repeat 0).
S5 Training cap: the reference XGBoost and LSTM trained with caps 100, 150, 200 and none (repeat 0).
S6 Regret = C_f * 1[failure] + waste, with C_f in {50, 100, 200, 500, 1000, 2000}, instead of the
cost rate.
S7 Matched accuracy: RMSE at 20 uniform truncation points per outer-test engine, against D.
S8 N-CMAPSS replication of E1-E3 (descriptive).
S9 E1 restricted to configurations whose leaderboard RMSE is within 2 cycles of the best.
Conventional leaderboard: reference configurations trained on the committed 80/20 split
(configs/splits.json) with seeds 0-4, for comparison with the literature.

## 8. N-CMAPSS (secondary)
Inputs: scenario descriptors W and physical sensors X_s only; never virtual sensors, health
parameters or the health state. One row per unit and flight (mean and standard deviation of each
input variable, flight duration, flight class), standardised within flight class on inner-fit
units. Cap: the median RUL at the first abnormal health state across units, rounded to 5 cycles,
computed before any model is trained. Five outer folds over pooled units, stratified by
sub-dataset; R = 3. Same pool and analyses; the leaderboard metric is RMSE over all cycles of
held-out units. The final preprocessing choices are written to DECISIONS.md before any N-CMAPSS
model is trained.

## 9. Failures, exclusions and stopping
A fit that raises an error or returns non-finite predictions is retried once with seed + 1000. A
configuration that still fails in any fold of a fleet is excluded from that fleet and listed. All
registered fits are run. No test or decision metric is computed from the confirmatory predictions
before the analysis code is tagged `analysis-frozen`; progress checks read fit counts, failures and
validation RMSE only.

## 10. Deviations
Every change after `prereg-v1` is logged in DEVIATIONS.md (date, section, change, reason, expected
impact). Affected results are labelled, and the paper reports every deviation.

## 11. Software and hardware
Python 3.11, dependencies pinned in pyproject.toml; per-fit package versions and hardware are stored
with each prediction file. One RTX 3090 (24 GB) and an 8-core CPU.
