# Decisions

## Research fork

**What changed.** At tag `fork-point` (commit 36c23b3) the engineering project maintadapt-rul became
the companion repository of one pre-registered study: do the accuracy metrics RUL papers rank models
by predict the maintenance value of the same models? The design is fixed in
[PREREGISTRATION.md](PREREGISTRATION.md) before any confirmatory fit, and every later change to it is
logged in [DEVIATIONS.md](DEVIATIONS.md). Research code goes in `src/maintadapt/research/`,
`scripts/research/` and `configs/research*`. The legacy pipeline stays and computes what it computed
at the fork; only its `make train`, `eval` and `report` targets were renamed `legacy-train`,
`legacy-eval` and `legacy-report`, and its README moved to
[docs/legacy/README_engineering.md](docs/legacy/README_engineering.md).

**Why.** The engineering project scored models on one fixed split, on the truncated official test
engines, with seed-only ± values, and evaluated maintenance decisions on those same truncated
engines. That cannot say whether a model that wins on the leaderboard also wins as a maintenance
policy. The research protocol keeps the two estimands apart: leaderboard metrics on the official test
engines exactly as RUL papers compute them, maintenance value only on complete run-to-failure
trajectories predicted out of fold, with uncertainty from resampling engines.

**Corrections.** Three legacy claims need correcting and one needs context: the XGBoost interval
called calibrated was not (its conformal margin was zero on three subsets), the decision analysis on
test engines rests on an extrapolation of truncated histories, and the ± values are seed spread
only. Details and the recomputed numbers, each with its source file, are in
[docs/legacy/ERRATA.md](docs/legacy/ERRATA.md).

## Legacy log (engineering project, tag `fork-point`)

Kept verbatim below.

A short log of every non-obvious choice in this repo and the reason for it. Numbers quoted here come
from files in `reports/`, which the `make` targets regenerate.

## Data and labels

**RUL cap of 125 cycles.** Training targets follow the piecewise-linear convention:
RUL = min(cycles to failure, 125). Early in life the sensors say nothing about how far away failure
is; in `reports/figures/degradation_trajectories.png` the fleet median of every informative sensor
is nearly flat until roughly 125 cycles before failure. An uncapped target would ask the model to
tell 300 cycles from 200 using noise. 125 is also the value most C-MAPSS papers use, which keeps
the numbers comparable.

**Splits by engine, fixed and committed.** 20% of each subset's run-to-failure engines are held out
for validation (early stopping, model selection and alarm-threshold selection). An engine's whole
trajectory lives in exactly one split. The split is drawn from `data.split_seed` and written to
`configs/splits.json`, which is committed so the exact partition can be reviewed. Loaders refuse a
split file written under different settings, and a test asserts zero overlap and that the file
regenerates exactly. The official test engines (`test_FD00x.txt` with `RUL_FD00x.txt`) are never
used to fit anything.

**Operating regimes found by clustering, not hardcoded.** FD002 and FD004 fly six
altitude/Mach/throttle regimes. KMeans on the three operating settings of training engines (k from
config) recovers them with a maximum deviation of 0.005 setting units from the cluster centres
(`reports/operating_conditions.csv`).

**Sensor screening by modal share, per regime.** A sensor is dropped if one value covers at least
95% of its training readings within any regime. Two alternatives were rejected. A distinct-value
count throws away real signal: sensor_10 in FD003 takes only a handful of values yet has a Spearman
correlation of -0.46 with RUL. A variance threshold is not comparable across sensors measured in
different units. "Within any regime" matters because condition-aware normalisation divides by the
within-regime spread, which is zero for a sensor frozen in one regime; sensor_16 in FD002 is
constant in 5 of 6 regimes. The result is 14 sensors for FD001 and FD002 (the set most of the
literature uses), 16 for FD003 and 15 for FD004. Every drop and its reason is listed in
`reports/sensor_screening.csv`.

**Normalisation fitted on training engines only.** Z-score statistics, per regime for FD002 and
FD004, are fitted on the training split and applied unchanged to validation and test.
`fit_normaliser` takes the list of training units explicitly, so statistics cannot silently be
computed on a concatenated frame, and `test_training_statistics_ignore_appended_validation_units`
guards the whole preparation step: regimes, sensor screening and statistics.

## Features and evaluation

**Rolling windows of 5, 10, 20 and 30 cycles.** Each kept sensor contributes its current normalised
reading plus the trailing mean, standard deviation and least-squares slope over each window. Short
windows react to recent change; long windows average out sensor noise, which is large relative to
the degradation trend. The longest window matches the typical test history: the shortest FD001
test engine has 31 observed cycles. Windows only look backwards, and on engines with less history
than a window they use what exists (a single reading has zero spread and a flat slope). Running
sums restart for every engine, so one engine's features never depend on another's data. Engine age
in cycles is also a feature: it is known at prediction time and carries the fleet's typical
lifetime.

**Condition-aware normalisation on FD002 and FD004.** In the multi-regime subsets a raw reading
mostly says which regime the engine is flying, not how worn it is: the six bands in
`reports/figures/condition_normalisation.png` are regimes. Z-scoring each sensor within its regime
removes that and leaves the degradation trend. With identical XGBoost settings it lowers test RMSE
from 15.91 to 12.60 on FD002 and from 17.39 to 13.49 on FD004, and roughly halves the NASA score
(`reports/normalisation_ablation.csv`). The globally normalised model also needed far more trees
before validation stopped improving, some close to the 2,000-tree cap, so it lost despite being
given more capacity, not less.

**Scoring at the last observed cycle, against capped truth.** Every model is scored on the final
observed cycle of each official test engine, the standard C-MAPSS protocol. Model metrics compare
against the truth capped at 125 cycles (`evaluation.cap_test_truth`), because the models are
trained to say "125 or more" for healthy engines; penalising them for not telling 150 from 190
would measure the label design rather than the model. This affects 11, 57, 15 and 67 test engines
in FD001 to FD004 (`reports/dataset_stats.csv`). The decision analysis uses the raw truth, because
discarded life is real life whether or not the label could see it.

**Ridge penalty and XGBoost stopping chosen on validation engines.** Ridge picks its penalty from a
small grid by validation RMSE; XGBoost stops adding trees once validation RMSE has not improved for
100 rounds. Ridge is deterministic given the split, so its standard deviation across seeds is zero,
and it is reported that way rather than hidden.

**MLflow on SQLite.** MLflow 3 no longer develops its plain-file backend, so runs are logged to
`mlruns/mlflow.db` inside the same gitignored folder.

## Sequence models

**Same protocol as the baselines.** The TCN and LSTM see the same engines, splits,
condition-aware normalisation, capped target, seeds and test protocol (the last observed cycle of
each official test engine) as ridge and XGBoost. Their inputs are the normalised raw sensor
readings rather than the engineered rolling features, so the comparison asks whether a network can
learn those features for itself.

**Windows of 30 cycles, padded with the first reading.** Each prediction reads the 30 cycles
ending at that cycle; 30 matches the shortest FD001 test history. Shorter histories, down to 21
cycles in FD002 and 19 in FD004, are left-padded by repeating the engine's first reading, the best
available estimate of an engine that was still healthy.

**Architectures fixed in advance, not tuned.** TCN: four residual blocks of causal dilated
convolutions (dilations 1, 2, 4 and 8, kernel 3, 64 channels), a receptive field of 61 cycles that
covers the window. LSTM: two layers of 64 units. Both use dropout 0.2, which the uncertainty study
reuses for MC dropout, Adam on the squared error of RUL/125, and early stopping on validation RMSE
with a patience of 15 epochs. Neither network was tuned: tuning only the deep models would bias the
comparison against the baselines, which were not tuned beyond a small ridge penalty grid either.

**Classical versus deep: XGBoost wins on every subset.** On per-seed test RMSE the TCN trails
XGBoost by 2.55, 1.75, 1.93 and 2.65 cycles on FD001 to FD004 and the LSTM by 2.38, 2.08, 1.17 and
1.76; every gap is significant under a Welch t-test (p ≤ 0.03, `reports/deep_vs_xgboost.csv`). The
closest call is the LSTM on FD003 (p = 0.025). The deep models' spread across seeds is 2 to 7 times
XGBoost's, so part of their deficit is instability rather than bias. Two caveats keep this from
being a verdict on deep learning for RUL in general: neither network was tuned, and seed variance
only captures training randomness, not how the ranking would move on a different set of test
engines. What it does show is that rolling features with gradient boosting are a demanding
baseline, and a deep model has to earn its extra complexity here rather than be assumed to.

## Transfer between regimes

**Source FD001, targets FD003 and FD004.** FD001 has one operating regime and one fault mode.
FD003 keeps the single regime but adds a second fault mode (fan degradation); FD004 adds both the
second fault mode and six regimes. The two targets separate the effect of a new failure mechanism
from that of new operating conditions.

**One shared input set.** A network trained on FD001 can only read the sensors FD001 kept, so all
three subsets use the sensors that survive screening in every one of them. Each subset is
normalised with its own training engines' statistics, per regime for FD004. That uses target sensor
readings but no target labels, so the zero-shot condition never sees a target RUL.

**Model chosen before any transfer run.** The transfer study uses the LSTM, which had the lower
FD001 validation RMSE in the sequence study: 12.95 against 14.66 for the TCN, averaged over five
seeds (`val_rmse` in `reports/sequence_per_seed.csv`). The choice used validation engines only; no
test result was consulted.

**Four conditions, five seeds each.** Source-only applies the FD001 model to the target's test
engines unchanged. Target-only trains from scratch on the target. Fine-tuned starts from the FD001
model of the same seed and updates every weight at a learning rate of 0.0003, a third of the
from-scratch rate, so the source features are adjusted rather than overwritten. Target-only and
fine-tuned are also run on 10%, 25% and 50% of the target's training engines. Budgets are drawn
per seed and nested, so a larger budget only ever adds engines and the curve compares like with
like. Every condition early-stops on the full target validation set; a real low-data project would
have fewer validation engines as well, which this setup does not model, but it treats both arms the
same way.

**Transfer outcome: a small gain that shrinks with target data, weakly supported.** Using the FD001
model unchanged fails on both targets (test RMSE 24.65 on FD003 and 26.60 on FD004,
`reports/transfer.csv`), worse than training from scratch on the smallest target budget. For FD003
the likely reason is visible in `reports/sensor_screening.csv`: sensors 7, 12, 15, 20 and 21 track
RUL strongly in FD001 (Spearman magnitude 0.67 to 0.71) but barely at all in FD003 (at most 0.07),
most likely because the second fault mode, fan degradation, moves them differently from the HPC
degradation FD001 was trained on. Fine-tuning beats training from scratch at every budget, by 0.9
to 3.3 cycles on FD003 and 0.2 to 1.3 on FD004, and the gap shrinks as target data grows. With five
seeds per condition only FD003 at 25% of the engines is significant (Welch p = 0.037). The honest
summary: pretraining on FD001 is worth doing when only a few dozen target engines exist, adds
little beyond half the target fleet, and never replaces target data. The target-only arm uses the
14 sensors shared by all three subsets, so its full-budget numbers differ slightly from the
sequence study, which gave FD003 and FD004 their own larger sensor sets.

## Uncertainty

**Quantile regression for XGBoost.** One booster trained with XGBoost's quantile loss predicts the
2.5th to 97.5th percentiles needed for central 50%, 80%, 90% and 95% intervals. Quantiles are
sorted per prediction, because quantiles fitted together can still cross.

**MC dropout for the sequence models.** The trained TCN and LSTM are each run 100 times with
dropout left on, and the spread of the draws gives the interval. MC dropout only captures model
uncertainty, not the noise in the data, so it was expected to under-cover; reporting it shows by
how much.

**Coverage measured on test engines, plus a conformal correction.** Coverage is the share of test
engines whose capped true RUL falls inside the interval. Neither method guarantees its nominal
level, so every interval is also widened by split conformal calibration: the margin is the
ceil((n + 1) * level)-th smallest nonconformity score over validation rows, which guarantees the
nominal coverage when calibration and test points are exchangeable. They are not quite: validation
rows cover every cycle of validation engines, test points are the last observed cycle of test
engines. The tables report the coverage actually achieved, raw and calibrated.

**Coverage outcome.** XGBoost's nominal 90% interval covers 83% to 93% of test engines; the sequence
models' MC dropout intervals cover 44% to 64%, as expected from a method that ignores the noise in
the data (`reports/uncertainty.csv`). Conformal calibration lifts MC dropout to 81% to 94% but
leaves XGBoost unchanged on FD001, FD003 and FD004, because its calibration margin comes out at
zero: many validation rows are early-life cycles at the 125-cycle cap, where the interval's upper
end sits on the cap and the nonconformity score is exactly zero. Calibrating on validation cut
points that resemble the test engines would address that. It was deliberately not tried after the
fact, because retuning the calibration once test coverage is known would turn the test set into a
tuning set.

## Maintenance decision

**Two-parameter cost model.** An unplanned failure costs C_f. Maintenance before failure costs C_p
for every cycle of remaining life thrown away. Only the ratio C_f / C_p matters, and it has units
of cycles: a failure costs as much as discarding that many cycles of healthy life. The headline
uses 100 cycles, about half the median FD001 lifetime; the sensitivity analysis spans 10 to 1,000.

**Alarm policy, tuned the same way it is evaluated.** An engine is serviced at the first cycle
whose predicted RUL is at or below the threshold. Test engines stop before failure, so for an
engine that has not alarmed by its last observed cycle the policy keeps its last booking: service
falls due when the last available prediction would reach the threshold, and the engine fails if its
true remaining life runs out first. The threshold is tuned on validation engines under exactly that
protocol: each validation history is cut at every cycle in turn, the same booking rule is applied at
each cut, and the outcomes are averaged per engine. Replaying the validation histories in full
instead would tune the threshold where predictions are sharpest, just before failure, and choose a
margin far too thin for the noisier single predictions most test engines are booked on. The
protocol is pessimistic for the predictive policy, since a real operator keeps predicting and later
predictions are sharper.

**References.** Run to failure costs C_f per engine by definition. The fixed schedule services every
engine at one age, tuned on training-engine lifetimes to minimise expected cost at the same ratio.

**Point prediction or interval bound.** The interval-aware variant alarms when the lower bound of
the calibrated 90% interval reaches the threshold, which pulls uncertain engines earlier. Both
variants have their thresholds tuned on validation engines, so the comparison is fair.

**Predictions from XGBoost.** The policy runs on XGBoost, the most accurate model on every subset,
with its quantile model supplying the interval bound.

**Decision outcome.** At a failure cost of 100 cycles, alarming on predicted RUL cuts expected cost
per test engine by 61% to 73% against running to failure and by 44% to 60% against the best fixed
schedule, and alarming on the lower bound of the calibrated interval is cheaper again on all four
fleets (`reports/decision_analysis.csv`). The tuned margins are modest: 11 to 21 cycles of predicted
RUL for the point alarm, and 2 to 4 for the interval alarm, whose lower bound already sits well
below the prediction. The fixed schedule does worst on FD004, whose lifetimes vary most: it has to
service every engine at age 225 and still loses 46% of them to failure. Prediction-driven
maintenance stops paying when failures are cheap: at a failure cost of 10 cycles it loses to
running to failure on every fleet, and at 20 cycles on FD002 and FD004.
