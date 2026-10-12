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

## Model registry

**One interface.** `fit(family, params, fold_data, seed, device, training)` in
`src/maintadapt/research/models/registry.py` trains any configuration of the pool and predicts every
cycle of inner_val, outer_test and official_test, clipped to [0, cap]. `FoldData`
(`src/maintadapt/research/fold_data.py`) runs the legacy `prepare_subset` with inner_fit as its
training engines, so regimes, sensor screening and z-scores never see another engine. inner_val
serves only early stopping (XGBoost and the networks); ridge, k-NN and the forest do not use it.
Epochs, patience and gradient clipping come from `training` in `configs/research.yaml`, so a smoke
run can shorten training without touching the frozen pool; batch size, learning rate and weight
decay come from the configuration. The legacy `train_model` now accepts `checkpoint=None` and then
keeps the best state in memory only.

**CNN after Li, Ding and Sun (2018).** Kept from the paper: each window is a one-channel image of
cycles by sensors; kernels span 10 cycles and one sensor, so sensors meet only in the
fully-connected layer; four layers of 10 filters, then one filter of 3 cycles; zero padding that
keeps the length; tanh; Xavier-normal initialisation; flatten, dropout 0.5, a fully-connected layer
of 100 tanh units and one linear output; Adam at 1e-3; batches of 512; RUL capped at 125. Our shared
protocol differs from theirs in five ways, and all ten families follow it:
1. inputs are z-scores within regime fitted on inner_fit engines, not min-max scaling to [-1, 1]
   per operating mode;
2. training stops early on inner_val RMSE (at most 100 epochs, patience 15) at a constant learning
   rate, instead of 250 epochs with the rate dropped from 1e-3 to 1e-4 for the last 50;
3. the loss is the MSE of RUL/125, with gradient clipping at 1.0 and Adam weight decay 1e-4;
4. the reference window is 30 cycles on every fleet (theirs: 30, 20, 30 and 15 on FD001 to FD004),
   and windows are left-padded with the engine's first reading so that every cycle is predicted;
5. the inputs are the sensors that pass screening on the fold's inner_fit engines, not a fixed
   list of 14.
These settings of the original are taken from its restatement by Javanmardi and Hüllermeier (2023,
arXiv:2212.14612), who reuse its architecture and parameters; the restatement does not mention the
initialisation. The original full text could not be opened when this entry was written, so the
list should be checked against its Sections 2-3.

**Transformer and MLP.** Neither follows a particular paper. Transformer: a linear projection of the
readings, a learned positional embedding, PyTorch encoder layers (post-norm, ReLU, `batch_first`)
and a prediction from the last cycle's representation; it needs no causal mask because every window
ends at the predicted cycle. MLP: ReLU layers with dropout on the rolling features, standardised by
a scaler fitted on inner_fit.

**What "identical" means in the tests.** The causality, alignment and leakage tests compare
predictions up to summation order. Float64 families agree to a relative 1e-12: ridge can differ in
the last bit because BLAS sums rows in blocks whose boundaries move when rows are removed. The
float32 networks agree to 1e-5 on their own output scale, RUL/125. The forest predicts with one
thread: scikit-learn adds the tree outputs of several threads in the order they finish, which
changed the last bits between two identical calls. KMeans regime centres also differ in the last
bits between identical runs, so the leakage test compares centres to 1e-12 and the regime of every
row exactly.

## Cross-fitting runs

**Archive.** `src/maintadapt/research/archive.py` stores each fit as
`runs/research/archive/<fleet>/<family>/<config_id>/r<repeat>_f<fold>.parquet`: unit, cycle,
uncapped RUL, split and prediction for every cycle of every inner_val, outer_test and official test
engine. A JSON sidecar beside it holds the fit's ids, seed, status, attempts, timings, validation
RMSE, package versions, hardware and git commit. Both files are written under a temporary name,
flushed to disk and renamed, the sidecar last, so a fit is finished exactly when its sidecar
exists and an interrupted run loses at most the fit in progress. `make research-validate` checks
every finished fit against the raw data, the committed folds and the frozen pool.

**Seeds and failures.** The seed of a fit is the first 31 bits of
sha256("fleet|repeat|fold|config_id"). A fit that raises an error or returns non-finite
predictions is retried once with seed + 1000 and then recorded as failed (PREREGISTRATION.md
section 9). A rerun skips failed fits as well as finished ones, so no fit is attempted more than
twice unless its sidecar is deleted by hand, which would be logged in DEVIATIONS.md.

**Two streams, with CPU load kept moderate.** The CPU stream fits ridge, k-NN and the random forest,
which scikit-learn runs on the CPU only, with half the logical CPUs (`training.cpu_jobs: null`). The
GPU stream fits XGBoost, XGBoost on raw readings and the five networks on the GPU with two CPU
threads (`training.gpu_threads`). Every fit also holds the BLAS and OpenMP thread pools to its
thread count with threadpoolctl; left alone they size themselves to every logical CPU whatever
`n_jobs` says. The runs last more than a day, and this keeps part of the CPU idle throughout.
XGBoost builds its histograms on the GPU with the same hyperparameters, including early stopping
on inner_val; its trees differ slightly from those of the CPU build, and the same seed gives the
same trees.

**Controls.** The smoke test adds two configurations outside the pool: `control_oracle` predicts
the capped truth and `control_constant` predicts 125 at every cycle. They exist for end-to-end
checks of the decision layer and never enter the confirmatory archive.

**R = 3 by the pre-registered rule.** The dry run of 2026-10-12 (`make research-dryrun`) timed
configurations 0, 1 and 2 of every family on FD001 and FD004 (repeat 0, fold 0) with both streams
running at once, then scaled FD003 from FD001 and FD002 from FD004 by their inner_fit rows.
`reports/research/compute_estimate.csv` projects 6.3 hours for the CPU stream and 25.6 hours for
the GPU stream at R = 3 (4.2 and 17.0 at R = 2). Both are below the 72 hours of
PREREGISTRATION.md section 4.4, so R = 3 and no configuration is dropped: 6,780 fits.
`configs/research.yaml` records `crossfit.repeats: 3` and `drop_late_configs: false`.

**Conventional leaderboard.** `make research-leaderboard` fits the reference configuration of
every family on the engineering project's split (`configs/splits.json`, regimes clustered with its
split seed 42) with seeds 0-4. It writes predictions to `runs/research/leaderboard`, outside the
confirmatory archive, and compares ridge, XGBoost, the LSTM and the TCN with `reports/baselines.csv`
and `reports/sequence.csv`. Two gaps are expected by design. Ridge's reference penalty is 1, while
the engineering project chose 1, 0.1, 0.1 and 100 on FD001 to FD004 by validation
(`reports/baselines_per_seed.csv`), so only FD001 should match. XGBoost now runs on the GPU, so its
differences include the change of implementation as well as seed noise.

## Decision layer and statistics

`src/maintadapt/research/policies.py`, `costs.py` and `stats.py` implement sections 5 and 6 of
PREREGISTRATION.md. Where those sections leave a detail open, it was fixed here, before any
decision metric exists:
- **Ties.** The candidate list keeps all 401 quantiles, duplicates included, between -inf and
  +inf. Among the cheapest candidates the median one wins, and of an even number the lower middle
  one. Under R2, alphas that reach the same cost rate go to the larger alpha, the lighter
  smoothing. Age replacement considers every distinct training lifetime and "never", which is the
  complete set of possible optima, with the same median rule.
- **No alarm.** An engine whose score never reaches the threshold carries a sentinel alarm time
  beyond any life plus lead time, so it fails without a special case.
- **Bootstrap p-values.** Twice the smaller tail share of the bootstrap differences, each tail
  counted as (count + 1) / (B + 1), so 2,000 replicates never report less than 0.001.
- **Selection.** Choosing by a validation metric takes the lowest value in each (repeat, fold);
  ties go to the configuration listed first in the pool.
- **The capped truth as a predictor.** It reaches S = 1 at every lead time below the cap: the
  threshold L puts the alarm at T - L and maintenance at T. At L = 125 it cannot, because the
  capped truth is 125 from cycle 1 until T - 125 and no threshold can tell those cycles apart. The
  registered lead times (0, 10, 20) are far below the cap.
- **Invariance.** The tests confirm that R1 outcomes on engines the policy never saw are
  unchanged by x + 15, 1.25x, x^2/125 and exp(x/50) applied to every prediction, and R2 outcomes
  by affine maps, so a monotone recalibration cannot change a model's maintenance value.

## Derived tables

`make research-outcomes` and `make research-metrics` turn an archive into the tables the analysis
reads, in `runs/research/derived/<archive>/<fleet>/`, so that no analysis reads predictions again:
- **Outcomes.** Every configuration, repeat and fold is tuned three ways: on its inner_val engines
  (validation, the primary), on the outer_test engines of the repeat's other folds (crossfit) and
  on every outer_test engine of the repeat (oracle). Each tuned policy is applied to the fold's
  outer_test engines; the table keeps theta, alpha, the in-sample cost rate (the validation cost
  of section 5.8) and every engine's failure, cycles used and waste. Age replacement is tuned per
  fold on the outer-train lifetimes and applied to the same outer_test engines.
- **Metrics.** Each fit's accuracy is stored as per-engine sums and counts (squared errors,
  NASA terms, jitter steps, late-phase errors), so every metric can be recomputed under the
  engine bootstrap. Jitter needs five qualifying cycles on an engine, a rule applied when the
  sums are combined. The 20 truncation points of each outer_test engine are drawn once per
  engine from `default_rng([2026, fleet number, unit])`, so every configuration, repeat and fold
  is scored at the same cycles.
- **Exclusions and the guard.** A configuration with a failed fit in any fold of a fleet is left
  out of that fleet and listed in `excluded.csv`; a pending fit stops the build. On the
  confirmatory archive both builders refuse to run until `make research-validate` passes and the
  tag `analysis-frozen` exists.

The smoke archive passes every check in `reports/research/smoke/sanity.md`: the capped-truth
control reaches S = 1 under all three tunings, and the constant control stays below 0.

## Confirmatory analysis

`src/maintadapt/research/analysis.py` and `scripts/research/run_confirmatory.py` compute E1-E3
and apply the interpretation rules of PREREGISTRATION.md section 6, which they quote verbatim (a
test holds them to the file). The details that section leaves to the implementation:
- **One formula for point and interval.** Every quantity is a function of per-engine outcomes or
  errors weighted by bootstrap counts; the point estimate passes a count of one per engine. D of a
  configuration is 1 minus S pooled over the five folds of each repeat and averaged over repeats;
  leaderboard RMSE and jitter are averaged over a configuration's fits.
- **Bootstrap.** 2,000 replicates per fleet: one multinomial draw of training engines from
  `default_rng([2026, fleet number, 0])`, shared by every configuration, repeat and selection,
  and one of official test engines from `default_rng([2026, fleet number, 1])`. A replicate in
  which tau-b is undefined (all values tied) is left out of the interval and the count of valid
  replicates is reported.
- **Jitter.** An official test engine enters jitter when it has at least five cycles whose previous
  true RUL is below 125; that depends on the truth alone, so every fit uses the same engines.
- **Verdict arithmetic.** "Interval excluding 0" means a lower bound above 0 or an upper bound
  below 0; "within [-0.02, 0.02]" includes the end points; the band cut points and every other
  comparison are as written in section 6. The tests check each rule on both sides of its
  boundary.
- **What the tag freezes.** The guard compares with the tagged versions not only `analysis.py`
  and `run_confirmatory.py` but all the code between the archive and the verdicts: `stats.py`,
  `policies.py`, `costs.py`, `derived.py`, `build_outcomes.py` and `build_metrics.py`. The
  outcome and metric builders apply the same guard, so `--exploratory` cannot reach the
  confirmatory archive before the tag either: it reads only derived tables, which cannot be
  built from that archive until then.

The smoke run (`reports/research/smoke/`, from `make research-confirmatory ROOT=smoke`) checks the
pipeline only: one fleet, one repeat, three training epochs for the networks.

**Leaderboard against the engineering project.** `reports/research/leaderboard_vs_legacy.csv`
(from `make research-leaderboard`) reproduces the LSTM and the TCN on all four fleets to within
0.001 cycles of RMSE, and ridge on FD001, where both use a penalty of 1. Ridge differs by design on
FD002 to FD004. XGBoost agrees within twice the standard error of the difference on FD002 to
FD004; on FD001 its RMSE is 0.18 cycles lower (11.96 against 12.14), just beyond twice that
standard error (0.15). The difference is the move of XGBoost to the GPU, whose histogram
algorithm builds slightly different trees with the same hyperparameters.

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
