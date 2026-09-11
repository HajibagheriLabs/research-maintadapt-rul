# Decisions

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
