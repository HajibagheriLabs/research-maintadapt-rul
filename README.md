# MaintAdapt — Predictive Maintenance and Remaining Useful Life

When should a turbofan engine come off the wing? Pull it too late and it fails in service, an
unplanned event that costs far more than any shop visit. Pull it too early and every cycle of
healthy life it had left is thrown away. MaintAdapt treats remaining-useful-life (RUL) prediction
on the NASA C-MAPSS fleet as an input to that decision, not as a regression leaderboard: models are
compared under unit-level splits with classical baselines first, their uncertainty is measured
against what actually happened, and their predictions are turned into a cost-optimal alarm
threshold under an explicit, openly stated cost model. The headline number is the maintenance cost
saved per engine against running to failure and against a fixed maintenance schedule.

## How it works

1. **Data** (`src/maintadapt/data.py`). Parse the four C-MAPSS subsets, label every cycle with its
   remaining useful life capped at 125, split engines (never cycles) into train and validation,
   find the operating regimes, and drop sensors that cannot show a trend.
2. **Features** (`features.py`). Normalise each sensor within its regime using training engines
   only, then add trailing means, spreads and slopes over 5 to 30 cycles.
3. **Models** (`baselines.py`, `sequence.py`). Ridge and XGBoost on the rolling features; a TCN and
   an LSTM on 30-cycle windows of normalised readings. Same splits, seeds and test protocol for all
   four.
4. **Transfer** (`transfer.py`). Pretrain on the single-regime FD001 fleet, then fine-tune on FD003
   and FD004 with 10% to 100% of their training engines.
5. **Uncertainty** (`uncertainty.py`). Quantile-regression and MC-dropout intervals, checked for
   coverage on the test engines and corrected by split conformal calibration.
6. **Decision** (`decision.py`). Replay each engine's predictions against an alarm threshold, price
   failures and discarded life, choose the threshold on validation engines, and compare the result
   with running to failure and with a fixed maintenance schedule.

## Results

Every number below is produced by the `make` targets and read from `reports/`; ± is the standard
deviation over five seeds. Sections marked TBD have not been run yet.

### Maintenance cost

TBD: produced by `make eval` (`reports/decision_analysis.md`).

### Remaining-useful-life models

Scored at the last observed cycle of every official test engine, against truth capped at 125
cycles (see [DECISIONS.md](DECISIONS.md)).

Test RMSE in cycles:

| model | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Ridge, rolling features | 14.99 ± 0.00 | 15.96 ± 0.00 | 16.29 ± 0.00 | 18.53 ± 0.00 |
| XGBoost, rolling features | 12.14 ± 0.09 | 12.60 ± 0.05 | 12.49 ± 0.18 | 13.49 ± 0.10 |
| TCN, raw windows | 14.69 ± 0.43 | 14.35 ± 0.30 | 14.42 ± 0.48 | 16.15 ± 0.23 |
| LSTM, raw windows | 14.52 ± 0.62 | 14.68 ± 0.32 | 13.66 ± 0.77 | 15.25 ± 0.32 |

NASA score, summed over test engines, where late predictions cost more than early ones:

| model | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Ridge, rolling features | 347 ± 0 | 1179 ± 0 | 508 ± 0 | 1596 ± 0 |
| XGBoost, rolling features | 214 ± 5 | 690 ± 12 | 262 ± 10 | 819 ± 21 |
| TCN, raw windows | 395 ± 65 | 1059 ± 130 | 553 ± 154 | 1703 ± 213 |
| LSTM, raw windows | 413 ± 52 | 1108 ± 194 | 436 ± 118 | 1408 ± 310 |

XGBoost beats ridge on every subset, by 2.9 to 5.0 cycles of RMSE, and roughly halves the NASA
score. Ridge is deterministic given the split, so its spread across seeds is zero.

**The deep sequence models do not beat XGBoost on any subset.** Trained on raw 30-cycle windows
under the same protocol and without tuning, the TCN and LSTM trail XGBoost by 1.2 to 2.7 cycles of
RMSE, and every gap is significant across seeds; their seed-to-seed spread is also 2 to 7 times
larger. Welch t-test on per-seed test RMSE (`reports/deep_vs_xgboost.csv`):

| subset | TCN minus XGBoost | p | LSTM minus XGBoost | p |
|---|---|---|---|---|
| FD001 | +2.55 | 0.0001 | +2.38 | 0.0009 |
| FD002 | +1.75 | 0.0001 | +2.08 | 0.0001 |
| FD003 | +1.93 | 0.0004 | +1.17 | 0.0254 |
| FD004 | +2.65 | < 0.0001 | +1.76 | 0.0001 |

Normalising each sensor within its operating regime is what makes the multi-regime subsets
tractable (XGBoost, same settings):

| subset | normalisation | RMSE | NASA score |
|---|---|---|---|
| FD002 | condition-aware | 12.60 ± 0.05 | 690 ± 12 |
| FD002 | global | 15.91 ± 0.13 | 1257 ± 31 |
| FD004 | condition-aware | 13.49 ± 0.10 | 819 ± 21 |
| FD004 | global | 17.39 ± 0.09 | 1713 ± 58 |

### Transfer between operating regimes

TBD: produced by `make train` (`reports/transfer.csv`).

### Prediction intervals

TBD: produced by `make eval` (`reports/uncertainty.csv`).

## Setup

Requires Python 3.11 and GNU make. `make setup` creates `.venv` and installs the pinned
dependencies from `pyproject.toml`:

```bash
make setup                                                    # CPU build of PyTorch
make setup TORCH_INDEX=https://download.pytorch.org/whl/cu128 # CUDA build of PyTorch
```

`PYTHON` selects the interpreter used to create the environment (default `python3.11`), for
example `make setup PYTHON="py -3.11"` on Windows. A GPU is optional; only the sequence models and
the transfer study use it.

## Data

The C-MAPSS turbofan degradation data (Saxena et al., PHM 2008) is not redistributed here. Download
data set 6, "Turbofan Engine Degradation Simulation", from the
[NASA Prognostics Center of Excellence data set repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
(or the Kaggle mirror `behrad3d/nasa-cmaps`), unzip it, and place the twelve text files under
`data/raw/CMAPSS/`:

```
data/raw/CMAPSS/train_FD001.txt   test_FD001.txt   RUL_FD001.txt
data/raw/CMAPSS/train_FD002.txt   test_FD002.txt   RUL_FD002.txt
data/raw/CMAPSS/train_FD003.txt   test_FD003.txt   RUL_FD003.txt
data/raw/CMAPSS/train_FD004.txt   test_FD004.txt   RUL_FD004.txt
```

`make data` checks the files, writes the unit-level train/validation splits to
`configs/splits.json`, and produces the dataset report in `reports/`.

## Usage

```bash
make data     # check the raw files, write configs/splits.json, dataset report and figures
make train    # ridge, XGBoost, TCN and LSTM on every subset and seed, then the transfer study
make eval     # prediction intervals and the cost-optimal maintenance policy
make report   # reports/results.csv and the markdown tables quoted in this README
make lint test
```

Every experiment reads `configs/default.yaml` (override with `CONFIG=...`) and writes the exact
config it used to `runs/<experiment>/config.yaml`. Finished runs are recorded in
`runs/<experiment>/progress.csv`, so an interrupted experiment resumes where it stopped as long as
the config is unchanged. Runs are also logged to a local MLflow store:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Limitations

- **Simulated engines.** C-MAPSS comes from a thermodynamic simulation with injected faults and
  additive noise. Real fleets have sensor drift and dropouts, maintenance events that reset
  degradation fully or partly, varying mission profiles, and far fewer run-to-failure histories,
  because engines are usually removed before they fail.
- **Assumed cost ratio.** The cost of an unplanned failure relative to a cycle of discarded life is
  an input, not a measurement. It depends on the operator, the engine type, secondary damage and
  spare availability. The sensitivity analysis shows how the threshold and the saving move with it,
  but a deployment needs the operator's own figures.
- **Simplified maintenance.** Maintenance is priced only through discarded life, is assumed to
  restore the engine fully, and can happen in the cycle the alarm fires. Lead times, shop capacity
  and the fixed cost of a shop visit are left out.
- **Censored test histories.** Test engines stop before failure, so for engines without an alarm
  the decision analysis has to extend the last prediction forward.
- **What real fleet data would change.** Labels would be censored, which calls for survival models
  rather than capped regression. Splits would be by time as well as by engine. Costs could be
  estimated from maintenance records rather than assumed. The threshold would be re-tuned per fleet
  and monitored as engines age, and interval calibration would need the same monitoring.

## License

MIT — see [LICENSE](LICENSE).
