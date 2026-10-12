# Research-MaintAdapt-RUL

Remaining-useful-life (RUL) papers rank models by RMSE and NASA score at the last observed cycle of
the official C-MAPSS test engines. This repository tests whether that ranking predicts the
maintenance value of the same models when their predictions drive a tuned maintenance alarm.

**Status:** in progress — pre-registration draft ([PREREGISTRATION.md](PREREGISTRATION.md)), no
confirmatory results. Changes to the design after registration go to [DEVIATIONS.md](DEVIATIONS.md).

## Provenance

This is a fork of the engineering project maintadapt-rul at tag `fork-point`. Its README, which
describes the engineering pipeline and its results, is kept in [docs/legacy/](docs/legacy/README_engineering.md),
and its decision log continues in [DECISIONS.md](DECISIONS.md). The legacy pipeline still runs
unchanged inside this repository.

## Corrections

Three claims of the engineering project are corrected, and one is put in context, in
[docs/legacy/ERRATA.md](docs/legacy/ERRATA.md): the XGBoost interval described as calibrated, the
decision analysis on truncated test engines, and the meaning of its ± values.

## Reproducing the legacy results

Requires Python 3.11 and GNU make; a GPU is optional. Place the twelve C-MAPSS text files under
`data/raw/CMAPSS/` (see [Data](docs/legacy/README_engineering.md#data) in the legacy README), then:

```bash
make setup TORCH_INDEX=https://download.pytorch.org/whl/cu128  # or plain `make setup` for CPU PyTorch
make data            # check the raw files, write configs/splits.json and the dataset report
make legacy-train    # ridge, XGBoost, TCN and LSTM on every subset and seed, then the transfer study
make legacy-eval     # prediction intervals and the alarm-threshold decision analysis
make legacy-report   # reports/results.csv and the markdown tables of the legacy README
make lint test
```

On Windows, pass the interpreter explicitly, for example `make setup PYTHON="py -3.11"`. `make help`
lists every target.

## License

MIT — see [LICENSE](LICENSE).
