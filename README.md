# MaintAdapt — Predictive Maintenance and Remaining Useful Life

When should a turbofan engine come off the wing? Pull it too late and it fails in service, an
unplanned event that costs far more than any shop visit. Pull it too early and every cycle of
healthy life it had left is thrown away. MaintAdapt treats remaining-useful-life (RUL) prediction
on the NASA C-MAPSS fleet as an input to that decision, not as a regression leaderboard: models are
compared under unit-level splits with classical baselines first, their uncertainty is measured
against what actually happened, and their predictions are turned into a cost-optimal alarm
threshold under an explicit, openly stated cost model. The headline number is the maintenance cost
saved per engine against running to failure and against a fixed maintenance schedule.

## Results

TBD — filled in as each stage of the pipeline is run.

## Setup

TBD

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

TBD

## License

MIT — see [LICENSE](LICENSE).
