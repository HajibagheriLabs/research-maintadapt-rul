ifeq ($(OS),Windows_NT)
PY := ".venv/Scripts/python.exe"
else
PY := .venv/bin/python
endif

PYTHON ?= python3.11
TORCH_INDEX ?=
CONFIG ?= configs/default.yaml
RESEARCH_CONFIG ?= configs/research.yaml
# cross-fitting stream (cpu, gpu or all), fleets such as "FD001 FD002", and an archive:
# archive (the default), smoke, dryrun or a path
STREAM ?= all
FLEETS ?=
ROOT ?=

# printed by make itself, so `make help` works under cmd and sh alike and before setup
define HELP
Targets (variables in brackets):
  help                 list every target with one line each
  setup                create .venv and install the pinned dependencies [PYTHON, TORCH_INDEX]
  lint                 ruff lint and format check
  format               ruff format and safe lint fixes
  test                 run the test suite
  data                 check the raw C-MAPSS files, write configs/splits.json and the dataset report [CONFIG]
  legacy-train         legacy: ridge, XGBoost, TCN and LSTM on every subset and seed, then transfer [CONFIG]
  legacy-eval          legacy: prediction intervals and the alarm-threshold decision analysis [CONFIG]
  legacy-report        legacy: reports/results.csv and the markdown tables of the engineering README [CONFIG]
  research-data        check the raw data against the manifest, write configs/research_folds.json [RESEARCH_CONFIG]
  research-pool        write the frozen model pool configs/research_pool.yaml [RESEARCH_CONFIG]
  research-smoke       FD001, repeat 0, configuration 0 of every family and two controls, 3 epochs, into runs/research/smoke [RESEARCH_CONFIG]
  research-dryrun      time configurations 0-2 on FD001 and FD004, both streams at once; write reports/research/compute_estimate.csv [RESEARCH_CONFIG]
  research-crossfit    fit the pool in every (fleet, repeat, fold); resumable [RESEARCH_CONFIG, STREAM, FLEETS]
  research-leaderboard reference configurations on the legacy split, seeds 0-4; reports/research/leaderboard_*.csv [RESEARCH_CONFIG, STREAM]
  research-progress    finished, failed and pending fits per fleet and family, no metrics [RESEARCH_CONFIG, ROOT]
  research-validate    check every finished fit of an archive [RESEARCH_CONFIG, ROOT]
  research-outcomes    tune policies and record maintenance outcomes into runs/research/derived [RESEARCH_CONFIG, ROOT]
  research-metrics     accuracy metrics as per-engine sums into runs/research/derived [RESEARCH_CONFIG, ROOT]
  research-sanity      check the smoke archive's derived tables; reports/research/smoke/sanity.md [RESEARCH_CONFIG]
  research-confirmatory  E1-E3 and the verdicts; refuses the archive until analysis-frozen [RESEARCH_CONFIG, ROOT]
endef

.PHONY: help setup test lint format data legacy-train legacy-eval legacy-report research-data research-pool research-smoke research-dryrun research-crossfit research-leaderboard research-progress research-validate research-outcomes research-metrics research-sanity research-confirmatory

help:
	$(info $(HELP))
	@cd .

setup:
	$(PYTHON) -m venv .venv
	$(PY) -m pip install --upgrade pip
ifneq ($(TORCH_INDEX),)
	$(PY) -m pip install torch --index-url $(TORCH_INDEX)
endif
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

data:
	$(PY) scripts/prepare_data.py --config $(CONFIG)
	$(PY) scripts/inspect_data.py --config $(CONFIG)

legacy-train:
	$(PY) scripts/train_baselines.py --config $(CONFIG)
	$(PY) scripts/train_sequence.py --config $(CONFIG)
	$(PY) scripts/run_transfer.py --config $(CONFIG)

legacy-eval:
	$(PY) scripts/run_uncertainty.py --config $(CONFIG)
	$(PY) scripts/run_decision.py --config $(CONFIG)

legacy-report:
	$(PY) scripts/make_report.py --config $(CONFIG)

research-data:
	$(PY) scripts/research/verify_data.py --config $(RESEARCH_CONFIG)
	$(PY) scripts/research/make_folds.py --config $(RESEARCH_CONFIG)

research-pool:
	$(PY) scripts/research/make_pool.py --config $(RESEARCH_CONFIG)

research-smoke:
	$(PY) scripts/research/run_crossfit.py --config $(RESEARCH_CONFIG) --smoke

research-dryrun:
	$(PY) scripts/research/run_crossfit.py --config $(RESEARCH_CONFIG) --dry-run

research-crossfit:
	$(PY) scripts/research/run_crossfit.py --config $(RESEARCH_CONFIG) --stream $(STREAM) $(if $(FLEETS),--fleets $(FLEETS))

research-leaderboard:
	$(PY) scripts/research/run_leaderboard.py --config $(RESEARCH_CONFIG) --stream $(STREAM)

research-progress:
	$(PY) scripts/research/progress.py --config $(RESEARCH_CONFIG) $(if $(ROOT),--root $(ROOT))

research-validate:
	$(PY) scripts/research/validate_archive.py --config $(RESEARCH_CONFIG) $(if $(ROOT),--root $(ROOT))

research-outcomes:
	$(PY) scripts/research/build_outcomes.py --config $(RESEARCH_CONFIG) $(if $(ROOT),--root $(ROOT))

research-metrics:
	$(PY) scripts/research/build_metrics.py --config $(RESEARCH_CONFIG) $(if $(ROOT),--root $(ROOT))

research-sanity:
	$(PY) scripts/research/smoke_sanity.py --config $(RESEARCH_CONFIG)

research-confirmatory:
	$(PY) scripts/research/run_confirmatory.py --config $(RESEARCH_CONFIG) $(if $(ROOT),--root $(ROOT))
