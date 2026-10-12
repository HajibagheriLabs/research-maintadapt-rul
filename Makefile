ifeq ($(OS),Windows_NT)
PY := ".venv/Scripts/python.exe"
else
PY := .venv/bin/python
endif

PYTHON ?= python3.11
TORCH_INDEX ?=
CONFIG ?= configs/default.yaml

# printed by make itself, so `make help` works under cmd and sh alike and before setup
define HELP
Targets (variables in brackets):
  help            list every target with one line each
  setup           create .venv and install the pinned dependencies [PYTHON, TORCH_INDEX]
  lint            ruff lint and format check
  format          ruff format and safe lint fixes
  test            run the test suite
  data            check the raw C-MAPSS files, write configs/splits.json and the dataset report [CONFIG]
  legacy-train    legacy: ridge, XGBoost, TCN and LSTM on every subset and seed, then transfer [CONFIG]
  legacy-eval     legacy: prediction intervals and the alarm-threshold decision analysis [CONFIG]
  legacy-report   legacy: reports/results.csv and the markdown tables of the engineering README [CONFIG]
endef

.PHONY: help setup test lint format data legacy-train legacy-eval legacy-report

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
