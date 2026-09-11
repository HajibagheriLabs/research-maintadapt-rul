ifeq ($(OS),Windows_NT)
PY := ".venv/Scripts/python.exe"
else
PY := .venv/bin/python
endif

PYTHON ?= python3.11
TORCH_INDEX ?=
CONFIG ?= configs/default.yaml

.PHONY: setup test lint format data train eval report

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

train:
	$(PY) scripts/train_baselines.py --config $(CONFIG)
	$(PY) scripts/train_sequence.py --config $(CONFIG)
	$(PY) scripts/run_transfer.py --config $(CONFIG)

eval:
	$(PY) scripts/run_uncertainty.py --config $(CONFIG)
	$(PY) scripts/run_decision.py --config $(CONFIG)

report:
	$(PY) scripts/make_report.py --config $(CONFIG)
