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
  research-sanity      check the derived tables against the controls; smoke unless ROOT is given [RESEARCH_CONFIG, ROOT]
  research-confirmatory  E1-E3 and the verdicts; refuses the archive until analysis-frozen [RESEARCH_CONFIG, ROOT]
  research-nested      S4: inner cross-fitting of the family representatives into runs/research/archive_nested; resumable [RESEARCH_CONFIG, STREAM]
  research-caps        S5: reference XGBoost and LSTM with training caps 100, 150, 200 and none into runs/research/archive_caps; resumable [RESEARCH_CONFIG]
  research-extra-estimate  project the hours of research-nested and research-caps; reports/research/compute_estimate_secondary.csv [RESEARCH_CONFIG]
  research-extra-validate  check every finished fit of the nested and training-cap archives [RESEARCH_CONFIG]
  research-secondary   S1-S7 and S9 of PREREGISTRATION.md section 7 into reports/research/secondary [RESEARCH_CONFIG]
  research-exploratory analyses outside the plan, labelled EXPLORATORY, into reports/research/exploratory [RESEARCH_CONFIG]
  research-ncmapss-inspect  N-CMAPSS groups, shapes, units, failure modes and the cap; reports/research/ncmapss/inspection.md [RESEARCH_CONFIG]
  research-ncmapss-data  N-CMAPSS per-flight frame from W and X_s, and configs/research_folds_ncmapss.json [RESEARCH_CONFIG]
  research-ncmapss-smoke  configuration 0 of every family and two controls, 3 epochs, on N-CMAPSS repeat 0 [RESEARCH_CONFIG]
  research-ncmapss-dryrun  time configurations 0-2 on N-CMAPSS, both streams; reports/research/ncmapss/compute_estimate.csv [RESEARCH_CONFIG]
  research-ncmapss     fit the pool on N-CMAPSS in every (repeat, fold); resumable [RESEARCH_CONFIG, STREAM]
  research-ncmapss-validate  check every finished N-CMAPSS fit [RESEARCH_CONFIG, ROOT]
  research-ncmapss-derived  N-CMAPSS outcomes and metrics into runs/research/derived [RESEARCH_CONFIG, ROOT]
  research-ncmapss-report  N-CMAPSS E1-E3, secondary and descriptive; reports/research/ncmapss [RESEARCH_CONFIG, ROOT]
  paper-assets         figures, tables, numbers.tex and RESULTS_DIGEST.md into paper_assets [RESEARCH_CONFIG]
  paper-check          recompute E1-E3 of every fleet from the archives with independent code [RESEARCH_CONFIG]
endef

.PHONY: help setup test lint format data legacy-train legacy-eval legacy-report research-data research-pool research-smoke research-dryrun research-crossfit research-leaderboard research-progress research-validate research-outcomes research-metrics research-sanity research-confirmatory research-nested research-caps research-extra-estimate research-extra-validate research-secondary research-exploratory research-ncmapss-inspect research-ncmapss-data research-ncmapss-smoke research-ncmapss-dryrun research-ncmapss research-ncmapss-validate research-ncmapss-derived research-ncmapss-report paper-assets paper-check

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
	$(PY) scripts/research/smoke_sanity.py --config $(RESEARCH_CONFIG) $(if $(ROOT),--root $(ROOT))

research-confirmatory:
	$(PY) scripts/research/run_confirmatory.py --config $(RESEARCH_CONFIG) $(if $(ROOT),--root $(ROOT))

research-nested:
	$(PY) scripts/research/run_crossfit.py --config $(RESEARCH_CONFIG) --nested --stream $(STREAM)

research-caps:
	$(PY) scripts/research/run_crossfit.py --config $(RESEARCH_CONFIG) --caps --stream gpu

research-extra-estimate:
	$(PY) scripts/research/run_crossfit.py --config $(RESEARCH_CONFIG) --estimate

research-extra-validate:
	$(PY) scripts/research/validate_archive.py --config $(RESEARCH_CONFIG) --nested
	$(PY) scripts/research/validate_archive.py --config $(RESEARCH_CONFIG) --caps

research-secondary:
	$(PY) scripts/research/run_secondary.py --config $(RESEARCH_CONFIG)

research-exploratory:
	$(PY) scripts/research/run_exploratory.py --config $(RESEARCH_CONFIG)

research-ncmapss-inspect:
	$(PY) scripts/research/inspect_ncmapss.py --config $(RESEARCH_CONFIG)

research-ncmapss-data:
	$(PY) scripts/research/make_ncmapss.py --config $(RESEARCH_CONFIG)

research-ncmapss-smoke:
	$(PY) scripts/research/run_crossfit.py --config $(RESEARCH_CONFIG) --ncmapss --smoke

research-ncmapss-dryrun:
	$(PY) scripts/research/run_crossfit.py --config $(RESEARCH_CONFIG) --ncmapss --dry-run

research-ncmapss:
	$(PY) scripts/research/run_crossfit.py --config $(RESEARCH_CONFIG) --ncmapss --stream $(STREAM)

research-ncmapss-validate:
	$(PY) scripts/research/validate_archive.py --config $(RESEARCH_CONFIG) --ncmapss $(if $(ROOT),--root $(ROOT))

research-ncmapss-derived:
	$(PY) scripts/research/build_ncmapss.py --config $(RESEARCH_CONFIG) $(if $(ROOT),--root $(ROOT))

research-ncmapss-report:
	$(PY) scripts/research/run_ncmapss.py --config $(RESEARCH_CONFIG) $(if $(ROOT),--root $(ROOT))

paper-assets:
	$(PY) scripts/research/make_paper_assets.py --config $(RESEARCH_CONFIG)

paper-check:
	$(PY) scripts/research/check_numbers.py --config $(RESEARCH_CONFIG)
