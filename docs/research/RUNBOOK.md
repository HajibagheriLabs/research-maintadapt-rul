# Runbook: the confirmatory cross-fitting runs

Everything here runs from the repository root. The runs fit every configuration of the frozen pool
in every (fleet, repeat, outer fold) and store one prediction per cycle for every inner_val,
outer_test and official test engine. They compute no test or decision metric; nothing may compute
one from `runs/research/archive` before the tag `analysis-frozen` exists (CLAUDE.md, rule 8).

## The full order

| step | command | time |
|---|---|---|
| 1 | `make research-crossfit STREAM=cpu` and `make research-crossfit STREAM=gpu`, at once | 6.3 h (CPU) and 25.6 h (GPU) at R = 3 |
| 2 | `make research-validate` | minutes |
| 3 | `make research-outcomes` | about 20 minutes |
| 4 | `make research-metrics` | about 5 minutes |
| 5 | `make research-confirmatory` | Prompt 7 |

The cross-fitting hours are the projection in `reports/research/compute_estimate.csv`. Steps 3 and
4 are scaled from their smoke runs (0.45 s per configuration and repeat for the outcomes, 0.06 s
per fit for the metrics, in `runs/research/derived/smoke/FD001/build_*.json`); FD002 and FD004
have about three times FD001's engines. Steps 3 to 5 refuse to run on the confirmatory archive
until `make research-validate` passes and the tag `analysis-frozen` exists, which is created
after Prompt 7 and before both streams finish. They write `runs/research/derived/archive/`.

The same chain on the smoke archive needs no tag and takes a few minutes:
`make research-smoke`, `make research-validate ROOT=smoke`, `make research-outcomes ROOT=smoke`,
`make research-metrics ROOT=smoke`, then `make research-sanity`, which writes
`reports/research/smoke/sanity.md`.

## Before the first launch

1. `make lint test` passes and `git status` shows no changes to tracked files. Every fit records the
   commit it ran from, with `+dirty` when tracked files had changed.
2. `make research-data` reports that the raw files match `docs/research/data_manifest.json`; it
   rewrites `configs/research_folds.json` byte for byte.
3. `configs/research.yaml` holds the number of repeats R (`crossfit.repeats`) and
   `crossfit.drop_late_configs` as set by the dry run under PREREGISTRATION.md section 4.4. The
   outcome and its source file are in DECISIONS.md.
4. `make research-progress` shows the expected total for that R: 6,780 fits for R = 3 or 4,520
   for R = 2.
5. The computer must not sleep for the whole run: on Windows set *Sleep* to *Never* under
   Settings > System > Power; a sleeping machine pauses both streams.

## Two streams at once

The **CPU stream** fits ridge, k-NN and the random forest, which scikit-learn runs on the CPU
only, with `training.cpu_jobs` threads (default: half the logical CPUs). The **GPU stream** fits
XGBoost, XGBoost on raw readings, the MLP, LSTM, TCN, CNN and transformer on the GPU with
`training.gpu_threads` CPU threads. Every fit also holds the BLAS and OpenMP thread pools to its
thread count, so the two streams together keep part of the CPU idle over a run of several days.
Run one process per stream, both at the same time; never two processes of the same stream, which
would fit the same fits twice.

### Linux or WSL, in tmux

A WSL checkout needs its own `make setup` (and its own copy of `data/raw/CMAPSS`); do not share the
Windows `.venv`.

```bash
mkdir -p runs/research
tmux new-session -d -s crossfit "make research-crossfit STREAM=cpu 2>&1 | tee -a runs/research/cpu.log"
tmux split-window -t crossfit "make research-crossfit STREAM=gpu 2>&1 | tee -a runs/research/gpu.log"
tmux attach -t crossfit
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t crossfit`.

### Windows, two terminals

In two PowerShell windows opened at the repository root:

```powershell
make research-crossfit STREAM=cpu | Tee-Object -FilePath runs\research\cpu.log -Append
```

```powershell
make research-crossfit STREAM=gpu | Tee-Object -FilePath runs\research\gpu.log -Append
```

A stream may also be split across machines by fleet, for example
`make research-crossfit STREAM=gpu FLEETS="FD001 FD003"` on one and `FLEETS="FD002 FD004"` on the
other; copy each machine's `runs/research/archive/<fleet>` folders back into one archive afterwards.

## Resuming

Run the same command again. A fit is finished once its JSON sidecar exists; finished fits are
skipped, and the first line of each stream says how many are planned and the last how many were
skipped. Predictions and sidecars are written to a `.tmp` name and then renamed, so Ctrl-C, a crash
or a power cut loses at most the fit in progress, which reruns from scratch with the same seed. A
stray `.tmp` file is harmless; the next write of that fit replaces it.

A fit that raises an error or returns non-finite predictions is retried once with seed + 1000 and
then recorded as failed (PREREGISTRATION.md section 9); failed fits are not retried by a rerun. To
retry one whose failure had nothing to do with the model (the machine ran out of memory because of
another program, say), delete its `.json` sidecar, rerun, and log the retry in DEVIATIONS.md.

## Checking progress without reading metrics

- `make research-progress`: finished, failed and pending fits per fleet, family and stream against
  the expected totals, and every failed fit with its ids. It reads sidecar statuses only.
- `make research-validate`: every finished fit holds each engine of its splits once, with cycles
  1..T, the true RUL, finite predictions in [0, 125] and the hash of a pool configuration. It reads
  the files only to check their structure.
- The console prints one line per fit with its ids, seconds, inner-validation RMSE and status.
  Inner-validation RMSE is the only accuracy figure allowed before `analysis-frozen`. Do not open
  the parquet files, plot predictions or compute any official-test, outer-test or cost figure from
  the archive until then.

## Disk

The archive takes about 1.5 GB for R = 3 (1.0 GB for R = 2). Each fit stores every cycle of its
inner_val, outer_test and official test engines: on average 21,344 rows on FD001, 55,491 on FD002,
26,479 on FD003 and 65,711 on FD004 (from the committed folds), at about 5 bytes per row after
compression (4.7 to 5.1 in the dry-run files), plus a 1 KB sidecar. The leaderboard adds well under
0.1 GB. Keep 5 GB free so that a full disk never interrupts a write.

## Time

`reports/research/compute_estimate.csv` projects 6.3 hours for the CPU stream and 25.6 hours for
the GPU stream at R = 3, from the dry run of 2026-10-12. The CPU stream finishes first; the GPU
stream's largest shares are the transformer (8.4 hours), the TCN (4.9) and the CNN (4.1). Check
progress a few times a day; there is nothing to do while it runs.

## Extra fits of the secondary analyses

Two secondary analyses need fits of their own, which go to archives beside the confirmatory one
and never into it. They choose and repeat confirmatory fits, so they run after both streams have
finished:

| step | command | time |
|---|---|---|
| 1 | `make research-nested STREAM=cpu` and `make research-nested STREAM=gpu`, at once | 0.16 h (CPU) and 0.84 h (GPU) |
| 2 | `make research-caps`, after the GPU stream of step 1 | 0.34 h |
| 3 | `make research-extra-validate` | about a minute |
| 4 | `make research-secondary` | about 10 minutes |

- `research-nested` (S4): for repeat 0 of FD001 and FD003, each family's configuration with the
  lowest inner_val RMSE in each outer fold, fitted on 4 inner folds of that fold's outer-train
  engines, into `runs/research/archive_nested/outer<k>/`. The inner folds are written to
  `runs/research/archive_nested/nested_folds.json`.
- `research-caps` (S5): configuration 0 of XGBoost and the LSTM with training caps 100, 150, 200
  and none, repeat 0, every fleet and fold, into `runs/research/archive_caps/<cap>/`. Both
  families run on the GPU.

The hours are the projection in `reports/research/compute_estimate_secondary.csv`
(`make research-extra-estimate`). Both targets resume like the cross-fitting runs. Until the fits
are complete, `make research-secondary` writes every other item and lists S4-nested and S5 as
pending in `reports/research/secondary/README.md`. `make research-exploratory` needs no extra
fits; its first run reads the archive once (about five minutes) and keeps per-fit tables in
`runs/research/derived/archive/<fleet>/`.

## N-CMAPSS (secondary, PREREGISTRATION.md section 8)

The N-CMAPSS files go in `data/raw/NCMAPSS/data_set/` (NASA's archive, extracted as it comes).
The cap and every preprocessing choice were committed to DECISIONS.md before the first fit.

| step | command | time |
|---|---|---|
| 1 | `make research-ncmapss-inspect` (writes `reports/research/ncmapss/inspection.md`) | under a minute |
| 2 | `make research-ncmapss-data` (per-flight frame and folds) | about a minute |
| 3 | `make research-ncmapss STREAM=cpu` and `make research-ncmapss STREAM=gpu`, at once | 0.40 h (CPU) and 2.48 h (GPU) |
| 4 | `make research-ncmapss-validate` | about a minute |
| 5 | `make research-ncmapss-derived` | a few minutes |
| 6 | `make research-ncmapss-report` (writes `reports/research/ncmapss/`) | under a minute |

All six steps have been run and their outputs committed. The hours are the
projection in `reports/research/ncmapss/compute_estimate.csv` (`make research-ncmapss-dryrun`).
The fits go to `runs/research/archive_ncmapss` and resume like the cross-fitting runs.
`make research-ncmapss-smoke` checks the pipeline in about two minutes.

## Paper assets

After every report above exists:

| step | command | time |
|---|---|---|
| 1 | `make paper-assets` (writes `paper_assets/`) | under a minute |
| 2 | `make paper-check` | about five minutes |

`make paper-assets` reads the committed reports, the derived tables and a few archive files.
It writes the figures (PDF with embedded fonts and PNG previews), the booktabs tables,
`numbers.tex` and `RESULTS_DIGEST.md`; the few numbers it computes itself go to
`paper_assets/computed/`. `make paper-check` recomputes the point estimates of E1-E3 for FD001-FD004
and N-CMAPSS from the archives, the fold files and the pool with
`src/maintadapt/research/recheck.py`, which imports nothing from maintadapt. It fails on any
difference above 1e-9 from the reports or from the macros of `numbers.tex`.

`make research-package` writes `dist/research-maintadapt-rul-predictions.tar.gz`: the
confirmatory archive, the archives of the nested, training-cap and N-CMAPSS fits, the fold files,
the pool and the data manifest, with `README_ARCHIVE.md`, `LICENSE_ARCHIVE.md` (CC BY 4.0) and a
SHA-256 for every file inside, plus `dist/<file>.sha256` and a copy of `README_ARCHIVE.md` beside
it. The tarball carries no times or owners, so packaging the same files twice gives the same
bytes. `dist/` is not committed; the file goes to Zenodo.

## Smoke test, dry run and leaderboard

- `make research-smoke`: FD001, repeat 0, all five folds, configuration 0 of every family and the
  two controls, with 3 epochs for the networks, into `runs/research/smoke`; about two minutes.
  Check it with `make research-validate ROOT=smoke`.
- `make research-dryrun`: times configurations 0-2 of every family on FD001 and FD004 (repeat 0,
  fold 0) with both streams running at once, writes `reports/research/compute_estimate.csv` and
  prints the outcome of the rule in PREREGISTRATION.md section 4.4.
- `make research-leaderboard`: the reference configuration of every family on the legacy split,
  seeds 0-4, into `runs/research/leaderboard` (never the confirmatory archive). It writes
  `reports/research/leaderboard_reference.csv` and `leaderboard_vs_legacy.csv`; run it with
  `STREAM=cpu` and `STREAM=gpu` in two terminals to halve the time. Resumable like the cross-fit.

## When both streams have finished

`make research-progress` shows no pending fit and `make research-validate` reports the archive
valid. Then follow steps 3 to 5 of the full order. Configurations that failed in any fold of a
fleet are excluded from that fleet and listed in `runs/research/derived/archive/<fleet>/
excluded.csv` (PREREGISTRATION.md section 9). The archive stays where it is: never delete or edit
`runs/research/archive`.
