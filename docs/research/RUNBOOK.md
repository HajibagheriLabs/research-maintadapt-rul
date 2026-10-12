# Runbook: the confirmatory cross-fitting runs

Everything here runs from the repository root. The runs fit every configuration of the frozen pool
in every (fleet, repeat, outer fold) and store one prediction per cycle for every inner_val,
outer_test and official test engine. They compute no test or decision metric; nothing may compute
one from `runs/research/archive` before the tag `analysis-frozen` exists (CLAUDE.md, rule 8).

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

## Smoke test, dry run and leaderboard

- `make research-smoke`: FD001, repeat 0, all five folds, configuration 0 of every family and the
  two controls, with 3 epochs for the networks, into `runs/research/smoke`; about two minutes.
  Check it with `make research-validate ROOT=runs/research/smoke`.
- `make research-dryrun`: times configurations 0-2 of every family on FD001 and FD004 (repeat 0,
  fold 0) with both streams running at once, writes `reports/research/compute_estimate.csv` and
  prints the outcome of the rule in PREREGISTRATION.md section 4.4.
- `make research-leaderboard`: the reference configuration of every family on the legacy split,
  seeds 0-4, into `runs/research/leaderboard` (never the confirmatory archive). It writes
  `reports/research/leaderboard_reference.csv` and `leaderboard_vs_legacy.csv`; run it with
  `STREAM=cpu` and `STREAM=gpu` in two terminals to halve the time. Resumable like the cross-fit.

## When both streams have finished

`make research-progress` shows no pending fit and `make research-validate` reports the archive
valid. Configurations that failed in any fold of a fleet are excluded from that fleet and listed
(PREREGISTRATION.md section 9). The archive stays where it is: never delete or edit
`runs/research/archive`.
