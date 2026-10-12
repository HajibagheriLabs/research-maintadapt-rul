"""Cross-fit the model pool: every configuration in every (fleet, repeat, outer fold).

Resumable: a fit whose sidecar exists is never refitted. Prints ids, seconds, validation RMSE
and status per fit, and never a test or decision metric.
"""

import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from maintadapt.data import load_subset
from maintadapt.experiment import exit_without_teardown
from maintadapt.research.archive import records
from maintadapt.research.compute import (
    decide_repeats,
    project_hours,
    stream_totals,
)
from maintadapt.research.config import ResearchConfig, load_research_config
from maintadapt.research.crossfit import (
    STREAMS,
    Job,
    control_configs,
    environment,
    plan,
    run_jobs,
    select_configs,
)
from maintadapt.research.fold_data import FoldData, build_fold_data
from maintadapt.research.folds import Fold, lifetimes, load_folds
from maintadapt.research.models.registry import (
    CPU_FAMILIES,
    GPU_FAMILIES,
    FitOutput,
    fit,
)
from maintadapt.research.models.spaces import FAMILY_NAMES, load_pool
from maintadapt.sequence import resolve_device

SMOKE_EPOCHS = 3
DRYRUN_FLEETS = ["FD001", "FD004"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--stream", choices=list(STREAMS), default="all")
    parser.add_argument("--fleets", nargs="+")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--repeats", nargs="+", type=int)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument(
        "--configs", nargs="+", type=int, help="configuration numbers within a family"
    )
    parser.add_argument("--limit", type=int, help="stop after this many new fits")
    parser.add_argument("--root", help="archive directory (default: from the mode)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="FD001, repeat 0, five folds, configuration 0 and two controls, "
        f"{SMOKE_EPOCHS} epochs",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="time configurations 0-2 on FD001 and FD004, repeat 0, fold 0, "
        "and project the wall time",
    )
    return parser.parse_args()


def settings(args: argparse.Namespace, cfg: ResearchConfig) -> dict:
    """Mode defaults, overridden by any option given explicitly."""
    if args.smoke:
        base = {
            "fleets": ["FD001"],
            "repeats": [0],
            "folds": list(range(cfg.folds.n_outer)),
            "configs": [0],
            "root": cfg.paths.smoke,
        }
    elif args.dry_run:
        base = {
            "fleets": DRYRUN_FLEETS,
            "repeats": [0],
            "folds": [0],
            "configs": [0, 1, 2],
            "root": cfg.paths.dryrun,
        }
    else:
        base = {
            "fleets": cfg.data.fleets,
            "repeats": list(range(cfg.crossfit.repeats)),
            "folds": list(range(cfg.folds.n_outer)),
            "configs": None,
            "root": cfg.paths.archive,
        }
    for name in base:
        if getattr(args, name) is not None:
            base[name] = getattr(args, name)
    return base


def jobs_for(args: argparse.Namespace, cfg: ResearchConfig, run: dict) -> list[Job]:
    families = [
        f for f in STREAMS[args.stream] if args.families is None or f in args.families
    ]
    pool = load_pool(cfg.paths.pool, cfg.pool.seed, cfg.pool.n_configs)
    confirmatory = not (args.smoke or args.dry_run)
    configs = select_configs(
        pool,
        families,
        run["configs"],
        cfg.crossfit.configs_run if confirmatory else None,
    )
    if args.smoke:
        configs += [c for c in control_configs() if c.family in families]
    return plan(configs, run["fleets"], run["repeats"], run["folds"])


def run_stream(args: argparse.Namespace, cfg: ResearchConfig) -> None:
    run = settings(args, cfg)
    jobs = jobs_for(args, cfg, run)
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)
    device = resolve_device(cfg.training.device)
    training = cfg.training
    if args.smoke:
        training = replace(training, max_epochs=SMOKE_EPOCHS)
    subsets = {}
    families, cpu_jobs = (
        {job.config.family for job in jobs},
        training.resolved_cpu_jobs(),
    )
    gpu_threads = training.gpu_threads if device.type == "cuda" else cpu_jobs
    # regime clustering and feature building get no more threads than the stream's fits
    stream_threads = cpu_jobs if families - set(GPU_FAMILIES) else gpu_threads

    def build_data(fleet: str, repeat: int, fold: int) -> FoldData:
        if fleet not in subsets:
            subsets[fleet] = load_subset(cfg.data.raw_dir, fleet, cfg.data.cap)
        with threadpool_limits(limits=stream_threads):
            return build_fold_data(
                subsets[fleet],
                folds[fleet][repeat][fold],
                cfg.data,
                cfg.folds.seed,
                repeat=repeat,
                index=fold,
            )

    def fit_job(job: Job, data: FoldData, seed: int) -> FitOutput:
        c = job.config
        return fit(c.family, c.params, data, seed, device, training)

    def device_for(family: str) -> str:
        return str(device) if family in GPU_FAMILIES else "cpu"

    parts = []
    if families & set(GPU_FAMILIES):
        parts.append(f"XGBoost and networks on {device} with {gpu_threads} CPU threads")
    if families - set(GPU_FAMILIES):
        parts.append(f"ridge, k-NN and forest with {cpu_jobs} CPU threads")
    print(
        f"{args.stream} stream: {len(jobs)} fits planned in {run['root']} "
        f"({'; '.join(parts)})",
        flush=True,
    )
    counts = run_jobs(
        jobs,
        run["root"],
        build_data,
        fit_job,
        environment(device),
        device_for,
        limit=args.limit,
        log=lambda line: print(line, flush=True),
    )
    print(
        f"{args.stream} stream: {counts['ran']} fitted ({counts['ok']} ok, "
        f"{counts['failed']} failed), {counts['skipped']} already finished, "
        f"{counts['left']} left by --limit",
        flush=True,
    )


def inner_fit_rows(lives: pd.Series, folds: list[list[Fold]]) -> list[int]:
    return [int(lives.loc[list(f.inner_fit)].sum()) for repeat in folds for f in repeat]


def estimate(cfg: ResearchConfig, root: str) -> None:
    """Project wall-clock hours per stream and apply PREREGISTRATION.md section 4.4."""
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)
    rows, measured = {}, {}
    for fleet in cfg.data.fleets:
        lives = lifetimes(load_subset(cfg.data.raw_dir, fleet, cfg.data.cap).train)
        per_fold = inner_fit_rows(lives, folds[fleet])
        rows[fleet] = sum(per_fold) / len(per_fold)
        # the dry run times repeat 0, fold 0, the first entry
        measured[fleet] = per_fold[0]
    # the controls run only in the smoke test, so they take no confirmatory time
    streams = {"cpu": CPU_FAMILIES, "gpu": GPU_FAMILIES}
    projection = project_hours(
        records(root), streams, cfg.pool.n_configs, rows, measured, cfg.folds.n_outer
    )
    out = Path(cfg.paths.reports) / "compute_estimate.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    projection.round(4).to_csv(out, index=False)
    totals = stream_totals(projection)
    decision = decide_repeats(totals)
    print(totals.round(1).to_string())
    print(
        f"wrote {out}; rule of section 4.4: R = {decision.repeats}, "
        f"configurations 8-11 dropped: {decision.drop_late_configs} ({decision.reason})"
    )


def dry_run_both_streams(args: argparse.Namespace, cfg: ResearchConfig) -> None:
    """Time the CPU and GPU streams side by side, as the confirmatory runs will go."""
    base = [sys.executable, str(Path(__file__).resolve()), "--config", args.config]
    base += ["--dry-run"]
    for name in ("fleets", "families", "configs", "root"):
        value = getattr(args, name)
        if value is not None:
            base += [
                f"--{name}",
                *map(str, value if isinstance(value, list) else [value]),
            ]
    children = [subprocess.Popen([*base, "--stream", s]) for s in ("cpu", "gpu")]
    codes = [child.wait() for child in children]
    if any(codes):
        raise SystemExit(f"a dry-run stream failed with exit codes {codes}")
    estimate(cfg, settings(args, cfg)["root"])


def main() -> None:
    args = parse_args()
    cfg = load_research_config(args.config)
    unknown = sorted(set(args.families or []) - set(FAMILY_NAMES) - set(STREAMS["cpu"]))
    if unknown:
        raise SystemExit(f"unknown families {unknown}")
    if args.dry_run and args.stream == "all":
        dry_run_both_streams(args, cfg)
        return
    run_stream(args, cfg)


if __name__ == "__main__":
    main()
    # CUDA libraries can crash while unloading on Windows after LSTM training
    exit_without_teardown()
