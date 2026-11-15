"""Cross-fit the model pool: every configuration in every (fleet, repeat, outer fold).

Resumable: a fit whose sidecar exists is never refitted. Prints ids, seconds, validation RMSE
and status per fit, and never a test or decision metric. --nested and --caps run the extra fits
of the secondary analyses S4 and S5 (PREREGISTRATION.md section 7) into archives of their own;
--estimate projects their hours from the confirmatory fits they repeat.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
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
from maintadapt.research.config import (
    NCMAPSS,
    FleetDataConfig,
    ResearchConfig,
    load_research_config,
)
from maintadapt.research.crossfit import (
    STREAMS,
    Job,
    control_configs,
    environment,
    family_representatives,
    nested_inner_folds,
    nested_root,
    nested_seed,
    plan,
    run_jobs,
    select_configs,
)
from maintadapt.research.fold_data import (
    FoldData,
    build_fold_data,
    build_ncmapss_fold_data,
)
from maintadapt.research.folds import Fold, lifetimes, load_folds
from maintadapt.research.models.registry import (
    CPU_FAMILIES,
    GPU_FAMILIES,
    FitOutput,
    fit,
)
from maintadapt.research.models.spaces import FAMILY_NAMES, load_pool
from maintadapt.research.ncmapss import load_subset as load_ncmapss
from maintadapt.sequence import resolve_device

SMOKE_EPOCHS = 3
DRYRUN_FLEETS = ["FD001", "FD004"]
FOLD_SPLITS = ("inner_fit", "inner_val", "outer_test")


@dataclass(frozen=True, eq=False)
class Part:
    """The fits of one archive: jobs, the data settings and folds they use, seeds, labels."""

    root: Path
    jobs: list[Job]
    data: FleetDataConfig
    # fleet -> repeat -> fold
    folds: Mapping[str, Mapping[int, Sequence[Fold]]]
    seed_of: Callable[[Job], int] | None = None
    # added to every sidecar
    extra: dict | None = None
    # builds a fold's data in place of the C-MAPSS preparation (N-CMAPSS)
    prepare: Callable[[str, int, int], FoldData] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--stream", choices=list(STREAMS), default="all")
    parser.add_argument("--fleets", nargs="+")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--repeats", nargs="+", type=int)
    parser.add_argument(
        "--folds", nargs="+", type=int, help="outer folds (with --nested too)"
    )
    parser.add_argument(
        "--configs", nargs="+", type=int, help="configuration numbers within a family"
    )
    parser.add_argument("--limit", type=int, help="stop after this many new fits")
    parser.add_argument("--root", help="archive directory (default: from the mode)")
    parser.add_argument(
        "--ncmapss",
        action="store_true",
        help="fleet NCMAPSS (PREREGISTRATION.md section 8), alone or with --smoke or "
        "--dry-run, into archives of its own",
    )
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
    mode.add_argument(
        "--nested",
        action="store_true",
        help="S4: inner cross-fitting of each family's representative "
        "(secondary.nested in the config)",
    )
    mode.add_argument(
        "--caps",
        action="store_true",
        help="S5: the reference configurations trained with other caps "
        "(secondary.caps in the config)",
    )
    mode.add_argument(
        "--estimate",
        action="store_true",
        help="project the hours of the --nested and --caps fits from the confirmatory "
        "fits they repeat",
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
    elif args.nested:
        n = cfg.secondary.nested
        base = {
            "fleets": n.fleets,
            "repeats": [n.repeat],
            # outer folds; every inner fold of each is fitted
            "folds": list(range(cfg.folds.n_outer)),
            "configs": None,
            "root": cfg.paths.archive_nested,
        }
    elif args.caps:
        base = {
            "fleets": cfg.data.fleets,
            "repeats": [cfg.secondary.caps.repeat],
            "folds": list(range(cfg.folds.n_outer)),
            "configs": [0],
            "root": cfg.paths.archive_caps,
        }
    else:
        base = {
            "fleets": cfg.data.fleets,
            "repeats": list(range(cfg.crossfit.repeats)),
            "folds": list(range(cfg.folds.n_outer)),
            "configs": None,
            "root": cfg.paths.archive,
        }
    if args.ncmapss:
        base["fleets"] = [NCMAPSS]
        base["root"] = (
            f"{cfg.paths.smoke}_ncmapss"
            if args.smoke
            else f"{cfg.paths.dryrun}_ncmapss"
            if args.dry_run
            else cfg.ncmapss.archive
        )
    for name in base:
        if getattr(args, name) is not None:
            base[name] = getattr(args, name)
    return base


def stream_families(args: argparse.Namespace) -> list[str]:
    return [
        f for f in STREAMS[args.stream] if args.families is None or f in args.families
    ]


def indexed(folds: dict[str, list[list[Fold]]]) -> dict[str, dict[int, list[Fold]]]:
    return {fleet: dict(enumerate(repeats)) for fleet, repeats in folds.items()}


def crossfit_part(args: argparse.Namespace, cfg: ResearchConfig, run: dict) -> Part:
    pool = load_pool(cfg.paths.pool, cfg.pool.seed, cfg.pool.n_configs)
    confirmatory = not (args.smoke or args.dry_run)
    configs = select_configs(
        pool,
        stream_families(args),
        run["configs"],
        cfg.crossfit.configs_run if confirmatory else None,
    )
    if args.smoke:
        configs += [c for c in control_configs() if c.family in stream_families(args)]
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)
    return Part(
        Path(run["root"]),
        plan(configs, run["fleets"], run["repeats"], run["folds"]),
        cfg.data,
        indexed(folds),
    )


def nested_parts(
    args: argparse.Namespace, cfg: ResearchConfig, run: dict
) -> list[Part]:
    """One archive per outer fold: each family's representative fitted on its inner folds."""
    n = cfg.secondary.nested
    pool = {
        c.id: c for c in load_pool(cfg.paths.pool, cfg.pool.seed, cfg.pool.n_configs)
    }
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)
    # representatives are chosen among the confirmatory fits by inner_val RMSE alone
    found = records(cfg.paths.archive, run["fleets"])
    parts = []
    for fleet in run["fleets"]:
        lives = lifetimes(load_subset(cfg.data.raw_dir, fleet, cfg.data.cap).train)
        for repeat in run["repeats"]:
            for outer in run["folds"]:
                inner = nested_inner_folds(cfg, lives, folds, fleet, repeat, outer)
                reps = family_representatives(found, fleet, repeat, outer, FAMILY_NAMES)
                configs = [
                    pool[c] for c in reps if pool[c].family in stream_families(args)
                ]
                parts.append(
                    Part(
                        nested_root(run["root"], outer),
                        plan(configs, [fleet], [repeat], range(n.inner_folds)),
                        cfg.data,
                        {fleet: {repeat: inner}},
                        seed_of=lambda job, outer=outer: nested_seed(
                            job.fleet, job.repeat, outer, job.fold, job.config.id
                        ),
                        extra={"outer_fold": outer},
                    )
                )
    return parts


def record_nested_folds(parts: Sequence[Part], root: str | Path) -> None:
    """Write the inner folds next to the fits; nested_folds reproduces them exactly."""
    path = Path(root) / "nested_folds.json"
    known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for part in parts:
        for fleet, repeats in part.folds.items():
            for repeat, inner in repeats.items():
                key = f"{fleet} r{repeat} f{part.extra['outer_fold']}"
                known[key] = [
                    {name: list(getattr(f, name)) for name in FOLD_SPLITS}
                    for f in inner
                ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(sorted(known.items())), indent=1) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def caps_parts(args: argparse.Namespace, cfg: ResearchConfig, run: dict) -> list[Part]:
    """One archive per training cap, with the confirmatory seeds, so only the cap changes."""
    caps = cfg.secondary.caps
    pool = load_pool(cfg.paths.pool, cfg.pool.seed, cfg.pool.n_configs)
    families = [f for f in stream_families(args) if f in caps.families]
    configs = select_configs(pool, families, run["configs"])
    folds = indexed(load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets))
    return [
        Part(
            Path(run["root"]) / caps.name(cap),
            plan(configs, run["fleets"], run["repeats"], run["folds"]),
            replace(cfg.data, cap=caps.training_cap(cap)),
            folds,
            extra={"training_cap": cap},
        )
        for cap in caps.values
    ]


def ncmapss_part(args: argparse.Namespace, cfg: ResearchConfig, run: dict) -> Part:
    """The pool on N-CMAPSS, with its own frame, folds, cap and preparation."""
    part = crossfit_part(args, cfg, run)
    cap = cfg.ncmapss.require_cap()
    subset, features = load_ncmapss(cfg.ncmapss.frame, cap)
    folds = load_folds(cfg.ncmapss.folds, cfg.folds, [NCMAPSS])
    data = replace(cfg.data, cap=cap)

    def prepare(fleet: str, repeat: int, fold: int) -> FoldData:
        return build_ncmapss_fold_data(
            subset,
            folds[fleet][repeat][fold],
            features,
            data,
            repeat=repeat,
            index=fold,
        )

    return replace(part, data=data, folds=indexed(folds), prepare=prepare)


def parts_for(args: argparse.Namespace, cfg: ResearchConfig) -> list[Part]:
    run = settings(args, cfg)
    if args.ncmapss:
        return [ncmapss_part(args, cfg, run)]
    if args.nested:
        return nested_parts(args, cfg, run)
    if args.caps:
        return caps_parts(args, cfg, run)
    return [crossfit_part(args, cfg, run)]


def run_stream(args: argparse.Namespace, cfg: ResearchConfig) -> None:
    parts = parts_for(args, cfg)
    if args.nested:
        record_nested_folds(parts, settings(args, cfg)["root"])
    device = resolve_device(cfg.training.device)
    training = cfg.training
    if args.smoke:
        training = replace(training, max_epochs=SMOKE_EPOCHS)
    subsets = {}
    families = {job.config.family for part in parts for job in part.jobs}
    cpu_jobs = training.resolved_cpu_jobs()
    gpu_threads = training.gpu_threads if device.type == "cuda" else cpu_jobs
    # regime clustering and feature building get no more threads than the stream's fits
    stream_threads = cpu_jobs if families - set(GPU_FAMILIES) else gpu_threads

    def fit_job(job: Job, data: FoldData, seed: int) -> FitOutput:
        c = job.config
        return fit(c.family, c.params, data, seed, device, training)

    def device_for(family: str) -> str:
        return str(device) if family in GPU_FAMILIES else "cpu"

    described = []
    if families & set(GPU_FAMILIES):
        described.append(
            f"XGBoost and networks on {device} with {gpu_threads} CPU threads"
        )
    if families - set(GPU_FAMILIES):
        described.append(f"ridge, k-NN and forest with {cpu_jobs} CPU threads")
    roots = sorted({part.root.as_posix() for part in parts})
    where = (
        roots[0] if len(roots) == 1 else f"{len(roots)} archives: {', '.join(roots)}"
    )
    print(
        f"{args.stream} stream: {sum(len(p.jobs) for p in parts)} fits planned in "
        f"{where} ({'; '.join(described)})",
        flush=True,
    )
    counts: Counter[str] = Counter()
    env = environment(device)
    for part in parts:

        def build_data(
            fleet: str, repeat: int, fold: int, part: Part = part
        ) -> FoldData:
            if part.prepare is not None:
                with threadpool_limits(limits=stream_threads):
                    return part.prepare(fleet, repeat, fold)
            key = (fleet, part.data.cap)
            if key not in subsets:
                subsets[key] = load_subset(cfg.data.raw_dir, fleet, part.data.cap)
            with threadpool_limits(limits=stream_threads):
                return build_fold_data(
                    subsets[key],
                    part.folds[fleet][repeat][fold],
                    part.data,
                    cfg.folds.seed,
                    repeat=repeat,
                    index=fold,
                )

        limit = None if args.limit is None else max(0, args.limit - counts["ran"])
        counts.update(
            run_jobs(
                part.jobs,
                part.root,
                build_data,
                fit_job,
                env,
                device_for,
                limit=limit,
                log=lambda line: print(line, flush=True),
                seed_of=part.seed_of,
                extra=part.extra,
            )
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
    # LF on every system, so a rerun on Windows or Linux writes the same bytes
    projection.round(4).to_csv(out, index=False, lineterminator="\n")
    totals = stream_totals(projection)
    decision = decide_repeats(totals)
    print(totals.round(1).to_string())
    print(
        f"wrote {out}; rule of section 4.4: R = {decision.repeats}, "
        f"configurations 8-11 dropped: {decision.drop_late_configs} ({decision.reason})"
    )


def estimate_secondary(args: argparse.Namespace, cfg: ResearchConfig) -> None:
    """Hours of the S4 and S5 fits, from the wall time of the confirmatory fits they repeat.

    A nested fit repeats its representative's fit of that outer fold on fewer engines and is
    scaled by inner_fit rows, as the dry run scaled fleets; a cap fit repeats the reference
    fit of the same fold unchanged.
    """
    found = records(cfg.paths.archive)
    seconds = found.set_index(["fleet", "config_id", "repeat", "fold"])["wall_seconds"]
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)
    lives = {
        f: lifetimes(load_subset(cfg.data.raw_dir, f, cfg.data.cap).train)
        for f in cfg.data.fleets
    }
    rows = []
    for analysis, flag in (("S4 nested", "nested"), ("S5 caps", "caps")):
        mode = argparse.Namespace(
            **{**vars(args), "nested": flag == "nested", "caps": flag == "caps"},
        )
        mode.stream, mode.estimate = "all", False
        for part in parts_for(mode, cfg):
            outer = (part.extra or {}).get("outer_fold")
            for job in part.jobs:
                fold = job.fold if outer is None else outer
                hours = seconds[(job.fleet, job.config.id, job.repeat, fold)] / 3600
                if outer is not None:
                    own = part.folds[job.fleet][job.repeat][job.fold].inner_fit
                    main = folds[job.fleet][job.repeat][outer].inner_fit
                    life = lives[job.fleet]
                    hours *= life.loc[list(own)].sum() / life.loc[list(main)].sum()
                stream = "gpu" if job.config.family in GPU_FAMILIES else "cpu"
                rows.append(
                    {"analysis": analysis, "stream": stream, "fits": 1, "hours": hours}
                )
    table = (
        pd.DataFrame(rows)
        .groupby(["analysis", "stream"], as_index=False)[["fits", "hours"]]
        .sum()
    )
    out = Path(cfg.paths.reports) / "compute_estimate_secondary.csv"
    table.round(2).to_csv(out, index=False, lineterminator="\n")
    print(table.round(2).to_string(index=False))
    print(f"wrote {out}")


def estimate_ncmapss(cfg: ResearchConfig, root: str) -> None:
    """Hours of the N-CMAPSS runs from its dry run, scaled by inner_fit flights as for C-MAPSS.

    A family's seconds per fit are the mean over its timed configurations (repeat 0, fold 0),
    times the mean inner_fit flights of all folds over those of the timed fold.
    """
    subset, _ = load_ncmapss(cfg.ncmapss.frame, cfg.ncmapss.require_cap())
    lives = lifetimes(subset.train)
    folds = load_folds(cfg.ncmapss.folds, cfg.folds, [NCMAPSS])[NCMAPSS]
    per_fold = inner_fit_rows(lives, folds)
    scale = (sum(per_fold) / len(per_fold)) / per_fold[0]
    found = records(root, [NCMAPSS])
    seconds = found[found["status"] == "ok"].groupby("family")["wall_seconds"].mean()
    streams = {"cpu": CPU_FAMILIES, "gpu": GPU_FAMILIES}
    rows = []
    for stream, families in streams.items():
        for family in families:
            if family not in seconds:
                raise SystemExit(f"the dry run has no finished {family} fit")
            fits = cfg.pool.n_configs[family] * cfg.folds.n_outer * cfg.crossfit.repeats
            hours = seconds[family] * scale * fits / 3600
            rows.append(
                {
                    "stream": stream,
                    "family": family,
                    "measured_seconds": seconds[family],
                    "rows_scale": scale,
                    "fits": fits,
                    "hours": hours,
                }
            )
    table = pd.DataFrame(rows)
    out = Path(cfg.paths.reports) / "ncmapss" / "compute_estimate.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out, index=False, lineterminator="\n")
    print(table.groupby("stream")[["fits", "hours"]].sum().round(2).to_string())
    print(f"wrote {out}")


def dry_run_both_streams(args: argparse.Namespace, cfg: ResearchConfig) -> None:
    """Time the CPU and GPU streams side by side, as the confirmatory runs will go."""
    base = [sys.executable, str(Path(__file__).resolve()), "--config", args.config]
    base += ["--dry-run", *(["--ncmapss"] if args.ncmapss else [])]
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
    if args.ncmapss:
        estimate_ncmapss(cfg, settings(args, cfg)["root"])
    else:
        estimate(cfg, settings(args, cfg)["root"])


def main() -> None:
    args = parse_args()
    cfg = load_research_config(args.config)
    unknown = sorted(set(args.families or []) - set(FAMILY_NAMES) - set(STREAMS["cpu"]))
    if unknown:
        raise SystemExit(f"unknown families {unknown}")
    if args.ncmapss and (args.nested or args.caps or args.estimate):
        raise SystemExit("--ncmapss runs the pool, its smoke test or its dry run only")
    if args.estimate:
        estimate_secondary(args, cfg)
        return
    if args.dry_run and args.stream == "all":
        dry_run_both_streams(args, cfg)
        return
    run_stream(args, cfg)


if __name__ == "__main__":
    main()
    # CUDA libraries can crash while unloading on Windows after LSTM training
    exit_without_teardown()
