"""Conventional leaderboard: each family's reference configuration on the legacy split.

Seeds 0-4 of configs/default.yaml on configs/splits.json; predictions go to
runs/research/leaderboard, never to the confirmatory archive. Writes RMSE and NASA score at the
last observed cycle of the official test engines (truth capped at 125) to
reports/research/leaderboard_reference.csv and the comparison with the engineering project's
tables to reports/research/leaderboard_vs_legacy.csv. Resumable.
"""

import argparse
from pathlib import Path

import pandas as pd

from maintadapt.config import load_config
from maintadapt.data import load_splits, load_subset
from maintadapt.evaluate import summarise
from maintadapt.experiment import exit_without_teardown
from maintadapt.research.archive import FitKey, fit_status
from maintadapt.research.config import load_research_config
from maintadapt.research.crossfit import STREAMS, environment, run_fit
from maintadapt.research.fold_data import build_fold_data
from maintadapt.research.folds import Fold
from maintadapt.research.leaderboard import (
    METRICS,
    compare_with_legacy,
    per_seed_metrics,
)
from maintadapt.research.models.registry import GPU_FAMILIES, fit
from maintadapt.research.models.spaces import load_pool
from maintadapt.sequence import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--legacy-config", default="configs/default.yaml")
    parser.add_argument("--stream", choices=list(STREAMS), default="all")
    parser.add_argument("--fleets", nargs="+")
    parser.add_argument("--limit", type=int, help="stop after this many new fits")
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    legacy = load_config(args.legacy_config)
    root = Path(cfg.paths.leaderboard)
    if root.resolve() == Path(cfg.paths.archive).resolve():
        raise SystemExit("the leaderboard must not write into the confirmatory archive")

    splits = load_splits(
        legacy.paths.splits, legacy.data.split_seed, legacy.data.val_fraction
    )
    pool = load_pool(cfg.paths.pool, cfg.pool.seed, cfg.pool.n_configs)
    references = [
        c for c in pool if c.id.endswith("-00") and c.family in STREAMS[args.stream]
    ]
    device = resolve_device(cfg.training.device)
    env = environment(device)
    ran = 0
    for fleet in args.fleets or cfg.data.fleets:
        todo = [
            (c, seed)
            for c in references
            for seed in legacy.seeds
            if fit_status(root, FitKey(fleet, c.family, c.id, f"seed{seed}")) is None
        ]
        if not todo or (args.limit is not None and ran >= args.limit):
            continue
        fold = Fold(
            inner_fit=tuple(splits[fleet]["train"]),
            inner_val=tuple(splits[fleet]["val"]),
            outer_test=(),
        )
        # the legacy split seed also seeded the legacy regime clustering
        data = build_fold_data(
            load_subset(cfg.data.raw_dir, fleet, cfg.data.cap),
            fold,
            cfg.data,
            legacy.data.split_seed,
            repeat=0,
            index=0,
        )
        for config, seed in todo:
            if args.limit is not None and ran >= args.limit:
                break
            identity = {
                "fleet": fleet,
                "family": config.family,
                "config_id": config.id,
                "config_hash": config.hash,
                "repeat": None,
                "fold": None,
            }
            record = run_fit(
                root,
                FitKey(fleet, config.family, config.id, f"seed{seed}"),
                identity,
                data,
                lambda s, c=config, d=data: fit(
                    c.family, c.params, d, s, device, cfg.training
                ),
                seed,
                env,
                str(device) if config.family in GPU_FAMILIES else "cpu",
            )
            ran += 1
            print(
                f"{fleet} {config.id:<16} seed {seed} {record['wall_seconds']:7.1f}s "
                f"{record['status']}",
                flush=True,
            )
    summarise_leaderboard(
        root, cfg.data.cap, Path(cfg.paths.reports), legacy.paths.reports
    )


def summarise_leaderboard(
    root: Path, cap: int, reports: Path, legacy_reports: str
) -> None:
    per_seed = per_seed_metrics(root, cap)
    if per_seed.empty:
        print("no finished leaderboard fits yet")
        return
    reports.mkdir(parents=True, exist_ok=True)
    # LF on every system, so a rerun on Windows or Linux writes the same bytes
    lf = {"index": False, "lineterminator": "\n"}
    per_seed.round(4).to_csv(reports / "leaderboard_reference_per_seed.csv", **lf)
    reference = summarise(per_seed, ["fleet", "family"], METRICS)
    reference.round(3).to_csv(reports / "leaderboard_reference.csv", **lf)
    legacy = pd.concat(
        [
            pd.read_csv(Path(legacy_reports) / name)
            for name in ("baselines.csv", "sequence.csv")
        ]
    ).rename(columns={"subset": "fleet", "model": "family"})
    comparison = compare_with_legacy(reference, legacy)
    comparison.round(4).to_csv(reports / "leaderboard_vs_legacy.csv", **lf)
    complete = per_seed.groupby(["fleet", "family"]).size()
    print(
        f"wrote {reports / 'leaderboard_reference.csv'} and leaderboard_vs_legacy.csv: "
        f"{len(per_seed)} fits, {int((complete == 5).sum())} of {len(complete)} "
        "(fleet, family) pairs with all five seeds"
    )


if __name__ == "__main__":
    main()
    exit_without_teardown()
