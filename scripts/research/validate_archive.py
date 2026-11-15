"""Validate a prediction archive: every engine of each split once per fit, cycles 1..T,
no duplicates, the true RUL, finite predictions in [0, cap], and pool hashes.

Reads truth and predictions only to check their structure; it computes no metric. --nested and
--caps check the archives of the secondary analyses S4 and S5 against their own folds and caps.
"""

import argparse
from collections.abc import Iterator
from pathlib import Path

from maintadapt.data import load_subset
from maintadapt.research.archive import validate, validate_fleet
from maintadapt.research.config import NCMAPSS, ResearchConfig, load_research_config
from maintadapt.research.crossfit import (
    control_configs,
    nested_inner_folds,
    nested_root,
)
from maintadapt.research.folds import lifetimes, load_folds
from maintadapt.research.guards import resolve_root
from maintadapt.research.models.spaces import load_pool
from maintadapt.research.ncmapss import load_subset as load_ncmapss

Checked = tuple[str, Path, list[str], dict[str, int]]


def pool_hashes(cfg: ResearchConfig) -> dict[str, str]:
    return {
        c.id: c.hash
        for c in load_pool(cfg.paths.pool, cfg.pool.seed, cfg.pool.n_configs)
    }


def nested(cfg: ResearchConfig, fleets: list[str]) -> Iterator[Checked]:
    n = cfg.secondary.nested
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)
    for fleet in fleets:
        subset = load_subset(cfg.data.raw_dir, fleet, cfg.data.cap)
        lives = lifetimes(subset.train)
        for outer in range(cfg.folds.n_outer):
            inner = nested_inner_folds(cfg, lives, folds, fleet, n.repeat, outer)
            root = nested_root(cfg.paths.archive_nested, outer)
            found, counts = validate_fleet(
                root, fleet, subset, {n.repeat: inner}, cfg.data.cap, pool_hashes(cfg)
            )
            yield fleet, root, found, counts


def ncmapss(cfg: ResearchConfig, root: Path) -> Iterator[Checked]:
    cap = cfg.ncmapss.require_cap()
    subset, _ = load_ncmapss(cfg.ncmapss.frame, cap)
    folds = load_folds(cfg.ncmapss.folds, cfg.folds, [NCMAPSS])[NCMAPSS]
    hashes = pool_hashes(cfg) | {c.id: c.hash for c in control_configs()}
    found, counts = validate_fleet(root, NCMAPSS, subset, folds, cap, hashes)
    yield NCMAPSS, root, found, counts


def caps(cfg: ResearchConfig, fleets: list[str]) -> Iterator[Checked]:
    c = cfg.secondary.caps
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)
    for cap in c.values:
        root = Path(cfg.paths.archive_caps) / c.name(cap)
        for fleet in fleets:
            subset = load_subset(cfg.data.raw_dir, fleet, c.training_cap(cap))
            found, counts = validate_fleet(
                root, fleet, subset, folds[fleet], c.training_cap(cap), pool_hashes(cfg)
            )
            yield fleet, root, found, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument(
        "--root", help="archive, smoke, dryrun or a path (default: archive)"
    )
    parser.add_argument("--fleets", nargs="+")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--nested", action="store_true", help="the S4 nested archive")
    which.add_argument(
        "--ncmapss",
        action="store_true",
        help="an N-CMAPSS archive: ncmapss.archive unless --root names another",
    )
    which.add_argument(
        "--caps", action="store_true", help="the S5 training-cap archives"
    )
    args = parser.parse_args()
    cfg = load_research_config(args.config)

    if args.nested:
        checked = nested(cfg, args.fleets or cfg.secondary.nested.fleets)
    elif args.caps:
        checked = caps(cfg, args.fleets or cfg.data.fleets)
    elif args.ncmapss:
        checked = ncmapss(cfg, Path(args.root or cfg.ncmapss.archive))
    else:
        root = resolve_root(args.root, cfg)
        checked = (
            (fleet, root, *validate(fleet, root, cfg))
            for fleet in args.fleets or cfg.data.fleets
        )
    problems, roots = [], set()
    for fleet, root, found, counts in checked:
        roots.add(root.as_posix())
        print(
            f"{root.as_posix()} {fleet}: {counts['ok']} fits checked, "
            f"{counts['failed']} failed fits, {counts['stray_tmp']} stray .tmp files, "
            f"{len(found)} problems"
        )
        problems += [f"{fleet} {p}" for p in found]
    where = ", ".join(sorted(roots))
    if problems:
        raise SystemExit(
            "\n".join(problems[:50]) + f"\n{len(problems)} problems in {where}"
        )
    print(f"{where}: valid")


if __name__ == "__main__":
    main()
