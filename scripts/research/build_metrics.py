"""Accuracy metrics of every fit of an archive, stored as per-engine sums and counts.

Official test engines: last-cycle error (truth capped at 125, at 130 and uncapped), NASA term,
all-cycle error, jitter and late-phase error. outer_test engines: error at 20 seeded uniform
truncation points per engine, jitter and late-phase error. inner_val: all-cycle error. Writes
runs/research/derived/<root>/<fleet>/: components (per engine), fit_metrics (per fit) and the
truncation points. The confirmatory archive is refused until it validates and analysis-frozen
exists.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from maintadapt.data import load_subset
from maintadapt.research.archive import ParquetSink, load, records, validate
from maintadapt.research.config import (
    ResearchConfig,
    fleet_number,
    load_research_config,
)
from maintadapt.research.crossfit import STREAMS
from maintadapt.research.derived import (
    fit_metrics,
    inventory,
    metric_components,
    truncation_points,
)
from maintadapt.research.folds import lifetimes
from maintadapt.research.guards import require_frozen_archive, resolve_root

COLUMNS = ["unit_id", "cycle", "rul", "split", "pred", "repeat", "fold"]
KEYS = ["config_id", "family", "repeat", "fold"]


def build_fleet(cfg: ResearchConfig, root: Path, fleet: str, out: Path) -> dict:
    start = time.perf_counter()
    found = records(root, [fleet])
    configs, repeats, excluded = inventory(found, cfg.folds.n_outer, STREAMS["all"])
    lives = lifetimes(load_subset(cfg.data.raw_dir, fleet, cfg.data.cap).train)
    # one draw per engine, shared by every configuration, repeat and fold
    truncation = truncation_points(lives, [cfg.stats.seed, fleet_number(fleet)])
    components = ParquetSink(out / "components.parquet")
    per_fit = []
    for config_id in configs:
        table = load(
            root, fleets=[fleet], config_ids=[config_id], columns=COLUMNS, found=found
        )
        parts = []
        for (repeat, fold), fit in table.groupby(["repeat", "fold"], sort=True):
            keys = {
                "config_id": config_id,
                "family": config_id.rsplit("-", 1)[0],
                "repeat": int(repeat),
                "fold": int(fold),
            }
            part = metric_components(fit, truncation, cfg.data.cap)
            parts.append(part.assign(**keys)[[*keys, *part.columns]])
        config_parts = pd.concat(parts, ignore_index=True)
        components.write(config_parts)
        per_fit.append(fit_metrics(config_parts, KEYS))
        print(
            f"{fleet} {config_id:<18} {time.perf_counter() - start:7.1f}s", flush=True
        )
    components.close()
    metrics = ParquetSink(out / "fit_metrics.parquet")
    metrics.write(pd.concat(per_fit, ignore_index=True))
    metrics.close()
    points = ParquetSink(out / "truncation.parquet")
    points.write(truncation)
    points.close()
    summary = {
        "fleet": fleet,
        "root": str(root),
        "configs": len(configs),
        "excluded": len(excluded),
        "repeats": repeats,
        "component_rows": components.rows,
        "fits": metrics.rows,
        "seconds": round(time.perf_counter() - start, 1),
    }
    (out / "build_metrics.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument(
        "--root", help="archive, smoke, dryrun or a path (default: archive)"
    )
    parser.add_argument("--fleets", nargs="+")
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    root = resolve_root(args.root, cfg)
    fleets = args.fleets or [f for f in cfg.data.fleets if (root / f).is_dir()]
    require_frozen_archive(root, cfg, fleets, validate)

    start = time.perf_counter()
    for fleet in fleets:
        summary = build_fleet(
            cfg, root, fleet, Path(cfg.paths.derived) / root.name / fleet
        )
        print(
            f"{fleet}: {summary['fits']} fits of {summary['configs']} configurations, "
            f"{summary['component_rows']} engine rows in {summary['seconds']} s",
            flush=True,
        )
    print(f"metrics built in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
