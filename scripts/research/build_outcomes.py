"""Tune alarm policies and record per-engine maintenance outcomes for every fit of an archive.

For every configuration, repeat, fold, rule (R1; R2 over the alpha grid), lead time and cost
ratio, with three tunings: validation (the fold's inner_val engines), crossfit (the outer_test
engines of the repeat's other folds) and oracle (every outer_test engine of the repeat). Writes
runs/research/derived/<root>/<fleet>/: policies, outcomes (outer_test engines), age (age
replacement tuned on outer-train lifetimes), engines (lifetimes) and excluded configurations.
The confirmatory archive is refused until it validates and analysis-frozen exists.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from maintadapt.data import load_subset
from maintadapt.research.archive import ParquetSink, load, records, validate
from maintadapt.research.config import ResearchConfig, load_research_config
from maintadapt.research.crossfit import STREAMS
from maintadapt.research.derived import age_table, inventory, repeat_outcomes
from maintadapt.research.folds import lifetimes, load_folds
from maintadapt.research.guards import require_frozen_archive, resolve_root

COLUMNS = ["unit_id", "cycle", "split", "pred", "repeat", "fold"]


def build_fleet(cfg: ResearchConfig, root: Path, fleet: str, out: Path) -> dict:
    start = time.perf_counter()
    found = records(root, [fleet])
    configs, repeats, excluded = inventory(found, cfg.folds.n_outer, STREAMS["all"])
    lives = lifetimes(load_subset(cfg.data.raw_dir, fleet, cfg.data.cap).train)
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)[fleet]
    policies = ParquetSink(out / "policies.parquet")
    outcomes = ParquetSink(out / "outcomes.parquet")
    for config_id in configs:
        family = config_id.rsplit("-", 1)[0]
        table = load(
            root,
            fleets=[fleet],
            config_ids=[config_id],
            splits=["inner_val", "outer_test"],
            columns=COLUMNS,
            found=found,
        )
        for repeat in repeats:
            in_repeat = table[table["repeat"] == repeat]
            fits = {
                f: in_repeat[in_repeat["fold"] == f] for f in range(cfg.folds.n_outer)
            }
            policy_table, outcome_table = repeat_outcomes(fits, cfg.decision)
            keys = {"config_id": config_id, "family": family, "repeat": repeat}
            policies.write(policy_table.assign(**keys)[[*keys, *policy_table.columns]])
            outcomes.write(
                outcome_table.assign(**keys)[[*keys, *outcome_table.columns]].astype(
                    {"unit_id": "int32", "used": "int32", "waste": "int32"}
                )
            )
        print(
            f"{fleet} {config_id:<18} {time.perf_counter() - start:7.1f}s", flush=True
        )
    policies.close()
    outcomes.close()
    age = age_table(lives, folds, repeats, cfg.decision.cost_ratios)
    sink = ParquetSink(out / "age.parquet")
    sink.write(age.astype({"unit_id": "int32", "used": "int32", "waste": "int32"}))
    sink.close()
    engines = ParquetSink(out / "engines.parquet")
    engines.write(pd.DataFrame({"unit_id": lives.index, "life": lives.to_numpy()}))
    engines.close()
    excluded.to_csv(out / "excluded.csv", index=False, lineterminator="\n")
    summary = {
        "fleet": fleet,
        "root": str(root),
        "configs": len(configs),
        "excluded": len(excluded),
        "repeats": repeats,
        "policy_rows": policies.rows,
        "outcome_rows": outcomes.rows,
        "seconds": round(time.perf_counter() - start, 1),
    }
    (out / "build_outcomes.json").write_text(
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
            f"{fleet}: {summary['configs']} configurations ({summary['excluded']} excluded), "
            f"repeats {summary['repeats']}, {summary['outcome_rows']} outcome rows in "
            f"{summary['seconds']} s",
            flush=True,
        )
    print(f"outcomes built in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
