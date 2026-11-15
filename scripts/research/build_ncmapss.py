"""Derived tables of an N-CMAPSS archive: the work of build_outcomes and build_metrics.

Those two scripts are frozen with the confirmatory analysis and read the C-MAPSS files and
folds; this one applies the same functions of derived.py (repeat_outcomes, age_table,
metric_components, fit_metrics, inventory) to the N-CMAPSS frame, folds and cap. Writes
runs/research/derived/<archive>/NCMAPSS/: policies, outcomes, age, engines, components,
fit_metrics, truncation and excluded, in the layout of the C-MAPSS fleets. N-CMAPSS has no
official test split, so the components hold inner_val and outer_test only.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from maintadapt.research.archive import ParquetSink, load, records
from maintadapt.research.config import NCMAPSS, fleet_number, load_research_config
from maintadapt.research.crossfit import STREAMS
from maintadapt.research.derived import (
    age_table,
    fit_metrics,
    inventory,
    metric_components,
    repeat_outcomes,
    truncation_points,
)
from maintadapt.research.folds import lifetimes, load_folds
from maintadapt.research.ncmapss import load_subset

COLUMNS = ["unit_id", "cycle", "rul", "split", "pred", "repeat", "fold"]
KEYS = ["config_id", "family", "repeat", "fold"]
INTS = {"unit_id": "int32", "used": "int32", "waste": "int32"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--root", help="an N-CMAPSS archive (default: ncmapss.archive)")
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    root = Path(args.root or cfg.ncmapss.archive)
    out = Path(cfg.paths.derived) / root.name / NCMAPSS
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cap = cfg.ncmapss.require_cap()
    subset, _ = load_subset(cfg.ncmapss.frame, cap)
    lives = lifetimes(subset.train)
    folds = load_folds(cfg.ncmapss.folds, cfg.folds, [NCMAPSS])[NCMAPSS]
    found = records(root, [NCMAPSS])
    configs, repeats, excluded = inventory(found, cfg.folds.n_outer, STREAMS["all"])
    # one draw per unit, shared by every configuration, repeat and fold
    truncation = truncation_points(lives, [cfg.stats.seed, fleet_number(NCMAPSS)])
    sinks = {
        name: ParquetSink(out / f"{name}.parquet")
        for name in ("policies", "outcomes", "components")
    }
    per_fit = []
    for config_id in configs:
        family = config_id.rsplit("-", 1)[0]
        table = load(
            root, fleets=[NCMAPSS], config_ids=[config_id], columns=COLUMNS, found=found
        )
        parts = []
        for repeat in repeats:
            in_repeat = table[table["repeat"] == repeat]
            fits = {
                f: in_repeat[in_repeat["fold"] == f] for f in range(cfg.folds.n_outer)
            }
            policy_table, outcome_table = repeat_outcomes(fits, cfg.decision)
            keys = {"config_id": config_id, "family": family, "repeat": repeat}
            sinks["policies"].write(
                policy_table.assign(**keys)[[*keys, *policy_table.columns]]
            )
            sinks["outcomes"].write(
                outcome_table.assign(**keys)[[*keys, *outcome_table.columns]].astype(
                    INTS
                )
            )
            for fold, fit in fits.items():
                part = metric_components(fit, truncation, cap)
                fit_keys = {**keys, "fold": fold}
                parts.append(part.assign(**fit_keys)[[*fit_keys, *part.columns]])
        config_parts = pd.concat(parts, ignore_index=True)
        sinks["components"].write(config_parts)
        per_fit.append(fit_metrics(config_parts, KEYS))
        print(f"{config_id:<20} {time.perf_counter() - start:7.1f}s", flush=True)
    for sink in sinks.values():
        sink.close()
    tables = {
        "fit_metrics": pd.concat(per_fit, ignore_index=True),
        "age": age_table(lives, folds, repeats, cfg.decision.cost_ratios).astype(INTS),
        "engines": pd.DataFrame({"unit_id": lives.index, "life": lives.to_numpy()}),
        "truncation": truncation,
    }
    for name, frame in tables.items():
        sink = ParquetSink(out / f"{name}.parquet")
        sink.write(frame)
        sink.close()
    excluded.to_csv(out / "excluded.csv", index=False, lineterminator="\n")
    summary = {
        "fleet": NCMAPSS,
        "root": str(root),
        "cap": cap,
        "configs": len(configs),
        "excluded": len(excluded),
        "repeats": repeats,
        "outcome_rows": sinks["outcomes"].rows,
        "component_rows": sinks["components"].rows,
        "seconds": round(time.perf_counter() - start, 1),
    }
    (out / "build_ncmapss.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
