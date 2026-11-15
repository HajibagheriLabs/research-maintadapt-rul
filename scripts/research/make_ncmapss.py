"""Build the N-CMAPSS per-flight frame and its folds (PREREGISTRATION.md section 8; DECISIONS.md).

Reads W and X_s one unit at a time from every readable file (ncmapss.build_frame), writes one
row per unit and flight to ncmapss.frame with a JSON summary beside it, and writes the folds,
five outer folds per repeat over the pooled units stratified by sub-dataset, to ncmapss.folds.
--folds-only rewrites the folds from an existing frame. Trains nothing.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from maintadapt.research.config import NCMAPSS, load_research_config
from maintadapt.research.folds import fleet_folds, save_folds, sort_by_stratum
from maintadapt.research.ncmapss import build_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--folds-only", action="store_true")
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    path = Path(cfg.ncmapss.frame)
    if args.folds_only:
        flights = pd.read_parquet(path)
    else:
        start = time.perf_counter()
        frame = build_frame(cfg.ncmapss.raw_dir)
        flights = frame.flights
        path.parent.mkdir(parents=True, exist_ok=True)
        flights.to_parquet(path, index=False)
        summary = {
            "units": int(flights["unit_id"].nunique()),
            "flights": len(flights),
            "sub_datasets": sorted(flights["sub_dataset"].unique()),
            "excluded": [{"file": f, "reason": why} for f, why in frame.excluded],
            "features": frame.features,
            "seconds": round(time.perf_counter() - start, 1),
        }
        path.with_suffix(".json").write_text(
            json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
        print(
            f"wrote {path}: {summary['units']} units, {summary['flights']} flights, "
            f"{len(frame.features)} features, excluded {[f for f, _ in frame.excluded]}"
        )
    units = flights.groupby("unit_id")
    lives, strata = units["cycle"].max(), units["sub_dataset"].first()
    folds = fleet_folds(lives, cfg.folds, NCMAPSS, sort_by_stratum(lives, strata))
    save_folds({NCMAPSS: folds}, cfg.folds, cfg.ncmapss.folds)
    sizes = [
        (len(f.outer_test), len(f.inner_val), len(f.inner_fit))
        for r in folds
        for f in r
    ]
    print(
        f"wrote {cfg.ncmapss.folds}: (outer_test, inner_val, inner_fit) {sorted(set(sizes))}"
    )


if __name__ == "__main__":
    main()
