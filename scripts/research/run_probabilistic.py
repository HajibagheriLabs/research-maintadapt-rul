"""Fit the probabilistic families of paper 2 into an archive of their own (make paper2-prob).

Every (fleet, repeat, outer fold) of the committed folds, families xgboost_quantile,
mc_dropout and ensemble (src/maintadapt/research/probabilistic.py), into
runs/research/archive_prob: the archive's columns plus one quantile column per level. Resumable
like the cross-fitting runs: a fit with a sidecar is skipped. A fit that fails is retried once
with seed + 1000. --smoke fits FD001, repeat 0, fold 0 with 3 training epochs into
runs/research/smoke_prob. Never reads or writes the confirmatory archive.
"""

import argparse
import time
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np

from maintadapt.data import load_subset
from maintadapt.research.archive import FAILED, OK, crossfit_key, write_fit
from maintadapt.research.config import load_research_config
from maintadapt.research.crossfit import RETRY_OFFSET, environment, fit_seed
from maintadapt.research.fold_data import build_fold_data
from maintadapt.research.folds import load_folds
from maintadapt.research.models.spaces import load_pool
from maintadapt.research.probabilistic import (
    BASES,
    fit_probabilistic,
    probabilistic_table,
)
from maintadapt.sequence import resolve_device

ROOTS = {"archive": "runs/research/archive_prob", "smoke": "runs/research/smoke_prob"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fleets", nargs="+")
    parser.add_argument("--families", nargs="+", default=list(BASES))
    parser.add_argument("--limit", type=int, help="stop after this many new fits")
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    training = cfg.training
    fleets = args.fleets or cfg.data.fleets
    repeats, folds_per = range(cfg.folds.n_repeats), range(cfg.folds.n_outer)
    root = ROOTS["smoke" if args.smoke else "archive"]
    if args.smoke:
        fleets, repeats, folds_per = ["FD001"], range(1), range(1)
        training = replace(training, max_epochs=3, patience=2)
    pool = {
        c.id: c for c in load_pool(cfg.paths.pool, cfg.pool.seed, cfg.pool.n_configs)
    }
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)
    device = resolve_device(training.device)
    env = environment(device)
    ran = 0
    for fleet in fleets:
        subset = load_subset(cfg.data.raw_dir, fleet, cfg.data.cap)
        for repeat in repeats:
            for fold in folds_per:
                data = None
                for family in args.families:
                    config_id = f"{family}-00"
                    key = crossfit_key(fleet, family, config_id, repeat, fold)
                    if key.sidecar(root).exists():
                        continue
                    if args.limit is not None and ran >= args.limit:
                        return
                    if data is None:
                        data = build_fold_data(
                            subset,
                            folds[fleet][repeat][fold],
                            cfg.data,
                            cfg.folds.seed,
                            repeat=repeat,
                            index=fold,
                        )
                    base = pool[BASES[family]]
                    seed = fit_seed(fleet, repeat, fold, config_id)
                    started = datetime.now(UTC).isoformat(timespec="seconds")
                    start = time.perf_counter()
                    errors, out, table, used = [], None, None, seed
                    for used in (seed, seed + RETRY_OFFSET):
                        try:
                            out = fit_probabilistic(
                                family, base.params, data, used, device, training
                            )
                            values = [*out.point.values(), *out.quantiles.values()]
                            if all(np.isfinite(v).all() for v in values):
                                table = probabilistic_table(data, out)
                                break
                            errors.append(f"seed {used}: non-finite predictions")
                        except Exception as exc:  # noqa: BLE001
                            errors.append(f"seed {used}: {type(exc).__name__}: {exc}")
                        out = None
                    record = {
                        "fleet": fleet,
                        "family": family,
                        "config_id": config_id,
                        "base_config": base.id,
                        "base_hash": base.hash,
                        "repeat": repeat,
                        "fold": fold,
                        "seed": used,
                        "status": OK if table is not None else FAILED,
                        "attempts": len(errors) + (table is not None),
                        "error": "; ".join(errors) or None,
                        "fit_seconds": None if out is None else out.fit_seconds,
                        "best_iteration_or_epoch": None
                        if out is None
                        else out.best_iteration,
                        "val_rmse": None if out is None else out.val_rmse,
                        "n_inputs": None if out is None else out.n_inputs,
                        "n_rows": None if table is None else table.num_rows,
                        "device": device.type,
                        **env,
                        "started_at": started,
                        "wall_seconds": time.perf_counter() - start,
                    }
                    write_fit(root, key, table, record)
                    ran += 1
                    print(
                        f"{fleet} r{repeat} f{fold} {config_id:<20} "
                        f"{record['wall_seconds']:6.1f}s val_rmse {record['val_rmse']} "
                        f"{record['status']}",
                        flush=True,
                    )
    print(f"{ran} fits written to {root}")


if __name__ == "__main__":
    main()
