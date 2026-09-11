"""Train the TCN and LSTM on every subset and seed under the baseline protocol."""

import argparse
from pathlib import Path

import pandas as pd

from maintadapt.config import load_config
from maintadapt.data import load_splits, load_subset
from maintadapt.evaluate import nasa_score, rmse, summarise
from maintadapt.experiment import (
    finished_runs,
    record_run,
    seed_everything,
    start_run_dir,
)
from maintadapt.features import last_cycle, prepare_subset
from maintadapt.sequence import (
    build_model,
    make_windows,
    predict,
    resolve_device,
    train_model,
)

METRICS = ["rmse", "nasa_score"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(parser.parse_args().config)
    run_dir = start_run_dir(cfg, "sequence")
    splits = load_splits(cfg.paths.splits, cfg.data.split_seed, cfg.data.val_fraction)
    device = resolve_device(cfg.sequence.device)
    seq, cap = cfg.sequence, cfg.data.rul_cap
    truth = "rul_capped" if cfg.evaluation.cap_test_truth else "rul"

    rows = finished_runs(run_dir)
    done = {(r["subset"], r["model"], r["seed"]) for r in rows}
    for name in cfg.data.subsets:
        todo = [
            (seed, kind)
            for seed in cfg.seeds
            for kind in seq.models
            if (name, kind, seed) not in done
        ]
        if not todo:
            continue
        p = prepare_subset(
            load_subset(cfg.paths.raw_dir, name, cap),
            splits[name],
            cfg.data.n_conditions[name],
            cfg.data.max_modal_share,
            cfg.data.split_seed,
            condition_aware=cfg.features.condition_aware,
        )
        train = (
            make_windows(p.train, p.sensors, seq.window),
            p.train["rul_capped"].to_numpy(),
        )
        val = (
            make_windows(p.val, p.sensors, seq.window),
            p.val["rul_capped"].to_numpy(),
        )
        x_test = make_windows(p.test, p.sensors, seq.window, last_only=True)
        y_test = p.test.loc[last_cycle(p.test), truth].to_numpy()
        for seed, kind in todo:
            seed_everything(seed)
            model = build_model(kind, len(p.sensors), seq)
            checkpoint = run_dir / name / f"{kind}_seed{seed}.pt"
            result = train_model(
                model, train, val, seq.training, cap, device, checkpoint
            )
            predicted = predict(
                result.model, x_test, cap, device, seq.training.batch_size
            )
            metrics = {
                "rmse": rmse(y_test, predicted),
                "nasa_score": nasa_score(y_test, predicted),
                "val_rmse": result.best_val_rmse,
            }
            key = {"subset": name, "model": kind, "seed": seed}
            row = {**key, **metrics, "best_epoch": result.best_epoch}
            record_run(run_dir, f"{name}-{kind}-seed{seed}", row, key, metrics)
            rows.append(row)
            print(f"{name} {kind:5s} seed {seed}: epoch {result.best_epoch} {metrics}")

    runs = pd.DataFrame(rows).sort_values(["subset", "model", "seed"])
    reports = Path(cfg.paths.reports)
    runs.round(4).to_csv(reports / "sequence_per_seed.csv", index=False)
    summary = summarise(runs, ["subset", "model"], METRICS).round(3)
    summary.to_csv(reports / "sequence.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
