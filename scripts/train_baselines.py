"""Train ridge and XGBoost on every subset and seed; write reports/baselines.csv."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from maintadapt.baselines import fit_ridge, fit_xgboost
from maintadapt.config import Config, load_config
from maintadapt.data import load_splits, load_subset
from maintadapt.evaluate import nasa_score, rmse, summarise
from maintadapt.experiment import log_run, seed_everything, start_run_dir
from maintadapt.features import (
    PreparedSubset,
    build_features,
    last_cycle,
    prepare_subset,
)

METRICS = ["rmse", "nasa_score"]


def design(p: PreparedSubset, cfg: Config) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    def make(frame: pd.DataFrame) -> pd.DataFrame:
        return build_features(
            frame, p.sensors, cfg.features.windows, cfg.features.include_cycle
        )

    final = last_cycle(p.test)
    truth = "rul_capped" if cfg.evaluation.cap_test_truth else "rul"
    return {
        "train": (make(p.train), p.train["rul_capped"]),
        "val": (make(p.val), p.val["rul_capped"]),
        "test": (make(p.test)[final], p.test.loc[final, truth]),
    }


def fit(model: str, data: dict, cfg: Config, seed: int) -> tuple[object, dict]:
    (x_train, y_train), (x_val, y_val) = data["train"], data["val"]
    if model == "ridge":
        fitted, alpha = fit_ridge(
            x_train, y_train, x_val, y_val, cfg.baselines.ridge_alphas
        )
        return fitted, {"alpha": alpha}
    fitted = fit_xgboost(x_train, y_train, x_val, y_val, cfg.baselines.xgboost, seed)
    return fitted, {"best_iteration": fitted.best_iteration}


def score(model: object, data: dict, cap: int) -> dict:
    x_test, y_test = data["test"]
    x_val, y_val = data["val"]
    predicted = np.clip(model.predict(x_test), 0, cap)
    return {
        "rmse": rmse(y_test.to_numpy(), predicted),
        "nasa_score": nasa_score(y_test.to_numpy(), predicted),
        "val_rmse": rmse(y_val.to_numpy(), np.clip(model.predict(x_val), 0, cap)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(parser.parse_args().config)
    run_dir = start_run_dir(cfg, "baselines")
    splits = load_splits(cfg.paths.splits, cfg.data.split_seed, cfg.data.val_fraction)
    reports = Path(cfg.paths.reports)

    rows = []
    for name in cfg.data.subsets:
        subset = load_subset(cfg.paths.raw_dir, name, cfg.data.rul_cap)
        main_setting = cfg.features.condition_aware
        # on multi-regime subsets, rerun XGBoost with the other normalisation as an ablation
        settings = [main_setting]
        if cfg.data.n_conditions[name] > 1:
            settings.append(not main_setting)
        for condition_aware in settings:
            p = prepare_subset(
                subset,
                splits[name],
                cfg.data.n_conditions[name],
                cfg.data.max_modal_share,
                cfg.data.split_seed,
                condition_aware=condition_aware,
            )
            data = design(p, cfg)
            models = (
                ["ridge", "xgboost"] if condition_aware == main_setting else ["xgboost"]
            )
            normalisation = "condition-aware" if condition_aware else "global"
            for seed in cfg.seeds:
                seed_everything(seed)
                for model in models:
                    fitted, extra = fit(model, data, cfg, seed)
                    metrics = score(fitted, data, cfg.data.rul_cap)
                    row = {
                        "subset": name,
                        "model": model,
                        "normalisation": normalisation,
                    }
                    rows.append({**row, "seed": seed, **metrics, **extra})
                    params = {
                        **row,
                        "seed": seed,
                        "n_features": data["train"][0].shape[1],
                    }
                    run_name = f"{name}-{model}-{normalisation}-seed{seed}"
                    log_run(run_name, params | extra, metrics)
                    print(
                        f"{name} {model:8s} {normalisation:15s} seed {seed}: {metrics}"
                    )

    runs = pd.DataFrame(rows)
    runs.to_csv(run_dir / "per_seed.csv", index=False)
    runs.round(4).to_csv(reports / "baselines_per_seed.csv", index=False)
    main_label = "condition-aware" if cfg.features.condition_aware else "global"
    main_runs = runs[runs["normalisation"] == main_label]
    summary = summarise(main_runs, ["subset", "model"], METRICS).round(3)
    summary.to_csv(reports / "baselines.csv", index=False)
    multi = [n for n in cfg.data.subsets if cfg.data.n_conditions[n] > 1]
    ablation = runs[(runs["model"] == "xgboost") & runs["subset"].isin(multi)]
    if not ablation.empty:
        table = summarise(ablation, ["subset", "normalisation"], METRICS).round(3)
        table.to_csv(reports / "normalisation_ablation.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
