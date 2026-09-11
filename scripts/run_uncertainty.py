"""Prediction intervals from quantile XGBoost and MC dropout, with coverage on test engines."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from maintadapt.baselines import fit_xgboost
from maintadapt.config import load_config
from maintadapt.data import load_splits, load_subset
from maintadapt.evaluate import summarise
from maintadapt.experiment import exit_without_teardown, seed_everything, start_run_dir
from maintadapt.features import build_features, last_cycle, prepare_subset
from maintadapt.plotting import MUTED, SERIES, apply_style
from maintadapt.sequence import (
    build_model,
    make_windows,
    predict_mc_dropout,
    resolve_device,
)
from maintadapt.uncertainty import (
    conformal_margin,
    coverage,
    fit_quantile_xgboost,
    interval_quantiles,
    sample_interval,
)

METRICS = ["coverage", "width", "coverage_conformal", "width_conformal"]
LABELS = {
    "xgboost": "XGBoost quantile regression",
    "tcn": "TCN MC dropout",
    "lstm": "LSTM MC dropout",
}


def interval_row(key: dict, level: float, val: tuple, test: tuple) -> dict:
    """Raw and conformal coverage at one level; val and test are (truth, lower, upper)."""
    margin = conformal_margin(*val, level)
    truth, lower, upper = test
    width = float(np.mean(upper - lower))
    return {
        **key,
        "level": level,
        "coverage": coverage(truth, lower, upper),
        "width": width,
        "coverage_conformal": coverage(truth, lower - margin, upper + margin),
        "width_conformal": width + 2 * margin,
        "margin": margin,
    }


def plot_reliability(summary: pd.DataFrame, subsets: list[str], path: Path) -> None:
    fig, axes = plt.subplots(
        1, len(subsets), figsize=(3.3 * len(subsets), 3.5), squeeze=False
    )
    for ax, subset in zip(axes[0], subsets, strict=True):
        ax.plot(
            [0.4, 1.0],
            [0.4, 1.0],
            color=MUTED,
            linewidth=1,
            label="perfect calibration",
        )
        rows = summary[summary["subset"] == subset]
        for colour, (model, group) in zip(
            SERIES, rows.groupby("model", sort=False), strict=False
        ):
            group = group.sort_values("level")
            ax.plot(
                group["level"],
                group["coverage_mean"],
                marker="o",
                color=colour,
                label=LABELS[model],
            )
            ax.plot(
                group["level"],
                group["coverage_conformal_mean"],
                marker="s",
                markerfacecolor="none",
                linestyle=":",
                color=colour,
                label=f"{LABELS[model]}, conformal",
            )
        ax.set_ylim(0, 1.02)
        ax.set_title(subset, loc="left")
        ax.set_xlabel("nominal interval level")
    axes[0][0].set_ylabel("empirical coverage on test engines")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(parser.parse_args().config)
    run_dir = start_run_dir(cfg, "uncertainty")
    splits = load_splits(cfg.paths.splits, cfg.data.split_seed, cfg.data.val_fraction)
    device = resolve_device(cfg.sequence.device)
    unc, cap, batch = (
        cfg.uncertainty,
        cfg.data.rul_cap,
        cfg.sequence.training.batch_size,
    )
    quantiles = interval_quantiles(unc.levels)
    truth = "rul_capped" if cfg.evaluation.cap_test_truth else "rul"

    def bounds(level: float) -> tuple[int, int]:
        return quantiles.index(round((1 - level) / 2, 6)), quantiles.index(
            round((1 + level) / 2, 6)
        )

    rows, predictions = [], []
    for name in cfg.data.subsets:
        p = prepare_subset(
            load_subset(cfg.paths.raw_dir, name, cap),
            splits[name],
            cfg.data.n_conditions[name],
            cfg.data.max_modal_share,
            cfg.data.split_seed,
            condition_aware=cfg.features.condition_aware,
        )
        rolling, with_age = cfg.features.windows, cfg.features.include_cycle
        x_train, x_val, x_test = [
            build_features(f, p.sensors, rolling, with_age)
            for f in (p.train, p.val, p.test)
        ]
        y_val = p.val["rul_capped"].to_numpy()
        final = last_cycle(p.test).to_numpy()
        y_test = p.test.loc[final, truth].to_numpy()

        for seed in cfg.seeds:
            seed_everything(seed)
            args = (
                x_train,
                p.train["rul_capped"],
                x_val,
                p.val["rul_capped"],
                cfg.baselines.xgboost,
                seed,
            )
            point = fit_xgboost(*args)
            booster = fit_quantile_xgboost(*args, quantiles)
            # quantiles are boosted jointly but can still cross, so sort before reading intervals
            q_val = np.sort(np.clip(booster.predict(x_val), 0, cap), axis=1)
            q_test = np.sort(np.clip(booster.predict(x_test), 0, cap), axis=1)
            key = {"subset": name, "model": "xgboost", "seed": seed}
            for level in unc.levels:
                lo, hi = bounds(level)
                val = (y_val, q_val[:, lo], q_val[:, hi])
                test = (y_test, q_test[final, lo], q_test[final, hi])
                rows.append(interval_row(key, level, val, test))
            lo, hi = bounds(unc.nominal)
            margin = conformal_margin(y_val, q_val[:, lo], q_val[:, hi], unc.nominal)
            for split, frame, x, q in (
                ("val", p.val, x_val, q_val),
                ("test", p.test, x_test, q_test),
            ):
                predictions.append(
                    pd.DataFrame(
                        {
                            "subset": name,
                            "seed": seed,
                            "split": split,
                            "unit_id": frame["unit_id"].to_numpy(),
                            "cycle": frame["cycle"].to_numpy(),
                            "rul": frame["rul"].to_numpy(),
                            "point": np.clip(point.predict(x), 0, cap),
                            "lower": q[:, lo] - margin,
                            "upper": q[:, hi] + margin,
                        }
                    )
                )

        windows_val = make_windows(p.val, p.sensors, cfg.sequence.window)
        windows_test = make_windows(
            p.test, p.sensors, cfg.sequence.window, last_only=True
        )
        for kind in cfg.sequence.models:
            for seed in cfg.seeds:
                path = (
                    Path(cfg.paths.runs) / "sequence" / name / f"{kind}_seed{seed}.pt"
                )
                if not path.is_file():
                    raise FileNotFoundError(
                        f"{path} is missing; run scripts/train_sequence.py first"
                    )
                model = build_model(kind, len(p.sensors), cfg.sequence).to(device)
                state = torch.load(path, map_location=device, weights_only=True)[
                    "state_dict"
                ]
                model.load_state_dict(state)
                seed_everything(seed)
                draws_val = predict_mc_dropout(
                    model, windows_val, cap, device, batch, unc.mc_samples
                )
                draws_test = predict_mc_dropout(
                    model, windows_test, cap, device, batch, unc.mc_samples
                )
                key = {"subset": name, "model": kind, "seed": seed}
                for level in unc.levels:
                    val = (y_val, *sample_interval(draws_val, level))
                    test = (y_test, *sample_interval(draws_test, level))
                    rows.append(interval_row(key, level, val, test))
            print(f"{name} {kind}: MC dropout done")
        print(f"{name}: intervals done")

    runs = pd.DataFrame(rows)
    reports = Path(cfg.paths.reports)
    runs.round(4).to_csv(reports / "uncertainty_per_seed.csv", index=False)
    summary = summarise(runs, ["subset", "model", "level"], METRICS).round(3)
    summary.to_csv(reports / "uncertainty.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_pickle(
        run_dir / "xgboost_predictions.pkl"
    )
    apply_style()
    plot_reliability(
        summary, cfg.data.subsets, reports / "figures" / "interval_coverage.png"
    )
    print(summary[summary["level"] == unc.nominal].to_string(index=False))


if __name__ == "__main__":
    main()
    exit_without_teardown()
