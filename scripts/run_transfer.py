"""Transfer from the source fleet to each target: zero-shot, scratch, fine-tuned, budgets."""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from maintadapt.config import Config, load_config
from maintadapt.data import load_splits, load_subset
from maintadapt.evaluate import nasa_score, rmse, summarise
from maintadapt.experiment import (
    exit_without_teardown,
    finished_runs,
    record_run,
    seed_everything,
    start_run_dir,
)
from maintadapt.features import PreparedSubset, last_cycle, prepare_subset
from maintadapt.plotting import MUTED, SERIES, apply_style
from maintadapt.sequence import (
    build_model,
    make_windows,
    predict,
    resolve_device,
    train_model,
)
from maintadapt.transfer import budget_units, shared_sensors

METRICS = ["rmse", "nasa_score"]
ARMS = ("target-only", "fine-tuned")


def prepare(
    cfg: Config, name: str, splits: dict, sensors: list[str] | None
) -> PreparedSubset:
    return prepare_subset(
        load_subset(cfg.paths.raw_dir, name, cfg.data.rul_cap),
        splits[name],
        cfg.data.n_conditions[name],
        cfg.data.max_modal_share,
        cfg.data.split_seed,
        condition_aware=cfg.features.condition_aware,
        sensors=sensors,
    )


def plot_budget_curves(
    summary: pd.DataFrame, units: dict[str, int], path: Path
) -> None:
    targets = list(summary["target"].unique())
    fig, axes = plt.subplots(
        1, len(targets), figsize=(5.6 * len(targets), 3.8), squeeze=False
    )
    for ax, target in zip(axes[0], targets, strict=True):
        rows = summary[summary["target"] == target]
        for colour, arm in zip(SERIES, ARMS, strict=False):
            curve = rows[rows["condition"] == arm].sort_values("budget")
            x, mean, std = curve["budget"] * 100, curve["rmse_mean"], curve["rmse_std"]
            ax.plot(x, mean, marker="o", markersize=6, color=colour, label=arm)
            ax.fill_between(
                x, mean - std, mean + std, color=colour, alpha=0.15, linewidth=0
            )
        zero_shot = rows.loc[rows["condition"] == "source-only", "rmse_mean"].iloc[0]
        ax.axhline(
            zero_shot,
            color=MUTED,
            linestyle="--",
            linewidth=1,
            label="source-only, zero-shot",
        )
        budgets = sorted(rows.loc[rows["condition"] == "fine-tuned", "budget"])
        ax.set_xscale("log")
        ax.set_xticks([b * 100 for b in budgets])
        ax.set_xticklabels(
            [f"{b:.0%}\n{round(b * units[target])} engines" for b in budgets]
        )
        ax.minorticks_off()
        ax.set_xlabel("share of target training engines available")
        ax.set_ylabel("test RMSE (cycles)")
        ax.set_title(f"FD001 to {target}", loc="left")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(parser.parse_args().config)
    run_dir = start_run_dir(cfg, "transfer")
    splits = load_splits(cfg.paths.splits, cfg.data.split_seed, cfg.data.val_fraction)
    device = resolve_device(cfg.sequence.device)
    tr, seq, cap = cfg.transfer, cfg.sequence, cfg.data.rul_cap
    truth = "rul_capped" if cfg.evaluation.cap_test_truth else "rul"
    names = [tr.source, *tr.targets]

    sensors = shared_sensors([prepare(cfg, n, splits, None) for n in names])
    prepared = {n: prepare(cfg, n, splits, sensors) for n in names}
    print(f"shared sensors ({len(sensors)}): {sensors}")

    def windows(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return make_windows(frame, sensors, seq.window), frame["rul_capped"].to_numpy()

    tests = {}
    for n in tr.targets:
        test = prepared[n].test
        x_test = make_windows(test, sensors, seq.window, last_only=True)
        tests[n] = (x_test, test.loc[last_cycle(test), truth].to_numpy())

    rows = finished_runs(run_dir)
    done = {
        (r["target"], r["condition"], float(r["budget"]), int(r["seed"])) for r in rows
    }

    def record(
        target: str, condition: str, budget: float, seed: int, model, extra: dict
    ) -> None:
        x_test, y_test = tests[target]
        predicted = predict(model, x_test, cap, device, seq.training.batch_size)
        metrics = {
            "rmse": rmse(y_test, predicted),
            "nasa_score": nasa_score(y_test, predicted),
        }
        key = {"target": target, "condition": condition, "budget": budget, "seed": seed}
        run_name = f"{target}-{condition}-{budget:g}-seed{seed}"
        record_run(run_dir, run_name, {**key, **metrics, **extra}, key, metrics)
        rows.append({**key, **metrics, **extra})
        print(f"{run_name}: {metrics}")

    source = prepared[tr.source]
    source_states = {}
    for seed in cfg.seeds:
        model = build_model(tr.model, len(sensors), seq).to(device)
        checkpoint = run_dir / f"source_{tr.source}_seed{seed}.pt"
        # a saved source model is reused only if its runs were recorded under this config
        recorded = all((t, "source-only", 0.0, seed) in done for t in tr.targets)
        if recorded and checkpoint.is_file():
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state["state_dict"])
        else:
            seed_everything(seed)
            train, val = windows(source.train), windows(source.val)
            model = train_model(
                model, train, val, seq.training, cap, device, checkpoint
            ).model
        source_states[seed] = copy.deepcopy(model.state_dict())
        for target in tr.targets:
            if (target, "source-only", 0.0, seed) not in done:
                record(target, "source-only", 0.0, seed, model, {"target_units": 0})

    for target in tr.targets:
        p = prepared[target]
        val = windows(p.val)
        for seed in cfg.seeds:
            for budget in tr.budgets:
                units = budget_units(splits[target]["train"], budget, seed)
                train = windows(p.train[p.train["unit_id"].isin(units)])
                for condition in ARMS:
                    if (target, condition, float(budget), seed) in done:
                        continue
                    seed_everything(seed)
                    model = build_model(tr.model, len(sensors), seq)
                    lr = None
                    if condition == "fine-tuned":
                        model.load_state_dict(source_states[seed])
                        lr = tr.finetune_lr
                    checkpoint = (
                        run_dir / target / f"{condition}_{budget:g}_seed{seed}.pt"
                    )
                    result = train_model(
                        model, train, val, seq.training, cap, device, checkpoint, lr=lr
                    )
                    extra = {
                        "target_units": len(units),
                        "best_epoch": result.best_epoch,
                    }
                    record(target, condition, float(budget), seed, result.model, extra)

    runs = pd.DataFrame(rows).sort_values(["target", "condition", "budget", "seed"])
    reports = Path(cfg.paths.reports)
    runs.round(4).to_csv(reports / "transfer_per_seed.csv", index=False)
    group = ["target", "condition", "budget"]
    summary = summarise(runs, group, METRICS)
    units = runs.groupby(group, sort=False)["target_units"].first().reset_index()
    summary = summary.merge(units, on=group).round(3)
    summary.to_csv(reports / "transfer.csv", index=False)
    apply_style()
    n_units = {t: len(splits[t]["train"]) for t in tr.targets}
    plot_budget_curves(summary, n_units, reports / "figures" / "transfer_budget.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
    exit_without_teardown()
