"""Report dataset statistics, operating regimes, sensor screening and degradation trends."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from maintadapt.config import load_config
from maintadapt.data import SETTINGS, Subset, load_splits, load_subset
from maintadapt.features import PreparedSubset, prepare_subset
from maintadapt.plotting import INK, MUTED, SERIES, apply_style

EXAMPLE_UNITS = 6
TOP_SENSORS = 3
DISPLAY_SMOOTHING = 10
HORIZON = 250


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def subset_stats(subset: Subset, prepared: PreparedSubset, cap: int) -> dict:
    life = subset.train.groupby("unit_id")["cycle"].max()
    seen = subset.test.groupby("unit_id")["cycle"].max()
    final = subset.test.groupby("unit_id")["rul"].min()
    dropped = prepared.screening.loc[prepared.screening["dropped"], "sensor"]
    return {
        "subset": subset.name,
        "train_units": prepared.train["unit_id"].nunique(),
        "val_units": prepared.val["unit_id"].nunique(),
        "test_units": len(seen),
        "train_file_rows": len(subset.train),
        "test_file_rows": len(subset.test),
        "lifetime_min": life.min(),
        "lifetime_median": life.median(),
        "lifetime_max": life.max(),
        "test_observed_min": seen.min(),
        "test_observed_median": seen.median(),
        "test_observed_max": seen.max(),
        "test_rul_min": final.min(),
        "test_rul_median": final.median(),
        "test_rul_max": final.max(),
        "test_rul_above_cap": int((final > cap).sum()),
        "operating_conditions": len(prepared.conditions.centers),
        "sensors_kept": len(prepared.sensors),
        "sensors_dropped": " ".join(dropped.str.replace("sensor_", "s")),
    }


def regime_table(prepared: PreparedSubset) -> pd.DataFrame:
    centers = prepared.conditions.centers_in_setting_units()
    train = prepared.train
    rows = []
    for k, center in enumerate(centers):
        members = train.loc[train["condition"] == k, SETTINGS].to_numpy()
        rows.append(
            {
                "subset": prepared.name,
                "condition": k,
                # adding 0.0 folds the -0.0 that rounding a tiny negative leaves behind
                **{
                    s: round(float(v), 4) + 0.0
                    for s, v in zip(SETTINGS, center, strict=True)
                },
                "train_rows": len(members),
                "share": round(len(members) / len(train), 4),
                "max_abs_deviation": round(float(np.abs(members - center).max()), 4),
            }
        )
    return pd.DataFrame(rows)


def plot_trajectories(
    prepared: dict[str, PreparedSubset], trends: dict, cap: int, path: Path
):
    fig, axes = plt.subplots(
        len(prepared), TOP_SENSORS, figsize=(11, 2.4 * len(prepared))
    )
    axes = np.atleast_2d(axes)
    for row, (name, p) in zip(axes, prepared.items(), strict=True):
        ranked = sorted(trends[name].items(), key=lambda kv: -abs(kv[1]))[:TOP_SENSORS]
        units = sorted(p.train["unit_id"].unique())[:EXAMPLE_UNITS]
        late = p.train[p.train["rul"] <= HORIZON]
        for ax, (sensor, rho) in zip(row, ranked, strict=True):
            for unit in units:
                trace = p.train[p.train["unit_id"] == unit]
                smooth = trace[sensor].rolling(DISPLAY_SMOOTHING, min_periods=1).mean()
                ax.plot(-trace["rul"], smooth, color=SERIES[0], alpha=0.45, linewidth=1)
            fleet = late.groupby("rul")[sensor].median().sort_index(ascending=False)
            fleet = fleet.rolling(DISPLAY_SMOOTHING, min_periods=1).mean()
            ax.plot(-fleet.index, fleet.to_numpy(), color=INK, linewidth=2)
            ax.axvline(-cap, color=MUTED, linewidth=1, linestyle="--")
            ax.set_xlim(-HORIZON, 0)
            ax.set_title(
                f"{name}  {sensor}  (Spearman with RUL {rho:+.2f})", loc="left"
            )
            ax.set_ylabel("normalised reading")
    for ax in axes[-1]:
        ax.set_xlabel("cycles before failure")
    handles = [
        plt.Line2D([], [], color=SERIES[0], alpha=0.6, linewidth=1),
        plt.Line2D([], [], color=INK, linewidth=2),
        plt.Line2D([], [], color=MUTED, linewidth=1, linestyle="--"),
    ]
    labels = [
        f"{EXAMPLE_UNITS} example engines ({DISPLAY_SMOOTHING}-cycle trailing mean)",
        "fleet median",
        f"RUL cap ({cap} cycles)",
    ]
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def plot_condition_normalisation(
    subset: Subset, prepared: PreparedSubset, sensor: str, path
):
    units = sorted(prepared.train["unit_id"].unique())[:3]
    fig, (raw_ax, norm_ax) = plt.subplots(1, 2, figsize=(11, 3.4))
    for colour, unit in zip(SERIES, units, strict=False):
        raw = subset.train[subset.train["unit_id"] == unit]
        norm = prepared.train[prepared.train["unit_id"] == unit]
        raw_ax.scatter(
            -raw["rul"], raw[sensor], s=6, color=colour, label=f"engine {unit}"
        )
        norm_ax.scatter(-norm["rul"], norm[sensor], s=6, color=colour, alpha=0.35)
        smooth = norm[sensor].rolling(DISPLAY_SMOOTHING, min_periods=1).mean()
        norm_ax.plot(
            -norm["rul"], smooth, color=colour, linewidth=1.6, label=f"engine {unit}"
        )
    raw_ax.set_title(
        f"{subset.name} {sensor}: raw reading jumps between regimes", loc="left"
    )
    norm_ax.set_title("normalised within each operating condition", loc="left")
    raw_ax.set_ylabel("raw reading")
    norm_ax.set_ylabel("z-score within condition")
    for ax in (raw_ax, norm_ax):
        ax.set_xlabel("cycles before failure")
    handles, labels = norm_ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(units), bbox_to_anchor=(0.5, 1.06)
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_lengths(subsets: dict[str, Subset], path: Path) -> None:
    fig, (life_ax, seen_ax) = plt.subplots(1, 2, figsize=(11, 3.2), sharey=False)
    bins = np.arange(0, 560, 20)
    for colour, (name, subset) in zip(SERIES, subsets.items(), strict=False):
        life = subset.train.groupby("unit_id")["cycle"].max()
        seen = subset.test.groupby("unit_id")["cycle"].max()
        life_ax.hist(
            life, bins=bins, histtype="step", linewidth=1.6, color=colour, label=name
        )
        seen_ax.hist(
            seen, bins=bins, histtype="step", linewidth=1.6, color=colour, label=name
        )
    life_ax.set_title("training engines: cycles until failure", loc="left")
    seen_ax.set_title("test engines: cycles observed before truncation", loc="left")
    for ax in (life_ax, seen_ax):
        ax.set_xlabel("cycles")
        ax.set_ylabel("engines")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(parser.parse_args().config)
    splits = load_splits(cfg.paths.splits, cfg.data.split_seed, cfg.data.val_fraction)
    reports = Path(cfg.paths.reports)
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    apply_style()

    subsets, prepared, trends = {}, {}, {}
    stats, screening, regimes = [], [], []
    for name in cfg.data.subsets:
        subset = load_subset(cfg.paths.raw_dir, name, cfg.data.rul_cap)
        p = prepare_subset(
            subset,
            splits[name],
            cfg.data.n_conditions[name],
            cfg.data.max_modal_share,
            cfg.data.split_seed,
            condition_aware=True,
        )
        subsets[name], prepared[name] = subset, p
        trends[name] = {
            s: spearman(p.train[s], p.train["rul_capped"]) for s in p.sensors
        }
        stats.append(subset_stats(subset, p, cfg.data.rul_cap))
        trend = p.screening["sensor"].map(trends[name]).round(3)
        screening.append(p.screening.assign(spearman_rul=trend).assign(subset=name))
        regimes.append(regime_table(p))

    stats_frame = pd.DataFrame(stats)
    stats_frame.to_csv(reports / "dataset_stats.csv", index=False)
    screening_frame = pd.concat(screening, ignore_index=True)
    columns = [
        "subset",
        "sensor",
        "min_modal_share",
        "max_modal_share",
        "dropped",
        "reason",
    ]
    screening_frame = screening_frame[[*columns, "spearman_rul"]].round(4)
    screening_frame.to_csv(reports / "sensor_screening.csv", index=False)
    pd.concat(regimes, ignore_index=True).to_csv(
        reports / "operating_conditions.csv", index=False
    )

    plot_trajectories(
        prepared, trends, cfg.data.rul_cap, figures / "degradation_trajectories.png"
    )
    plot_lengths(subsets, figures / "trajectory_lengths.png")
    multi = [n for n in cfg.data.subsets if cfg.data.n_conditions[n] > 1]
    if multi:
        name = multi[0]
        sensor = max(trends[name], key=lambda s: abs(trends[name][s]))
        path = figures / "condition_normalisation.png"
        plot_condition_normalisation(subsets[name], prepared[name], sensor, path)

    print(stats_frame.to_string(index=False))
    print(screening_frame[screening_frame["dropped"]].to_string(index=False))


if __name__ == "__main__":
    main()
