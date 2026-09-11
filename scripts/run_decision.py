"""Turn RUL predictions into a maintenance policy: sweep, optimum, savings and sensitivity."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from maintadapt.config import Config, load_config
from maintadapt.data import load_splits, load_subset
from maintadapt.decision import (
    CostModel,
    Outcomes,
    fixed_schedule,
    replay_alarms,
    replay_snapshots,
)
from maintadapt.evaluate import summarise
from maintadapt.plotting import INK, INK_SECONDARY, MUTED, SERIES, apply_style

POINT, LOWER = "alarm on predicted RUL", "alarm on lower interval bound"
POLICIES = {POINT: "point", LOWER: "lower"}
REFERENCES = ("run to failure", "fixed schedule")
METRICS = [
    "expected_cost",
    "setting",
    "failure_rate",
    "discarded_cycles",
    "saving_vs_run_to_failure",
    "saving_vs_fixed_schedule",
]


def engines(frame: pd.DataFrame, column: str) -> list[tuple[np.ndarray, np.ndarray]]:
    ordered = frame.sort_values(["unit_id", "cycle"])
    return [
        (unit[column].to_numpy(), unit["rul"].to_numpy(dtype=float))
        for _, unit in ordered.groupby("unit_id", sort=True)
    ]


def outcome(outcomes: Outcomes, costs: CostModel, index: int) -> dict:
    failed = outcomes.failed[index]
    maintained = outcomes.discarded[index][~failed]
    return {
        "expected_cost": float(outcomes.cost(costs)[index]),
        "failure_rate": float(failed.mean()),
        "discarded_cycles": float(maintained.mean()) if maintained.size else np.nan,
    }


def evaluate_policies(cfg: Config, predictions: pd.DataFrame, splits: dict) -> tuple:
    dec, cap = cfg.decision, cfg.data.rul_cap
    thresholds = np.arange(0.0, cap + dec.threshold_step, dec.threshold_step)
    default = dec.failure_cost / dec.cycle_cost
    ratios = sorted(set(dec.cost_ratios) | {default})
    rows, curves, median_life = [], [], {}
    for name in cfg.data.subsets:
        train = load_subset(cfg.paths.raw_dir, name, cap).train
        train = train[train["unit_id"].isin(splits[name]["train"])]
        lives = train.groupby("unit_id")["cycle"].max().to_numpy(dtype=float)
        median_life[name] = float(np.median(lives))
        ages = np.arange(1.0, lives.max() + 1)
        subset = predictions[predictions["subset"] == name]
        first = subset[(subset["split"] == "test") & (subset["seed"] == cfg.seeds[0])]
        test_lives = (first["cycle"] + first["rul"]).groupby(first["unit_id"]).first()

        for ratio in ratios:
            costs = CostModel(failure=ratio * dec.cycle_cost, per_cycle=dec.cycle_cost)
            # the schedule is tuned on training lifetimes, as a fleet manager would tune it
            age = ages[int(np.argmin(fixed_schedule(lives, ages).cost(costs)))]
            schedule = fixed_schedule(test_lives.to_numpy(float), np.array([age]))
            for seed in cfg.seeds:
                base = {"subset": name, "cost_ratio": ratio, "seed": seed}
                failure = {
                    "setting": np.nan,
                    "expected_cost": costs.failure,
                    "failure_rate": 1.0,
                    "discarded_cycles": np.nan,
                }
                rows.append({**base, "policy": REFERENCES[0], **failure})
                fixed = {"setting": age, **outcome(schedule, costs, 0)}
                rows.append({**base, "policy": REFERENCES[1], **fixed})

        for seed in cfg.seeds:
            seeded = subset[subset["seed"] == seed]
            val, test = (
                seeded[seeded["split"] == "val"],
                seeded[seeded["split"] == "test"],
            )
            for policy, column in POLICIES.items():
                on_val = replay_snapshots(engines(val, column), thresholds)
                on_test = replay_alarms(
                    engines(test, column), thresholds, run_to_failure=False
                )
                for ratio in ratios:
                    costs = CostModel(
                        failure=ratio * dec.cycle_cost, per_cycle=dec.cycle_cost
                    )
                    val_cost, test_cost = on_val.cost(costs), on_test.cost(costs)
                    # tuned on validation engines cut at every cycle, never on the test fleet
                    best = int(np.argmin(val_cost))
                    rows.append(
                        {
                            "subset": name,
                            "cost_ratio": ratio,
                            "seed": seed,
                            "policy": policy,
                            "setting": thresholds[best],
                            **outcome(on_test, costs, best),
                            "test_optimal_setting": thresholds[
                                int(np.argmin(test_cost))
                            ],
                            "test_optimal_cost": float(test_cost.min()),
                        }
                    )
                    if ratio == default:
                        curves.append(
                            pd.DataFrame(
                                {
                                    "subset": name,
                                    "policy": policy,
                                    "seed": seed,
                                    "threshold": thresholds,
                                    "test_cost": test_cost,
                                    "val_cost": val_cost,
                                }
                            )
                        )
    runs = pd.DataFrame(rows)
    key = ["subset", "cost_ratio", "seed"]
    for reference in REFERENCES:
        ref = runs.loc[runs["policy"] == reference, [*key, "expected_cost"]]
        cost = runs[key].merge(ref, on=key, how="left")["expected_cost"].to_numpy()
        column = f"saving_vs_{reference.replace(' ', '_')}"
        runs[column] = 1.0 - runs["expected_cost"].to_numpy() / cost
    return runs, pd.concat(curves, ignore_index=True), default, median_life


def plot_cost_curves(
    curve: pd.DataFrame, summary: pd.DataFrame, default: float, path: Path
) -> None:
    subsets = list(curve["subset"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), squeeze=False)
    for ax, name in zip(axes.flat, subsets, strict=False):
        here = summary[(summary["subset"] == name) & (summary["cost_ratio"] == default)]
        here = here.set_index("policy")
        failure = here.loc[REFERENCES[0], "expected_cost_mean"]
        fixed = here.loc[REFERENCES[1], "expected_cost_mean"]
        chosen_lines = []
        for colour, policy in zip(SERIES, POLICIES, strict=False):
            rows = curve[(curve["subset"] == name) & (curve["policy"] == policy)]
            x, mean, std = (
                rows["threshold"],
                rows["test_cost_mean"],
                rows["test_cost_std"],
            )
            ax.plot(x, mean, color=colour, label=policy)
            ax.fill_between(
                x, mean - std, mean + std, color=colour, alpha=0.15, linewidth=0
            )
            chosen = here.loc[policy, "setting_mean"]
            cost = here.loc[policy, "expected_cost_mean"]
            ax.plot(
                [chosen], [cost], "o", markersize=8, color=colour, markeredgecolor=INK
            )
            chosen_lines.append(f"{policy}: threshold {chosen:.0f}, cost {cost:.1f}")
        # high thresholds are always costly, so the lower right corner stays free for labels
        ax.text(
            0.98,
            0.04,
            "\n".join(chosen_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=INK_SECONDARY,
        )
        ax.axhline(
            failure, color=MUTED, linestyle="--", linewidth=1, label=REFERENCES[0]
        )
        ax.axhline(
            fixed, color=INK, linestyle=":", linewidth=1.2, label="best fixed schedule"
        )
        ax.set_ylim(0, failure * 1.08)
        ax.set_title(f"{name}", loc="left")
        ax.set_xlabel("alarm threshold (cycles of predicted RUL)")
        ax.set_ylabel("expected cost per engine")
    handles, labels = axes[0][0].get_legend_handles_labels()
    title = f"Expected maintenance cost per test engine, failure costs {default:g} cycles of life"
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def plot_sensitivity(summary: pd.DataFrame, path: Path) -> None:
    rows = summary[summary["policy"] == POINT]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = [
        ("saving_vs_run_to_failure_mean", "saving against run to failure"),
        ("saving_vs_fixed_schedule_mean", "saving against the best fixed schedule"),
        ("setting_mean", "chosen alarm threshold (cycles)"),
    ]
    for ax, (column, title) in zip(axes, panels, strict=True):
        for colour, (name, group) in zip(
            SERIES, rows.groupby("subset", sort=True), strict=False
        ):
            ax.plot(
                group["cost_ratio"], group[column], marker="o", color=colour, label=name
            )
        ax.set_xscale("log")
        ax.set_xlabel("failure cost in cycles of discarded life")
        ax.set_title(title, loc="left")
        if column.startswith("saving"):
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def markdown(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |"]
    lines.append("|" + "|".join("---" for _ in frame.columns) + "|")
    lines += [
        "| " + " | ".join(map(str, r)) + " |" for r in frame.itertuples(index=False)
    ]
    return "\n".join(lines)


def results_table(at: pd.DataFrame) -> str:
    def pct(mean: float, std: float) -> str:
        return "" if np.isnan(mean) else f"{mean:.0%} ± {std:.0%}"

    rows = []
    for r in at.itertuples(index=False):
        setting = (
            ""
            if np.isnan(r.setting_mean)
            else f"{r.setting_mean:.0f} ± {r.setting_std:.0f}"
        )
        discarded = (
            ""
            if np.isnan(r.discarded_cycles_mean)
            else f"{r.discarded_cycles_mean:.1f}"
        )
        rows.append(
            {
                "subset": r.subset,
                "policy": r.policy,
                "threshold or age": setting,
                "expected cost per engine": f"{r.expected_cost_mean:.1f} ± {r.expected_cost_std:.1f}",
                "failure rate": f"{r.failure_rate_mean:.0%}",
                "cycles discarded per maintained engine": discarded,
                "saving vs run to failure": pct(
                    r.saving_vs_run_to_failure_mean, r.saving_vs_run_to_failure_std
                ),
                "saving vs fixed schedule": pct(
                    r.saving_vs_fixed_schedule_mean, r.saving_vs_fixed_schedule_std
                ),
            }
        )
    return markdown(pd.DataFrame(rows))


def sensitivity_table(summary: pd.DataFrame, column: str, fmt: str) -> str:
    rows = summary[summary["policy"] == POINT]
    table = rows.pivot(index="cost_ratio", columns="subset", values=column)
    table = table.map(lambda v: format(v, fmt)).reset_index()
    table["cost_ratio"] = table["cost_ratio"].map(lambda v: f"{v:g}")
    return markdown(table.rename(columns={"cost_ratio": "failure cost (cycles)"}))


def break_even_note(summary: pd.DataFrame) -> str:
    """Operator warning naming the cost ratios at which the alarm loses to running to failure."""
    losing = summary["saving_vs_run_to_failure_mean"] <= 0
    rows = summary[(summary["policy"] == POINT) & losing]
    if rows.empty:
        return "The alarm beats running to failure at every cost ratio studied."
    losses = [
        f"{ratio:g} cycles ({', '.join(group['subset'])})"
        for ratio, group in rows.groupby("cost_ratio", sort=True)
    ]
    return (
        "Do not use the alarm when failures are cheap: it costs more than running to "
        f"failure at a failure cost of {'; '.join(losses)}."
    )


def operator_note(
    cfg: Config, summary: pd.DataFrame, default: float, median_life: dict
) -> str:
    dec = cfg.decision
    at = summary[summary["cost_ratio"] == default]
    lead = cfg.data.subsets[0]
    head = at[at["subset"] == lead].set_index("policy")
    point, lower = head.loc[POINT], head.loc[LOWER]
    per_subset = at.pivot(index="subset", columns="policy", values="expected_cost_mean")
    better = int((per_subset[LOWER] < per_subset[POINT]).sum())
    n = len(per_subset)
    if better > n / 2:
        interval_advice = (
            f"Alarming on the lower bound of the calibrated {cfg.uncertainty.nominal:.0%} "
            f"interval is cheaper than alarming on the point prediction on {better} of {n} "
            "subsets, so prefer it: it pulls engines whose prediction is uncertain a little "
            "earlier and leaves confident ones alone."
        )
    else:
        interval_advice = (
            f"Alarming on the lower bound of the calibrated {cfg.uncertainty.nominal:.0%} "
            f"interval is cheaper than the point prediction on only {better} of {n} subsets. "
            "Once the threshold is tuned, the interval adds little here: the tuned threshold "
            "already carries the safety margin the interval would supply."
        )
    life = median_life[lead]
    return f"""# Decision analysis: when to pull an engine for maintenance

Generated by `scripts/run_decision.py` from `reports/decision_analysis.csv`. Rerun `make eval` to
refresh every number below.

## The decision

After every flight cycle the operator sees a predicted remaining useful life (RUL) for each engine
and decides whether to schedule maintenance now. Pulling an engine early throws away the cycles it
had left. Leaving it too long risks an unplanned failure in service.

## Cost model (assumed, not measured)

- Unplanned failure: {dec.failure_cost:g} cost units per event.
- Premature maintenance: {dec.cycle_cost:g} cost unit for every cycle of remaining life discarded.
- Only the ratio matters. The headline assumes one unplanned failure costs as much as throwing
  away {default:g} cycles of healthy life, about {default / life:.0%} of the median {lead}
  lifetime of {life:.0f} cycles. The sensitivity section shows how the answer moves if the true
  ratio is different.

## Policy and how it is evaluated

Raise a maintenance alarm the first time the predicted RUL is at or below a threshold. Test
histories stop before failure, so when no alarm has fired by the last observed cycle, maintenance
is booked for the cycle at which the last prediction says the engine will reach the threshold; if
the engine's true life runs out first, it fails. The threshold is tuned on validation engines under
that same rule: every validation history is cut at each cycle in turn, booked the same way, and
scored against its known failure time. It is then applied unchanged to the official test engines.
Predictions come from the XGBoost baseline; the interval-aware variant alarms on the lower end of
its calibrated {cfg.uncertainty.nominal:.0%} interval instead of the point prediction.

Two reference policies:

- **Run to failure**: never maintain early, so every engine fails in service.
- **Fixed schedule**: maintain every engine at the same age, tuned on training-engine lifetimes to
  minimise expected cost.

## Results at a failure cost of {default:g} cycles (mean ± std over {len(cfg.seeds)} seeds)

{results_table(at)}

## What an operator should do

1. Raise a maintenance alarm when an engine's predicted RUL falls to about
   **{point["setting_mean"]:.0f} cycles** ({lead}). On the test fleet this cuts expected
   maintenance cost per engine by **{point["saving_vs_run_to_failure_mean"]:.0%}** against running
   to failure and by **{point["saving_vs_fixed_schedule_mean"]:.0%}** against the best fixed
   schedule, while {point["failure_rate_mean"]:.0%} of engines still fail in service and each
   maintained engine gives up {point["discarded_cycles_mean"]:.0f} cycles on average.
2. {interval_advice} On {lead}, the interval-aware alarm costs
   {lower["expected_cost_mean"]:.1f} per engine against {point["expected_cost_mean"]:.1f} for the
   point alarm.
3. Re-derive the threshold for your own cost ratio from the tables below. The more a failure
   costs relative to a cycle of life, the earlier the alarm should fire.
4. Add your maintenance lead time to the threshold. The model assumes an engine can be serviced in
   the cycle the alarm fires.
5. {break_even_note(summary)}

## Sensitivity to the assumed cost ratio (alarm on predicted RUL)

Chosen alarm threshold, in cycles of predicted RUL:

{sensitivity_table(summary, "setting_mean", ".0f")}

Saving against running to failure:

{sensitivity_table(summary, "saving_vs_run_to_failure_mean", ".0%")}

Saving against the best fixed schedule:

{sensitivity_table(summary, "saving_vs_fixed_schedule_mean", ".0%")}

## Assumptions and limits

- The cost ratio is an assumption. Real ratios depend on the operator, the engine type, secondary
  damage and whether a spare engine is available.
- Maintenance is priced only through discarded life; its fixed cost, capacity limits and lead time
  are left out, and it is assumed to restore the engine completely.
- C-MAPSS engines are simulated, one fault mode or two, with clean labels. Real fleets add sensor
  drift, maintenance history and censored failures.
- Test histories stop before failure, so for engines without an alarm the booking relies on the
  last available prediction. That is conservative for the predictive policy: a real operator would
  keep re-predicting, and predictions sharpen as failure approaches.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(parser.parse_args().config)
    source = Path(cfg.paths.runs) / "uncertainty" / "xgboost_predictions.pkl"
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} is missing; run scripts/run_uncertainty.py first"
        )
    splits = load_splits(cfg.paths.splits, cfg.data.split_seed, cfg.data.val_fraction)
    runs, curves, default, median_life = evaluate_policies(
        cfg, pd.read_pickle(source), splits
    )

    reports = Path(cfg.paths.reports)
    runs.round(4).to_csv(reports / "decision_per_seed.csv", index=False)
    summary = summarise(runs, ["subset", "cost_ratio", "policy"], METRICS).round(4)
    summary.to_csv(reports / "decision_analysis.csv", index=False)
    curve = summarise(
        curves, ["subset", "policy", "threshold"], ["test_cost", "val_cost"]
    )
    curve.round(4).to_csv(reports / "decision_curve.csv", index=False)
    apply_style()
    plot_cost_curves(
        curve, summary, default, reports / "figures" / "cost_vs_threshold.png"
    )
    plot_sensitivity(summary, reports / "figures" / "cost_sensitivity.png")
    note = operator_note(cfg, summary, default, median_life)
    (reports / "decision_analysis.md").write_text(note, encoding="utf-8", newline="\n")
    print(summary[summary["cost_ratio"] == default].to_string(index=False))


if __name__ == "__main__":
    main()
