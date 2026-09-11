"""Assemble reports/results.csv and the markdown tables quoted in the README."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from maintadapt.config import Config, load_config

MODEL_NAMES = {
    "ridge": "Ridge, rolling features",
    "xgboost": "XGBoost, rolling features",
    "tcn": "TCN, raw windows",
    "lstm": "LSTM, raw windows",
}
INTERVAL_NAMES = {
    "xgboost": "XGBoost quantile regression",
    "tcn": "TCN MC dropout",
    "lstm": "LSTM MC dropout",
}
SIGNIFICANCE = 0.05


def cell(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def share(mean: float, std: float) -> str:
    return f"{mean:.0%} ± {std:.0%}"


def markdown(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |"]
    lines.append("|" + "|".join("---" for _ in frame.columns) + "|")
    lines += [
        "| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)
    ]
    return "\n".join(lines)


def metric_table(results: pd.DataFrame, metric: str, digits: int) -> pd.DataFrame:
    means, stds = results[f"{metric}_mean"], results[f"{metric}_std"]
    cells = [cell(m, s, digits) for m, s in zip(means, stds, strict=True)]
    table = results.assign(value=cells).pivot(
        index="model", columns="subset", values="value"
    )
    table = table.reindex([m for m in MODEL_NAMES if m in table.index])
    return table.rename(index=MODEL_NAMES).reset_index(names="model")


def ablation_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    for metric, digits in (("rmse", 2), ("nasa_score", 0)):
        pairs = zip(table[f"{metric}_mean"], table[f"{metric}_std"], strict=True)
        table[metric] = [cell(m, s, digits) for m, s in pairs]
    return table[["subset", "normalisation", "rmse", "nasa_score"]]


def deep_versus_xgboost(reports: Path, normalisation: str) -> pd.DataFrame:
    """Welch t-test of each deep model's per-seed test RMSE against XGBoost's."""
    baselines = pd.read_csv(reports / "baselines_per_seed.csv")
    baselines = baselines[baselines["normalisation"] == normalisation]
    runs = pd.concat([baselines, pd.read_csv(reports / "sequence_per_seed.csv")])
    rows = []
    for subset, group in runs.groupby("subset"):
        reference = group.loc[group["model"] == "xgboost", "rmse"]
        for model in ("tcn", "lstm"):
            deep = group.loc[group["model"] == model, "rmse"]
            if deep.empty:
                continue
            gap = deep.mean() - reference.mean()
            p_value = float(ttest_ind(deep, reference, equal_var=False).pvalue)
            if p_value >= SIGNIFICANCE:
                verdict = "no significant difference"
            else:
                verdict = f"{model.upper()} better" if gap < 0 else "XGBoost better"
            rows.append(
                {
                    "subset": subset,
                    "model": model.upper(),
                    "rmse_gap_vs_xgboost": round(gap, 2),
                    "welch_p_value": round(p_value, 4),
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


def p_label(p_value: float) -> str:
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def transfer_table(path: Path, source: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    runs = pd.read_csv(path.with_name("transfer_per_seed.csv"))
    pairs = zip(table["rmse_mean"], table["rmse_std"], strict=True)
    table["rmse"] = [cell(m, s, 2) for m, s in pairs]
    gain = "gain from fine-tuning (Welch)"
    rows = []
    for target, group in table.groupby("target", sort=True):
        zero = group.loc[group["condition"] == "source-only", "rmse"].iloc[0]
        rows.append(
            {
                "target": target,
                "target engines used": "none",
                "trained from scratch": "",
                f"fine-tuned from {source}": "",
                gain: "",
                f"{source} model, zero-shot": zero,
            }
        )
        trained = group[group["condition"] != "source-only"]
        for budget, by_budget in trained.groupby("budget", sort=True):
            arms = by_budget.set_index("condition")
            engines = int(arms["target_units"].iloc[0])
            here = runs[(runs["target"] == target) & np.isclose(runs["budget"], budget)]
            scratch = here.loc[here["condition"] == "target-only", "rmse"]
            tuned = here.loc[here["condition"] == "fine-tuned", "rmse"]
            p_value = float(ttest_ind(tuned, scratch, equal_var=False).pvalue)
            rows.append(
                {
                    "target": target,
                    "target engines used": f"{engines} ({budget:.0%})",
                    "trained from scratch": arms.loc["target-only", "rmse"],
                    f"fine-tuned from {source}": arms.loc["fine-tuned", "rmse"],
                    gain: f"{scratch.mean() - tuned.mean():+.2f} ({p_label(p_value)})",
                    f"{source} model, zero-shot": "",
                }
            )
    return pd.DataFrame(rows)


def coverage_table(path: Path, level: float) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table[np.isclose(table["level"], level)]
    return pd.DataFrame(
        {
            "subset": table["subset"],
            "interval": table["model"].map(INTERVAL_NAMES),
            "coverage": [
                share(m, s)
                for m, s in zip(table["coverage_mean"], table["coverage_std"])
            ],
            "mean width (cycles)": [
                cell(m, s, 1) for m, s in zip(table["width_mean"], table["width_std"])
            ],
            "coverage after conformal calibration": [
                share(m, s)
                for m, s in zip(
                    table["coverage_conformal_mean"], table["coverage_conformal_std"]
                )
            ],
            "calibrated width (cycles)": [
                cell(m, s, 1)
                for m, s in zip(
                    table["width_conformal_mean"], table["width_conformal_std"]
                )
            ],
        }
    )


def decision_table(path: Path, ratio: float) -> pd.DataFrame:
    table = pd.read_csv(path)
    at = table[np.isclose(table["cost_ratio"], ratio)]
    rows = []
    for r in at.itertuples(index=False):
        rows.append(
            {
                "subset": r.subset,
                "policy": r.policy,
                "threshold or age": ""
                if np.isnan(r.setting_mean)
                else f"{r.setting_mean:.0f}",
                "expected cost per engine": cell(
                    r.expected_cost_mean, r.expected_cost_std, 1
                ),
                "failure rate": f"{r.failure_rate_mean:.0%}",
                "saving vs run to failure": ""
                if np.isnan(r.saving_vs_run_to_failure_mean)
                else share(
                    r.saving_vs_run_to_failure_mean, r.saving_vs_run_to_failure_std
                ),
                "saving vs fixed schedule": ""
                if np.isnan(r.saving_vs_fixed_schedule_mean)
                else share(
                    r.saving_vs_fixed_schedule_mean, r.saving_vs_fixed_schedule_std
                ),
            }
        )
    return pd.DataFrame(rows)


def sections(cfg: Config, reports: Path) -> list[str]:
    parts = [reports / "baselines.csv", reports / "sequence.csv"]
    results = pd.concat(
        [pd.read_csv(p) for p in parts if p.exists()], ignore_index=True
    )
    results.to_csv(reports / "results.csv", index=False)
    out = [
        "## Test RMSE (mean ± std over seeds)",
        markdown(metric_table(results, "rmse", 2)),
        "## Test NASA score (mean ± std over seeds; lower is better)",
        markdown(metric_table(results, "nasa_score", 0)),
    ]
    if (reports / "normalisation_ablation.csv").exists():
        table = ablation_table(reports / "normalisation_ablation.csv")
        out += ["## XGBoost normalisation ablation", markdown(table)]
    if (reports / "sequence_per_seed.csv").exists():
        label = "condition-aware" if cfg.features.condition_aware else "global"
        comparison = deep_versus_xgboost(reports, label)
        comparison.to_csv(reports / "deep_vs_xgboost.csv", index=False)
        title = "## Deep models against XGBoost (Welch t-test on per-seed test RMSE)"
        out += [title, markdown(comparison)]
    if (reports / "transfer.csv").exists():
        table = transfer_table(reports / "transfer.csv", cfg.transfer.source)
        out += ["## Transfer: test RMSE on the target fleet", markdown(table)]
    if (reports / "uncertainty.csv").exists():
        level = cfg.uncertainty.nominal
        table = coverage_table(reports / "uncertainty.csv", level)
        out += [
            f"## Coverage of the nominal {level:.0%} interval on test",
            markdown(table),
        ]
    if (reports / "decision_analysis.csv").exists():
        ratio = cfg.decision.failure_cost / cfg.decision.cycle_cost
        table = decision_table(reports / "decision_analysis.csv", ratio)
        out += [
            f"## Maintenance policy at a failure cost of {ratio:g} cycles",
            markdown(table),
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(parser.parse_args().config)
    reports = Path(cfg.paths.reports)
    text = "\n\n".join(sections(cfg, reports)) + "\n"
    (reports / "results.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
