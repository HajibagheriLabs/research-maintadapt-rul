"""Assemble reports/results.csv and the markdown tables quoted in the README."""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind

from maintadapt.config import load_config

MODEL_NAMES = {
    "ridge": "Ridge, rolling features",
    "xgboost": "XGBoost, rolling features",
    "tcn": "TCN, raw windows",
    "lstm": "LSTM, raw windows",
}
SIGNIFICANCE = 0.05


def cell(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(parser.parse_args().config)
    reports = Path(cfg.paths.reports)

    parts = [reports / "baselines.csv", reports / "sequence.csv"]
    results = pd.concat(
        [pd.read_csv(p) for p in parts if p.exists()], ignore_index=True
    )
    results.to_csv(reports / "results.csv", index=False)

    sections = [
        "## Test RMSE (mean ± std over seeds)",
        markdown(metric_table(results, "rmse", 2)),
        "## Test NASA score (mean ± std over seeds; lower is better)",
        markdown(metric_table(results, "nasa_score", 0)),
    ]
    ablation = reports / "normalisation_ablation.csv"
    if ablation.exists():
        sections += [
            "## XGBoost normalisation ablation",
            markdown(ablation_table(ablation)),
        ]
    if (reports / "sequence_per_seed.csv").exists():
        label = "condition-aware" if cfg.features.condition_aware else "global"
        comparison = deep_versus_xgboost(reports, label)
        comparison.to_csv(reports / "deep_vs_xgboost.csv", index=False)
        sections += [
            "## Deep models against XGBoost (Welch t-test on per-seed test RMSE)",
            markdown(comparison),
        ]

    text = "\n\n".join(sections) + "\n"
    (reports / "results.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
