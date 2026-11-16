"""Paper assets from the committed reports and the archive (make paper-assets).

Writes paper_assets/figures (PDF with embedded fonts, PNG previews), paper_assets/tables
(booktabs), paper_assets/numbers.tex, paper_assets/RESULTS_DIGEST.md and, for the few numbers
computed here rather than read from reports/research, paper_assets/data/. Assets only: no
manuscript prose. Every number names its source file, column and command in the digest.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyArrowPatch, Rectangle

from maintadapt.data import load_subset
from maintadapt.plotting import GRID, INK, INK_SECONDARY, MUTED, SERIES
from maintadapt.research.analysis import (
    choose,
    decision_loss,
    estimate,
    fleet_inputs,
    selected_share,
)
from maintadapt.research.archive import load
from maintadapt.research.config import (
    NCMAPSS,
    ResearchConfig,
    fleet_number,
    load_research_config,
)
from maintadapt.research.crossfit import CPU_FAMILIES, git_commit
from maintadapt.research.derived import (
    age_table,
    fit_metrics,
    metric_components,
    repeat_outcomes,
    savings,
    truncation_points,
)
from maintadapt.research.folds import lifetimes, load_folds
from maintadapt.research.models.spaces import FAMILY_NAMES
from maintadapt.research.paper import (
    DOUBLE,
    FAMILY_LABELS,
    FLEETS,
    HOLLOW,
    VARIANTS,
    apply_paper_style,
    booktabs,
    family_handle,
    family_of,
    family_style,
    interval,
    number,
    numbers_tex,
    panel_label,
    save,
    scatter_style,
    tex,
    zero_line,
)
from maintadapt.research.secondary import pool_order
from maintadapt.research.stats import bootstrap_counts

COMMANDS = {
    "reports/research/confirmatory": "make research-confirmatory",
    "reports/research/secondary": "make research-secondary",
    "reports/research/exploratory": "make research-exploratory",
    "reports/research/ncmapss": "make research-ncmapss-report",
    "reports/research/leaderboard": "make research-leaderboard",
    "reports/research/compute_estimate": "make research-dryrun",
    "paper_assets/data": "make paper-assets",
    "configs/research_pool.yaml": "make research-pool",
}
VERDICT_FILE = "reports/research/confirmatory/verdicts.md"
F1 = {
    "fleet": "FD001",
    "config_id": "xgboost-00",
    "repeat": 0,
    "shift": 15.0,
    "scale": 1.25,
}
CAPS_ORDER = ["100", "125", "150", "200", "none"]


def command_for(source: str) -> str:
    for prefix, command in COMMANDS.items():
        if source.startswith(prefix):
            return command
    raise KeyError(f"no make target is known for {source}")


def read(source: str) -> pd.DataFrame:
    return pd.read_csv(source, comment="#")


@dataclass
class Numbers:
    """The macros of numbers.tex and the rows of RESULTS_DIGEST.md, filled together."""

    values: dict[str, str] = field(default_factory=dict)
    rows: dict[str, list[dict]] = field(default_factory=dict)

    def add(
        self,
        section: str,
        key: str,
        quantity: str,
        value: float | str,
        source: str,
        column: str,
        decimals: int = 3,
        lo: float | None = None,
        hi: float | None = None,
    ) -> None:
        text = value if isinstance(value, str) else number(value, decimals)
        if key in self.values:
            raise ValueError(f"{key} is defined twice")
        self.values[key] = text
        shown = interval(lo, hi, decimals) if lo is not None else ""
        if lo is not None:
            self.values[f"{key}_lo"] = number(lo, decimals)
            self.values[f"{key}_hi"] = number(hi, decimals)
            self.values[f"{key}_ci"] = shown
        fleet = key.split(".")[1]
        self.rows.setdefault(section, []).append(
            {
                "quantity": quantity,
                "fleet": fleet,
                "estimate": text,
                "interval": shown,
                "macro": f"\\val{{{key}}}",
                "source": source,
                "column": column,
            }
        )


def plain(text: str) -> str:
    """A macro value as Markdown shows it."""
    return re.sub(r"\\ensuremath\{([^}]*)\}", r"\1", text)


# ---------------------------------------------------------------- numbers from reports


def endpoint_numbers(n: Numbers, cfg: ResearchConfig) -> None:
    for folder, fleets, label in (
        ("reports/research/confirmatory", FLEETS, "confirmatory"),
        ("reports/research/ncmapss", [NCMAPSS], "secondary S8, descriptive"),
    ):
        e1, e2, e3 = (
            read(f"{folder}/{name}.csv").set_index("fleet")
            for name in ("e1", "e2", "e3")
        )
        for fleet in fleets:
            a, b, c = e1.loc[fleet], e2.loc[fleet], e3.loc[fleet]
            n.add(
                "RQ1",
                f"e1.{fleet}.tau_b",
                f"E1 tau-b(leaderboard RMSE, D), {label}",
                a["tau_b"],
                f"{folder}/e1.csv",
                "tau_b, lo, hi",
                lo=a["lo"],
                hi=a["hi"],
            )
            n.add(
                "RQ1",
                f"e1.{fleet}.band",
                "E1 band",
                a["band"],
                f"{folder}/e1.csv",
                "band",
            )
            n.add(
                "RQ1",
                f"e1.{fleet}.configs",
                "configurations",
                int(a["configs"]),
                f"{folder}/e1.csv",
                "configs",
                decimals=0,
            )
            n.add(
                "RQ2",
                f"e2.{fleet}.selection_loss",
                f"E2 selection loss, {label}",
                b["selection_loss"],
                f"{folder}/e2.csv",
                "selection_loss, lo, hi",
                lo=b["lo"],
                hi=b["hi"],
            )
            for column, what in (
                ("s_by_cost", "validation cost"),
                ("s_by_rmse", "validation RMSE"),
            ):
                n.add(
                    "RQ2",
                    f"e2.{fleet}.{column}",
                    f"S of the models chosen by {what}",
                    b[column],
                    f"{folder}/e2.csv",
                    f"{column}, {column}_lo, {column}_hi",
                    lo=b[f"{column}_lo"],
                    hi=b[f"{column}_hi"],
                )
            n.add(
                "RQ3",
                f"e3.{fleet}.delta_tau",
                f"E3 delta-tau, {label}",
                c["delta_tau"],
                f"{folder}/e3.csv",
                "delta_tau, lo, hi",
                lo=c["lo"],
                hi=c["hi"],
            )
            n.add(
                "RQ3",
                f"e3.{fleet}.tau_jitter",
                "tau-b(jitter, D)",
                c["tau_jitter"],
                f"{folder}/e3.csv",
                "tau_jitter, tau_jitter_lo, tau_jitter_hi",
                lo=c["tau_jitter_lo"],
                hi=c["tau_jitter_hi"],
            )
    text = Path(VERDICT_FILE).read_text(encoding="utf-8")
    for h, section in (("H1", "RQ1"), ("H2", "RQ2"), ("H3", "RQ3")):
        verdict = re.search(rf"^## {h}: (\w+)", text, re.MULTILINE).group(1)
        n.add(
            section,
            f"verdict.all.{h}",
            f"pre-registered verdict on {h}",
            verdict,
            VERDICT_FILE,
            f"heading '## {h}'",
        )


def secondary_numbers(n: Numbers) -> None:
    base = "reports/research/secondary"
    s1 = read(f"{base}/s1_accuracy_metrics.csv")
    for _, r in s1.iterrows():
        n.add(
            "RQ1",
            f"s1.{r['fleet']}.{r['metric']}_tau_b",
            f"S1 tau-b({r['metric']}, D)",
            r["tau_b"],
            f"{base}/s1_accuracy_metrics.csv",
            f"tau_b, tau_b_lo, tau_b_hi (metric = {r['metric']})",
            lo=r["tau_b_lo"],
            hi=r["tau_b_hi"],
        )
    grid = read(f"{base}/s2_settings_grid.csv")
    for (fleet, rule), g in grid.groupby(["fleet", "rule"]):
        for column, what, section in (
            ("e1_tau_b", "E1", "RQ1"),
            ("e2_selection_loss", "E2", "RQ2"),
            ("e3_delta_tau", "E3", "RQ3"),
        ):
            stem = f"s2.{fleet}.{rule}_{column}"
            n.add(
                section,
                f"{stem}_min",
                f"S2 smallest {what} over the 21 settings of {rule}",
                g[column].min(),
                f"{base}/s2_settings_grid.csv",
                f"{column} (rule = {rule})",
            )
            n.add(
                section,
                f"{stem}_max",
                f"S2 largest {what} over the 21 settings of {rule}",
                g[column].max(),
                f"{base}/s2_settings_grid.csv",
                f"{column} (rule = {rule})",
            )
        n.add(
            "RQ1",
            f"s2.{fleet}.{rule}_weak_settings",
            f"S2 settings of {rule} with E1 in the weak band (of 21)",
            int((g["e1_band"] == "weak").sum()),
            f"{base}/s2_settings_grid.csv",
            f"e1_band (rule = {rule})",
            decimals=0,
        )
    rules = read(f"{base}/s2_rules.csv")
    for h, section in (("H1", "RQ1"), ("H2", "RQ2"), ("H3", "RQ3")):
        for outcome, count in rules[f"{h}_rule"].value_counts().items():
            n.add(
                section,
                f"s2.all.{h}_{outcome}_settings",
                f"S2 settings (of {len(rules)}) where the {h} rule reads {outcome}",
                int(count),
                f"{base}/s2_rules.csv",
                f"{h}_rule",
                decimals=0,
            )
    families = read(f"{base}/s3_families.csv")
    for _, r in families.iterrows():
        stem = f"s3.{r['fleet']}.{r['family']}"
        src = f"{base}/s3_families.csv"
        n.add(
            "S3",
            f"{stem}_savings_share",
            f"S3 S of the {r['family']} representative",
            r["savings_share"],
            src,
            "savings_share, _lo, _hi",
            lo=r["savings_share_lo"],
            hi=r["savings_share_hi"],
        )
        n.add(
            "S3",
            f"{stem}_leaderboard_rmse",
            f"S3 leaderboard RMSE, {r['family']}",
            r["leaderboard_rmse"],
            src,
            "leaderboard_rmse, _lo, _hi",
            decimals=2,
            lo=r["leaderboard_rmse_lo"],
            hi=r["leaderboard_rmse_hi"],
        )
        for column, decimals in (
            ("rank_rmse", 0),
            ("rank_value", 0),
            ("failure_rate", 3),
            ("waste_per_maintenance", 1),
        ):
            n.add(
                "S3",
                f"{stem}_{column}",
                f"S3 {column}, {r['family']}",
                r[column],
                src,
                column,
                decimals=decimals,
            )
    fe1 = read(f"{base}/s3_family_e1.csv")
    for _, r in fe1.iterrows():
        n.add(
            "RQ1",
            f"s3.{r['fleet']}.family_tau_b",
            "S3 tau-b(leaderboard RMSE, D) across the family representatives",
            r["tau_b"],
            f"{base}/s3_family_e1.csv",
            "tau_b, tau_b_lo, tau_b_hi",
            lo=r["tau_b_lo"],
            hi=r["tau_b_hi"],
        )
    pairs = read(f"{base}/s3_pairwise.csv")
    for fleet, g in pairs.groupby("fleet"):
        n.add(
            "S3",
            f"s3.{fleet}.inversions",
            "S3 inverted family pairs (Benjamini-Hochberg)",
            int(g["inversion"].sum()),
            f"{base}/s3_pairwise.csv",
            "inversion",
            decimals=0,
        )
        n.add(
            "S3",
            f"s3.{fleet}.inversions_unadjusted",
            "S3 inverted family pairs (intervals)",
            int(g["inversion_unadjusted"].sum()),
            f"{base}/s3_pairwise.csv",
            "inversion_unadjusted",
            decimals=0,
        )
    variants = read(f"{base}/s4_threshold_variants.csv")
    for _, r in variants.iterrows():
        stem = f"s4.{r['fleet']}.{r['tuning']}"
        src = f"{base}/s4_threshold_variants.csv"
        for column, section in (
            ("e1_tau_b", "RQ1"),
            ("e2_selection_loss", "RQ2"),
            ("e3_delta_tau", "RQ3"),
        ):
            n.add(
                section,
                f"{stem}_{column}",
                f"S4 {column} with {r['tuning']} thresholds",
                r[column],
                src,
                f"{column}, _lo, _hi (tuning = {r['tuning']})",
                lo=r[f"{column}_lo"],
                hi=r[f"{column}_hi"],
            )
        n.add(
            "RQ2",
            f"{stem}_mean_savings_share",
            f"S4 mean S over configurations, {r['tuning']}",
            r["mean_savings_share"],
            src,
            f"mean_savings_share (tuning = {r['tuning']})",
        )
    nested = read(f"{base}/s4_nested_e1.csv")
    for _, r in nested.iterrows():
        n.add(
            "RQ1",
            f"s4.{r['fleet']}.{r['tuning']}_family_tau_b",
            f"S4 tau-b across family representatives, {r['tuning']} (repeat 0)",
            r["tau_b"],
            f"{base}/s4_nested_e1.csv",
            f"tau_b, _lo, _hi (tuning = {r['tuning']})",
            lo=r["tau_b_lo"],
            hi=r["tau_b_hi"],
        )
    caps = read(f"{base}/s5_caps.csv")
    for _, r in caps.iterrows():
        stem = f"s5.{r['fleet']}.{r['family']}_cap{r['training_cap']}"
        src = f"{base}/s5_caps.csv"
        n.add(
            "S5",
            f"{stem}_leaderboard_rmse",
            f"S5 leaderboard RMSE, {r['family']}, training cap {r['training_cap']}",
            r["leaderboard_rmse"],
            src,
            "leaderboard_rmse, _lo, _hi",
            decimals=2,
            lo=r["leaderboard_rmse_lo"],
            hi=r["leaderboard_rmse_hi"],
        )
        n.add(
            "S5",
            f"{stem}_savings_share",
            f"S5 S, {r['family']}, training cap {r['training_cap']}",
            r["savings_share"],
            src,
            "savings_share, _lo, _hi",
            lo=r["savings_share_lo"],
            hi=r["savings_share_hi"],
        )
    regret = read(f"{base}/s6_regret.csv")
    for _, r in regret.iterrows():
        stem = f"s6.{r['fleet']}.cf{int(r['failure_cost'])}"
        src = f"{base}/s6_regret.csv"
        for column, section in (
            ("e1_tau_b", "RQ1"),
            ("e2_selection_loss", "RQ2"),
            ("e3_delta_tau", "RQ3"),
        ):
            n.add(
                section,
                f"{stem}_{column}",
                f"S6 {column} with regret, C_f = {int(r['failure_cost'])}",
                r[column],
                src,
                f"{column}, _lo, _hi (failure_cost = {int(r['failure_cost'])})",
                lo=r[f"{column}_lo"],
                hi=r[f"{column}_hi"],
            )
    s7 = read(f"{base}/s7_matched_accuracy.csv")
    for _, r in s7.iterrows():
        src = f"{base}/s7_matched_accuracy.csv"
        n.add(
            "RQ1",
            f"s7.{r['fleet']}.tau_matched",
            "S7 tau-b(matched-accuracy RMSE, D)",
            r["tau_matched"],
            src,
            "tau_matched, _lo, _hi",
            lo=r["tau_matched_lo"],
            hi=r["tau_matched_hi"],
        )
        n.add(
            "RQ1",
            f"s7.{r['fleet']}.difference",
            "S7 matched minus leaderboard tau-b",
            r["difference"],
            src,
            "difference, _lo, _hi",
            lo=r["difference_lo"],
            hi=r["difference_hi"],
        )
    s9 = read(f"{base}/s9_competitive.csv")
    for _, r in s9.iterrows():
        src = f"{base}/s9_competitive.csv"
        n.add(
            "RQ1",
            f"s9.{r['fleet']}.tau_b",
            "S9 E1 among configurations within 2 cycles of the best",
            r["tau_b"],
            src,
            "tau_b, tau_b_lo, tau_b_hi",
            lo=r["tau_b_lo"],
            hi=r["tau_b_hi"],
        )
        n.add(
            "RQ1",
            f"s9.{r['fleet']}.configs",
            "S9 configurations within the margin",
            int(r["configs"]),
            src,
            "configs",
            decimals=0,
        )
        n.add(
            "RQ1",
            f"s9.{r['fleet']}.families",
            "S9 families within the margin",
            int(r["families"]),
            src,
            "families",
            decimals=0,
        )
        n.add(
            "RQ1",
            f"s9.{r['fleet']}.best_rmse",
            "S9 best leaderboard RMSE",
            r["best_rmse"],
            src,
            "best_rmse",
            decimals=2,
        )


def exploratory_numbers(n: Numbers) -> None:
    base = "reports/research/exploratory"
    x1 = read(f"{base}/x1_isotonic_late_rmse.csv")
    for _, r in x1[x1["proxy"] != "leaderboard_rmse"].iterrows():
        n.add(
            "X",
            f"x1.{r['fleet']}.{r['proxy']}_tau_b",
            f"X1 tau-b({r['proxy']}, D)",
            r["tau_b"],
            f"{base}/x1_isotonic_late_rmse.csv",
            f"tau_b, _lo, _hi (proxy = {r['proxy']})",
            lo=r["tau_b_lo"],
            hi=r["tau_b_hi"],
        )
    x3 = read(f"{base}/x3_legacy_booking.csv")
    for _, r in x3.iterrows():
        for stem in (
            "legacy_booking_cost_vs_leaderboard_rmse",
            "legacy_booking_cost_vs_decision_loss",
        ):
            column = f"tau_{stem}"
            n.add(
                "X",
                f"x3.{r['fleet']}.{column}",
                f"X3 {column}",
                r[column],
                f"{base}/x3_legacy_booking.csv",
                f"{column}, _lo, _hi",
                lo=r[f"{column}_lo"],
                hi=r[f"{column}_hi"],
            )


def leaderboard_numbers(n: Numbers) -> None:
    src = "reports/research/leaderboard_reference.csv"
    for _, r in read(src).iterrows():
        stem = f"leaderboard.{r['fleet']}.{r['family']}"
        n.add(
            "LB",
            f"{stem}_rmse",
            f"reference {r['family']}: RMSE, mean of 5 seeds",
            r["rmse_mean"],
            src,
            "rmse_mean",
            decimals=2,
        )
        n.add(
            "LB",
            f"{stem}_rmse_sd",
            "standard deviation over seeds",
            r["rmse_std"],
            src,
            "rmse_std",
            decimals=2,
        )
        n.add(
            "LB",
            f"{stem}_nasa",
            "NASA score, mean of 5 seeds",
            r["nasa_score_mean"],
            src,
            "nasa_score_mean",
            decimals=0,
        )
    src = "reports/research/leaderboard_vs_legacy.csv"
    for _, r in read(src).iterrows():
        n.add(
            "LB",
            f"leaderboard.{r['fleet']}.{r['family']}_rmse_legacy",
            f"legacy RMSE, {r['family']}",
            r["rmse_legacy"],
            src,
            "rmse_legacy",
            decimals=2,
        )


# ---------------------------------------------------------------- computed here


def invariance(cfg: ResearchConfig) -> tuple[pd.DataFrame, dict]:
    """F1: one configuration's predictions shifted and scaled, through the real policy code."""
    fleet, config_id, repeat = F1["fleet"], F1["config_id"], F1["repeat"]
    p = cfg.decision.primary
    lives = lifetimes(load_subset(cfg.data.raw_dir, fleet, cfg.data.cap).train)
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)[fleet]
    table = load(
        cfg.paths.archive,
        fleets=[fleet],
        config_ids=[config_id],
        repeats=[repeat],
        columns=["unit_id", "cycle", "rul", "split", "pred", "repeat", "fold"],
    )
    age = age_table(lives, folds, [repeat], cfg.decision.cost_ratios)
    truncation = truncation_points(lives, [cfg.stats.seed, fleet_number(fleet)])
    transforms: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "original": lambda y: y,
        "shifted": lambda y: y + F1["shift"],
        "scaled": lambda y: y * F1["scale"],
    }
    rows, outcomes, policies, example = [], {}, {}, {}
    for name, f in transforms.items():
        fits = {
            k: g.assign(pred=f(g["pred"].to_numpy(float)))
            for k, g in table.groupby("fold")
        }
        policy_table, outcome_table = repeat_outcomes(fits, cfg.decision)
        outcomes[name] = outcome_table.sort_values(
            ["fold", "tuning", "rule", "lead", "cost_ratio", "unit_id"]
        ).reset_index(drop=True)
        policies[name] = policy_table
        primary = outcome_table[
            (outcome_table["tuning"] == p.tuning)
            & (outcome_table["rule"] == p.rule)
            & (outcome_table["lead"] == p.lead_time)
            & (outcome_table["cost_ratio"] == p.cost_ratio)
        ].assign(config_id=config_id, repeat=repeat)
        value = savings(primary, age, lives).iloc[0]
        parts = pd.concat(
            [
                metric_components(fit, truncation, cfg.data.cap).assign(fold=k)
                for k, fit in fits.items()
            ]
        )
        metrics = fit_metrics(parts, ["fold"])
        rows.append(
            {
                "variant": name,
                "transform": {
                    "original": "y",
                    "shifted": f"y + {F1['shift']:g}",
                    "scaled": f"{F1['scale']:g} y",
                }[name],
                "leaderboard_rmse": metrics["rmse_last"].mean(),
                "savings_share": value["savings_share"],
                "cost_rate": value["cost_rate"],
                "failures": int(primary["failed"].sum()),
                "engines": len(primary),
            }
        )
        example[name] = fits
    frame = pd.DataFrame(rows)
    base = outcomes["original"]
    for name in transforms:
        other = outcomes[name]
        same = (
            other[["failed", "used", "waste"]] == base[["failed", "used", "waste"]]
        ).all(axis=1)
        for rule in ("R1", "R2"):
            keep = base["rule"] == rule
            frame.loc[frame["variant"] == name, f"outcomes_differing_{rule}"] = int(
                (~same[keep]).sum()
            )
        frame.loc[frame["variant"] == name, "outcomes_compared"] = len(base)
    # an outer_test engine of fold 0 with the median life, for the trajectory panel
    test = table[(table["fold"] == 0) & (table["split"] == "outer_test")]
    life = test.groupby("unit_id")["cycle"].max().sort_values(kind="stable")
    unit = int(life.index[(len(life) - 1) // 2])
    detail = {
        "unit": unit,
        "life": int(life.loc[unit]),
        "fits": example,
        "policies": policies,
    }
    return frame, detail


def oracle_selection(cfg: ResearchConfig) -> pd.DataFrame:
    """F5: S of the best configuration in hindsight minus S of each validation selection."""
    p = cfg.decision.primary
    keep = [
        ("tuning", "==", p.tuning),
        ("rule", "==", p.rule),
        ("lead", "==", p.lead_time),
        ("cost_ratio", "==", p.cost_ratio),
    ]
    rows = []
    for fleet in FLEETS:
        source = Path(cfg.paths.derived) / "archive" / fleet
        tables = {
            "outcomes": pd.read_parquet(source / "outcomes.parquet", filters=keep),
            "policies": pd.read_parquet(source / "policies.parquet", filters=keep),
            "age": pd.read_parquet(source / "age.parquet"),
            "engines": pd.read_parquet(source / "engines.parquet"),
            "components": pd.read_parquet(
                source / "components.parquet",
                filters=[("split", "in", ["official_test", "inner_val"])],
            ),
        }
        configs = pool_order(list(tables["outcomes"]["config_id"].unique()))
        inputs = fleet_inputs(tables, p, configs)
        n = inputs.cost.shape[-1]
        # the confirmatory draws of training engines, so the intervals pair with E2's
        counts = bootstrap_counts(
            n, cfg.stats.n_boot, [cfg.stats.seed, fleet_number(fleet), 0]
        )
        once = np.ones((1, n))
        point = 1.0 - decision_loss(inputs, once)[0]
        best = int(np.argmax(point))
        oracle = 1.0 - decision_loss(inputs, counts)[:, best]
        row = {"fleet": fleet, "oracle_config": configs[best], "oracle_s": point[best]}
        for name, metric in (("cost", inputs.val_cost), ("rmse", inputs.val_rmse)):
            choice = choose(metric)
            chosen = selected_share(inputs, choice, counts)
            gap = estimate(
                point[best] - selected_share(inputs, choice, once)[0],
                oracle - chosen,
                cfg.stats.ci,
            )
            row |= {
                f"gap_{name}": gap.point,
                f"gap_{name}_lo": gap.lo,
                f"gap_{name}_hi": gap.hi,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def compute_summary(cfg: ResearchConfig) -> pd.DataFrame:
    """Summed fit time per archive and family, from the sidecars."""
    archives = {
        "confirmatory": cfg.paths.archive,
        "S4 nested": cfg.paths.archive_nested,
        "S5 caps": cfg.paths.archive_caps,
        "N-CMAPSS": cfg.ncmapss.archive,
    }
    rows = []
    for name, root in archives.items():
        for path in sorted(Path(root).rglob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if "status" not in record or "family" not in record:
                continue
            rows.append(
                {
                    "archive": name,
                    "family": record["family"],
                    "device": record.get("device", ""),
                    "status": record["status"],
                    "seconds": record.get("wall_seconds", 0.0),
                }
            )
    found = pd.DataFrame(rows)
    out = (
        found.groupby(["archive", "family"], sort=False)
        .agg(
            fits=("seconds", "size"),
            failed=("status", lambda s: int((s != "ok").sum())),
            hours=("seconds", lambda s: s.sum() / 3600.0),
            median_seconds=("seconds", "median"),
            device=("device", lambda s: "/".join(sorted(set(s)))),
        )
        .reset_index()
    )
    out["stream"] = np.where(out["family"].isin(CPU_FAMILIES), "cpu", "gpu")
    return out


def span(sizes: Sequence[int]) -> str:
    """Fold sizes as one number or a range: 20, or 49--50."""
    low, high = min(sizes), max(sizes)
    return str(low) if low == high else f"{low}--{high}"


def data_summary(cfg: ResearchConfig) -> pd.DataFrame:
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)
    rows = []
    for fleet in FLEETS:
        subset = load_subset(cfg.data.raw_dir, fleet, cfg.data.cap)
        lives = lifetimes(subset.train)
        sizes = {
            split: span([len(getattr(f, split)) for r in folds[fleet] for f in r])
            for split in ("outer_test", "inner_val", "inner_fit")
        }
        rows.append(
            {
                "fleet": fleet,
                "train_units": len(lives),
                "test_units": subset.test["unit_id"].nunique(),
                "regimes": cfg.data.n_conditions[fleet],
                "life_min": int(lives.min()),
                "life_median": float(lives.median()),
                "life_max": int(lives.max()),
                "cap": cfg.data.cap,
                "flights": 0,
                "repeats": len(folds[fleet]),
                **sizes,
            }
        )
    engines = pd.read_parquet(
        Path(cfg.paths.derived) / "archive_ncmapss" / NCMAPSS / "engines.parquet"
    )
    nfolds = json.loads(Path(cfg.ncmapss.folds).read_text(encoding="utf-8"))["fleets"][
        NCMAPSS
    ]
    sizes = {
        split: span([len(f[split]) for r in nfolds for f in r])
        for split in ("outer_test", "inner_val", "inner_fit")
    }
    units = pd.read_csv("reports/research/ncmapss/inspection_units.csv")
    rows.append(
        {
            "fleet": NCMAPSS,
            "train_units": len(engines),
            "test_units": 0,
            "regimes": int(units["flight_classes"].nunique()),
            "life_min": int(engines["life"].min()),
            "life_median": float(engines["life"].median()),
            "life_max": int(engines["life"].max()),
            "cap": cfg.ncmapss.require_cap(),
            "flights": int(units["flights"].sum()),
            "repeats": len(nfolds),
            **sizes,
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures


def fig_invariance(
    frame: pd.DataFrame, detail: dict, cfg: ResearchConfig, out: Path
) -> None:
    p = cfg.decision.primary
    fig, axes = plt.subplots(
        1, 3, figsize=(DOUBLE, 2.4), gridspec_kw={"width_ratios": [2.0, 1.05, 1.05]}
    )
    ax = axes[0]
    unit, life = detail["unit"], detail["life"]
    alarms = []
    for style, name in zip(VARIANTS, frame["variant"], strict=True):
        fit = detail["fits"][name][0]
        rows = fit[(fit["split"] == "outer_test") & (fit["unit_id"] == unit)]
        rows = rows.sort_values("cycle")
        pol = detail["policies"][name]
        theta = pol[
            (pol["fold"] == 0)
            & (pol["tuning"] == p.tuning)
            & (pol["rule"] == p.rule)
            & (pol["lead"] == p.lead_time)
            & (pol["cost_ratio"] == p.cost_ratio)
        ]["theta"].iloc[0]
        label = frame.loc[frame["variant"] == name, "transform"].iloc[0]
        ax.plot(
            rows["cycle"],
            rows["pred"],
            color=style["color"],
            linestyle=style["linestyle"],
            label=label,
        )
        ax.axhline(
            theta, color=style["color"], linestyle=style["linestyle"], linewidth=0.7
        )
        alarm = int(rows.loc[rows["pred"].to_numpy() <= theta, "cycle"].min())
        alarms.append(alarm)
        ax.plot(
            [alarm],
            [theta],
            marker=style["marker"],
            color=style["color"],
            markerfacecolor="white",
            markersize=5,
            zorder=5,
        )
    truth = np.minimum(rows["rul"], cfg.data.cap)
    ax.plot(rows["cycle"], truth, color=INK, linewidth=0.8, label="true RUL (capped)")
    alarm = alarms[0]
    ax.axvline(alarm, color=MUTED, linewidth=0.6, zorder=0)
    ax.axvline(
        alarm + p.lead_time, color=MUTED, linewidth=0.6, linestyle="--", zorder=0
    )
    top = max(ax.get_ylim()[1], 160)
    ax.set_ylim(-5, top)
    ax.text(
        alarm - 1, top, "alarm a", va="top", ha="right", fontsize=7, color=INK_SECONDARY
    )
    ax.text(
        alarm + p.lead_time + 1,
        top,
        f"m = a + {p.lead_time}",
        va="top",
        ha="left",
        fontsize=7,
        color=INK_SECONDARY,
    )
    ax.set_xlim(0, life + 25)
    ax.set_xlabel(f"cycle (outer_test engine {unit}, life {life})")
    ax.set_ylabel("RUL (cycles)")
    ax.legend(loc="lower left", handlelength=2.2)
    panel_label(ax, "a")
    for ax, column, ylabel, letter, decimals in (
        (axes[1], "leaderboard_rmse", "leaderboard RMSE (cycles)", "b", 2),
        (axes[2], "savings_share", "savings share S", "c", 3),
    ):
        x = np.arange(len(frame))
        for i, style in enumerate(VARIANTS):
            value = frame[column].iloc[i]
            ax.bar(
                x[i],
                value,
                color="white",
                edgecolor=style["color"],
                hatch=["", "////", "...."][i],
                linewidth=1.0,
                width=0.7,
            )
            ax.text(
                x[i],
                value,
                f"{value:.{decimals}f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_xticks(
            x, [t.replace(" + ", "+").replace(" ", "") for t in frame["transform"]]
        )
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, frame[column].max() * 1.15)
        ax.grid(axis="x", visible=False)
        panel_label(ax, letter)
    fig.tight_layout(w_pad=1.2)
    save(fig, out, "fig1_invariance")


def _box(ax, x, y, w, h, text="", hatch="", face="white", size=7):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=face,
            edgecolor=INK_SECONDARY,
            hatch=hatch,
            linewidth=0.8,
        )
    )
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8}
            if hatch
            else None,
        )


def _arrow(ax, start, end, connection="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            color=INK_SECONDARY,
            linewidth=0.8,
            connectionstyle=connection,
            shrinkA=0,
            shrinkB=0,
        )
    )


def fig_protocol(out: Path) -> None:
    """F2: where each set of engines goes, with one hatch per set."""
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 41)
    ax.axis("off")
    sets = {
        "outer_test": ("////", "#d9d9d9"),
        "inner_val": ("....", "#f0f0f0"),
        "inner_fit": ("", "white"),
        "official": ("xxxx", "#bdbdbd"),
    }
    ax.text(
        1,
        38.5,
        "run-to-failure training engines of a fleet, one repeat",
        fontsize=8,
        va="center",
    )
    width = 9.0
    for k in range(5):
        hatch, face = sets["outer_test"] if k == 4 else ("", "white")
        _box(
            ax,
            1 + k * width,
            31,
            width,
            5,
            "outer_test" if k == 4 else f"fold {k + 1}",
            hatch=hatch,
            face=face,
        )
    ax.text(
        1 + 4.5 * width,
        29.8,
        "outer fold k (each in turn)",
        ha="center",
        va="top",
        fontsize=7,
        color=INK_SECONDARY,
    )
    ax.text(
        6,
        26.6,
        "outer-train engines (folds other than k)",
        fontsize=7,
        va="center",
        color=INK_SECONDARY,
    )
    _arrow(ax, (4, 31), (4, 21))
    _box(ax, 1, 16, 27, 5, "inner_fit (75%)", *sets["inner_fit"])
    _box(ax, 28, 16, 9, 5, "inner_val (25%)", *sets["inner_val"], size=6.5)
    _box(ax, 1, 3, 45, 5, "official test engines (truncated)", *sets["official"])
    uses = [
        (
            33.5,
            "decision metrics: cost rate, S, D\n(complete trajectories, out of fold)",
            sets["outer_test"],
        ),
        (
            24.5,
            "fitting: regimes, sensor screening,\nnormalisation, model",
            sets["inner_fit"],
        ),
        (
            13.0,
            "early stopping; threshold tuning;\nvalidation RMSE and cost (selection)",
            sets["inner_val"],
        ),
        (
            5.5,
            "leaderboard metrics: last-cycle RMSE,\nNASA score, jitter",
            sets["official"],
        ),
    ]
    for y, text, (hatch, face) in uses:
        _box(ax, 58, y - 3, 41, 6)
        ax.add_patch(
            Rectangle(
                (58, y - 3),
                2.5,
                6,
                facecolor=face,
                hatch=hatch,
                edgecolor=INK_SECONDARY,
                linewidth=0.8,
            )
        )
        ax.text(80, y, text, ha="center", va="center", fontsize=7)
    _arrow(ax, (46, 33.5), (58, 33.5))
    _arrow(ax, (14, 21), (58, 24.5), "angle,angleA=90,angleB=180,rad=0")
    _arrow(ax, (32.5, 16), (58, 13.0), "angle,angleA=-90,angleB=180,rad=0")
    _arrow(ax, (46, 5.5), (58, 5.5))
    save(fig, out, "fig2_protocol")


def fig_bump(families: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.7))
    for ax, fleet in zip(axes.ravel(), FLEETS, strict=True):
        g = families[families["fleet"] == fleet]
        for _, r in g.iterrows():
            st = family_style(r["family"])
            ax.plot(
                [0, 1],
                [r["rank_rmse"], r["rank_value"]],
                color=st["color"],
                marker=st["marker"],
                markersize=5,
                linewidth=1.0,
                markerfacecolor="white" if r["family"] in HOLLOW else st["color"],
            )
            ax.text(
                -0.06, r["rank_rmse"], st["label"], ha="right", va="center", fontsize=7
            )
            ax.text(
                1.06, r["rank_value"], st["label"], ha="left", va="center", fontsize=7
            )
        ax.set_xlim(-0.75, 1.75)
        ax.set_ylim(len(g) + 0.6, 0.4)
        ax.set_yticks(range(1, len(g) + 1))
        ax.set_xticks([0, 1], ["rank by\nleaderboard RMSE", "rank by S"])
        ax.grid(axis="x", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel_label(ax, fleet)
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    save(fig, out, "fig3_rank_bump")


def fig_tau_by_cost(grid: pd.DataFrame, cfg: ResearchConfig, out: Path) -> None:
    p = cfg.decision.primary
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.1), sharex=True, sharey=True)
    leads = sorted(grid["lead"].unique())
    for ax, fleet in zip(axes.ravel(), FLEETS, strict=True):
        g = grid[(grid["fleet"] == fleet) & (grid["rule"] == "R1")]
        for i, (lead, style) in enumerate(zip(leads, VARIANTS, strict=True)):
            h = g[g["lead"] == lead].sort_values("cost_ratio")
            x = h["cost_ratio"].to_numpy() * (1.0 + 0.05 * (i - 1))
            ax.errorbar(
                x,
                h["e1_tau_b"],
                yerr=[
                    h["e1_tau_b"] - h["e1_tau_b_lo"],
                    h["e1_tau_b_hi"] - h["e1_tau_b"],
                ],
                capsize=1.5,
                elinewidth=0.6,
                label=f"L = {lead}",
                markerfacecolor="white" if i == 1 else style["color"],
                **style,
            )
        star = g[(g["lead"] == p.lead_time) & (g["cost_ratio"] == p.cost_ratio)]
        ax.plot(
            star["cost_ratio"] * (1.0 + 0.05 * (leads.index(p.lead_time) - 1)),
            star["e1_tau_b"],
            marker="o",
            color=INK,
            markersize=8,
            linestyle="none",
            markerfacecolor="none",
            markeredgewidth=1.0,
            label="primary setting",
        )
        for edge in (0.4, 0.7):
            ax.axhline(
                edge, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)), zorder=0
            )
        zero_line(ax)
        ax.set_xscale("log")
        ticks = sorted(g["cost_ratio"].unique())
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        panel_label(ax, fleet)
    for ax in axes[1]:
        ax.set_xlabel("cost ratio K")
    for ax in axes[:, 0]:
        ax.set_ylabel("E1: tau-b(leaderboard RMSE, D)")
    fig.tight_layout(h_pad=1.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "fig4_tau_by_cost")


def fig_selection(
    e2: pd.DataFrame, gap: pd.DataFrame, nc: pd.DataFrame, out: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    ax = axes[0]
    rows = pd.concat([e2, nc], ignore_index=True)
    x = np.arange(len(rows))
    ax.axhspan(-0.02, 0.02, color=GRID, zorder=0, linewidth=0)
    for i, r in rows.iterrows():
        secondary = r["fleet"] == NCMAPSS
        colour = INK_SECONDARY if secondary else SERIES[0]
        ax.errorbar(
            [x[i]],
            [r["selection_loss"]],
            yerr=[[r["selection_loss"] - r["lo"]], [r["hi"] - r["selection_loss"]]],
            marker="D" if secondary else "o",
            color=colour,
            markerfacecolor="white" if secondary else colour,
            capsize=2,
            linestyle="none",
        )
    zero_line(ax)
    ax.set_xticks(
        x, [f if f != NCMAPSS else "N-CMAPSS\n(secondary)" for f in rows["fleet"]]
    )
    ax.set_ylabel("E2: S(by validation cost)\n- S(by validation RMSE)")
    ax.grid(axis="x", visible=False)
    panel_label(ax, "a")
    ax = axes[1]
    x = np.arange(len(gap))
    for i, (name, label) in enumerate(
        (("cost", "chosen by validation cost"), ("rmse", "chosen by validation RMSE"))
    ):
        style = VARIANTS[i]
        ax.errorbar(
            x + (i - 0.5) * 0.25,
            gap[f"gap_{name}"],
            yerr=[
                gap[f"gap_{name}"] - gap[f"gap_{name}_lo"],
                gap[f"gap_{name}_hi"] - gap[f"gap_{name}"],
            ],
            marker=style["marker"],
            color=style["color"],
            capsize=2,
            linestyle="none",
            markerfacecolor="white" if i else style["color"],
            label=label,
        )
    zero_line(ax)
    top = max(gap["gap_cost_hi"].max(), gap["gap_rmse_hi"].max())
    ax.set_ylim(top=top * 1.4)
    ax.set_xticks(x, gap["fleet"])
    ax.set_ylabel("S(best configuration in hindsight)\n- S(selected)")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    panel_label(ax, "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "fig5_selection")


def _family_scatter(
    axes_row, table: pd.DataFrame, fleets: Sequence[str], column: str, label: bool
) -> None:
    for ax, fleet in zip(axes_row, fleets, strict=True):
        g = table[table["fleet"] == fleet]
        for family in FAMILY_NAMES:
            h = g[g["config_id"].map(family_of) == family]
            ax.scatter(
                h[column],
                h["savings_share"],
                s=13,
                linewidths=0.7,
                **scatter_style(family),
            )
        zero_line(ax)
        ax.margins(0.06)
        if label:
            panel_label(ax, fleet if fleet != NCMAPSS else "N-CMAPSS (secondary)")


def _family_legend(fig, ncol: int = 5, y: float = 0.0) -> None:
    fig.legend(
        handles=[family_handle(f) for f in FAMILY_NAMES],
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=(0.5, y),
        handletextpad=0.3,
        columnspacing=1.2,
    )


def fig_scatter(
    table: pd.DataFrame, out: Path, name: str, fleets: Sequence[str]
) -> None:
    fig, axes = plt.subplots(2, len(fleets), figsize=(DOUBLE, 4.0), squeeze=False)
    _family_scatter(axes[0], table, fleets, "leaderboard_rmse", label=True)
    _family_scatter(axes[1], table, fleets, "jitter", label=False)
    unit = "flights" if list(fleets) == [NCMAPSS] else "cycles"
    for ax in axes[0]:
        ax.set_xlabel(f"leaderboard RMSE ({unit})")
    for ax in axes[1]:
        ax.set_xlabel(f"jitter ({unit})")
    for ax in axes[:, 0]:
        ax.set_ylabel("savings share S")
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    _family_legend(fig)
    save(fig, out, name)


def fig_caps(caps: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 3.7), sharex=True)
    x = np.arange(len(CAPS_ORDER))
    for j, fleet in enumerate(FLEETS):
        for i, (column, ylabel) in enumerate(
            (
                ("leaderboard_rmse", "leaderboard RMSE (cycles)"),
                ("savings_share", "savings share S"),
            )
        ):
            ax = axes[i, j]
            for k, family in enumerate(("xgboost", "lstm")):
                g = caps[(caps["fleet"] == fleet) & (caps["family"] == family)]
                g = g.set_index("training_cap").loc[CAPS_ORDER]
                st = family_style(family)
                ax.errorbar(
                    x + (k - 0.5) * 0.18,
                    g[column],
                    yerr=[g[column] - g[f"{column}_lo"], g[f"{column}_hi"] - g[column]],
                    color=st["color"],
                    marker=st["marker"],
                    label=st["label"],
                    linestyle=["-", "--"][k],
                    capsize=1.5,
                    elinewidth=0.6,
                    markerfacecolor="white" if k else st["color"],
                )
            ax.axvline(CAPS_ORDER.index("125"), color=GRID, linewidth=3, zorder=0)
            ax.set_xticks(x, CAPS_ORDER)
            if i == 1:
                ax.set_xlabel("training cap")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                panel_label(ax, fleet)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "fig7_caps")


def fig_heatmaps(grid: pd.DataFrame, cfg: ResearchConfig, out: Path) -> None:
    p = cfg.decision.primary
    ks = sorted(grid["cost_ratio"].unique())
    leads = sorted(grid["lead"].unique())
    fig, axes = plt.subplots(4, 2, figsize=(DOUBLE, 5.4), sharex=True, sharey=True)
    lo, hi = grid["e1_tau_b"].min(), grid["e1_tau_b"].max()
    for j, rule in enumerate(("R1", "R2")):
        for i, fleet in enumerate(FLEETS):
            ax = axes[i, j]
            g = grid[(grid["fleet"] == fleet) & (grid["rule"] == rule)]
            values = g.pivot(index="lead", columns="cost_ratio", values="e1_tau_b")
            values = values.loc[leads, ks]
            image = ax.imshow(
                values.to_numpy(), cmap="cividis", vmin=lo, vmax=hi, aspect="auto"
            )
            for a in range(len(leads)):
                for b in range(len(ks)):
                    v = values.iloc[a, b]
                    shade = (v - lo) / (hi - lo)
                    ax.text(
                        b,
                        a,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if shade < 0.55 else INK,
                    )
            if rule == p.rule:
                ax.add_patch(
                    Rectangle(
                        (ks.index(p.cost_ratio) - 0.5, leads.index(p.lead_time) - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=INK,
                        linewidth=1.4,
                    )
                )
            ax.set_xticks(range(len(ks)), [f"{k:g}" for k in ks])
            ax.set_yticks(range(len(leads)), [str(v) for v in leads])
            ax.grid(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.tick_params(length=0)
            panel_label(ax, f"{fleet}, rule {rule}")
            if i == len(FLEETS) - 1:
                ax.set_xlabel("cost ratio K")
            if j == 0:
                ax.set_ylabel("lead time L")
    fig.tight_layout(h_pad=0.9, w_pad=0.8)
    bar = fig.colorbar(image, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    bar.set_label("E1: tau-b(leaderboard RMSE, D)")
    bar.outline.set_visible(False)
    save(fig, out, "figA1_tau_heatmaps")


def fig_threshold_variants(variants: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.3))
    tunings = ["validation", "crossfit", "oracle"]
    x = np.arange(len(FLEETS))
    for ax, (column, ylabel) in zip(
        axes,
        (
            ("e1_tau_b", "E1: tau-b"),
            ("e2_selection_loss", "E2: selection loss"),
            ("e3_delta_tau", "E3: delta-tau"),
        ),
        strict=True,
    ):
        for i, (tuning, style) in enumerate(zip(tunings, VARIANTS, strict=True)):
            g = (
                variants[variants["tuning"] == tuning]
                .set_index("fleet")
                .loc[list(FLEETS)]
            )
            ax.errorbar(
                x + (i - 1) * 0.22,
                g[column],
                yerr=[g[column] - g[f"{column}_lo"], g[f"{column}_hi"] - g[column]],
                color=style["color"],
                marker=style["marker"],
                linestyle="none",
                capsize=1.5,
                elinewidth=0.7,
                label=f"{tuning} thresholds",
                markerfacecolor="white" if i == 1 else style["color"],
            )
        zero_line(ax)
        ax.set_xticks(x, FLEETS)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "figA2_threshold_variants")


# ---------------------------------------------------------------- tables


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def table_data(data: pd.DataFrame, out: Path) -> None:
    rows = []
    for _, r in data.iterrows():
        rows.append(
            [
                "N-CMAPSS" if r["fleet"] == NCMAPSS else r["fleet"],
                str(r["train_units"]),
                str(r["test_units"]) if r["test_units"] else "--",
                str(r["regimes"]),
                f"{r['life_min']}, {r['life_median']:g}, {r['life_max']}",
                str(r["cap"]),
                r["outer_test"],
                r["inner_val"],
                r["inner_fit"],
            ]
        )
    write(
        out / "tab1_data_folds.tex",
        booktabs(
            [
                "fleet",
                "run-to-failure",
                "official test",
                "regimes",
                "life (min, median, max)",
                "cap",
                "outer\\_test",
                "inner\\_val",
                "inner\\_fit",
            ],
            rows,
            "lrrrlrrrr",
            "paper_assets/data/data_folds.csv (raw files, configs/research_folds*.json); N-CMAPSS: "
            "units, flight classes and flights",
        ),
    )


def table_leaderboard(out: Path) -> None:
    ref = read("reports/research/leaderboard_reference.csv")
    legacy = read("reports/research/leaderboard_vs_legacy.csv").set_index(
        ["fleet", "family"]
    )
    rows = []
    for family in FAMILY_NAMES:
        cells = [tex(FAMILY_LABELS[family])]
        for fleet in FLEETS:
            r = ref[(ref["fleet"] == fleet) & (ref["family"] == family)].iloc[0]
            cells.append(f"{r['rmse_mean']:.2f} ({r['rmse_std']:.2f})")
            key = (fleet, family)
            cells.append(
                f"{legacy.loc[key, 'rmse_legacy']:.2f}" if key in legacy.index else "--"
            )
        rows.append(cells)
    columns = ["family"] + [f"{f} {c}" for f in FLEETS for c in ("RMSE (sd)", "legacy")]
    write(
        out / "tab2_leaderboard.tex",
        booktabs(
            columns,
            rows,
            "l" + "rr" * 4,
            "reports/research/leaderboard_reference.csv (rmse_mean, rmse_std; 5 seeds) and "
            "leaderboard_vs_legacy.csv (rmse_legacy); make research-leaderboard",
        ),
    )
    rows = []
    for family in FAMILY_NAMES:
        cells = [tex(FAMILY_LABELS[family])]
        for fleet in FLEETS:
            r = ref[(ref["fleet"] == fleet) & (ref["family"] == family)].iloc[0]
            cells.append(f"{r['nasa_score_mean']:.0f} ({r['nasa_score_std']:.0f})")
        rows.append(cells)
    write(
        out / "tab2b_leaderboard_nasa.tex",
        booktabs(
            ["family", *[f"{f} NASA (sd)" for f in FLEETS]],
            rows,
            "lrrrr",
            "reports/research/leaderboard_reference.csv (nasa_score_mean, nasa_score_std)",
        ),
    )


def table_endpoints(n: Numbers, out: Path) -> None:
    v = n.values

    def cell(key: str) -> str:
        return f"{v[key]} {v[key + '_ci']}"

    rows = []
    for fleet in FLEETS:
        rows.append(
            [
                fleet,
                cell(f"e1.{fleet}.tau_b"),
                v[f"e1.{fleet}.band"],
                cell(f"e2.{fleet}.selection_loss"),
                cell(f"e3.{fleet}.delta_tau"),
            ]
        )
    rows.append("\\midrule")
    rows.append(
        [
            "verdict",
            f"\\multicolumn{{2}}{{l}}{{H1 {v['verdict.all.H1']}}}",
            f"H2 {v['verdict.all.H2']}",
            f"H3 {v['verdict.all.H3']}",
        ]
    )
    rows.append("\\midrule")
    rows.append(
        [
            "N-CMAPSS$^{a}$",
            cell(f"e1.{NCMAPSS}.tau_b"),
            v[f"e1.{NCMAPSS}.band"],
            cell(f"e2.{NCMAPSS}.selection_loss"),
            cell(f"e3.{NCMAPSS}.delta_tau"),
        ]
    )
    body = booktabs(
        [
            "fleet",
            "E1 tau-b [95\\% CI]",
            "band",
            "E2 selection loss [95\\% CI]",
            "E3 delta-tau [95\\% CI]",
        ],
        rows,
        "lllll",
        "reports/research/confirmatory/e1.csv, e2.csv, e3.csv, verdicts.md "
        "(make research-confirmatory); N-CMAPSS: reports/research/ncmapss/ "
        "(make research-ncmapss-report)",
    )
    body += "% a: secondary analysis S8, descriptive; not part of the verdicts\n"
    write(out / "tab3_endpoints.tex", body)


def table_families(families: pd.DataFrame, out: Path) -> None:
    rows = []
    for fleet in FLEETS:
        g = (
            families[families["fleet"] == fleet]
            .set_index("family")
            .loc[list(FAMILY_NAMES)]
        )
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{fleet}}}}} \\\\")
        for family, r in g.iterrows():
            rows.append(
                [
                    tex(FAMILY_LABELS[family]),
                    f"{r['leaderboard_rmse']:.2f}",
                    f"{number(r['savings_share'])} {interval(r['savings_share_lo'], r['savings_share_hi'])}",
                    f"{r['failure_rate']:.3f}",
                    f"{r['waste_per_maintenance']:.1f}",
                    str(int(r["rank_rmse"])),
                    str(int(r["rank_value"])),
                ]
            )
        if fleet != FLEETS[-1]:
            rows.append("\\midrule")
    write(
        out / "tab4_families.tex",
        booktabs(
            [
                "family",
                "leaderboard RMSE",
                "S [95\\% CI]",
                "failure rate",
                "waste per maintenance",
                "rank RMSE",
                "rank S",
            ],
            rows,
            "lrlrrrr",
            "reports/research/secondary/s3_families.csv (make research-secondary); family "
            "representatives at K = 10, L = 10, R1, validation-tuned thresholds",
        ),
    )


def table_variants(variants: pd.DataFrame, out: Path) -> None:
    rows = []
    for fleet in FLEETS:
        for tuning in ("validation", "crossfit", "oracle"):
            r = variants[
                (variants["fleet"] == fleet) & (variants["tuning"] == tuning)
            ].iloc[0]
            rows.append(
                [
                    fleet if tuning == "validation" else "",
                    tuning,
                    f"{number(r['e1_tau_b'])} {interval(r['e1_tau_b_lo'], r['e1_tau_b_hi'])}",
                    (
                        f"{number(r['e2_selection_loss'])} "
                        f"{interval(r['e2_selection_loss_lo'], r['e2_selection_loss_hi'])}"
                    ),
                    f"{number(r['e3_delta_tau'])} {interval(r['e3_delta_tau_lo'], r['e3_delta_tau_hi'])}",
                    number(r["mean_savings_share"]),
                ]
            )
    write(
        out / "tab5_threshold_variants.tex",
        booktabs(
            [
                "fleet",
                "thresholds",
                "E1 [95\\% CI]",
                "E2 [95\\% CI]",
                "E3 [95\\% CI]",
                "mean S",
            ],
            rows,
            "llllll",
            "reports/research/secondary/s4_threshold_variants.csv (make research-secondary)",
        ),
    )


def _params(params: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in params.items())


def table_pool(cfg: ResearchConfig, out: Path) -> None:
    pool = yaml.safe_load(Path(cfg.paths.pool).read_text(encoding="utf-8"))
    families = sorted(pool["families"].items(), key=lambda kv: kv[1]["index"])
    rows = []
    for name, f in families:
        space = "; ".join(f"{k} {v}" for k, v in f["space"].items())
        rows.append(
            [
                tex(FAMILY_LABELS[name]),
                tex(f["inputs"]),
                str(f["n_configs"]),
                tex(space),
                tex(_params(f["all"])) if f["all"] else "--",
            ]
        )
    write(
        out / "tabA1_search_spaces.tex",
        booktabs(
            ["family", "inputs", "n", "search space", "all configurations"],
            rows,
            "llrp{6cm}p{3cm}",
            f"{cfg.paths.pool} (make research-pool)",
        ),
    )
    rows = []
    for name, f in families:
        for c in f["configs"]:
            rows.append(
                [tex(c["id"]), tex(_params(c["params"])), f"\\texttt{{{c['hash']}}}"]
            )
    write(
        out / "tabA2_pool.tex",
        booktabs(
            ["configuration", "parameters", "hash"],
            rows,
            "lp{10cm}l",
            f"{cfg.paths.pool} (make research-pool); configuration 0 of each family is its "
            "reference configuration",
            environment="longtable",
        ),
    )


def table_compute(compute: pd.DataFrame, out: Path) -> None:
    estimate_csv = read("reports/research/compute_estimate.csv")
    projected = estimate_csv.groupby("stream")["hours_r3"].sum()
    rows = []
    for archive, g in compute.groupby("archive", sort=False):
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{tex(archive)}}}}} \\\\")
        for _, r in g.iterrows():
            rows.append(
                [
                    tex(FAMILY_LABELS.get(r["family"], r["family"])),
                    r["stream"],
                    tex(r["device"]),
                    str(int(r["fits"])),
                    f"{r['hours']:.2f}",
                    f"{r['median_seconds']:.1f}",
                ]
            )
        rows.append(
            ["total", "", "", str(int(g["fits"].sum())), f"{g['hours'].sum():.2f}", ""]
        )
        rows.append("\\midrule")
    rows.append(
        ["projected (dry run), cpu", "", "", "", f"{projected.get('cpu', 0):.2f}", ""]
    )
    rows.append(
        ["projected (dry run), gpu", "", "", "", f"{projected.get('gpu', 0):.2f}", ""]
    )
    write(
        out / "tabA3_compute.tex",
        booktabs(
            [
                "family",
                "stream",
                "device",
                "fits",
                "summed fit hours",
                "median s per fit",
            ],
            rows,
            "lllrrr",
            "paper_assets/data/compute.csv (sidecars of every archive, wall_seconds) and "
            "reports/research/compute_estimate.csv (hours_r3)",
        ),
    )


def table_deviations(out: Path) -> None:
    text = Path("DEVIATIONS.md").read_text(encoding="utf-8")
    entries = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.startswith("|") and not line.startswith("|---") and "date" not in line
    ]
    rows = [[tex(c) for c in e] for e in entries] or [
        "\\multicolumn{5}{l}{none recorded in DEVIATIONS.md} \\\\"
    ]
    write(
        out / "tabA4_deviations.tex",
        booktabs(
            ["date", "section", "change", "reason", "expected impact"],
            rows,
            "llp{4cm}p{4cm}p{3cm}",
            "DEVIATIONS.md",
        ),
    )


# ---------------------------------------------------------------- digest


SECTIONS = {
    "RQ1": "RQ1: agreement between leaderboard rank and maintenance-value rank (H1)",
    "RQ2": "RQ2: selecting by validation RMSE instead of validation cost (H2)",
    "RQ3": "RQ3: jitter against leaderboard RMSE (H3)",
    "S3": "Family level (S3, secondary)",
    "S5": "Training cap (S5, secondary)",
    "F": "Numbers computed by make paper-assets",
    "LB": "Conventional leaderboard (reference configurations, legacy split)",
    "D": "Data, pool and compute",
    "X": "Exploratory (not pre-registered, uncorrected)",
}


def digest(n: Numbers, commit: str) -> str:
    lines = [
        "# Results digest",
        "",
        (
            "Facts only. Every number below is a macro of `paper_assets/numbers.tex` "
            "(`\\val{<analysis>.<fleet>.<quantity>}`), with its source file, column and the "
            "command that wrote the source. Intervals are percentile 95% bootstrap intervals; "
            "`_lo`, `_hi` and `_ci` macros exist for every number shown with one. Confirmatory "
            "numbers are `e1`, `e2`, `e3` and `verdict` on FD001-FD004; `s*` are pre-registered "
            "secondary analyses, NCMAPSS is secondary analysis S8 (descriptive), `x*` are "
            "exploratory, and `f1`, `f5`, `data` and `compute` are computed by "
            f"`make paper-assets`. Written by make paper-assets at commit {commit}."
        ),
        "",
    ]
    lines += [
        "Definitions of the numbers computed here:",
        "",
        (
            f"- F1: {F1['config_id']} on {F1['fleet']}, repeat {F1['repeat']}, with its "
            f"predictions y replaced by y + {F1['shift']:g} and by {F1['scale']:g} y before "
            "the policy code of derived.py (repeat_outcomes, age_table, savings) and the "
            "metric code (metric_components, fit_metrics) run; S and leaderboard RMSE are "
            "for that repeat alone."
        ),
        (
            "- F5 (descriptive, not pre-registered): the oracle selection takes the "
            "configuration with the highest S in hindsight (point estimate, the same in "
            "every fold); the gap is its S minus the S of the models chosen by validation "
            "cost or by validation RMSE, with the confirmatory bootstrap draws of training "
            "engines and every selection held fixed."
        ),
        "",
    ]
    for section, title in SECTIONS.items():
        rows = n.rows.get(section, [])
        if not rows:
            continue
        lines += [f"## {title}", ""]
        if section in ("RQ1", "RQ2", "RQ3"):
            h = {"RQ1": "H1", "RQ2": "H2", "RQ3": "H3"}[section]
            lines += [
                (
                    f"Pre-registered verdict: **{n.values[f'verdict.all.{h}']}** "
                    f"(`{VERDICT_FILE}`, make research-confirmatory)."
                ),
                "",
            ]
        lines += [
            "| quantity | fleet | estimate | 95% interval | macro | source | column | command |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            cells = [
                r["quantity"],
                r["fleet"],
                plain(r["estimate"]),
                plain(r["interval"]),
                f"`{r['macro']}`",
                f"`{r['source']}`",
                r["column"],
                command_for(r["source"]),
            ]
            lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    root = Path(cfg.paths.paper_assets)
    figures, tables, data = root / "figures", root / "tables", root / "data"
    for d in (figures, tables, data):
        d.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    commit = git_commit()
    n = Numbers()

    endpoint_numbers(n, cfg)
    secondary_numbers(n)
    exploratory_numbers(n)
    leaderboard_numbers(n)

    frame, detail = invariance(cfg)
    frame.to_csv(data / "f1_invariance.csv", index=False, lineterminator="\n")
    for _, r in frame.iterrows():
        stem = f"f1.{F1['fleet']}.{r['variant']}"
        src = "paper_assets/data/f1_invariance.csv"
        n.add(
            "F",
            f"{stem}_leaderboard_rmse",
            f"F1 leaderboard RMSE of {F1['config_id']}, repeat {F1['repeat']}, predictions "
            f"{r['transform']}",
            r["leaderboard_rmse"],
            src,
            "leaderboard_rmse",
            decimals=2,
        )
        n.add(
            "F",
            f"{stem}_savings_share",
            f"F1 S, predictions {r['transform']}",
            r["savings_share"],
            src,
            "savings_share",
        )
        n.add(
            "F",
            f"{stem}_outcomes_differing",
            "F1 outcomes (engine x setting x tuning) that differ from the original, R1 and R2",
            f"{int(r['outcomes_differing_R1'])} and {int(r['outcomes_differing_R2'])} of "
            f"{int(r['outcomes_compared'])}",
            src,
            "outcomes_differing_R1, outcomes_differing_R2, outcomes_compared",
        )
    fig_invariance(frame, detail, cfg, figures)

    gap = oracle_selection(cfg)
    gap.to_csv(data / "f5_oracle_selection.csv", index=False, lineterminator="\n")
    for _, r in gap.iterrows():
        src = "paper_assets/data/f5_oracle_selection.csv"
        n.add(
            "RQ2",
            f"f5.{r['fleet']}.oracle_config",
            "F5 configuration with the highest S in hindsight",
            tex(r["oracle_config"]),
            src,
            "oracle_config",
        )
        n.add(
            "RQ2",
            f"f5.{r['fleet']}.oracle_s",
            "F5 its S",
            r["oracle_s"],
            src,
            "oracle_s",
        )
        for name, what in (("cost", "validation cost"), ("rmse", "validation RMSE")):
            n.add(
                "RQ2",
                f"f5.{r['fleet']}.gap_{name}",
                "F5 oracle-selection gap (descriptive, not pre-registered): "
                f"S(best in hindsight) - S(chosen by {what})",
                r[f"gap_{name}"],
                src,
                f"gap_{name}, _lo, _hi",
                lo=r[f"gap_{name}_lo"],
                hi=r[f"gap_{name}_hi"],
            )

    data_table = data_summary(cfg)
    data_table.to_csv(data / "data_folds.csv", index=False, lineterminator="\n")
    for _, r in data_table.iterrows():
        src = "paper_assets/data/data_folds.csv"
        for column in (
            "train_units",
            "test_units",
            "regimes",
            "life_min",
            "life_max",
            "cap",
            "flights",
            "repeats",
        ):
            n.add(
                "D",
                f"data.{r['fleet']}.{column}",
                f"{column}",
                int(r[column]),
                src,
                column,
                decimals=0,
            )
        n.add(
            "D",
            f"data.{r['fleet']}.life_median",
            "median life",
            r["life_median"],
            src,
            "life_median",
            decimals=1,
        )
        for column in ("outer_test", "inner_val", "inner_fit"):
            n.add(
                "D",
                f"data.{r['fleet']}.{column}",
                f"{column} engines per fold",
                str(r[column]),
                src,
                column,
            )
    pool = yaml.safe_load(Path(cfg.paths.pool).read_text(encoding="utf-8"))
    n.add(
        "D",
        "data.all.configs",
        "configurations in the pool",
        int(pool["total"]),
        cfg.paths.pool,
        "total",
        decimals=0,
    )
    n.add(
        "D",
        "data.all.families",
        "families in the pool",
        len(pool["families"]),
        cfg.paths.pool,
        "families",
        decimals=0,
    )

    compute = compute_summary(cfg)
    compute.to_csv(data / "compute.csv", index=False, lineterminator="\n")
    for archive, g in compute.groupby("archive", sort=False):
        key = archive.lower().replace(" ", "_").replace("-", "")
        src = "paper_assets/data/compute.csv"
        n.add(
            "D",
            f"compute.all.{key}_fits",
            f"fits in the {archive} archive",
            int(g["fits"].sum()),
            src,
            f"fits (archive = {archive})",
            decimals=0,
        )
        n.add(
            "D",
            f"compute.all.{key}_failed",
            f"failed fits in the {archive} archive",
            int(g["failed"].sum()),
            src,
            f"failed (archive = {archive})",
            decimals=0,
        )
        for stream, h in g.groupby("stream"):
            n.add(
                "D",
                f"compute.all.{key}_{stream}_hours",
                f"summed fit hours, {archive}, {stream} stream",
                h["hours"].sum(),
                src,
                f"hours (archive = {archive}, stream = {stream})",
                decimals=2,
            )

    families = read("reports/research/secondary/s3_families.csv")
    grid = read("reports/research/secondary/s2_settings_grid.csv")
    variants = read("reports/research/secondary/s4_threshold_variants.csv")
    configurations = read("reports/research/confirmatory/configurations.csv")
    fig_protocol(figures)
    fig_bump(families, figures)
    fig_tau_by_cost(grid, cfg, figures)
    fig_selection(
        read("reports/research/confirmatory/e2.csv"),
        gap,
        read("reports/research/ncmapss/e2.csv"),
        figures,
    )
    fig_scatter(configurations, figures, "fig6_value_vs_accuracy", FLEETS)
    fig_caps(read("reports/research/secondary/s5_caps.csv"), figures)
    fig_heatmaps(grid, cfg, figures)
    fig_threshold_variants(variants, figures)
    ncmapss = read("reports/research/ncmapss/configurations.csv")
    fig_scatter(ncmapss, figures, "figA3_ncmapss", [NCMAPSS])

    table_data(data_table, tables)
    table_leaderboard(tables)
    table_endpoints(n, tables)
    table_families(families, tables)
    table_variants(variants, tables)
    table_pool(cfg, tables)
    table_compute(compute, tables)
    table_deviations(tables)

    header = (
        "% Numbers of the paper, written by make paper-assets at commit "
        f"{commit}; do not edit.\n% Sources and columns: paper_assets/RESULTS_DIGEST.md."
    )
    write(root / "numbers.tex", numbers_tex(n.values, header))
    write(root / "RESULTS_DIGEST.md", digest(n, commit))
    print(
        f"{len(n.values)} numbers, {len(list(figures.glob('*.pdf')))} figures, "
        f"{len(list(tables.glob('*.tex')))} tables in {root}"
    )


if __name__ == "__main__":
    main()
