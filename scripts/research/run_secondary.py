"""Secondary analyses S1-S7 and S9 (PREREGISTRATION.md section 7) of the confirmatory archive.

Reads the derived tables in runs/research/derived/archive (make research-outcomes, make
research-metrics); builds the regret tables of S6 from the archive once and keeps them there;
reads the nested (S4) and training-cap (S5) archives once their fits are complete. Writes
reports/research/secondary/ and a README that maps every file to its item. Every quantity uses
the confirmatory bootstrap draws and formulas (analysis.py, unchanged). Secondary tests use
Benjamini-Hochberg at stats.fdr within each analysis; nothing here is confirmatory.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from maintadapt.data import load_subset
from maintadapt.research.analysis import (
    Estimate,
    FleetInputs,
    analyse_fleet,
    band,
    decision_loss,
    estimate,
    fleet_inputs,
    jitter,
    leaderboard_rmse,
    verdict_h1,
    verdict_h2,
    verdict_h3,
)
from maintadapt.research.archive import OK, load, records
from maintadapt.research.config import (
    PrimaryConfig,
    ResearchConfig,
    fleet_number,
    load_research_config,
)
from maintadapt.research.crossfit import git_commit, nested_root
from maintadapt.research.derived import TUNINGS, metric_components, repeat_outcomes
from maintadapt.research.folds import lifetimes, load_folds
from maintadapt.research.guards import tag_exists
from maintadapt.research.secondary import (
    Draws,
    analyse_regret,
    check_oof,
    competitive,
    failure_and_waste,
    family_level,
    family_tables,
    inversions,
    matched_values,
    nasa_score,
    nested_outcomes,
    official_values,
    outcomes_at,
    pool_order,
    pooled_rmse,
    regret_age_table,
    regret_inputs,
    regret_outcomes,
    representatives,
    settings_grid,
    split_by_setting,
    tau,
    with_validation_selection,
)

ITEMS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9")
PREDICTION_COLUMNS = ["unit_id", "cycle", "rul", "split", "pred", "repeat", "fold"]
LEADERBOARD_CAP = 125


def status(item: str) -> str:
    return (
        f"SECONDARY {item}: pre-registered secondary analysis (PREREGISTRATION.md section "
        "7); not confirmatory"
    )


def write_csv(frame: pd.DataFrame, path: Path, item: str) -> None:
    text = f"# {status(item)}\n" + frame.to_csv(index=False, lineterminator="\n")
    path.write_text(text, encoding="utf-8", newline="\n")


def columns(name: str, e: Estimate) -> dict[str, float]:
    return {name: e.point, f"{name}_lo": e.lo, f"{name}_hi": e.hi}


def read_tables(source: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(source / f"{name}.parquet")
        for name in ("outcomes", "policies", "age", "engines", "components")
    }


def with_outcomes(tables: dict, found: pd.DataFrame) -> dict:
    return {**tables, "outcomes": found}


class Fleet:
    """One fleet's tables, its primary inputs and the confirmatory bootstrap draws."""

    def __init__(self, cfg: ResearchConfig, derived: Path, fleet: str) -> None:
        self.cfg, self.fleet, self.source = cfg, fleet, derived / fleet
        tables = read_tables(self.source)
        self.parts = split_by_setting(tables.pop("outcomes"))
        # every use takes the outcomes of one setting from parts
        self.tables = tables
        self.configs = pool_order(
            list(outcomes_at(self.parts, cfg.decision.primary)["config_id"].unique())
        )
        self.primary = cfg.decision.primary
        self.seed = [cfg.stats.seed, fleet_number(fleet)]
        self.base = self.inputs(self.primary)
        self.draws = Draws.of(self.base, cfg.stats.n_boot, self.seed)
        d = self.draws
        self.loss = decision_loss(self.base, d.train)
        self.loss_point = decision_loss(self.base, d.once_train)[0]
        self.rmse = leaderboard_rmse(self.base, d.test)
        self.rmse_point = leaderboard_rmse(self.base, d.once_test)[0]

    def inputs(
        self, setting: PrimaryConfig, configs: list[str] | None = None
    ) -> FleetInputs:
        found = outcomes_at(self.parts, setting)
        return fleet_inputs(
            with_outcomes(self.tables, found), setting, configs or self.configs
        )

    def analyse(self, inputs: FleetInputs):
        return analyse_fleet(
            inputs, self.cfg.stats.n_boot, self.cfg.stats.ci, self.seed
        )

    def e1(self) -> tuple[Estimate, np.ndarray]:
        return tau(
            self.rmse_point, self.rmse, self.loss_point, self.loss, self.cfg.stats.ci
        )


def s1(f: Fleet) -> list[dict]:
    """E1 with the NASA score and with all-cycle RMSE in place of last-cycle RMSE."""
    comps, d, ci = f.tables["components"], f.draws, f.cfg.stats.ci
    nasa = official_values(f.base, comps, "nasa_last")
    sse = official_values(f.base, comps, "sse_all")
    rows = official_values(f.base, comps, "n_all")
    metrics = {
        "leaderboard_rmse": (f.rmse_point, f.rmse),
        "nasa_score": (nasa_score(nasa, d.once_test)[0], nasa_score(nasa, d.test)),
        "rmse_all_cycles": (
            pooled_rmse(sse, rows, d.once_test)[0],
            pooled_rmse(sse, rows, d.test),
        ),
    }
    out = []
    for name, (point, reps) in metrics.items():
        e, _ = tau(point, reps, f.loss_point, f.loss, ci)
        out.append(
            {"fleet": f.fleet, "metric": name, "configs": len(f.configs)}
            | columns("tau_b", e)
            | {"valid_replicates": e.valid}
        )
    return out


def s2(f: Fleet) -> list[dict]:
    """E1-E3 at every rule, lead time and cost ratio, validation-tuned."""
    out = []
    for setting in settings_grid(f.cfg.decision, f.primary.tuning):
        r = f.analyse(f.inputs(setting))
        out.append(
            {
                "fleet": f.fleet,
                "rule": setting.rule,
                "lead": setting.lead_time,
                "cost_ratio": setting.cost_ratio,
                "primary": setting == f.primary,
            }
            | columns("e1_tau_b", r.e1)
            | {"e1_band": band(r.e1.point)}
            | columns("e2_selection_loss", r.e2)
            | columns("s_by_cost", r.s_by_cost)
            | columns("s_by_rmse", r.s_by_rmse)
            | columns("e3_delta_tau", r.e3)
            | columns("tau_jitter", r.tau_jitter)
        )
    return out


def s2_rules(grid: pd.DataFrame) -> pd.DataFrame:
    """The section 6 rules read at every setting, descriptively, from the S2 estimates."""

    def as_estimates(group: pd.DataFrame, name: str) -> dict[str, Estimate]:
        return {
            row["fleet"]: Estimate(row[name], row[f"{name}_lo"], row[f"{name}_hi"], 0)
            for _, row in group.iterrows()
        }

    rows = []
    for (rule, lead, k), group in grid.groupby(
        ["rule", "lead", "cost_ratio"], sort=False
    ):
        rows.append(
            {
                "rule": rule,
                "lead": lead,
                "cost_ratio": k,
                "H1_rule": verdict_h1(as_estimates(group, "e1_tau_b")),
                "H2_rule": verdict_h2(as_estimates(group, "e2_selection_loss")),
                "H3_rule": verdict_h3(as_estimates(group, "e3_delta_tau")),
            }
        )
    return pd.DataFrame(rows)


def heatmaps(grid: pd.DataFrame, cfg: ResearchConfig) -> str:
    """E1, E2 and E3 as cost ratio by lead time grids, one per fleet and rule."""
    lines = [
        "# SECONDARY S2: E1-E3 over the settings grid",
        "",
        (
            f"**{status('S2')}.** Point estimates from `s2_settings_grid.csv`; validation-tuned "
            "thresholds; * marks a 95% interval that excludes 0; the primary setting is in bold."
        ),
        "",
    ]
    p = cfg.decision.primary
    for name, title in (
        ("e1_tau_b", "E1, tau-b(leaderboard RMSE, D)"),
        ("e2_selection_loss", "E2, selection loss"),
        ("e3_delta_tau", "E3, delta-tau"),
    ):
        lines += [f"## {title}", ""]
        for (fleet, rule), group in grid.groupby(["fleet", "rule"], sort=False):
            leads = sorted(group["lead"].unique())
            lines += [
                f"{fleet}, rule {rule}:",
                "",
                "| K | " + " | ".join(f"L = {lead}" for lead in leads) + " |",
                "|---|" + "---|" * len(leads),
            ]
            for k, row_group in group.groupby("cost_ratio", sort=True):
                cells = []
                for lead in leads:
                    row = row_group[row_group["lead"] == lead].iloc[0]
                    text = f"{row[name]:.2f}"
                    if row[f"{name}_lo"] > 0 or row[f"{name}_hi"] < 0:
                        text += "*"
                    if rule == p.rule and lead == p.lead_time and k == p.cost_ratio:
                        text = f"**{text}**"
                    cells.append(text)
                lines.append(f"| {k:g} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def s3(f: Fleet) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Family representatives: points and ranks, pairwise differences, family-level E1."""
    reps = representatives(f.base)
    tables = family_tables(
        with_outcomes(f.tables, outcomes_at(f.parts, f.primary)), reps
    )
    families = list(dict.fromkeys(reps["family"]))
    inputs = fleet_inputs(tables, f.primary, families)
    result = family_level(inputs, f.draws, f.cfg.stats.ci)
    extra = failure_and_waste(tables["outcomes"]).rename_axis("family").reset_index()
    points = result.points.merge(extra, on="family").assign(fleet=f.fleet)
    pairs = result.pairs.assign(fleet=f.fleet)
    e1 = {"fleet": f.fleet, "families": len(families)} | columns("tau_b", result.e1)
    return points, pairs, e1


def s4_variants(f: Fleet) -> list[dict]:
    """E1-E3 under each tuning of the derived tables; E2 still selects by validation."""
    out = []
    for tuning in TUNINGS:
        setting = replace(f.primary, tuning=tuning)
        inputs = with_validation_selection(f.inputs(setting), f.base)
        r = f.analyse(inputs)
        out.append(
            {"fleet": f.fleet, "tuning": tuning, "configs": len(f.configs)}
            | columns("e1_tau_b", r.e1)
            | columns("e2_selection_loss", r.e2)
            | columns("s_by_cost", r.s_by_cost)
            | columns("s_by_rmse", r.s_by_rmse)
            | columns("e3_delta_tau", r.e3)
            | {"mean_savings_share": float(r.points["savings_share"].mean())}
        )
    return out


def complete(found: pd.DataFrame, expected: set[tuple[str, int, int]]) -> bool:
    if found.empty:
        return False
    ok = found[found["status"] == OK]
    seen = set(zip(ok["config_id"], ok["repeat"], ok["fold"], strict=True))
    return expected <= seen


def s4_nested(f: Fleet) -> tuple[list[dict], list[dict]] | None:
    """Nested tuning of the family representatives against the other tunings (repeat 0)."""
    cfg, n = f.cfg, f.cfg.secondary.nested
    reps = representatives(f.base)
    reps = reps[reps["repeat"] == n.repeat].reset_index(drop=True)
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)[f.fleet][n.repeat]
    oof, test = {}, {}
    for outer in sorted(reps["fold"].unique()):
        root = nested_root(cfg.paths.archive_nested, outer)
        found = records(root, [f.fleet])
        chosen = reps[reps["fold"] == outer]
        if not found.empty and set(found["config_id"]) != set(chosen["config_id"]):
            raise SystemExit(
                f"{root} holds {sorted(set(found['config_id']))}, not the representatives "
                f"{sorted(chosen['config_id'])}"
            )
        expected = {
            (c, n.repeat, j) for c in chosen["config_id"] for j in range(n.inner_folds)
        }
        if not complete(found, expected):
            return None
        for config_id in chosen["config_id"]:
            inner = load(
                root,
                fleets=[f.fleet],
                config_ids=[config_id],
                splits=["outer_test"],
                columns=PREDICTION_COLUMNS,
                found=found,
            )
            check_oof(inner, folds[outer])
            oof[(config_id, outer)] = inner
    archive = records(cfg.paths.archive, [f.fleet])
    for config_id, outer in oof:
        test[(config_id, outer)] = load(
            cfg.paths.archive,
            fleets=[f.fleet],
            config_ids=[config_id],
            repeats=[n.repeat],
            folds=[outer],
            splits=["outer_test"],
            columns=PREDICTION_COLUMNS,
            found=archive,
        )
    policies, found = [], []
    for family, group in reps.groupby("family", sort=False):
        pairs = list(zip(group["config_id"], group["fold"], strict=True))
        pol, out = nested_outcomes(
            {outer: oof[(c, outer)] for c, outer in pairs},
            {outer: test[(c, outer)] for c, outer in pairs},
            cfg.decision,
        )
        config_of = dict(zip(group["fold"], group["config_id"], strict=True))
        for table, sink in ((pol, policies), (out, found)):
            sink.append(
                table.assign(
                    config_id=table["fold"].map(config_of),
                    family=family,
                    repeat=n.repeat,
                )
            )
    nested_tables = {
        **f.tables,
        "outcomes": pd.concat(found, ignore_index=True),
        "policies": pd.concat(policies, ignore_index=True),
    }
    families = list(dict.fromkeys(reps["family"]))
    rows, taus = [], []
    for tuning in (*TUNINGS, "nested"):
        setting = replace(f.primary, tuning=tuning)
        if tuning == "nested":
            source = nested_tables
        else:
            at = outcomes_at(f.parts, setting)
            source = with_outcomes(f.tables, at[at["repeat"] == n.repeat])
        inputs = fleet_inputs(family_tables(source, reps), setting, families)
        loss = decision_loss(inputs, f.draws.train)
        loss_point = decision_loss(inputs, f.draws.once_train)[0]
        rmse = leaderboard_rmse(inputs, f.draws.test)
        rmse_point = leaderboard_rmse(inputs, f.draws.once_test)[0]
        for i, family in enumerate(families):
            e = estimate(1.0 - loss_point[i], 1.0 - loss[:, i], cfg.stats.ci)
            rows.append(
                {"fleet": f.fleet, "family": family, "tuning": tuning}
                | columns("savings_share", e)
            )
        e1, _ = tau(rmse_point, rmse, loss_point, loss, cfg.stats.ci)
        taus.append(
            {"fleet": f.fleet, "tuning": tuning, "families": len(families)}
            | columns("tau_b", e1)
        )
    return rows, taus


def cap_fits(
    cfg: ResearchConfig, root: Path, fleet: str, config_id: str, repeat: int
) -> dict[int, pd.DataFrame] | None:
    found = records(root, [fleet])
    expected = {(config_id, repeat, k) for k in range(cfg.folds.n_outer)}
    if not complete(found, expected):
        return None
    table = load(
        root,
        fleets=[fleet],
        config_ids=[config_id],
        repeats=[repeat],
        columns=PREDICTION_COLUMNS,
        found=found,
    )
    return {int(k): fit for k, fit in table.groupby("fold", sort=True)}


def s5(f: Fleet) -> list[dict] | None:
    """The reference XGBoost and LSTM trained with other caps, against the confirmatory 125."""
    cfg, caps = f.cfg, f.cfg.secondary.caps
    truncation = pd.read_parquet(f.source / "truncation.parquet")
    runs = [(cfg.data.cap, Path(cfg.paths.archive))] + [
        (cap, Path(cfg.paths.archive_caps) / caps.name(cap)) for cap in caps.values
    ]
    found, policies, components, labels = [], [], [], {}
    for family in caps.families:
        config_id = f"{family}-00"
        for cap, root in runs:
            fits = cap_fits(cfg, root, f.fleet, config_id, caps.repeat)
            if fits is None:
                return None
            label = f"{config_id}@{caps.name(cap)}"
            labels[label] = (family, caps.name(cap))
            pol, out = repeat_outcomes(fits, cfg.decision)
            keys = {"config_id": label, "repeat": caps.repeat}
            found.append(out.assign(**keys))
            policies.append(pol.assign(**keys))
            for k, fit in fits.items():
                part = metric_components(fit, truncation, LEADERBOARD_CAP)
                components.append(part.assign(fold=k, **keys))
    tables = {
        **f.tables,
        "outcomes": pd.concat(found, ignore_index=True),
        "policies": pd.concat(policies, ignore_index=True),
        "components": pd.concat(components, ignore_index=True),
    }
    names = list(labels)
    inputs = fleet_inputs(tables, f.primary, names)
    d, ci = f.draws, cfg.stats.ci
    raw = official_values(inputs, tables["components"], "sq_last_raw")
    raw_inputs = replace(inputs, sq_last=raw)
    metrics = {
        "leaderboard_rmse": (
            leaderboard_rmse(inputs, d.once_test)[0],
            leaderboard_rmse(inputs, d.test),
        ),
        "rmse_last_uncapped": (
            leaderboard_rmse(raw_inputs, d.once_test)[0],
            leaderboard_rmse(raw_inputs, d.test),
        ),
        "jitter": (
            jitter(inputs, d.once_test[:, inputs.jitter_engines])[0],
            jitter(inputs, d.test[:, inputs.jitter_engines]),
        ),
        "savings_share": (
            1.0 - decision_loss(inputs, d.once_train)[0],
            1.0 - decision_loss(inputs, d.train),
        ),
    }
    rows = []
    for i, name in enumerate(names):
        family, cap = labels[name]
        row = {"fleet": f.fleet, "family": family, "config_id": f"{family}-00"}
        row |= {"training_cap": cap, "confirmatory": cap == str(cfg.data.cap)}
        for metric, (point, reps) in metrics.items():
            row |= columns(metric, estimate(point[i], reps[:, i], ci))
        rows.append(row)
    return rows


def build_regret(
    cfg: ResearchConfig, fleet: str, configs: list[str], out: Path
) -> None:
    """Regret-tuned R1 policies of every fit at the primary lead time; kept in out."""
    start = time.perf_counter()
    root = Path(cfg.paths.archive)
    found = records(root, [fleet])
    d, lead = cfg.decision, cfg.decision.primary.lead_time
    policies, rows = [], []
    for config_id in configs:
        table = load(
            root,
            fleets=[fleet],
            config_ids=[config_id],
            splits=["inner_val", "outer_test"],
            columns=PREDICTION_COLUMNS,
            found=found,
        )
        for repeat, in_repeat in table.groupby("repeat", sort=True):
            fits = {int(k): fit for k, fit in in_repeat.groupby("fold", sort=True)}
            pol, outcome = regret_outcomes(
                fits, lead, d.regret_failure_costs, d.threshold_levels
            )
            keys = {"config_id": config_id, "repeat": int(repeat)}
            policies.append(pol.assign(**keys))
            rows.append(outcome.assign(**keys))
    lives = lifetimes(load_subset(cfg.data.raw_dir, fleet, cfg.data.cap).train)
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)[fleet]
    repeats = sorted(int(r) for r in found["repeat"].unique())
    age = regret_age_table(lives, folds, repeats, d.regret_failure_costs)
    pd.concat(policies, ignore_index=True).to_parquet(out / "regret_policies.parquet")
    pd.concat(rows, ignore_index=True).to_parquet(out / "regret_outcomes.parquet")
    age.to_parquet(out / "regret_age.parquet")
    summary = {
        "fleet": fleet,
        "configs": len(configs),
        "lead": lead,
        "failure_costs": d.regret_failure_costs,
        "seconds": round(time.perf_counter() - start, 1),
    }
    (out / "build_regret.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n"
    )


def s6(f: Fleet) -> list[dict]:
    """E1-E3 with regret in place of the cost rate, at every C_f of section 7."""
    paths = [f.source / f"regret_{n}.parquet" for n in ("outcomes", "policies", "age")]
    if not all(p.exists() for p in paths):
        build_regret(f.cfg, f.fleet, f.configs, f.source)
    found, policies, age = (pd.read_parquet(p) for p in paths)
    out = []
    for cf in f.cfg.decision.regret_failure_costs:
        r = analyse_regret(
            f.base,
            regret_inputs(f.base, found, policies, age, float(cf)),
            f.draws,
            f.cfg.stats.ci,
        )
        out.append(
            {"fleet": f.fleet, "failure_cost": cf, "lead": f.primary.lead_time}
            | columns("e1_tau_b", r.e1)
            | columns("e2_selection_loss", r.e2)
            | columns("s_by_regret", r.s_by_regret)
            | columns("s_by_rmse", r.s_by_rmse)
            | columns("e3_delta_tau", r.e3)
        )
    return out


def s7(f: Fleet) -> dict:
    """RMSE at the truncation points of the outer_test engines, against D on those engines."""
    comps, d, ci = f.tables["components"], f.draws, f.cfg.stats.ci
    sse = matched_values(f.base, comps, "sse_trunc")
    rows = matched_values(f.base, comps, "n_trunc")
    matched = pooled_rmse(sse, rows, d.train)
    matched_point = pooled_rmse(sse, rows, d.once_train)[0]
    e_matched, reps_matched = tau(matched_point, matched, f.loss_point, f.loss, ci)
    e1, reps_e1 = f.e1()
    return (
        {"fleet": f.fleet, "configs": len(f.configs)}
        | columns("tau_matched", e_matched)
        | columns("tau_leaderboard", e1)
        | columns(
            "difference",
            estimate(e_matched.point - e1.point, reps_matched - reps_e1, ci),
        )
    )


def s9(f: Fleet) -> dict:
    """E1 among the configurations within the margin of the best leaderboard RMSE."""
    margin = f.cfg.secondary.competitive_margin
    points = pd.DataFrame({"config_id": f.configs, "leaderboard_rmse": f.rmse_point})
    chosen = competitive(points, margin)
    row = {
        "fleet": f.fleet,
        "margin": margin,
        "best_rmse": float(points["leaderboard_rmse"].min()),
        "configs": len(chosen),
        "families": len({c.rsplit("-", 1)[0] for c in chosen}),
    }
    if len(chosen) < 3:
        return row | {"tau_b": np.nan, "tau_b_lo": np.nan, "tau_b_hi": np.nan}
    r = f.analyse(f.inputs(f.primary, chosen))
    return row | columns("tau_b", r.e1) | {"valid_replicates": r.e1.valid}


FILES = {
    "s1_accuracy_metrics.csv": (
        "S1",
        (
            "E1 with the NASA score and with RMSE over all observed cycles of the official test "
            "engines; the leaderboard_rmse row repeats E1 from the same replicates"
        ),
    ),
    "s2_settings_grid.csv": (
        "S2",
        (
            "E1-E3 (with S by validation cost and by validation RMSE) at every rule, lead time "
            "and cost ratio, validation-tuned"
        ),
    ),
    "s2_rules.csv": (
        "S2",
        "the interpretation rules of section 6 read descriptively at every setting",
    ),
    "s2_heatmaps.md": ("S2", "E1, E2 and E3 as K by L grids per fleet and rule"),
    "s3_families.csv": (
        "S3",
        (
            "family representatives: leaderboard RMSE, S and jitter with intervals, ranks by "
            "leaderboard RMSE and by S (bump-chart data), failure share and waste"
        ),
    ),
    "s3_pairwise.csv": (
        "S3",
        (
            "every pair of families: differences in leaderboard RMSE and in D with intervals, "
            "bootstrap p-values, Benjamini-Hochberg q-values over the whole analysis, inversions"
        ),
    ),
    "s3_family_e1.csv": (
        "S3",
        "tau-b(leaderboard RMSE, D) across the family representatives",
    ),
    "s4_threshold_variants.csv": (
        "S4",
        "E1-E3 under validation, crossfit and oracle tuning; E2 keeps the validation selection",
    ),
    "s4_nested.csv": (
        "S4",
        (
            "S of each family representative under validation, crossfit, oracle and nested tuning "
            "(nested fleets, repeat 0)"
        ),
    ),
    "s4_nested_e1.csv": (
        "S4",
        "tau-b(leaderboard RMSE, D) across the family representatives per tuning (repeat 0)",
    ),
    "s5_caps.csv": (
        "S5",
        (
            "reference XGBoost and LSTM per training cap: leaderboard RMSE (truth capped at 125), "
            "RMSE against the uncapped truth, jitter and S (repeat 0)"
        ),
    ),
    "s6_regret.csv": (
        "S6",
        "E1-E3 with regret (C_f * failure + waste) in place of the cost rate, per C_f",
    ),
    "s7_matched_accuracy.csv": (
        "S7",
        "tau-b(RMSE at 20 truncation points per outer_test engine, D) against E1",
    ),
    "s9_competitive.csv": (
        "S9",
        "E1 among the configurations within the margin of the best leaderboard RMSE",
    ),
}
PENDING = {
    "s4_nested.csv": "make research-nested STREAM=cpu and STREAM=gpu",
    "s4_nested_e1.csv": "make research-nested STREAM=cpu and STREAM=gpu",
    "s5_caps.csv": "make research-caps",
}


def readme(cfg: ResearchConfig, out: Path) -> str:
    c = cfg.secondary
    lines = [
        "# Secondary analyses",
        "",
        (
            "Pre-registered secondary analyses S1-S7 and S9 of PREREGISTRATION.md section 7, written "
            f"by `make research-secondary` at commit {git_commit()}. None of them is confirmatory "
            "and none changes a verdict. S8 (N-CMAPSS) is not part of this folder."
        ),
        "",
        (
            "Every estimate uses the derived tables of `runs/research/derived/archive/<fleet>/` and "
            "the confirmatory bootstrap: 2,000 replicates with training engines from "
            "`default_rng([2026, fleet number, 0])` and official test engines from `[..., 1]`, "
            "shared by every configuration, so each interval comes from the same replicates as E1-E3. "
            "Rows are point estimates with percentile 95% intervals (`_lo`, `_hi`). The first line "
            "of every CSV states its item."
        ),
        "",
        "| file | item | content |",
        "|---|---|---|",
    ]
    for name, (item, text) in FILES.items():
        state = (
            ""
            if (out / name).exists()
            else f" Pending: needs {PENDING.get(name, 'a run')}."
        )
        lines.append(f"| `{name}` | {item} | {text}.{state} |")
    lines += [
        "",
        "## Settings of each item",
        "",
        (
            "- S2: the grid of section 5.5, K in {1.5, 2, 3, 5, 10, 20, 50} and L in {0, 10, 20}, "
            "for rules R1 and R2, all with validation-tuned thresholds."
        ),
        (
            "- S3: in each (repeat, fold) the family's configuration with the lowest inner_val RMSE "
            "represents the family; the representatives' outcomes are pooled like a configuration's. "
            f"Benjamini-Hochberg at {cfg.stats.fdr} runs over every p-value of S3 (both metrics, all "
            "pairs and fleets). A pair is inverted when both differences are rejected with opposite "
            "signs; `inversion_unadjusted` uses the intervals instead."
        ),
        (
            "- S4: crossfit and oracle thresholds come from the derived tables. Nested thresholds "
            f"are tuned on {c.nested.inner_folds}-fold inner cross-fitted predictions of the "
            f"outer-train engines ({', '.join(c.nested.fleets)}, repeat {c.nested.repeat}, family "
            "representatives) and applied to the confirmatory outer_test predictions."
        ),
        (
            f"- S5: caps {', '.join(c.caps.name(v) for v in c.caps.values)} besides the "
            f"confirmatory {cfg.data.cap}, families {', '.join(c.caps.families)}, configuration 0, "
            f"repeat {c.caps.repeat}, with the confirmatory seeds; metrics on the official test "
            f"engines use the truth capped at {LEADERBOARD_CAP} except `rmse_last_uncapped`."
        ),
        (
            "- S6: thresholds and age replacement are tuned to the lowest mean regret, rule R1, "
            f"lead time {cfg.decision.primary.lead_time}; the regret share is 1 - regret / age "
            "replacement's regret, pooled over the folds of a repeat and averaged over repeats "
            "(perfect prognostics has zero regret). E2 selects by in-sample validation regret."
        ),
        (
            "- S7: each configuration's RMSE at the truncation points of its outer_test engines, "
            "pooled per repeat and averaged, resampled with the training engines like D."
        ),
        (
            f"- S9: margin {c.competitive_margin:g} cycles of leaderboard RMSE (point estimates); "
            "the set is held fixed in the bootstrap."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--only", nargs="+", choices=ITEMS, help="items to (re)compute")
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    if not tag_exists():
        raise SystemExit(
            "secondary analyses run only after the tag analysis-frozen exists"
        )
    derived = Path(cfg.paths.derived) / Path(cfg.paths.archive).name
    fleets = cfg.data.fleets
    missing = [f for f in fleets if not (derived / f / "outcomes.parquet").exists()]
    if missing:
        raise SystemExit(
            f"no derived tables for {missing}; run research-outcomes first"
        )
    items = args.only or ITEMS
    out = Path(cfg.paths.reports) / "secondary"
    out.mkdir(parents=True, exist_ok=True)
    collected: dict[str, list] = {name: [] for name in FILES}
    timings: dict[str, float] = {}
    confirmatory = Path(cfg.paths.reports) / "confirmatory" / "e1.csv"
    reference = (
        pd.read_csv(confirmatory, comment="#").set_index("fleet")
        if confirmatory.exists()
        else None
    )

    def timed(item: str, work, *args):
        start = time.perf_counter()
        result = work(*args)
        timings[item] = timings.get(item, 0.0) + time.perf_counter() - start
        return result

    for fleet in fleets:
        f = Fleet(cfg, derived, fleet)
        e1, _ = f.e1()
        # same tables, draws and formula as the confirmatory E1, so the same number
        if (
            reference is not None
            and abs(e1.point - reference.loc[fleet, "tau_b"]) > 1e-12
        ):
            raise SystemExit(f"{fleet}: E1 differs from the confirmatory e1.csv")
        if "S1" in items:
            collected["s1_accuracy_metrics.csv"] += timed("S1", s1, f)
        if "S2" in items:
            collected["s2_settings_grid.csv"] += timed("S2", s2, f)
        if "S3" in items:
            points, pairs, fam_e1 = timed("S3", s3, f)
            collected["s3_families.csv"].append(points)
            collected["s3_pairwise.csv"].append(pairs)
            collected["s3_family_e1.csv"].append(fam_e1)
        if "S4" in items:
            collected["s4_threshold_variants.csv"] += timed("S4", s4_variants, f)
            if fleet in cfg.secondary.nested.fleets:
                nested = timed("S4", s4_nested, f)
                if nested is not None:
                    collected["s4_nested.csv"] += nested[0]
                    collected["s4_nested_e1.csv"] += nested[1]
        if "S5" in items:
            caps = timed("S5", s5, f)
            if caps is not None:
                collected["s5_caps.csv"] += caps
        if "S6" in items:
            collected["s6_regret.csv"] += timed("S6", s6, f)
        if "S7" in items:
            collected["s7_matched_accuracy.csv"].append(timed("S7", s7, f))
        if "S9" in items:
            collected["s9_competitive.csv"].append(timed("S9", s9, f))
        print(
            f"{fleet}: {', '.join(f'{k} {v:.0f}s' for k, v in timings.items())}",
            flush=True,
        )
        del f

    for name, rows in collected.items():
        item = FILES[name][0]
        if item not in items or not rows:
            continue
        if name == "s3_pairwise.csv":
            frame = inversions(pd.concat(rows, ignore_index=True), cfg.stats.fdr)
            frame = frame[["fleet", *[c for c in frame.columns if c != "fleet"]]]
        elif name == "s3_families.csv":
            frame = pd.concat(rows, ignore_index=True)
            frame = frame[["fleet", *[c for c in frame.columns if c != "fleet"]]]
        else:
            frame = pd.DataFrame(rows)
        write_csv(frame, out / name, item)
        if name == "s2_settings_grid.csv":
            write_csv(s2_rules(frame), out / "s2_rules.csv", item)
            (out / "s2_heatmaps.md").write_text(
                heatmaps(frame, cfg), encoding="utf-8", newline="\n"
            )
    (out / "README.md").write_text(readme(cfg, out), encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
