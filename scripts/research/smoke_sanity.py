"""Check the derived tables of an archive against what the controls and the design force.

The controls must behave as their definitions force: the oracle (prediction = capped truth)
reaches S = 1 under oracle tuning at every lead time, the constant (prediction = cap) has S <= 0;
every table has the expected rows and no missing values, and every engine appears once per repeat
in outer_test. Checks every fleet with derived tables. The smoke archive writes
reports/research/smoke/sanity.md, the confirmatory archive reports/research/archive_sanity.md.
Exits with an error if any check fails.
"""

import argparse
from pathlib import Path

import pandas as pd

from maintadapt.research.config import ResearchConfig, load_research_config
from maintadapt.research.derived import RULES, TUNINGS, savings
from maintadapt.research.guards import is_confirmatory, resolve_root

OUTCOME_KEYS = ["config_id", "repeat", "tuning", "rule", "lead", "cost_ratio"]


def describe(values: pd.Series) -> str:
    return f"{values.min():.6g} to {values.max():.6g}"


def check_fleet(cfg: ResearchConfig, source: Path) -> list[tuple[str, str, str, str]]:
    read = {
        name: pd.read_parquet(source / f"{name}.parquet")
        for name in (
            "policies",
            "outcomes",
            "age",
            "engines",
            "components",
            "fit_metrics",
        )
    }
    lives = read["engines"].set_index("unit_id")["life"]

    d = cfg.decision
    n_configs = read["outcomes"]["config_id"].nunique()
    n_repeats = read["outcomes"]["repeat"].nunique()
    n_engines = len(lives)
    n_policies = len(TUNINGS) * len(RULES) * len(d.lead_times) * len(d.cost_ratios)
    n_official = (
        read["components"].query("split == 'official_test'")["unit_id"].nunique()
    )
    fits = n_configs * n_repeats * cfg.folds.n_outer
    per_fold_val = (
        read["components"]
        .query("split == 'inner_val'")
        .groupby(["config_id", "fold"])
        .size()
    )
    expected_rows = {
        "policies": n_configs * n_repeats * cfg.folds.n_outer * n_policies,
        "outcomes": n_configs * n_repeats * n_engines * n_policies,
        "age": n_repeats * len(d.cost_ratios) * n_engines,
        "components": fits * n_official
        + n_configs * n_repeats * n_engines
        + int(per_fold_val.sum()),
        "fit_metrics": fits,
    }
    checks = []

    def check(name: str, expected: str, observed: str, ok: bool) -> None:
        checks.append((name, expected, observed, "pass" if ok else "FAIL"))

    # the controls run only in the smoke test (DECISIONS.md, Cross-fitting runs)
    controls = read["outcomes"][
        read["outcomes"]["config_id"].str.startswith("control_")
    ]
    if len(controls):
        share = savings(controls, read["age"], lives)
        oracle = share[share["config_id"] == "control_oracle-00"]
        for tuning in TUNINGS:
            s = oracle.loc[oracle["tuning"] == tuning, "savings_share"]
            if tuning == "oracle":
                check(
                    "oracle control, oracle tuning: S at every rule, lead time and K",
                    "1 exactly",
                    describe(s),
                    bool(len(s) and (s == 1.0).all()),
                )
            else:
                check(
                    f"oracle control, {tuning} tuning: S at every rule, lead time and K",
                    "close to 1",
                    describe(s),
                    bool(len(s) and s.min() > 0.95),
                )
        constant = share.loc[
            share["config_id"] == "control_constant-00", "savings_share"
        ]
        check(
            "constant control: S at every tuning, rule, lead time and K",
            "<= 0",
            describe(constant),
            bool(len(constant) and (constant <= 0).all()),
        )
    for name, rows in expected_rows.items():
        check(
            f"rows in {name}", str(rows), str(len(read[name])), len(read[name]) == rows
        )
    for name in ("policies", "outcomes", "age", "fit_metrics"):
        missing = int(read[name].isna().sum().sum())
        check(f"missing values in {name}", "0", str(missing), missing == 0)
    applicable = {
        "official_test": ["sq_last_125", "nasa_last", "sse_all", "jit_n", "n_late"],
        "outer_test": ["sse_trunc", "n_trunc", "jit_n", "n_late"],
        "inner_val": ["sse_all", "n_all"],
    }
    gaps = sum(
        int(
            read["components"]
            .loc[read["components"]["split"] == split, cols]
            .isna()
            .sum()
            .sum()
        )
        for split, cols in applicable.items()
    )
    check(
        "missing values in the components that apply to each split",
        "0",
        str(gaps),
        gaps == 0,
    )
    coverage = (
        read["outcomes"].groupby(OUTCOME_KEYS)["unit_id"].agg(["size", "nunique"])
    )
    once = bool(
        ((coverage["size"] == n_engines) & (coverage["nunique"] == n_engines)).all()
    )
    check(
        "every engine once per repeat in outer_test, for every policy",
        f"{n_engines} engines, each once",
        f"{len(coverage)} policies checked",
        once,
    )
    trunc = read["components"].query("split == 'outer_test'")["n_trunc"]
    check(
        "truncation points per outer_test engine",
        "20",
        describe(trunc),
        bool((trunc == 20).all()),
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument(
        "--root", default="smoke", help="smoke (default), archive or a path"
    )
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    root = resolve_root(args.root, cfg)
    derived = Path(cfg.paths.derived) / root.name
    fleets = [f for f in cfg.data.fleets if (derived / f / "outcomes.parquet").exists()]
    if not fleets:
        raise SystemExit(
            f"no derived tables under {derived}; run research-outcomes first"
        )
    checks = []
    for fleet in fleets:
        checks += [
            (f"{fleet}: {name}", *rest)
            for name, *rest in check_fleet(cfg, derived / fleet)
        ]
    flag = f" ROOT={args.root}"
    # research-sanity defaults to the smoke archive; the builders default to the confirmatory one
    own = "" if root.name == "smoke" else flag
    if root.name == "smoke":
        title = "# Smoke test: sanity of the derived tables"
        scope = [
            "NOT CONFIRMATORY. These checks run on the smoke archive (FD001, repeat 0, all five",
            "folds, configuration 0 of every family with 3 training epochs for the networks, and",
            "two controls). They test the pipeline and say nothing about the study's questions.",
        ]
        out = Path(cfg.paths.reports) / "smoke" / "sanity.md"
    else:
        title = f"# Sanity of the derived tables of `{root.as_posix()}`"
        scope = [
            "Checks of the pipeline, not results; nothing here bears on the hypotheses. The two",
            "controls run only in the smoke test (`reports/research/smoke/sanity.md`), so the",
            "checks here are structural.",
        ]
        name = "archive" if is_confirmatory(root, cfg) else root.name
        out = Path(cfg.paths.reports) / f"{name}_sanity.md"
    lines = [
        title,
        "",
        *scope,
        "",
        f"Sources: `{derived.as_posix()}/<fleet>/` (written by",
        f"`make research-outcomes{flag}` and `make research-metrics{flag}`); this file is written",
        f"by `make research-sanity{own}`.",
        "S pools the outcomes of the five folds; CR_age uses A* tuned on each fold's outer-train",
        "lifetimes.",
        "",
        "| check | expected | observed | result |",
        "|---|---|---|---|",
        *[f"| {a} | {b} | {c} | {r} |" for a, b, c, r in checks],
        "",
    ]
    failures = [c for c in checks if c[3] != "pass"]
    lines.append(
        f"{len(checks) - len(failures)} of {len(checks)} checks pass."
        + ("" if not failures else " Failing checks are marked FAIL.")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(f"{len(failures)} sanity checks failed; see {out}")


if __name__ == "__main__":
    main()
