"""Confirmatory endpoints E1-E3 and the verdicts on H1-H3 (PREREGISTRATION.md section 6).

Reads the derived tables of an archive and writes e1.csv, e2.csv, e3.csv, configurations.csv,
selections.csv and verdicts.md. On the confirmatory archive it runs only when every fleet
validates, the tag analysis-frozen exists and the analysis code equals its tagged version; it
then writes reports/research/confirmatory/. --exploratory skips those checks and writes
reports/research/exploratory/ with EXPLORATORY at the top of every file. The smoke archive
writes reports/research/smoke/, labelled as smoke output.
"""

import argparse
from pathlib import Path

import pandas as pd

from maintadapt.research.analysis import (
    HYPOTHESES,
    RULES,
    Estimate,
    FleetResult,
    analyse_fleet,
    band,
    fleet_inputs,
    verdict_h1,
    verdict_h2,
    verdict_h3,
)
from maintadapt.research.archive import validate
from maintadapt.research.config import (
    ResearchConfig,
    fleet_number,
    load_research_config,
)
from maintadapt.research.crossfit import git_commit
from maintadapt.research.guards import (
    is_confirmatory,
    require_frozen_archive,
    resolve_root,
)
from maintadapt.research.models.spaces import FAMILY_NAMES

STATUS = {
    "confirmatory": "CONFIRMATORY: the pre-registered analysis of the confirmatory archive",
    "exploratory": "EXPLORATORY: not the pre-registered confirmatory analysis",
    "smoke": "SMOKE: pipeline check on the smoke archive (FD001, repeat 0, 3 training epochs "
    "for the networks); not confirmatory and says nothing about the hypotheses",
}


def read_fleet(source: Path, cfg: ResearchConfig) -> dict[str, pd.DataFrame]:
    p = cfg.decision.primary
    primary = [
        ("tuning", "==", p.tuning),
        ("rule", "==", p.rule),
        ("lead", "==", p.lead_time),
        ("cost_ratio", "==", p.cost_ratio),
    ]
    return {
        "outcomes": pd.read_parquet(source / "outcomes.parquet", filters=primary),
        "policies": pd.read_parquet(source / "policies.parquet", filters=primary),
        "age": pd.read_parquet(source / "age.parquet"),
        "engines": pd.read_parquet(source / "engines.parquet"),
        "components": pd.read_parquet(
            source / "components.parquet",
            filters=[("split", "in", ["official_test", "inner_val"])],
        ),
    }


def pool_order(config_ids: list[str]) -> list[str]:
    """Pool configurations in pool order; the controls are not part of any endpoint."""
    rank = {f: i for i, f in enumerate(FAMILY_NAMES)}
    kept = [c for c in config_ids if c.rsplit("-", 1)[0] in rank]
    return sorted(kept, key=lambda c: (rank[c.rsplit("-", 1)[0]], c))


def interval(e: Estimate) -> str:
    return f"[{e.lo:.3f}, {e.hi:.3f}]"


def write_csv(frame: pd.DataFrame, path: Path, status: str) -> None:
    text = f"# {status}\n" + frame.to_csv(index=False, lineterminator="\n")
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--root", help="archive (default), smoke, or a path")
    parser.add_argument("--exploratory", action="store_true")
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    root = resolve_root(args.root, cfg)
    if args.exploratory:
        mode = "exploratory"
    elif is_confirmatory(root, cfg):
        mode = "confirmatory"
        require_frozen_archive(root, cfg, cfg.data.fleets, validate)
    elif root.resolve() == Path(cfg.paths.smoke).resolve():
        mode = "smoke"
    else:
        raise SystemExit(
            f"{root} is neither the confirmatory nor the smoke archive; add --exploratory"
        )
    derived = Path(cfg.paths.derived) / root.name
    fleets = [f for f in cfg.data.fleets if (derived / f / "outcomes.parquet").exists()]
    if mode == "confirmatory" and fleets != cfg.data.fleets:
        raise SystemExit(
            f"derived tables exist only for {fleets}; run research-outcomes and research-metrics"
        )
    out = Path(cfg.paths.reports) / mode
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, FleetResult] = {}
    sizes, excluded, configs_of = {}, {}, {}
    for fleet in fleets:
        tables = read_fleet(derived / fleet, cfg)
        configs = pool_order(list(tables["outcomes"]["config_id"].unique()))
        inputs = fleet_inputs(tables, cfg.decision.primary, configs)
        results[fleet] = analyse_fleet(
            inputs,
            cfg.stats.n_boot,
            cfg.stats.ci,
            [cfg.stats.seed, fleet_number(fleet)],
        )
        sizes[fleet] = len(configs)
        configs_of[fleet] = configs
        listed = pd.read_csv(derived / fleet / "excluded.csv")
        excluded[fleet] = listed["config_id"].tolist()
        print(f"{fleet}: {len(configs)} configurations analysed", flush=True)

    status = STATUS[mode]
    e1 = {f: r.e1 for f, r in results.items()}
    e2 = {f: r.e2 for f, r in results.items()}
    e3 = {f: r.e3 for f, r in results.items()}
    write_csv(
        pd.DataFrame(
            [
                {
                    "fleet": f,
                    "configs": sizes[f],
                    "tau_b": e.point,
                    "lo": e.lo,
                    "hi": e.hi,
                    "valid_replicates": e.valid,
                    "band": band(e.point),
                }
                for f, e in e1.items()
            ]
        ),
        out / "e1.csv",
        status,
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "fleet": f,
                    "configs": sizes[f],
                    "selection_loss": r.e2.point,
                    "lo": r.e2.lo,
                    "hi": r.e2.hi,
                    "valid_replicates": r.e2.valid,
                    "s_by_cost": r.s_by_cost.point,
                    "s_by_cost_lo": r.s_by_cost.lo,
                    "s_by_cost_hi": r.s_by_cost.hi,
                    "s_by_rmse": r.s_by_rmse.point,
                    "s_by_rmse_lo": r.s_by_rmse.lo,
                    "s_by_rmse_hi": r.s_by_rmse.hi,
                }
                for f, r in results.items()
            ]
        ),
        out / "e2.csv",
        status,
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "fleet": f,
                    "configs": sizes[f],
                    "delta_tau": r.e3.point,
                    "lo": r.e3.lo,
                    "hi": r.e3.hi,
                    "valid_replicates": r.e3.valid,
                    "tau_jitter": r.tau_jitter.point,
                    "tau_jitter_lo": r.tau_jitter.lo,
                    "tau_jitter_hi": r.tau_jitter.hi,
                    "tau_rmse": r.e1.point,
                }
                for f, r in results.items()
            ]
        ),
        out / "e3.csv",
        status,
    )
    write_csv(
        pd.concat(
            [
                r.points.assign(fleet=f)[["fleet", *r.points.columns]]
                for f, r in results.items()
            ],
            ignore_index=True,
        ),
        out / "configurations.csv",
        status,
    )
    selections = []
    for f, r in results.items():
        for i in range(r.by_cost.shape[0]):
            for j in range(r.by_cost.shape[1]):
                selections.append(
                    {
                        "fleet": f,
                        "repeat_index": i,
                        "fold": j,
                        "by_validation_cost": configs_of[f][r.by_cost[i, j]],
                        "by_validation_rmse": configs_of[f][r.by_rmse[i, j]],
                    }
                )
    write_csv(pd.DataFrame(selections), out / "selections.csv", status)

    p = cfg.decision.primary
    verdicts = {"H1": verdict_h1(e1), "H2": verdict_h2(e2), "H3": verdict_h3(e3)}
    title = {
        "confirmatory": "# Confirmatory verdicts",
        "exploratory": "# EXPLORATORY: endpoints E1-E3 outside the confirmatory analysis",
        "smoke": "# Smoke run of the confirmatory analysis",
    }[mode]
    lines = [
        title,
        "",
        f"**{status}.**",
        "",
        (
            f"Primary setting: K = {p.cost_ratio:g}, L = {p.lead_time}, rule {p.rule}, "
            f"{p.tuning}-tuned thresholds. {cfg.stats.n_boot:,} engine bootstrap replicates "
            f"and percentile {cfg.stats.ci:.0%} intervals: training engines are resampled "
            "once per fleet and shared by every configuration and repeat, official test "
            "engines independently; models, thresholds and selections are held fixed "
            "(PREREGISTRATION.md section 6)."
        ),
        "",
        (
            f"Sources: `{derived.as_posix()}/<fleet>/` (make research-outcomes, make "
            "research-metrics). This file, e1.csv, e2.csv, e3.csv, configurations.csv and "
            "selections.csv were written by make research-confirmatory at commit "
            f"{git_commit()}."
        ),
        "",
    ]
    lines += [
        f"## H1: {verdicts['H1']}",
        "",
        f"H1: {HYPOTHESES['H1']}",
        "",
        f'Rule: "{RULES["H1"]}"',
        "",
        "| fleet | configurations | tau-b (E1) | 95% interval | band |",
        "|---|---|---|---|---|",
        *[
            f"| {f} | {sizes[f]} | {e.point:.3f} | {interval(e)} | {band(e.point)} |"
            for f, e in e1.items()
        ],
        "",
        f"## H2: {verdicts['H2']}",
        "",
        f"H2: {HYPOTHESES['H2']}",
        "",
        f'Rule: "{RULES["H2"]}"',
        "",
        (
            "| fleet | selection loss (E2) | 95% interval | S by validation cost | "
            "S by validation RMSE |"
        ),
        "|---|---|---|---|---|",
        *[
            f"| {f} | {r.e2.point:.3f} | {interval(r.e2)} | {r.s_by_cost.point:.3f} "
            f"{interval(r.s_by_cost)} | {r.s_by_rmse.point:.3f} {interval(r.s_by_rmse)} |"
            for f, r in results.items()
        ],
        "",
        f"## H3: {verdicts['H3']}",
        "",
        f"H3: {HYPOTHESES['H3']}",
        "",
        f'Rule: "{RULES["H3"]}"',
        "",
        "| fleet | delta-tau (E3) | 95% interval | tau-b(jitter, D) | tau-b(RMSE, D) |",
        "|---|---|---|---|---|",
        *[
            f"| {f} | {r.e3.point:.3f} | {interval(r.e3)} | {r.tau_jitter.point:.3f} "
            f"{interval(r.tau_jitter)} | {r.e1.point:.3f} |"
            for f, r in results.items()
        ],
        "",
        "## Excluded configurations",
        "",
        *[
            f"- {f}: {', '.join(excluded[f]) if excluded[f] else 'none'}"
            for f in fleets
        ],
        "",
    ]
    (out / "verdicts.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    for h, v in verdicts.items():
        print(f"{h}: {v}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
