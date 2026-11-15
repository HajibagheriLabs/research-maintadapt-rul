"""E1-E3 on N-CMAPSS (S8 of PREREGISTRATION.md section 7; section 8), secondary and descriptive.

Reads the derived tables of an N-CMAPSS archive (make research-ncmapss-derived) and writes
e1.csv, e2.csv, e3.csv, configurations.csv, selections.csv and results.md, each labelled
secondary and descriptive. The confirmatory archive's results go to reports/research/ncmapss/,
any other archive's to reports/research/ncmapss/<archive>/. Section 6's rules count fleets
and decide the verdicts on C-MAPSS; here each fleet-level condition is only read.
"""

import argparse
from pathlib import Path

import pandas as pd

from maintadapt.research.analysis import RULES, Estimate, band
from maintadapt.research.config import NCMAPSS, fleet_number, load_research_config
from maintadapt.research.crossfit import git_commit
from maintadapt.research.ncmapss_analysis import analyse_ncmapss, ncmapss_inputs
from maintadapt.research.secondary import pool_order


def status(archive: str, primary: bool) -> str:
    text = (
        "SECONDARY S8: N-CMAPSS replication of E1-E3 (PREREGISTRATION.md section 8), "
        "descriptive; not confirmatory"
    )
    return text if primary else f"{text}; pipeline check on {archive}"


def write_csv(frame: pd.DataFrame, path: Path, label: str) -> None:
    path.write_text(
        f"# {label}\n" + frame.to_csv(index=False, lineterminator="\n"),
        encoding="utf-8",
        newline="\n",
    )


def interval(e: Estimate) -> str:
    return f"{e.point:.3f} [{e.lo:.3f}, {e.hi:.3f}]"


def excludes_zero(e: Estimate) -> bool:
    return bool(e.lo > 0 or e.hi < 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--root", help="an N-CMAPSS archive (default: ncmapss.archive)")
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    root = Path(args.root or cfg.ncmapss.archive)
    primary_archive = root.resolve() == Path(cfg.ncmapss.archive).resolve()
    source = Path(cfg.paths.derived) / root.name / NCMAPSS
    if not (source / "outcomes.parquet").exists():
        raise SystemExit(f"no derived tables in {source}; run research-ncmapss-derived")
    out = Path(cfg.paths.reports) / "ncmapss"
    if not primary_archive:
        out = out / root.name
    out.mkdir(parents=True, exist_ok=True)
    label = status(root.as_posix(), primary_archive)

    p = cfg.decision.primary
    keep = [
        ("tuning", "==", p.tuning),
        ("rule", "==", p.rule),
        ("lead", "==", p.lead_time),
        ("cost_ratio", "==", p.cost_ratio),
    ]
    tables = {
        "outcomes": pd.read_parquet(source / "outcomes.parquet", filters=keep),
        "policies": pd.read_parquet(source / "policies.parquet", filters=keep),
        "age": pd.read_parquet(source / "age.parquet"),
        "engines": pd.read_parquet(source / "engines.parquet"),
        "components": pd.read_parquet(source / "components.parquet"),
    }
    configs = pool_order(list(tables["outcomes"]["config_id"].unique()))
    inputs = ncmapss_inputs(tables, p, configs)
    r = analyse_ncmapss(
        inputs, cfg.stats.n_boot, cfg.stats.ci, [cfg.stats.seed, fleet_number(NCMAPSS)]
    )
    excluded = pd.read_csv(source / "excluded.csv")["config_id"].tolist()
    units = len(tables["engines"])

    def row(name: str, e: Estimate, **extra) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"fleet": NCMAPSS, "configs": len(configs), name: e.point, "lo": e.lo}
                | {"hi": e.hi, "valid_replicates": e.valid}
                | extra
            ]
        )

    write_csv(row("tau_b", r.e1, band=band(r.e1.point)), out / "e1.csv", label)
    write_csv(
        row(
            "selection_loss",
            r.e2,
            s_by_cost=r.s_by_cost.point,
            s_by_cost_lo=r.s_by_cost.lo,
            s_by_cost_hi=r.s_by_cost.hi,
            s_by_rmse=r.s_by_rmse.point,
            s_by_rmse_lo=r.s_by_rmse.lo,
            s_by_rmse_hi=r.s_by_rmse.hi,
        ),
        out / "e2.csv",
        label,
    )
    write_csv(
        row(
            "delta_tau",
            r.e3,
            tau_jitter=r.tau_jitter.point,
            tau_jitter_lo=r.tau_jitter.lo,
            tau_jitter_hi=r.tau_jitter.hi,
            tau_rmse=r.e1.point,
        ),
        out / "e3.csv",
        label,
    )
    write_csv(r.points.assign(fleet=NCMAPSS), out / "configurations.csv", label)
    write_csv(
        pd.DataFrame(
            [
                {
                    "repeat_index": i,
                    "fold": j,
                    "by_validation_cost": configs[r.by_cost[i, j]],
                    "by_validation_rmse": configs[r.by_rmse[i, j]],
                }
                for i in range(r.by_cost.shape[0])
                for j in range(r.by_cost.shape[1])
            ]
        ),
        out / "selections.csv",
        label,
    )
    conditions = [
        (
            "E1 weak with interval upper bound below 0.6 (counts towards H1 supported)",
            band(r.e1.point) == "weak" and r.e1.hi < 0.6,
        ),
        (
            "E1 strong with interval lower bound above 0.5 (counts towards H1 rejected)",
            band(r.e1.point) == "strong" and r.e1.lo > 0.5,
        ),
        (
            "E2 at least 0.05 with an interval excluding 0 (counts towards H2 supported)",
            r.e2.point >= 0.05 and excludes_zero(r.e2),
        ),
        (
            "E2 interval within [-0.02, 0.02] (counts towards H2 rejected)",
            r.e2.lo >= -0.02 and r.e2.hi <= 0.02,
        ),
        (
            "E3 above 0 with an interval excluding 0 (counts towards H3 supported)",
            r.e3.point > 0 and excludes_zero(r.e3),
        ),
        ("E3 at most 0 (counts towards H3 rejected)", r.e3.point <= 0),
    ]
    lines = [
        "# N-CMAPSS: E1-E3 (secondary, descriptive)",
        "",
        f"**{label}.**",
        "",
        (
            f"{units} units pooled from the readable sub-datasets "
            "(`reports/research/ncmapss/inspection.md`), cap "
            f"{cfg.ncmapss.require_cap()}, {len(configs)} configurations. Primary setting: "
            f"K = {p.cost_ratio:g}, L = {p.lead_time} flights, rule {p.rule}, "
            f"{p.tuning}-tuned thresholds. Leaderboard RMSE is the RMSE over every flight of "
            "a fit's outer_test units against the capped truth, averaged over fits; jitter "
            f"uses the same units. {cfg.stats.n_boot:,} bootstrap replicates draw the units "
            "once for D and the accuracy metrics alike; percentile "
            f"{cfg.stats.ci:.0%} intervals (DECISIONS.md, N-CMAPSS)."
        ),
        "",
        (
            f"Sources: `{source.as_posix()}/` (make research-ncmapss-derived); written by "
            f"make research-ncmapss-report at commit {git_commit()}."
        ),
        "",
        "| endpoint | estimate [95% interval] |",
        "|---|---|",
        f"| E1, tau-b(leaderboard RMSE, D) | {interval(r.e1)} ({band(r.e1.point)}) |",
        f"| E2, selection loss | {interval(r.e2)} |",
        f"| S chosen by validation cost | {interval(r.s_by_cost)} |",
        f"| S chosen by validation RMSE | {interval(r.s_by_rmse)} |",
        f"| E3, delta-tau | {interval(r.e3)} |",
        f"| tau-b(jitter, D) | {interval(r.tau_jitter)} |",
        "",
        "## Section 6's fleet-level conditions, read for this fleet",
        "",
        (
            "The verdicts on H1-H3 count C-MAPSS fleets and are in "
            "`reports/research/confirmatory/verdicts.md`; N-CMAPSS does not enter them. Rules "
            "quoted from PREREGISTRATION.md section 6:"
        ),
        "",
        *[f'- "{RULES[h]}"' for h in ("H1", "H2", "H3")],
        "",
        "| condition | holds here |",
        "|---|---|",
        *[f"| {c} | {'yes' if ok else 'no'} |" for c, ok in conditions],
        "",
        f"Excluded configurations: {', '.join(excluded) if excluded else 'none'}.",
        "",
    ]
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
