"""EXPLORATORY analyses outside the pre-registered plan; every file says so in its first line.

X1: late-phase RMSE after isotonic recalibration on inner_val, as a leaderboard-compatible proxy.
X2: EMA smoothing at fixed alphas: its effect on maintenance value against its effect on RMSE.
X3: the engineering project's booking evaluation on truncated official test engines
    (decision.replay_alarms) against the trajectory evaluation of this study.
Per-fit quantities are built from the confirmatory archive once and kept in
runs/research/derived/archive/<fleet>/exploratory_*.parquet; the analysis writes
reports/research/exploratory/x*.csv and a README. Intervals reuse the confirmatory bootstrap
draws and are not corrected for multiplicity.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from maintadapt.config import load_config
from maintadapt.decision import CostModel
from maintadapt.research.analysis import (
    FleetInputs,
    decision_loss,
    estimate,
    fleet_inputs,
    leaderboard_rmse,
)
from maintadapt.research.archive import load, records
from maintadapt.research.config import (
    ResearchConfig,
    fleet_number,
    load_research_config,
)
from maintadapt.research.crossfit import git_commit
from maintadapt.research.exploratory import fit_quantities
from maintadapt.research.guards import tag_exists
from maintadapt.research.secondary import (
    Draws,
    grid_values,
    official_values,
    pool_order,
    pooled_rmse,
    regret_outcomes,
    tau,
)
from maintadapt.research.stats import kendall_tau_b

COLUMNS = ["unit_id", "cycle", "rul", "split", "pred", "repeat", "fold"]
TABLES = ("official", "outer", "policies", "regret")
FILES = {
    "x1_isotonic_late_rmse.csv": (
        "X1",
        (
            "tau-b(proxy, D) for late-phase RMSE (true RUL <= 50) on the official test "
            "engines, raw and after isotonic recalibration on each fit's inner_val engines, "
            "beside E1"
        ),
    ),
    "x2_ema_configurations.csv": (
        "X2",
        (
            "per configuration and fixed alpha: leaderboard RMSE of the smoothed predictions "
            "and S of rule R1 on them, with the changes from alpha = 1"
        ),
    ),
    "x2_ema_summary.csv": (
        "X2",
        (
            "per fleet and alpha: median changes in leaderboard RMSE and in S, the shares of "
            "configurations whose RMSE rises, whose S rises and both, and tau-b between the "
            "changes"
        ),
    ),
    "x3_legacy_booking.csv": (
        "X3",
        (
            "per fleet: tau-b among the legacy booking cost on truncated official test "
            "engines, the trajectory regret at the same failure cost, leaderboard RMSE and D"
        ),
    ),
    "x3_legacy_configurations.csv": (
        "X3",
        (
            "per configuration: legacy booking cost per engine, the trajectory counterpart "
            "(mean regret per engine), leaderboard RMSE and S"
        ),
    ),
}


def status(item: str) -> str:
    return (
        f"EXPLORATORY {item}: not part of the pre-registered plan; intervals uncorrected for "
        "multiplicity"
    )


def write_csv(frame: pd.DataFrame, path: Path, item: str) -> None:
    text = f"# {status(item)}\n" + frame.to_csv(index=False, lineterminator="\n")
    path.write_text(text, encoding="utf-8", newline="\n")


def columns(name: str, e) -> dict[str, float]:
    return {name: e.point, f"{name}_lo": e.lo, f"{name}_hi": e.hi}


class Legacy:
    """The engineering project's decision settings (configs/default.yaml)."""

    def __init__(self, path: str) -> None:
        legacy = load_config(path)
        d = legacy.decision
        self.costs = CostModel(failure=d.failure_cost, per_cycle=d.cycle_cost)
        cap = legacy.data.rul_cap
        self.thresholds = np.arange(0.0, cap + d.threshold_step, d.threshold_step)
        self.cap = cap


def build(cfg: ResearchConfig, legacy: Legacy, fleet: str, out: Path) -> None:
    start = time.perf_counter()
    root = Path(cfg.paths.archive)
    found = records(root, [fleet])
    configs = pool_order(list(found["config_id"].unique()))
    d = cfg.decision
    alphas = sorted(set(d.ema_alphas), reverse=True)
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLES}
    for config_id in configs:
        table = load(
            root, fleets=[fleet], config_ids=[config_id], columns=COLUMNS, found=found
        )
        for repeat, in_repeat in table.groupby("repeat", sort=True):
            fits = {int(k): fit for k, fit in in_repeat.groupby("fold", sort=True)}
            keys = {"config_id": config_id, "repeat": int(repeat)}
            for fold, fit in fits.items():
                got = fit_quantities(
                    fit,
                    alphas,
                    d.primary.lead_time,
                    d.primary.cost_ratio,
                    d.threshold_levels,
                    cfg.data.cap,
                    legacy.costs,
                    legacy.thresholds,
                )
                for name in ("official", "outer", "policies"):
                    parts[name].append(got[name].assign(fold=fold, **keys))
            # the trajectory counterpart of the legacy protocol: same failure cost, no lead
            _, regret = regret_outcomes(
                fits, 0, [legacy.costs.failure], d.threshold_levels
            )
            parts["regret"].append(regret.assign(**keys))
        print(
            f"{fleet} {config_id:<18} {time.perf_counter() - start:7.1f}s", flush=True
        )
    for name in TABLES:
        pd.concat(parts[name], ignore_index=True).to_parquet(
            out / f"exploratory_{name}.parquet"
        )
    summary = {
        "fleet": fleet,
        "configs": len(configs),
        "alphas": alphas,
        "legacy_failure_cost": legacy.costs.failure,
        "seconds": round(time.perf_counter() - start, 1),
    }
    (out / "build_exploratory.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n"
    )


class Fleet:
    def __init__(self, cfg: ResearchConfig, source: Path, fleet: str) -> None:
        self.cfg, self.fleet = cfg, fleet
        p = cfg.decision.primary
        keep = [
            ("tuning", "==", p.tuning),
            ("rule", "==", p.rule),
            ("lead", "==", p.lead_time),
            ("cost_ratio", "==", p.cost_ratio),
        ]
        self.tables = {
            "outcomes": pd.read_parquet(source / "outcomes.parquet", filters=keep),
            "policies": pd.read_parquet(source / "policies.parquet", filters=keep),
            "age": pd.read_parquet(source / "age.parquet"),
            "engines": pd.read_parquet(source / "engines.parquet"),
            "components": pd.read_parquet(source / "components.parquet"),
        }
        self.extra = {
            name: pd.read_parquet(source / f"exploratory_{name}.parquet")
            for name in TABLES
        }
        self.configs = pool_order(list(self.tables["outcomes"]["config_id"].unique()))
        self.base = fleet_inputs(self.tables, p, self.configs)
        self.draws = Draws.of(
            self.base, cfg.stats.n_boot, [cfg.stats.seed, fleet_number(fleet)]
        )
        d = self.draws
        self.loss = decision_loss(self.base, d.train)
        self.loss_point = decision_loss(self.base, d.once_train)[0]
        self.rmse = leaderboard_rmse(self.base, d.test)
        self.rmse_point = leaderboard_rmse(self.base, d.once_test)[0]

    def official(self, table: pd.DataFrame, column: str) -> np.ndarray:
        """configs x fits x official test engines; the exploratory tables hold no split."""
        if "split" not in table:
            table = table.assign(split="official_test")
        return official_values(self.base, table, column)


def x1(f: Fleet) -> list[dict]:
    ci, d = f.cfg.stats.ci, f.draws
    proxies = {
        "late_rmse": (
            f.official(f.tables["components"], "sse_late"),
            f.official(f.tables["components"], "n_late"),
        ),
        "late_rmse_isotonic": (
            f.official(f.extra["official"], "sse_late_iso"),
            f.official(f.extra["official"], "n_late"),
        ),
    }
    e1, reps_e1 = tau(f.rmse_point, f.rmse, f.loss_point, f.loss, ci)
    rows = [
        {"fleet": f.fleet, "proxy": "leaderboard_rmse"}
        | columns("tau_b", e1)
        | {"difference_from_e1": 0.0, "difference_lo": 0.0, "difference_hi": 0.0}
    ]
    for name, (sse, n) in proxies.items():
        point = pooled_rmse(sse, n, d.once_test)[0]
        reps = pooled_rmse(sse, n, d.test)
        e, reps_tau = tau(point, reps, f.loss_point, f.loss, ci)
        diff = estimate(e.point - e1.point, reps_tau - reps_e1, ci)
        rows.append(
            {"fleet": f.fleet, "proxy": name}
            | columns("tau_b", e)
            | {
                "difference_from_e1": diff.point,
                "difference_lo": diff.lo,
                "difference_hi": diff.hi,
            }
        )
    return rows


def at_alpha(f: Fleet, alpha: float) -> FleetInputs:
    """FleetInputs whose outcomes are rule R1 on predictions smoothed with a fixed alpha."""
    p = f.cfg.decision.primary
    setting = {
        "tuning": p.tuning,
        "rule": p.rule,
        "lead": p.lead_time,
        "cost_ratio": p.cost_ratio,
    }
    outer, pol = f.extra["outer"], f.extra["policies"]
    tables = {
        **f.tables,
        "outcomes": outer[outer["alpha"] == alpha].assign(**setting),
        "policies": pol[pol["alpha"] == alpha].assign(**setting),
    }
    inputs = fleet_inputs(tables, p, f.configs)
    sq = f.official(f.extra["official"], f"sq_last_ema_{alpha:g}")
    return replace(inputs, sq_last=sq)


def x2(f: Fleet) -> tuple[list[dict], list[dict]]:
    once_train, once_test = f.draws.once_train, f.draws.once_test
    per_alpha = {}
    for alpha in sorted(f.extra["outer"]["alpha"].unique(), reverse=True):
        inputs = at_alpha(f, float(alpha))
        per_alpha[float(alpha)] = (
            leaderboard_rmse(inputs, once_test)[0],
            1.0 - decision_loss(inputs, once_train)[0],
        )
    rmse_1, s_1 = per_alpha[1.0]
    # alpha = 1 is rule R1 itself: the same thresholds, outcomes and errors as the tables
    if not (
        np.allclose(rmse_1, f.rmse_point, rtol=0, atol=1e-9)
        and np.allclose(s_1, 1.0 - f.loss_point, rtol=0, atol=1e-12)
    ):
        raise SystemExit(f"{f.fleet}: alpha = 1 does not reproduce the derived tables")
    rows, summary = [], []
    for alpha, (rmse, share) in per_alpha.items():
        d_rmse, d_s = rmse - rmse_1, share - s_1
        for i, config_id in enumerate(f.configs):
            rows.append(
                {
                    "fleet": f.fleet,
                    "config_id": config_id,
                    "alpha": alpha,
                    "leaderboard_rmse": rmse[i],
                    "savings_share": share[i],
                    "rmse_change": d_rmse[i],
                    "savings_share_change": d_s[i],
                }
            )
        if alpha == 1.0:
            continue
        summary.append(
            {
                "fleet": f.fleet,
                "alpha": alpha,
                "configs": len(f.configs),
                "median_rmse_change": float(np.median(d_rmse)),
                "median_savings_share_change": float(np.median(d_s)),
                "share_rmse_rises": float(np.mean(d_rmse > 0)),
                "share_savings_share_rises": float(np.mean(d_s > 0)),
                "share_both_rise": float(np.mean((d_rmse > 0) & (d_s > 0))),
                "tau_b_changes": kendall_tau_b(d_rmse, d_s),
            }
        )
    return rows, summary


def x3(f: Fleet) -> tuple[dict, list[dict]]:
    ci, d = f.cfg.stats.ci, f.draws
    legacy_cube = f.official(f.extra["official"], "legacy_cost")
    ones = np.ones_like(legacy_cube)

    def legacy(counts: np.ndarray) -> np.ndarray:
        # mean cost per official test engine of each fit, averaged over the fits
        n = legacy_cube.shape[-1]
        total = counts @ legacy_cube.reshape(-1, n).T
        engines = counts @ ones.reshape(-1, n).T
        return (total / engines).reshape(len(counts), *legacy_cube.shape[:2]).mean(-1)

    found = f.extra["regret"].assign(
        regret=lambda t: t["failure_cost"] * t["failed"] + t["waste"]
    )
    units = np.sort(found["unit_id"].unique())
    regret_cube = grid_values(
        found,
        ["config_id", "repeat", "unit_id"],
        [f.configs, f.base.repeats, units],
        "regret",
    )

    def trajectory(counts: np.ndarray) -> np.ndarray:
        # mean regret per engine pooled over the folds of a repeat, averaged over repeats
        n = regret_cube.shape[-1]
        mean = (counts @ regret_cube.reshape(-1, n).T) / counts.sum(axis=1)[:, None]
        return mean.reshape(len(counts), *regret_cube.shape[:2]).mean(-1)

    metrics = {
        "legacy_booking_cost": (legacy(d.once_test)[0], legacy(d.test)),
        "trajectory_regret": (trajectory(d.once_train)[0], trajectory(d.train)),
        "leaderboard_rmse": (f.rmse_point, f.rmse),
        "decision_loss": (f.loss_point, f.loss),
    }
    row = {"fleet": f.fleet, "configs": len(f.configs)}
    for a, b in (
        ("legacy_booking_cost", "trajectory_regret"),
        ("legacy_booking_cost", "leaderboard_rmse"),
        ("trajectory_regret", "leaderboard_rmse"),
        ("legacy_booking_cost", "decision_loss"),
    ):
        e, _ = tau(*metrics[a], *metrics[b], ci)
        row |= columns(f"tau_{a}_vs_{b}", e)
    per_config = [
        {
            "fleet": f.fleet,
            "config_id": c,
            "legacy_booking_cost": metrics["legacy_booking_cost"][0][i],
            "trajectory_regret": metrics["trajectory_regret"][0][i],
            "leaderboard_rmse": f.rmse_point[i],
            "savings_share": 1.0 - f.loss_point[i],
        }
        for i, c in enumerate(f.configs)
    ]
    return row, per_config


def readme(cfg: ResearchConfig, legacy: Legacy, out: Path) -> str:
    alphas = ", ".join(
        f"{a:g}" for a in sorted(set(cfg.decision.ema_alphas), reverse=True)
    )
    p = cfg.decision.primary
    lines = [
        "# Exploratory analyses",
        "",
        (
            "**EXPLORATORY.** None of these analyses is in PREREGISTRATION.md; they are reported "
            "as exploration, with percentile 95% intervals from the confirmatory bootstrap draws "
            f"and no correction for multiplicity. Written by `make research-exploratory` at commit "
            f"{git_commit()} from the confirmatory archive, after the tag `analysis-frozen`."
        ),
        "",
        "| file | analysis | content |",
        "|---|---|---|",
        *[
            f"| `{name}` | {item} | {text}. |"
            for name, (item, text) in FILES.items()
            if (out / name).exists()
        ],
        "",
        "## Definitions",
        "",
        (
            "- X1: for each fit, an isotonic regression of the truth capped at "
            f"{cfg.data.cap} on the inner_val predictions maps the official test predictions; "
            "late-phase RMSE pools the cycles with true RUL <= 50 over the official test engines "
            "of a fit and is averaged over fits, like leaderboard RMSE. The recalibration never "
            "reverses the order of two predictions, but it may tie predictions that differed; "
            "it changes the proxy only, never a policy."
        ),
        (
            f"- X2: predictions smoothed by the causal EMA at each fixed alpha in {{{alphas}}}; "
            f"rule R1 thresholds tuned on the smoothed inner_val predictions at K = "
            f"{p.cost_ratio:g} and L = {p.lead_time}; S pooled and averaged as for D. The "
            "leaderboard RMSE uses the smoothed prediction at the last observed cycle. alpha = 1 "
            "reproduces the derived tables, which the script checks."
        ),
        (
            "- X3: the engineering project's protocol (`scripts/run_decision.py`, "
            "`configs/default.yaml`): thresholds 0 to "
            f"{legacy.cap} in steps of {legacy.thresholds[1] - legacy.thresholds[0]:g} tuned on "
            "every inner_val history cut at every cycle (`decision.replay_snapshots`), applied to "
            "the truncated official test engines, which keep their last booking "
            "(`decision.replay_alarms`); cost per engine = "
            f"{legacy.costs.failure:g} x failure + cycles thrown away, no lead time. Its "
            "trajectory counterpart tunes the threshold on complete inner_val trajectories for the "
            "lowest mean regret at the same failure cost and no lead time, and scores the complete "
            "outer_test trajectories."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--legacy-config", default="configs/default.yaml")
    parser.add_argument("--fleets", nargs="+")
    parser.add_argument(
        "--rebuild", action="store_true", help="recompute the per-fit tables"
    )
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    if not tag_exists():
        raise SystemExit(
            "exploratory analyses run only after the tag analysis-frozen exists"
        )
    legacy = Legacy(args.legacy_config)
    derived = Path(cfg.paths.derived) / Path(cfg.paths.archive).name
    out = Path(cfg.paths.reports) / "exploratory"
    out.mkdir(parents=True, exist_ok=True)
    collected: dict[str, list] = {name: [] for name in FILES}
    for fleet in args.fleets or cfg.data.fleets:
        source = derived / fleet
        if not (source / "outcomes.parquet").exists():
            raise SystemExit(
                f"no derived tables for {fleet}; run research-outcomes first"
            )
        built = all((source / f"exploratory_{n}.parquet").exists() for n in TABLES)
        if args.rebuild or not built:
            build(cfg, legacy, fleet, source)
        f = Fleet(cfg, source, fleet)
        collected["x1_isotonic_late_rmse.csv"] += x1(f)
        rows, summary = x2(f)
        collected["x2_ema_configurations.csv"] += rows
        collected["x2_ema_summary.csv"] += summary
        row, per_config = x3(f)
        collected["x3_legacy_booking.csv"].append(row)
        collected["x3_legacy_configurations.csv"] += per_config
        print(f"{fleet}: done", flush=True)
    for name, rows in collected.items():
        write_csv(pd.DataFrame(rows), out / name, FILES[name][0])
    (out / "README.md").write_text(
        readme(cfg, legacy, out), encoding="utf-8", newline="\n"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
