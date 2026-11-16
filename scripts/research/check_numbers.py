"""Recompute the point estimates of E1-E3 for every fleet and compare them (make paper-check).

maintadapt.research.recheck rebuilds them from the prediction archives, the fold files and the
pool with its own code; it imports nothing from analysis, stats, policies, costs or derived.
Every estimate, every configuration's leaderboard RMSE, jitter and S, and every selection must
match reports/research/confirmatory/ (FD001-FD004) and reports/research/ncmapss/ (N-CMAPSS)
within 1e-9, and every E1-E3 macro of paper_assets/numbers.tex must print the recomputed value.
Exits with status 1 on any difference.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from maintadapt.research.recheck import (
    FleetPoints,
    Setting,
    fleet_folds,
    pool_configs,
    recompute_fleet,
)

TOLERANCE = 1e-9
ENDPOINTS = {
    "e1": {"tau_b": "tau_b"},
    "e2": {
        "selection_loss": "selection_loss",
        "s_by_cost": "s_by_cost",
        "s_by_rmse": "s_by_rmse",
    },
    "e3": {"delta_tau": "delta_tau", "tau_jitter": "tau_jitter"},
}
PER_CONFIG = ["leaderboard_rmse", "jitter", "savings_share"]
MACRO = re.compile(r"\\defval\{([^}]*)\}\{([^}]*)\}")


def read_report(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def printed(value: str) -> float | None:
    """The number a macro prints: \\ensuremath{-0.123} gives -0.123."""
    found = re.search(r"-?\d+(\.\d+)?", value)
    return float(found.group()) if found else None


def compare(
    fleet: str, points: FleetPoints, reports: Path, macros: dict[str, str]
) -> list[str]:
    problems = []
    found = points.endpoints
    for name, columns in ENDPOINTS.items():
        row = read_report(reports / f"{name}.csv").set_index("fleet").loc[fleet]
        for column, key in columns.items():
            diff = abs(found[key] - float(row[column]))
            print(
                f"{fleet:8} {name}.{column:15} {found[key]: .12f} "
                f"report {float(row[column]): .12f} diff {diff:.1e}"
            )
            if not diff <= TOLERANCE:
                problems.append(f"{fleet} {name}.{column}: differs by {diff:.3g}")
            macro = f"{name}.{fleet}.{column}"
            if macro in macros:
                shown = macros[macro]
                decimals = len(shown.split(".")[1]) if "." in shown else 0
                if printed(shown) != round(found[key], decimals):
                    problems.append(f"numbers.tex {macro} prints {shown}")
            else:
                problems.append(f"numbers.tex has no {macro}")
    table = read_report(reports / "configurations.csv")
    table = table[table["fleet"] == fleet].set_index("config_id")
    if list(table.index) != points.configs:
        problems.append(f"{fleet}: the configurations differ from the report")
    else:
        for column in PER_CONFIG:
            diff = np.max(np.abs(getattr(points, column) - table[column].to_numpy()))
            print(f"{fleet:8} max |{column}| difference over configurations {diff:.1e}")
            if not diff <= TOLERANCE:
                problems.append(f"{fleet} {column}: differs by up to {diff:.3g}")
    chosen = read_report(reports / "selections.csv")
    if "fleet" in chosen:
        chosen = chosen[chosen["fleet"] == fleet]
    for _, row in chosen.iterrows():
        r, f = int(row["repeat_index"]), int(row["fold"])
        for column, choice in (
            ("by_validation_cost", points.by_cost),
            ("by_validation_rmse", points.by_rmse),
        ):
            if points.configs[choice[r, f]] != row[column]:
                problems.append(f"{fleet} r{r} f{f} {column}: {row[column]} differs")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--fleets", nargs="*", help="default: FD001-FD004 and NCMAPSS")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    primary = cfg["decision"]["primary"]
    if primary["rule"] != "R1" or primary["tuning"] != "validation":
        raise SystemExit("the recomputation covers rule R1 with validation tuning only")
    pool = pool_configs(cfg["paths"]["pool"])
    reports = Path(cfg["paths"]["reports"])
    numbers = Path(cfg["paths"]["paper_assets"]) / "numbers.tex"
    macros = dict(MACRO.findall(numbers.read_text(encoding="utf-8")))
    common = {
        "cost_ratio": float(primary["cost_ratio"]),
        "lead": int(primary["lead_time"]),
        "levels": int(cfg["decision"]["threshold_levels"]),
    }
    plan = {
        fleet: (
            cfg["paths"]["archive"],
            cfg["paths"]["folds"],
            Setting(**common, cap=int(cfg["data"]["cap"]), leaderboard="official_last"),
            reports / "confirmatory",
        )
        for fleet in cfg["data"]["fleets"]
    }
    plan["NCMAPSS"] = (
        cfg["ncmapss"]["archive"],
        cfg["ncmapss"]["folds"],
        Setting(**common, cap=int(cfg["ncmapss"]["cap"]), leaderboard="outer_all"),
        reports / "ncmapss",
    )
    problems = []
    for fleet in args.fleets or list(plan):
        root, folds, setting, source = plan[fleet]
        start = time.perf_counter()
        points = recompute_fleet(
            root, fleet, fleet_folds(folds, fleet), pool, setting, log=lambda _: None
        )
        print(
            f"{fleet}: {len(points.configs)} configurations recomputed "
            f"in {time.perf_counter() - start:.0f} s"
        )
        problems += compare(fleet, points, source, macros)
    if problems:
        print("\n".join(["DIFFERENCES:", *problems]))
        sys.exit(1)
    print("paper-check: every recomputed point estimate agrees within 1e-9")


if __name__ == "__main__":
    main()
