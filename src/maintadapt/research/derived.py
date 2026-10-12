"""Derived tables: tuned policies, per-engine outcomes, age replacement and metric components.

Built from the predictions of complete fits; the archive is read once, and everything the
analysis needs under the engine bootstrap is stored per engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from maintadapt.research.config import DecisionGridConfig
from maintadapt.research.costs import (
    age_outcomes,
    perfect_rate,
    savings_share,
    tune_age,
)
from maintadapt.research.folds import Fold
from maintadapt.research.policies import (
    Trial,
    TunedPolicy,
    alarm_times,
    ema,
    outcomes,
    pick,
    trial,
)

TUNINGS = ("validation", "crossfit", "oracle")
RULES = ("R1", "R2")
POLICY_KEYS = ["repeat", "fold", "tuning", "rule", "lead", "cost_ratio"]
# official test metrics follow the leaderboard convention: truth capped at 125
CAPS = {"125": 125, "130": 130}
LATE_RUL = 50
MIN_JITTER_CYCLES = 5
TRUNCATION_POINTS = 20
TRUNCATION_START = 31


@dataclass(frozen=True, eq=False)
class EngineSet:
    """Complete trajectories of some engines: predictions at cycles 1..T, in unit order."""

    units: np.ndarray
    predictions: list[np.ndarray]
    lives: np.ndarray

    @staticmethod
    def concat(sets: Sequence[EngineSet]) -> EngineSet:
        return EngineSet(
            np.concatenate([s.units for s in sets]),
            [p for s in sets for p in s.predictions],
            np.concatenate([s.lives for s in sets]),
        )


def engines_of(table: pd.DataFrame, split: str) -> EngineSet:
    """The engines of one split of one fit, from its archive rows."""
    rows = table[table["split"].astype(str) == split].sort_values(["unit_id", "cycle"])
    units, starts, lives = np.unique(
        rows["unit_id"].to_numpy(), return_index=True, return_counts=True
    )
    if not (rows.groupby("unit_id")["cycle"].max().to_numpy() == lives).all():
        raise ValueError(f"{split}: every engine must hold cycles 1..T")
    predictions = np.split(rows["pred"].to_numpy(float), starts[1:])
    return EngineSet(units, predictions, lives.astype(np.int64))


def _alphas(decision: DecisionGridConfig) -> list[float]:
    # the lightest smoothing first, so ties between alphas go to it
    return sorted(set(decision.ema_alphas) | {1.0}, reverse=True)


def _scored(
    engines: EngineSet, alphas: Sequence[float]
) -> dict[float, list[np.ndarray]]:
    return {a: [ema(p, a) for p in engines.predictions] for a in alphas}


def _tune(
    scored: Mapping[float, Sequence[np.ndarray]],
    lives: np.ndarray,
    decision: DecisionGridConfig,
) -> dict[tuple[str, int, float], TunedPolicy]:
    """Policies for R1 (alpha 1) and R2 (every alpha) at every lead time and cost ratio."""
    alphas = list(scored)
    trials: dict[float, Trial] = {
        a: trial(scored[a], a, decision.threshold_levels) for a in alphas
    }
    tuned = {}
    for lead in decision.lead_times:
        r1 = pick([trials[1.0]], lives, lead, decision.cost_ratios)
        r2 = pick([trials[a] for a in alphas], lives, lead, decision.cost_ratios)
        for k in decision.cost_ratios:
            tuned[("R1", lead, k)] = r1[k]
            tuned[("R2", lead, k)] = r2[k]
    return tuned


def _apply(
    tuned: Mapping[tuple[str, int, float], TunedPolicy],
    scored: Mapping[float, Sequence[np.ndarray]],
    engines: EngineSet,
) -> tuple[list[tuple], np.ndarray, np.ndarray, np.ndarray]:
    """Outcomes of every tuned policy on engines; one row per policy, one column per engine."""
    keys = list(tuned)
    failed = np.empty((len(keys), len(engines.units)), dtype=bool)
    used = np.empty((len(keys), len(engines.units)), dtype=np.int64)
    waste = np.empty_like(used)
    groups: dict[tuple[float, int], list[int]] = {}
    for i, (_, lead, _) in enumerate(keys):
        groups.setdefault((tuned[keys[i]].alpha, lead), []).append(i)
    for (alpha, lead), rows in groups.items():
        thetas = np.array([tuned[keys[i]].theta for i in rows])
        found = outcomes(alarm_times(scored[alpha], thetas), engines.lives, lead)
        failed[rows], used[rows], waste[rows] = found.failed, found.used, found.waste
    return keys, failed, used, waste


def repeat_outcomes(
    fits: Mapping[int, pd.DataFrame], decision: DecisionGridConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Policies and outer_test outcomes of one configuration in one repeat, every tuning.

    validation tunes on the fold's inner_val engines; crossfit on the outer_test engines of
    the repeat's other folds; oracle on the outer_test engines of every fold of the repeat.
    """
    folds = sorted(fits)
    alphas = _alphas(decision)
    val = {f: engines_of(fits[f], "inner_val") for f in folds}
    test = {f: engines_of(fits[f], "outer_test") for f in folds}
    val_scored = {f: _scored(val[f], alphas) for f in folds}
    test_scored = {f: _scored(test[f], alphas) for f in folds}

    def pooled(
        chosen: Sequence[int],
    ) -> tuple[dict[float, list[np.ndarray]], np.ndarray]:
        scored = {a: [s for f in chosen for s in test_scored[f][a]] for a in alphas}
        return scored, np.concatenate([test[f].lives for f in chosen])

    oracle = _tune(*pooled(folds), decision)
    policies, rows = [], []
    for f in folds:
        others = [g for g in folds if g != f]
        by_tuning = {
            "validation": _tune(val_scored[f], val[f].lives, decision),
            "crossfit": _tune(*pooled(others), decision),
            "oracle": oracle,
        }
        for tuning, tuned in by_tuning.items():
            keys, failed, used, waste = _apply(tuned, test_scored[f], test[f])
            for rule, lead, k in keys:
                p = tuned[(rule, lead, k)]
                policies.append(
                    (f, tuning, rule, lead, k, p.theta, p.alpha, p.cost_rate)
                )
            n = len(test[f].units)
            rows.append(
                pd.DataFrame(
                    {
                        "fold": f,
                        "tuning": tuning,
                        "rule": np.repeat([k[0] for k in keys], n),
                        "lead": np.repeat([k[1] for k in keys], n),
                        "cost_ratio": np.repeat([k[2] for k in keys], n),
                        "unit_id": np.tile(test[f].units, len(keys)),
                        "failed": failed.ravel(),
                        "used": used.ravel(),
                        "waste": waste.ravel(),
                    }
                )
            )
    policy_table = pd.DataFrame(
        policies,
        columns=[
            "fold",
            "tuning",
            "rule",
            "lead",
            "cost_ratio",
            "theta",
            "alpha",
            "tuned_cost_rate",
        ],
    )
    return policy_table, pd.concat(rows, ignore_index=True)


def age_table(
    lives: pd.Series,
    folds: Sequence[Sequence[Fold]],
    repeats: Sequence[int],
    cost_ratios: Sequence[float],
) -> pd.DataFrame:
    """Age replacement per (repeat, fold, cost ratio): A* tuned on the outer-train lifetimes."""
    rows = []
    for r in repeats:
        for f, fold in enumerate(folds[r]):
            train = lives.loc[[*fold.inner_fit, *fold.inner_val]].to_numpy()
            units = np.array(sorted(fold.outer_test))
            test = lives.loc[units].to_numpy()
            for k in cost_ratios:
                age = tune_age(train, k)
                found = age_outcomes(age, test)
                rows.append(
                    pd.DataFrame(
                        {
                            "repeat": r,
                            "fold": f,
                            "cost_ratio": k,
                            "age": age,
                            "unit_id": units,
                            "failed": found.failed[0],
                            "used": found.used[0],
                            "waste": found.waste[0],
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def savings(
    found: pd.DataFrame,
    age: pd.DataFrame,
    lives: pd.Series,
    by: Sequence[str] = ("config_id", "repeat", "tuning", "rule", "lead", "cost_ratio"),
) -> pd.DataFrame:
    """Savings share per group, from outcomes pooled over the folds of each repeat.

    CR and CR_age pool costs and cycles over every outer_test engine of the repeat; CR* is
    N / sum of T over the same engines.
    """
    by = list(by)

    def pool(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        costs = np.where(table["failed"], table["cost_ratio"].astype(float), 1.0)
        grouped = table.assign(cost=costs).groupby(keys, observed=True)
        totals = grouped[["cost", "used"]].sum()
        totals["engines"] = grouped["unit_id"].nunique()
        return totals.reset_index()

    reference = pool(age, ["repeat", "cost_ratio"]).rename(
        columns={"cost": "age_cost", "used": "age_used", "engines": "age_engines"}
    )
    model = pool(found, by).merge(reference, on=["repeat", "cost_ratio"], how="left")
    perfect = {
        r: perfect_rate(lives.loc[np.unique(group["unit_id"])].to_numpy())
        for r, group in age.groupby("repeat")
    }
    model["cost_rate"] = model["cost"] / model["used"]
    model["age_rate"] = model["age_cost"] / model["age_used"]
    model["perfect_rate"] = model["repeat"].map(perfect)
    model["savings_share"] = savings_share(
        model["cost_rate"], model["age_rate"], model["perfect_rate"]
    )
    return model


def truncation_points(
    lives: pd.Series, seed: Sequence[int], n: int = TRUNCATION_POINTS
) -> pd.DataFrame:
    """n cycles per engine, uniform on min(31, T)..T with replacement, the same for every fit."""
    rows = []
    for unit, life in lives.items():
        rng = np.random.default_rng([*seed, int(unit)])
        cycles = rng.integers(min(TRUNCATION_START, life), life + 1, size=n)
        rows.append(pd.DataFrame({"unit_id": unit, "cycle": cycles}))
    return pd.concat(rows, ignore_index=True)


def _nasa_terms(error: np.ndarray) -> np.ndarray:
    # PHM08: an early prediction by d cycles costs exp(d / 13) - 1, a late one exp(d / 10) - 1
    return np.where(error < 0, np.expm1(-error / 13.0), np.expm1(error / 10.0))


def metric_components(
    table: pd.DataFrame, truncation: pd.DataFrame, cap: int
) -> pd.DataFrame:
    """Per engine of one fit: the sums and counts every accuracy metric is built from.

    official_test: last-cycle squared errors (truth capped at 125, at 130 and uncapped) and
    NASA term, all-cycle squared errors, jitter, late-phase errors. outer_test: squared errors
    at the truncation points, jitter, late-phase errors. inner_val: all-cycle squared errors.
    Columns that do not apply to a split are NaN.
    """
    rows = table.assign(split=table["split"].astype(str)).sort_values(
        ["split", "unit_id", "cycle"]
    )
    pred = rows["pred"].to_numpy(float)
    rul = rows["rul"].to_numpy(float)
    capped = np.minimum(rul, cap)
    same_unit = (rows["unit_id"].to_numpy()[1:] == rows["unit_id"].to_numpy()[:-1]) & (
        rows["split"].to_numpy()[1:] == rows["split"].to_numpy()[:-1]
    )
    # jitter at cycle t: |y_t - y_(t-1) + 1| where the previous true RUL is below the cap
    step = np.full(len(rows), np.nan)
    keep = same_unit & (rul[:-1] < cap)
    step[1:][keep] = np.abs(pred[1:] - pred[:-1] + 1.0)[keep]
    late = rul <= LATE_RUL
    work = pd.DataFrame(
        {
            "split": rows["split"].to_numpy(),
            "unit_id": rows["unit_id"].to_numpy(),
            "sq": (pred - capped) ** 2,
            "jit": step,
            "jit_n": ~np.isnan(step),
            "sq_late": np.where(late, (pred - rul) ** 2, 0.0),
            "err_late": np.where(late, pred - rul, 0.0),
            "n_late": late,
        }
    )
    grouped = work.groupby(["split", "unit_id"], sort=True)
    out = pd.DataFrame(
        {
            "sse_all": grouped["sq"].sum(),
            "n_all": grouped["sq"].size(),
            "jit_sum": grouped["jit"].sum(),
            "jit_n": grouped["jit_n"].sum(),
            "sse_late": grouped["sq_late"].sum(),
            "err_late": grouped["err_late"].sum(),
            "n_late": grouped["n_late"].sum(),
        }
    ).reset_index()
    last = (
        rows.groupby(["split", "unit_id"], sort=True)
        .tail(1)
        .set_index(["split", "unit_id"])
    )
    error = {
        name: last["pred"] - np.minimum(last["rul"], c) for name, c in CAPS.items()
    }
    out = out.join(
        pd.DataFrame(
            {
                "sq_last_125": error["125"] ** 2,
                "sq_last_130": error["130"] ** 2,
                "sq_last_raw": (last["pred"] - last["rul"]) ** 2,
                "nasa_last": _nasa_terms(error["125"].to_numpy()),
            },
            index=last.index,
        ),
        on=["split", "unit_id"],
    )
    test_rows = rows[rows["split"] == "outer_test"]
    hits = truncation.merge(test_rows, on=["unit_id", "cycle"], how="inner")
    hits_sq = (hits["pred"] - np.minimum(hits["rul"], cap)) ** 2
    trunc = (
        hits.assign(sq=hits_sq)
        .groupby("unit_id")["sq"]
        .agg(sse_trunc="sum", n_trunc="size")
    )
    out = out.join(trunc, on="unit_id")
    official = out["split"] == "official_test"
    outer = out["split"] == "outer_test"
    inner = out["split"] == "inner_val"
    for column in ("sq_last_125", "sq_last_130", "sq_last_raw", "nasa_last"):
        out.loc[~official, column] = np.nan
    for column in ("sse_trunc", "n_trunc"):
        out.loc[~outer, column] = np.nan
    for column in ("jit_sum", "jit_n", "sse_late", "err_late", "n_late"):
        out.loc[inner, column] = np.nan
    return out


def fit_metrics(components: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Point metrics per fit from its components."""
    by = list(by)
    official = components[components["split"] == "official_test"].groupby(by)
    outer = components[components["split"] == "outer_test"]
    inner = components[components["split"] == "inner_val"].groupby(by)

    def jitter(table: pd.DataFrame) -> pd.Series:
        ok = table[table["jit_n"] >= MIN_JITTER_CYCLES]
        return (ok["jit_sum"] / ok["jit_n"]).groupby([ok[c] for c in by]).mean()

    out = pd.DataFrame(
        {
            "rmse_last": np.sqrt(official["sq_last_125"].mean()),
            "rmse_last_130": np.sqrt(official["sq_last_130"].mean()),
            "rmse_last_raw": np.sqrt(official["sq_last_raw"].mean()),
            "nasa": official["nasa_last"].sum(),
            "rmse_all": np.sqrt(official["sse_all"].sum() / official["n_all"].sum()),
            "jitter": jitter(components[components["split"] == "official_test"]),
            "late_rmse": np.sqrt(official["sse_late"].sum() / official["n_late"].sum()),
            "late_bias": official["err_late"].sum() / official["n_late"].sum(),
            "rmse_trunc_outer": np.sqrt(
                outer.groupby(by)["sse_trunc"].sum()
                / outer.groupby(by)["n_trunc"].sum()
            ),
            "jitter_outer": jitter(outer),
            "late_rmse_outer": np.sqrt(
                outer.groupby(by)["sse_late"].sum() / outer.groupby(by)["n_late"].sum()
            ),
            "val_rmse": np.sqrt(inner["sse_all"].sum() / inner["n_all"].sum()),
        }
    )
    return out.reset_index()


def inventory(
    found: pd.DataFrame, n_outer: int, family_order: Sequence[str]
) -> tuple[list[str], list[int], pd.DataFrame]:
    """Configurations to build, the repeats present, and the configurations excluded.

    A configuration with a failed fit in any (repeat, fold) of the fleet is excluded
    (PREREGISTRATION.md section 9). A fit with no sidecar means the runs are unfinished, and
    nothing is built from an unfinished archive.
    """
    if found.empty:
        raise ValueError("the archive holds no fits for this fleet")
    repeats = sorted(int(r) for r in found["repeat"].unique())
    expected = {(r, f) for r in repeats for f in range(n_outer)}
    configs, excluded = [], []
    for config_id, group in found.groupby("config_id", sort=False):
        seen = {
            (int(r), int(f)): s
            for r, f, s in group[["repeat", "fold", "status"]].values
        }
        missing = sorted(expected - set(seen))
        if missing:
            raise ValueError(
                f"{config_id}: fits {missing} (repeat, fold) are still pending"
            )
        failed = sorted(k for k, s in seen.items() if s != "ok")
        if failed:
            excluded.append({"config_id": config_id, "failed_fits": str(failed)})
        else:
            configs.append(config_id)
    families = {c: c.rsplit("-", 1)[0] for c in configs}
    rank = {f: i for i, f in enumerate(family_order)}
    configs.sort(key=lambda c: (rank.get(families[c], len(rank)), c))
    return (
        configs,
        repeats,
        pd.DataFrame(excluded, columns=["config_id", "failed_fits"]),
    )
