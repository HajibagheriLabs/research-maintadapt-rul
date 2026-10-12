"""Confirmatory analysis: endpoints E1-E3 of PREREGISTRATION.md section 6 and their verdicts.

Frozen at the tag analysis-frozen; any later change is a deviation. Every quantity is computed by
one formula from per-engine outcomes and errors weighted by bootstrap counts. The point estimate
is the same formula with every engine counted once, so the two can never disagree in definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from maintadapt.research.config import PrimaryConfig
from maintadapt.research.costs import savings_share
from maintadapt.research.derived import MIN_JITTER_CYCLES
from maintadapt.research.stats import bootstrap_counts, kendall_tau_b

# PREREGISTRATION.md section 2, quoted
HYPOTHESES = {
    "H1": "across a realistic pool of RUL models, ranking by leaderboard RMSE agrees only "
    "weakly with ranking by maintenance value.",
    "H2": "selecting a model by validation RMSE yields materially lower maintenance value than "
    "selecting by validation maintenance cost.",
    "H3": "the temporal stability of predictions (jitter) is more strongly associated with "
    "maintenance value than leaderboard RMSE is.",
}
# PREREGISTRATION.md section 6, "Interpretation rules, fixed now", quoted
RULES = {
    "H1": "E1 bands by point estimate: strong >= 0.7; moderate 0.4-0.7; weak < 0.4. H1 is "
    "supported if at least three fleets are weak with interval upper bounds below 0.6; rejected "
    "if at least three fleets are strong with interval lower bounds above 0.5; otherwise mixed "
    "and reported per fleet.",
    "H2": "E2: H2 is supported if the selection loss is at least 0.05 with an interval "
    "excluding 0 on at least two fleets; rejected if the interval lies within [-0.02, 0.02] on "
    "at least three fleets; otherwise inconclusive.",
    "H3": "E3: H3 is supported if delta-tau > 0 with an interval excluding 0 on at least three "
    "fleets; rejected if delta-tau <= 0 on at least three fleets; otherwise inconclusive.",
}


@dataclass(frozen=True, eq=False)
class FleetInputs:
    """One fleet's per-engine quantities at the primary setting, in fixed orders."""

    configs: list[str]
    repeats: list[int]
    folds: list[int]
    lives: np.ndarray
    # configs x repeats x training engines: cost (1 or K) and cycles used on outer_test
    cost: np.ndarray
    used: np.ndarray
    # repeats x training engines: position in folds of the outer fold that tested each engine
    fold_of: np.ndarray
    # repeats x training engines: age replacement tuned on that fold's outer-train lifetimes
    age_cost: np.ndarray
    age_used: np.ndarray
    # configs x repeats x folds: section 5.8's validation cost and validation RMSE
    val_cost: np.ndarray
    val_rmse: np.ndarray
    # configs x fits x official test engines, fits ordered by (repeat, fold)
    sq_last: np.ndarray
    # configs x fits x jitter engines: each engine's mean jitter step
    jitter: np.ndarray
    # positions, among the official test engines, of those with enough jitter cycles
    jitter_engines: np.ndarray


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float
    # bootstrap replicates with a finite value, out of all
    valid: int


def _primary(table: pd.DataFrame, primary: PrimaryConfig) -> pd.DataFrame:
    keep = (
        (table["tuning"] == primary.tuning)
        & (table["rule"] == primary.rule)
        & (table["lead"] == primary.lead_time)
        & (table["cost_ratio"] == primary.cost_ratio)
    )
    return table[keep]


def _cube(
    table: pd.DataFrame, keys: Sequence[str], levels: Sequence[Sequence], column: str
) -> np.ndarray:
    """Values of column on the full grid of keys; every grid point must appear exactly once."""
    index = pd.MultiIndex.from_product([list(level) for level in levels], names=keys)
    values = table.set_index(list(keys))[column].reindex(index)
    if values.isna().any():
        raise ValueError(
            f"{column}: {int(values.isna().sum())} of the expected rows are missing"
        )
    return values.to_numpy(float).reshape([len(level) for level in levels])


def fleet_inputs(
    tables: Mapping[str, pd.DataFrame], primary: PrimaryConfig, configs: Sequence[str]
) -> FleetInputs:
    """Arrange a fleet's derived tables (outcomes, policies, age, engines, components)."""
    configs = list(configs)
    found = _primary(tables["outcomes"], primary)
    found = found[found["config_id"].isin(configs)]
    found = found.assign(cost=np.where(found["failed"], primary.cost_ratio, 1.0))
    units = np.sort(tables["engines"]["unit_id"].to_numpy())
    lives = tables["engines"].set_index("unit_id").loc[units, "life"].to_numpy(float)
    repeats = sorted(int(r) for r in found["repeat"].unique())
    grid = ["config_id", "repeat", "unit_id"]
    cost = _cube(found, grid, [configs, repeats, units], "cost")
    used = _cube(found, grid, [configs, repeats, units], "used")
    fold_cube = _cube(found, grid, [configs, repeats, units], "fold").astype(int)
    if not (fold_cube == fold_cube[0]).all():
        raise ValueError("configurations disagree on which fold tested an engine")
    age = tables["age"][tables["age"]["cost_ratio"] == primary.cost_ratio]
    age = age.assign(cost=np.where(age["failed"], primary.cost_ratio, 1.0))
    pol = _primary(tables["policies"], primary)
    folds = sorted(int(f) for f in pol["fold"].unique())
    if not set(np.unique(fold_cube)) <= set(folds):
        raise ValueError("outcomes name folds that have no tuned policy")
    fit_grid = ["config_id", "repeat", "fold"]
    parts = tables["components"]
    inner = (
        parts[parts["split"] == "inner_val"]
        .groupby(fit_grid)[["sse_all", "n_all"]]
        .sum()
    )
    inner["val_rmse"] = np.sqrt(inner["sse_all"] / inner["n_all"])
    official = parts[parts["split"] == "official_test"]
    test_units = np.sort(official["unit_id"].unique())
    fits = [(r, f) for r in repeats for f in folds]
    sq = _cube(
        official,
        [*fit_grid, "unit_id"],
        [configs, repeats, folds, test_units],
        "sq_last_125",
    )
    steps = _cube(
        official, [*fit_grid, "unit_id"], [configs, repeats, folds, test_units], "jit_n"
    )
    sums = _cube(
        official,
        [*fit_grid, "unit_id"],
        [configs, repeats, folds, test_units],
        "jit_sum",
    )
    # qualifying cycles depend on the truth alone, so every fit keeps the same engines
    qualifies = steps >= MIN_JITTER_CYCLES
    if not (qualifies == qualifies[:1, :1, :1]).all():
        raise ValueError(
            "fits disagree on which official test engines have jitter cycles"
        )
    keep = qualifies[0, 0, 0]
    shape = (len(configs), len(fits), len(test_units))
    return FleetInputs(
        configs=configs,
        repeats=repeats,
        folds=folds,
        lives=lives,
        cost=cost,
        used=used,
        fold_of=np.searchsorted(folds, fold_cube[0]),
        age_cost=_cube(age, ["repeat", "unit_id"], [repeats, units], "cost"),
        age_used=_cube(age, ["repeat", "unit_id"], [repeats, units], "used"),
        val_cost=_cube(pol, fit_grid, [configs, repeats, folds], "tuned_cost_rate"),
        val_rmse=_cube(
            inner.reset_index(), fit_grid, [configs, repeats, folds], "val_rmse"
        ),
        sq_last=sq.reshape(shape),
        jitter=(sums / np.where(qualifies, steps, 1.0)).reshape(shape)[:, :, keep],
        jitter_engines=np.flatnonzero(keep),
    )


def _rates(counts: np.ndarray, cost: np.ndarray, used: np.ndarray) -> np.ndarray:
    """Pooled cost rates of every row of cost and used under every replicate: B x rows."""
    n = cost.shape[-1]
    return (counts @ cost.reshape(-1, n).T) / (counts @ used.reshape(-1, n).T)


def _shares(
    inputs: FleetInputs, counts: np.ndarray, cost: np.ndarray, used: np.ndarray
) -> np.ndarray:
    """Savings shares of (..., repeats, engines) outcomes, averaged over repeats: B x ...."""
    b = len(counts)
    rate = _rates(counts, cost, used).reshape(b, *cost.shape[:-1])
    middle = (1,) * (rate.ndim - 2)
    age = _rates(counts, inputs.age_cost, inputs.age_used).reshape(b, *middle, -1)
    perfect = (counts.sum(axis=1) / (counts @ inputs.lives)).reshape(b, *middle, 1)
    return savings_share(rate, age, perfect).mean(axis=-1)


def decision_loss(inputs: FleetInputs, counts: np.ndarray) -> np.ndarray:
    """D = 1 - S per configuration, S pooled over folds per repeat and averaged: B x configs."""
    return 1.0 - _shares(inputs, counts, inputs.cost, inputs.used)


def leaderboard_rmse(inputs: FleetInputs, counts: np.ndarray) -> np.ndarray:
    """Last-cycle RMSE on official test engines, averaged over each configuration's fits."""
    n = inputs.sq_last.shape[-1]
    mse = (counts @ inputs.sq_last.reshape(-1, n).T) / counts.sum(axis=1)[:, None]
    return np.sqrt(mse).reshape(len(counts), *inputs.sq_last.shape[:2]).mean(axis=-1)


def jitter(inputs: FleetInputs, counts_qualifying: np.ndarray) -> np.ndarray:
    """Mean jitter over qualifying official test engines, averaged over each configuration's fits.

    counts_qualifying holds the bootstrap counts of the qualifying engines only.
    """
    n = inputs.jitter.shape[-1]
    mean = (counts_qualifying @ inputs.jitter.reshape(-1, n).T) / counts_qualifying.sum(
        axis=1
    )[:, None]
    return mean.reshape(len(counts_qualifying), *inputs.jitter.shape[:2]).mean(axis=-1)


def choose(metric: np.ndarray) -> np.ndarray:
    """repeats x folds: the configuration with the lowest metric; ties to the first listed."""
    if np.isnan(metric).any():
        raise ValueError("a validation metric is missing")
    return np.argmin(metric, axis=0)


def selected_share(
    inputs: FleetInputs, choice: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """S of the chosen configurations: each fold's choice supplies its outer_test outcomes."""
    picked = choice[np.arange(len(inputs.repeats))[:, None], inputs.fold_of]
    rows = np.arange(len(inputs.repeats))[:, None]
    engines = np.arange(inputs.cost.shape[-1])[None, :]
    cost = inputs.cost[picked, rows, engines]
    used = inputs.used[picked, rows, engines]
    return _shares(inputs, counts, cost, used)


def _taus(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([kendall_tau_b(a, b) for a, b in zip(x, y, strict=True)])


def estimate(point: float, replicates: np.ndarray, ci: float) -> Estimate:
    """Percentile interval over the finite replicates."""
    finite = replicates[np.isfinite(replicates)]
    tail = (1.0 - ci) / 2.0
    lo, hi = (
        np.quantile(finite, [tail, 1.0 - tail]) if len(finite) else (np.nan, np.nan)
    )
    return Estimate(float(point), float(lo), float(hi), len(finite))


@dataclass(frozen=True, eq=False)
class FleetResult:
    e1: Estimate
    e2: Estimate
    e3: Estimate
    tau_jitter: Estimate
    s_by_cost: Estimate
    s_by_rmse: Estimate
    by_cost: np.ndarray
    by_rmse: np.ndarray
    # point estimates per configuration
    points: pd.DataFrame


def analyse_fleet(
    inputs: FleetInputs, n_boot: int, ci: float, seed: Sequence[int]
) -> FleetResult:
    """E1, E2 and E3 for one fleet with the engine bootstrap of section 6.

    Training engines are drawn once (shared by every configuration and repeat) and official
    test engines independently; outcomes, thresholds and selections stay fixed.
    """
    n_train, n_test = inputs.cost.shape[-1], inputs.sq_last.shape[-1]
    train = bootstrap_counts(n_train, n_boot, [*seed, 0])
    test = bootstrap_counts(n_test, n_boot, [*seed, 1])
    once_train, once_test = np.ones((1, n_train)), np.ones((1, n_test))

    loss = decision_loss(inputs, train)
    loss_point = decision_loss(inputs, once_train)[0]
    rmse = leaderboard_rmse(inputs, test)
    rmse_point = leaderboard_rmse(inputs, once_test)[0]
    jit = jitter(inputs, test[:, inputs.jitter_engines])
    jit_point = jitter(inputs, once_test[:, inputs.jitter_engines])[0]

    tau_rmse = _taus(rmse, loss)
    tau_rmse_point = kendall_tau_b(rmse_point, loss_point)
    tau_jit = _taus(jit, loss)
    tau_jit_point = kendall_tau_b(jit_point, loss_point)

    by_cost, by_rmse = choose(inputs.val_cost), choose(inputs.val_rmse)
    s_cost = selected_share(inputs, by_cost, train)
    s_rmse = selected_share(inputs, by_rmse, train)
    s_cost_point = selected_share(inputs, by_cost, once_train)[0]
    s_rmse_point = selected_share(inputs, by_rmse, once_train)[0]

    points = pd.DataFrame(
        {
            "config_id": inputs.configs,
            "leaderboard_rmse": rmse_point,
            "jitter": jit_point,
            "savings_share": 1.0 - loss_point,
            "decision_loss": loss_point,
        }
    )
    return FleetResult(
        e1=estimate(tau_rmse_point, tau_rmse, ci),
        e2=estimate(s_cost_point - s_rmse_point, s_cost - s_rmse, ci),
        e3=estimate(tau_jit_point - tau_rmse_point, tau_jit - tau_rmse, ci),
        tau_jitter=estimate(tau_jit_point, tau_jit, ci),
        s_by_cost=estimate(s_cost_point, s_cost, ci),
        s_by_rmse=estimate(s_rmse_point, s_rmse, ci),
        by_cost=by_cost,
        by_rmse=by_rmse,
        points=points,
    )


def band(tau: float) -> str:
    """E1 band of a point estimate: strong >= 0.7; moderate 0.4-0.7; weak < 0.4."""
    if tau >= 0.7:
        return "strong"
    if tau >= 0.4:
        return "moderate"
    return "weak"


def _excludes_zero(lo: float, hi: float) -> bool:
    return bool(lo > 0 or hi < 0)


def verdict_h1(e1: Mapping[str, Estimate]) -> str:
    weak = sum(band(e.point) == "weak" and e.hi < 0.6 for e in e1.values())
    strong = sum(band(e.point) == "strong" and e.lo > 0.5 for e in e1.values())
    if weak >= 3:
        return "supported"
    if strong >= 3:
        return "rejected"
    return "mixed"


def verdict_h2(e2: Mapping[str, Estimate]) -> str:
    material = sum(e.point >= 0.05 and _excludes_zero(e.lo, e.hi) for e in e2.values())
    negligible = sum(e.lo >= -0.02 and e.hi <= 0.02 for e in e2.values())
    if material >= 2:
        return "supported"
    if negligible >= 3:
        return "rejected"
    return "inconclusive"


def verdict_h3(e3: Mapping[str, Estimate]) -> str:
    stronger = sum(e.point > 0 and _excludes_zero(e.lo, e.hi) for e in e3.values())
    not_stronger = sum(e.point <= 0 for e in e3.values())
    if stronger >= 3:
        return "supported"
    if not_stronger >= 3:
        return "rejected"
    return "inconclusive"
