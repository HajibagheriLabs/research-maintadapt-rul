"""Secondary analyses S1-S7 and S9 of PREREGISTRATION.md section 7.

They reuse the confirmatory functions of analysis.py unchanged: the same derived tables, the
same bootstrap draws (training engines from default_rng([2026, fleet number, 0]), official test
engines from [..., 1]) and the same formulas, applied at other settings, to other accuracy
metrics, to family representatives, to other thresholds and to other training caps. Regret (S6)
replaces the cost rate, so its formulas are written out here in parallel to analysis.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from maintadapt.research.analysis import (
    Estimate,
    FleetInputs,
    choose,
    decision_loss,
    estimate,
    jitter,
    leaderboard_rmse,
)
from maintadapt.research.config import DecisionGridConfig, PrimaryConfig
from maintadapt.research.costs import age_outcomes, regret
from maintadapt.research.derived import (
    RULES,
    EngineSet,
    engines_of,
    repeat_outcomes,
)
from maintadapt.research.folds import Fold
from maintadapt.research.models.spaces import FAMILY_NAMES
from maintadapt.research.policies import (
    alarm_times,
    lives_of,
    median_minimiser,
    outcomes,
    trial,
)
from maintadapt.research.stats import (
    benjamini_hochberg,
    bootstrap_counts,
    kendall_tau_b,
    pairwise_differences,
)

FIT_KEYS = ["config_id", "repeat", "fold"]
SETTING_KEYS = ["tuning", "rule", "lead", "cost_ratio"]


def family_of(config_id: str) -> str:
    return config_id.rsplit("-", 1)[0]


def pool_order(config_ids: Sequence[str]) -> list[str]:
    """Pool configurations in pool order; anything outside the pool (the controls) is dropped."""
    rank = {f: i for i, f in enumerate(FAMILY_NAMES)}
    kept = [c for c in config_ids if family_of(c) in rank]
    return sorted(kept, key=lambda c: (rank[family_of(c)], c))


def settings_grid(decision: DecisionGridConfig, tuning: str) -> list[PrimaryConfig]:
    """Every rule, lead time and cost ratio of section 5.5 under one tuning."""
    return [
        PrimaryConfig(cost_ratio=k, lead_time=lead, rule=rule, tuning=tuning)
        for rule in RULES
        for lead in decision.lead_times
        for k in decision.cost_ratios
    ]


def split_by_setting(
    found: pd.DataFrame,
) -> dict[tuple[str, str, int, float], pd.DataFrame]:
    """Outcome rows per (tuning, rule, lead, cost ratio), so each setting is cut once."""
    return {
        (str(t), str(r), int(lead), float(k)): group
        for (t, r, lead, k), group in found.groupby(SETTING_KEYS, sort=False)
    }


def outcomes_at(
    parts: Mapping[tuple[str, str, int, float], pd.DataFrame], setting: PrimaryConfig
) -> pd.DataFrame:
    key = (setting.tuning, setting.rule, setting.lead_time, float(setting.cost_ratio))
    if key not in parts:
        raise KeyError(f"no outcomes at {key}")
    return parts[key]


@dataclass(frozen=True, eq=False)
class Draws:
    """One fleet's bootstrap counts: the confirmatory draws, and counts of one for the point."""

    train: np.ndarray
    test: np.ndarray

    @staticmethod
    def of(inputs: FleetInputs, n_boot: int, seed: Sequence[int]) -> Draws:
        """The draws of analysis.analyse_fleet with the same seed: identical replicates."""
        return Draws(
            bootstrap_counts(inputs.cost.shape[-1], n_boot, [*seed, 0]),
            bootstrap_counts(inputs.sq_last.shape[-1], n_boot, [*seed, 1]),
        )

    @property
    def once_train(self) -> np.ndarray:
        return np.ones((1, self.train.shape[1]))

    @property
    def once_test(self) -> np.ndarray:
        return np.ones((1, self.test.shape[1]))


def tau(
    x_point: np.ndarray,
    x_reps: np.ndarray,
    y_point: np.ndarray,
    y_reps: np.ndarray,
    ci: float,
) -> tuple[Estimate, np.ndarray]:
    """Kendall's tau-b of two per-configuration metrics, with its replicates."""
    reps = np.array([kendall_tau_b(a, b) for a, b in zip(x_reps, y_reps, strict=True)])
    return estimate(kendall_tau_b(x_point, y_point), reps, ci), reps


def grid_values(
    table: pd.DataFrame, keys: Sequence[str], levels: Sequence[Sequence], column: str
) -> np.ndarray:
    """Values of column on the full grid of keys; every grid point must appear exactly once."""
    index = pd.MultiIndex.from_product([list(v) for v in levels], names=list(keys))
    values = table.set_index(list(keys))[column].reindex(index)
    if values.isna().any():
        raise ValueError(
            f"{column}: {int(values.isna().sum())} expected rows are missing"
        )
    return values.to_numpy(float).reshape([len(v) for v in levels])


# S1: other leaderboard metrics


def official_values(
    inputs: FleetInputs, components: pd.DataFrame, column: str
) -> np.ndarray:
    """configs x fits x official test engines, ordered as FleetInputs.sq_last."""
    official = components[components["split"] == "official_test"]
    units = np.sort(official["unit_id"].unique())
    values = grid_values(
        official,
        [*FIT_KEYS, "unit_id"],
        [inputs.configs, inputs.repeats, inputs.folds, units],
        column,
    )
    return values.reshape(len(inputs.configs), -1, len(units))


def nasa_score(nasa: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """NASA score summed over official test engines, averaged over fits: B x configs."""
    n = nasa.shape[-1]
    total = counts @ nasa.reshape(-1, n).T
    return total.reshape(len(counts), *nasa.shape[:2]).mean(axis=-1)


def pooled_rmse(sse: np.ndarray, rows: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """RMSE pooled over engines per fit (or repeat), averaged over the middle axis."""
    n = sse.shape[-1]
    mse = (counts @ sse.reshape(-1, n).T) / (counts @ rows.reshape(-1, n).T)
    return np.sqrt(mse).reshape(len(counts), *sse.shape[:2]).mean(axis=-1)


# S3 and S4: family representatives


def representatives(inputs: FleetInputs) -> pd.DataFrame:
    """Per (repeat, fold) and family, the configuration with the lowest validation RMSE.

    Ties go to the configuration listed first, as in the selection of E2.
    """
    families = [family_of(c) for c in inputs.configs]
    rows = []
    for family in dict.fromkeys(families):
        members = [i for i, f in enumerate(families) if f == family]
        best = choose(inputs.val_rmse[members])
        for r, repeat in enumerate(inputs.repeats):
            for k, fold in enumerate(inputs.folds):
                rows.append(
                    {
                        "family": family,
                        "repeat": repeat,
                        "fold": fold,
                        "config_id": inputs.configs[members[best[r, k]]],
                    }
                )
    return pd.DataFrame(rows)


def family_tables(
    tables: Mapping[str, pd.DataFrame], reps: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Tables in which each family is one configuration: its representative in every fold.

    fleet_inputs and every confirmatory formula then apply to families unchanged.
    """
    out = dict(tables)
    for name in ("outcomes", "policies", "components"):
        table = tables[name].drop(columns=["family"], errors="ignore")
        merged = table.merge(reps, on=FIT_KEYS, how="inner")
        out[name] = merged.assign(config_id=merged["family"])
    return out


def failure_and_waste(found: pd.DataFrame) -> pd.DataFrame:
    """Per configuration: failure share and cycles wasted per preventive maintenance.

    Each is pooled over the outer_test engines of a repeat and averaged over repeats.
    """
    preventive = found.assign(kept=found["waste"].where(~found["failed"]))
    per_repeat = preventive.groupby(["config_id", "repeat"]).agg(
        failure_rate=("failed", "mean"), waste_per_maintenance=("kept", "mean")
    )
    return per_repeat.groupby("config_id").mean()


@dataclass(frozen=True, eq=False)
class FamilyResult:
    # one row per family: point estimates, intervals and ranks
    points: pd.DataFrame
    # every pair of families: leaderboard RMSE and D differences, before correction
    pairs: pd.DataFrame
    # tau-b between the families' leaderboard RMSE and D
    e1: Estimate


def family_level(inputs: FleetInputs, draws: Draws, ci: float) -> FamilyResult:
    """S3 on inputs whose configurations are families (see family_tables)."""
    names = inputs.configs
    rmse = leaderboard_rmse(inputs, draws.test)
    rmse_point = leaderboard_rmse(inputs, draws.once_test)[0]
    loss = decision_loss(inputs, draws.train)
    loss_point = decision_loss(inputs, draws.once_train)[0]
    jit = jitter(inputs, draws.test[:, inputs.jitter_engines])
    jit_point = jitter(inputs, draws.once_test[:, inputs.jitter_engines])[0]
    e1, _ = tau(rmse_point, rmse, loss_point, loss, ci)
    points = pd.DataFrame({"family": names})
    for name, point, reps in (
        ("leaderboard_rmse", rmse_point, rmse),
        ("savings_share", 1.0 - loss_point, 1.0 - loss),
        ("jitter", jit_point, jit),
    ):
        found = [estimate(p, reps[:, i], ci) for i, p in enumerate(point)]
        points[name] = [e.point for e in found]
        points[f"{name}_lo"] = [e.lo for e in found]
        points[f"{name}_hi"] = [e.hi for e in found]
    # rank 1 is the best: the lowest leaderboard RMSE, the highest S
    points["rank_rmse"] = points["leaderboard_rmse"].rank(method="min").astype(int)
    points["rank_value"] = (-points["savings_share"]).rank(method="min").astype(int)
    by_rmse = pairwise_differences(
        pd.Series(rmse_point, index=names), pd.DataFrame(rmse, columns=names), ci
    )
    by_loss = pairwise_differences(
        pd.Series(loss_point, index=names), pd.DataFrame(loss, columns=names), ci
    )
    pairs = by_rmse.merge(by_loss, on=["a", "b"], suffixes=("_rmse", "_loss"))
    return FamilyResult(points, pairs, e1)


def inversions(pairs: pd.DataFrame, fdr: float) -> pd.DataFrame:
    """Benjamini-Hochberg over every p-value of the analysis, then the inverted pairs.

    A pair is inverted when both differences are rejected and point in opposite directions:
    one family has the lower leaderboard RMSE, the other the lower decision loss.
    """
    p = np.concatenate([pairs["p_value_rmse"], pairs["p_value_loss"]])
    adjusted, rejected = benjamini_hochberg(p, fdr)
    n = len(pairs)
    out = pairs.assign(
        q_value_rmse=adjusted[:n],
        q_value_loss=adjusted[n:],
        rejected_rmse=rejected[:n],
        rejected_loss=rejected[n:],
    )
    opposite = np.sign(out["difference_rmse"]) * np.sign(out["difference_loss"]) < 0
    out["inversion"] = out["rejected_rmse"] & out["rejected_loss"] & opposite

    def excludes_zero(lo: pd.Series, hi: pd.Series) -> pd.Series:
        return (lo > 0) | (hi < 0)

    out["inversion_unadjusted"] = (
        excludes_zero(out["lo_rmse"], out["hi_rmse"])
        & excludes_zero(out["lo_loss"], out["hi_loss"])
        & opposite
    )
    return out


# S4: threshold variants


def with_validation_selection(
    variant: FleetInputs, validation: FleetInputs
) -> FleetInputs:
    """Outcomes under another tuning, with models still chosen by the validation metrics.

    Under crossfit or oracle tuning the in-sample cost of a policy is measured on outer_test
    engines, so it cannot serve as a validation metric; E2 keeps section 5.8's choice.
    """
    if variant.configs != validation.configs or variant.repeats != validation.repeats:
        raise ValueError(
            "the variant and the validation inputs must hold the same fits"
        )
    return replace(variant, val_cost=validation.val_cost, val_rmse=validation.val_rmse)


def nested_outcomes(
    oof: Mapping[int, pd.DataFrame],
    test: Mapping[int, pd.DataFrame],
    decision: DecisionGridConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Policies tuned on nested out-of-fold predictions, applied to the outer_test fit.

    oof[fold] holds the inner cross-fitted predictions of the fold's outer-train engines,
    test[fold] the outer_test predictions of the confirmatory fit. derived.repeat_outcomes
    tunes on whatever is labelled inner_val, so the nested predictions take that label and
    its validation tuning becomes the nested tuning; its other tunings are discarded.
    """
    fits = {
        fold: pd.concat(
            [
                oof[fold].assign(split="inner_val"),
                test[fold].assign(split="outer_test"),
            ],
            ignore_index=True,
        )
        for fold in sorted(test)
    }
    policies, found = repeat_outcomes(fits, decision)

    def nested(table: pd.DataFrame) -> pd.DataFrame:
        kept = table[table["tuning"] == "validation"]
        return kept.assign(tuning="nested").reset_index(drop=True)

    return nested(policies), nested(found)


def check_oof(oof: pd.DataFrame, outer: Fold) -> None:
    """Nested predictions must cover every outer-train engine exactly once."""
    expected = {*outer.inner_fit, *outer.inner_val}
    seen = oof.groupby("unit_id")["cycle"].agg(["size", "max"])
    if set(seen.index) != expected or not (seen["size"] == seen["max"]).all():
        raise ValueError("nested predictions must cover each outer-train engine once")


# S6: regret instead of the cost rate


def tune_regret(
    scores: Sequence[np.ndarray],
    lead: int,
    failure_costs: Sequence[float],
    levels: int,
) -> dict[float, tuple[float, float]]:
    """Per C_f, the R1 threshold with the lowest mean regret, and that regret.

    The candidates and the tie rule are those of the cost-rate tuning (section 5.4).
    """
    t = trial(scores, 1.0, levels)
    found = outcomes(t.alarms, lives_of(scores), lead)
    tuned = {}
    for cf in failure_costs:
        mean = regret(found, cf).mean(axis=-1)
        i = median_minimiser(mean)
        tuned[float(cf)] = (float(t.candidates[i]), float(mean[i]))
    return tuned


def tune_age_regret(lifetimes: np.ndarray, failure_cost: float) -> float:
    """Age replacement with the lowest mean regret; candidates as in costs.tune_age."""
    lifetimes = np.asarray(lifetimes)
    candidates = np.concatenate([np.unique(lifetimes).astype(float), [np.inf]])
    mean = np.array(
        [regret(age_outcomes(a, lifetimes), failure_cost).mean() for a in candidates]
    )
    return float(candidates[median_minimiser(mean)])


def regret_outcomes(
    fits: Mapping[int, pd.DataFrame],
    lead: int,
    failure_costs: Sequence[float],
    levels: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regret-tuned R1 policies of one configuration in one repeat, on its inner_val engines.

    Returns the policies (fold, failure cost, theta, in-sample mean regret) and the outer_test
    outcomes (fold, failure cost, engine, failed, waste).
    """
    policies, rows = [], []
    for fold in sorted(fits):
        val = engines_of(fits[fold], "inner_val")
        test: EngineSet = engines_of(fits[fold], "outer_test")
        tuned = tune_regret(val.predictions, lead, failure_costs, levels)
        thetas = np.array([theta for theta, _ in tuned.values()])
        found = outcomes(alarm_times(test.predictions, thetas), test.lives, lead)
        for i, (cf, (theta, mean)) in enumerate(tuned.items()):
            policies.append((fold, cf, theta, mean))
            rows.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "failure_cost": cf,
                        "unit_id": test.units,
                        "failed": found.failed[i],
                        "waste": found.waste[i],
                    }
                )
            )
    return (
        pd.DataFrame(
            policies, columns=["fold", "failure_cost", "theta", "tuned_regret"]
        ),
        pd.concat(rows, ignore_index=True),
    )


def regret_age_table(
    lives: pd.Series,
    folds: Sequence[Sequence[Fold]],
    repeats: Sequence[int],
    failure_costs: Sequence[float],
) -> pd.DataFrame:
    """Regret-tuned age replacement per (repeat, fold, C_f) on the outer-train lifetimes."""
    rows = []
    for r in repeats:
        for f, fold in enumerate(folds[r]):
            train = lives.loc[[*fold.inner_fit, *fold.inner_val]].to_numpy()
            units = np.array(sorted(fold.outer_test))
            for cf in failure_costs:
                age = tune_age_regret(train, cf)
                found = age_outcomes(age, lives.loc[units].to_numpy())
                rows.append(
                    pd.DataFrame(
                        {
                            "repeat": r,
                            "fold": f,
                            "failure_cost": float(cf),
                            "age": age,
                            "unit_id": units,
                            "failed": found.failed[0],
                            "waste": found.waste[0],
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


@dataclass(frozen=True, eq=False)
class RegretInputs:
    """One fleet's per-engine regret at one C_f, in the orders of its FleetInputs."""

    # configs x repeats x training engines
    regret: np.ndarray
    # repeats x training engines: regret-tuned age replacement
    age: np.ndarray
    # configs x repeats x folds: mean regret on inner_val with the threshold tuned there
    val_regret: np.ndarray


def regret_inputs(
    base: FleetInputs,
    found: pd.DataFrame,
    policies: pd.DataFrame,
    age: pd.DataFrame,
    failure_cost: float,
) -> RegretInputs:
    """Arrange regret tables (from regret_outcomes and regret_age_table) like base."""
    units = np.arange(base.cost.shape[-1])

    def at(table: pd.DataFrame) -> pd.DataFrame:
        kept = table[table["failure_cost"] == failure_cost]
        return kept.assign(regret=failure_cost * kept["failed"] + kept["waste"])

    found, age = at(found), at(age)
    unit_ids = np.sort(found["unit_id"].unique())
    if len(unit_ids) != len(units):
        raise ValueError("regret outcomes must cover the same engines as the inputs")
    grid = ["config_id", "repeat", "unit_id"]
    levels = [base.configs, base.repeats, unit_ids]
    folds = grid_values(found, grid, levels, "fold").astype(int)
    if not (np.searchsorted(base.folds, folds) == base.fold_of[None]).all():
        raise ValueError(
            "regret outcomes disagree with the inputs on the testing folds"
        )
    pol = policies[policies["failure_cost"] == failure_cost]
    return RegretInputs(
        regret=grid_values(found, grid, levels, "regret"),
        age=grid_values(age, ["repeat", "unit_id"], [base.repeats, unit_ids], "regret"),
        val_regret=grid_values(
            pol, FIT_KEYS, [base.configs, base.repeats, base.folds], "tuned_regret"
        ),
    )


def regret_loss(found: np.ndarray, age: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Mean regret relative to age replacement's, per repeat, averaged: B x ....

    Perfect prognostics has zero regret, so 1 minus this is the regret analogue of S.
    """
    b, n = len(counts), found.shape[-1]
    engines = counts.sum(axis=1)[:, None]
    mean = (counts @ found.reshape(-1, n).T) / engines
    reference = (counts @ age.reshape(-1, n).T) / engines
    middle = (1,) * (found.ndim - 2)
    ratio = mean.reshape(b, *found.shape[:-1]) / reference.reshape(b, *middle, -1)
    return ratio.mean(axis=-1)


def regret_selected(
    base: FleetInputs, r: RegretInputs, choice: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Regret share of the chosen configurations, as analysis.selected_share does for S."""
    rows = np.arange(len(base.repeats))[:, None]
    picked = choice[rows, base.fold_of]
    engines = np.arange(r.regret.shape[-1])[None, :]
    return 1.0 - regret_loss(r.regret[picked, rows, engines], r.age, counts)


@dataclass(frozen=True, eq=False)
class RegretResult:
    e1: Estimate
    e2: Estimate
    e3: Estimate
    s_by_regret: Estimate
    s_by_rmse: Estimate
    points: pd.DataFrame


def analyse_regret(
    base: FleetInputs, r: RegretInputs, draws: Draws, ci: float
) -> RegretResult:
    """E1-E3 with the regret share in place of S; models chosen by validation regret."""
    loss = regret_loss(r.regret, r.age, draws.train)
    loss_point = regret_loss(r.regret, r.age, draws.once_train)[0]
    rmse = leaderboard_rmse(base, draws.test)
    rmse_point = leaderboard_rmse(base, draws.once_test)[0]
    jit = jitter(base, draws.test[:, base.jitter_engines])
    jit_point = jitter(base, draws.once_test[:, base.jitter_engines])[0]
    e1, tau_rmse = tau(rmse_point, rmse, loss_point, loss, ci)
    e3_jit, tau_jit = tau(jit_point, jit, loss_point, loss, ci)
    by_regret, by_rmse = choose(r.val_regret), choose(base.val_rmse)
    s_regret = regret_selected(base, r, by_regret, draws.train)
    s_rmse = regret_selected(base, r, by_rmse, draws.train)
    s_regret_point = regret_selected(base, r, by_regret, draws.once_train)[0]
    s_rmse_point = regret_selected(base, r, by_rmse, draws.once_train)[0]
    return RegretResult(
        e1=e1,
        e2=estimate(s_regret_point - s_rmse_point, s_regret - s_rmse, ci),
        e3=estimate(e3_jit.point - e1.point, tau_jit - tau_rmse, ci),
        s_by_regret=estimate(s_regret_point, s_regret, ci),
        s_by_rmse=estimate(s_rmse_point, s_rmse, ci),
        points=pd.DataFrame(
            {"config_id": base.configs, "regret_share": 1.0 - loss_point}
        ),
    )


# S7: accuracy on the engines the decision metric uses


def matched_values(
    inputs: FleetInputs, components: pd.DataFrame, column: str
) -> np.ndarray:
    """configs x repeats x training engines, from each engine's outer_test fit."""
    outer = components[components["split"] == "outer_test"]
    units = np.sort(outer["unit_id"].unique())
    if len(units) != inputs.cost.shape[-1]:
        raise ValueError("outer_test components must cover every training engine")
    return grid_values(
        outer,
        ["config_id", "repeat", "unit_id"],
        [inputs.configs, inputs.repeats, units],
        column,
    )


# S9: the competitive region


def competitive(points: pd.DataFrame, margin: float) -> list[str]:
    """Configurations whose leaderboard RMSE is within margin cycles of the best, in order."""
    best = points["leaderboard_rmse"].min()
    return points.loc[points["leaderboard_rmse"] <= best + margin, "config_id"].tolist()
