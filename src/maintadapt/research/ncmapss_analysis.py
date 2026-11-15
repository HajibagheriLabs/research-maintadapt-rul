"""E1-E3 on N-CMAPSS (PREREGISTRATION.md section 8), secondary and descriptive.

N-CMAPSS has no official test engines. Its leaderboard RMSE is the RMSE over every flight of a
fit's outer_test units, and jitter uses the same units. Maintenance value, the selections of E2
and every statistic come from analysis.py unchanged. Only the two accuracy metrics are written
here, because each fit scores a different set of units. Since D and the accuracy metrics are
measured on the same units, one draw of units resamples both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from maintadapt.research.analysis import (
    FleetInputs,
    FleetResult,
    choose,
    decision_loss,
    estimate,
    fleet_inputs,
    selected_share,
)
from maintadapt.research.config import PrimaryConfig
from maintadapt.research.derived import MIN_JITTER_CYCLES
from maintadapt.research.stats import bootstrap_counts, kendall_tau_b

# fleet_inputs reads official test engines; N-CMAPSS gives it one placeholder that is never used
PLACEHOLDER_UNIT = -1


def with_placeholder(components: pd.DataFrame) -> pd.DataFrame:
    """The components plus one empty official_test row per fit, for fleet_inputs alone."""
    fits = components[["config_id", "repeat", "fold"]].drop_duplicates()
    rows = fits.assign(
        split="official_test",
        unit_id=PLACEHOLDER_UNIT,
        sq_last_125=0.0,
        jit_sum=0.0,
        jit_n=0.0,
    )
    return pd.concat([components, rows], ignore_index=True)


@dataclass(frozen=True, eq=False)
class NcmapssInputs:
    # the decision inputs of analysis.fleet_inputs; its official-test fields are placeholders
    base: FleetInputs
    # configs x repeats x units: each unit's outer_test sums in the fold that tested it
    sse: np.ndarray
    rows: np.ndarray
    jit_sum: np.ndarray
    jit_n: np.ndarray
    # repeats x folds x units: whether a unit is in that fold's outer_test
    members: np.ndarray


def _grid(table: pd.DataFrame, levels: Sequence[Sequence], column: str) -> np.ndarray:
    keys = ["config_id", "repeat", "unit_id"]
    index = pd.MultiIndex.from_product([list(v) for v in levels], names=keys)
    values = table.set_index(keys)[column].reindex(index)
    if values.isna().any():
        raise ValueError(
            f"{column}: {int(values.isna().sum())} expected rows are missing"
        )
    return values.to_numpy(float).reshape([len(v) for v in levels])


def ncmapss_inputs(
    tables: Mapping[str, pd.DataFrame], primary: PrimaryConfig, configs: Sequence[str]
) -> NcmapssInputs:
    """Arrange an N-CMAPSS fleet's derived tables (outcomes, policies, age, engines, components)."""
    components = tables["components"]
    base = fleet_inputs(
        {**tables, "components": with_placeholder(components)}, primary, configs
    )
    outer = components[components["split"] == "outer_test"]
    units = np.sort(tables["engines"]["unit_id"].to_numpy())
    levels = [list(configs), base.repeats, units]
    members = np.stack(
        [base.fold_of == k for k in range(len(base.folds))], axis=1
    ).astype(float)
    return NcmapssInputs(
        base=base,
        sse=_grid(outer, levels, "sse_all"),
        rows=_grid(outer, levels, "n_all"),
        jit_sum=_grid(outer, levels, "jit_sum"),
        jit_n=_grid(outer, levels, "jit_n"),
        members=members,
    )


def _per_fit(values: np.ndarray, members: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Sums over each fit's units under every replicate: B x configs x repeats x folds."""
    c, r, n = values.shape
    spread = values[:, :, None, :] * members[None]
    return (counts @ spread.reshape(-1, n).T).reshape(len(counts), c, r, -1)


def leaderboard_rmse(inputs: NcmapssInputs, counts: np.ndarray) -> np.ndarray:
    """RMSE over every flight of a fit's outer_test units, averaged over fits: B x configs."""
    sse = _per_fit(inputs.sse, inputs.members, counts)
    rows = _per_fit(inputs.rows, inputs.members, counts)
    # a replicate that draws no unit of some fold is undefined and left out, like tau-b's
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt(sse / rows)
    return rmse.reshape(len(counts), sse.shape[1], -1).mean(axis=-1)


def jitter(inputs: NcmapssInputs, counts: np.ndarray) -> np.ndarray:
    """Mean jitter over a fit's qualifying outer_test units, averaged over fits: B x configs.

    A unit qualifies with at least MIN_JITTER_CYCLES steps, which depends on the truth alone.
    """
    qualifies = inputs.jit_n >= MIN_JITTER_CYCLES
    if not (qualifies == qualifies[:1]).all():
        raise ValueError("configurations disagree on which units have jitter cycles")
    step = np.where(
        qualifies, inputs.jit_sum / np.where(qualifies, inputs.jit_n, 1.0), 0.0
    )
    total = _per_fit(step, inputs.members, counts)
    units = _per_fit(qualifies.astype(float), inputs.members, counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / units
    return mean.reshape(len(counts), total.shape[1], -1).mean(axis=-1)


def _taus(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([kendall_tau_b(a, b) for a, b in zip(x, y, strict=True)])


def analyse_ncmapss(
    inputs: NcmapssInputs, n_boot: int, ci: float, seed: Sequence[int]
) -> FleetResult:
    """E1-E3 as analysis.analyse_fleet computes them, with the units drawn once for all."""
    base = inputs.base
    n = base.cost.shape[-1]
    counts = bootstrap_counts(n, n_boot, [*seed, 0])
    once = np.ones((1, n))
    loss, loss_point = decision_loss(base, counts), decision_loss(base, once)[0]
    rmse, rmse_point = (
        leaderboard_rmse(inputs, counts),
        leaderboard_rmse(inputs, once)[0],
    )
    jit, jit_point = jitter(inputs, counts), jitter(inputs, once)[0]
    tau_rmse, tau_rmse_point = _taus(rmse, loss), kendall_tau_b(rmse_point, loss_point)
    tau_jit, tau_jit_point = _taus(jit, loss), kendall_tau_b(jit_point, loss_point)
    by_cost, by_rmse = choose(base.val_cost), choose(base.val_rmse)
    s_cost, s_rmse = (selected_share(base, c, counts) for c in (by_cost, by_rmse))
    s_cost_point = selected_share(base, by_cost, once)[0]
    s_rmse_point = selected_share(base, by_rmse, once)[0]
    return FleetResult(
        e1=estimate(tau_rmse_point, tau_rmse, ci),
        e2=estimate(s_cost_point - s_rmse_point, s_cost - s_rmse, ci),
        e3=estimate(tau_jit_point - tau_rmse_point, tau_jit - tau_rmse, ci),
        tau_jitter=estimate(tau_jit_point, tau_jit, ci),
        s_by_cost=estimate(s_cost_point, s_cost, ci),
        s_by_rmse=estimate(s_rmse_point, s_rmse, ci),
        by_cost=by_cost,
        by_rmse=by_rmse,
        points=pd.DataFrame(
            {
                "config_id": base.configs,
                "leaderboard_rmse": rmse_point,
                "jitter": jit_point,
                "savings_share": 1.0 - loss_point,
                "decision_loss": loss_point,
            }
        ),
    )
