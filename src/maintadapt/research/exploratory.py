"""Exploratory analyses outside the pre-registered plan; every output is labelled EXPLORATORY.

X1: late-phase RMSE after isotonic recalibration on inner_val, as an accuracy proxy blind to
    order-preserving distortions; strictly increasing ones cannot change R1 outcomes
    (DECISIONS.md, Decision layer and statistics).
X2: EMA smoothing at fixed alphas: its effect on maintenance value against its effect on RMSE.
X3: the engineering project's booking evaluation (thresholds tuned on validation snapshots,
    decision.replay_alarms on truncated official test engines) against the trajectory
    evaluation of this study, under the same failure cost and no lead time.
Per-fit quantities are computed here; the analysis is in scripts/research/run_exploratory.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.isotonic import IsotonicRegression

from maintadapt.decision import CostModel, replay_alarms, replay_snapshots
from maintadapt.research.costs import cost_rate
from maintadapt.research.derived import LATE_RUL
from maintadapt.research.policies import (
    alarm_times,
    lives_of,
    median_minimiser,
    outcomes,
    trial,
)


def ema_all(predictions: Sequence[np.ndarray], alpha: float) -> list[np.ndarray]:
    """policies.ema of many engines at once, as one recursive filter over a padded matrix.

    s_1 = y_1 and s_t = alpha * y_t + (1 - alpha) * s_(t-1); the padding after an engine's
    last cycle never reaches its own values.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
    if alpha == 1.0:
        return [np.asarray(p, dtype=float) for p in predictions]
    lengths = [len(p) for p in predictions]
    matrix = np.zeros((len(lengths), max(lengths)))
    for i, p in enumerate(predictions):
        matrix[i, : len(p)] = p
    start = (1.0 - alpha) * matrix[:, :1]
    smoothed, _ = lfilter([alpha], [1.0, alpha - 1.0], matrix, axis=1, zi=start)
    return [smoothed[i, :n] for i, n in enumerate(lengths)]


def recalibrate(
    val_pred: np.ndarray, val_truth: np.ndarray, pred: np.ndarray
) -> np.ndarray:
    """Isotonic regression of the truth on the inner_val predictions, applied to pred.

    Non-decreasing, so it never reverses the order of two predictions; beyond the range seen
    on inner_val it holds the end values.
    """
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(np.asarray(val_pred, dtype=float), np.asarray(val_truth, dtype=float))
    return model.predict(np.asarray(pred, dtype=float))


@dataclass(frozen=True, eq=False)
class Split:
    """One split of a fit as arrays: each engine's predictions and true RUL, in unit order."""

    units: np.ndarray
    predictions: list[np.ndarray]
    rul: list[np.ndarray]

    @property
    def lives(self) -> np.ndarray:
        return lives_of(self.predictions)

    def booking(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """(predictions, true RUL) per engine, as the engineering project reads them."""
        return list(zip(self.predictions, self.rul, strict=True))


def splits_of(fit: pd.DataFrame) -> dict[str, Split]:
    """Every split of one fit's rows, which the archive stores in unit, then cycle, order."""
    # the archive stores split as a dictionary column; its codes avoid comparing strings
    split = fit["split"].astype("category")
    codes = split.cat.codes.to_numpy()
    out = {}
    for code, name in enumerate(split.cat.categories):
        rows = fit[codes == code]
        if rows.empty:
            continue
        units, cycles = rows["unit_id"].to_numpy(), rows["cycle"].to_numpy()
        step, same = np.diff(cycles), np.diff(units) == 0
        if (np.diff(units) < 0).any() or (same & (step != 1)).any():
            raise ValueError(
                f"{name}: each engine's cycles must follow each other in order"
            )
        ids, starts = np.unique(units, return_index=True)
        out[str(name)] = Split(
            ids,
            np.split(rows["pred"].to_numpy(float), starts[1:]),
            np.split(rows["rul"].to_numpy(float), starts[1:]),
        )
    return out


def late_squared_errors(
    rul: Sequence[np.ndarray], predictions: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Per engine: squared errors summed over cycles whose true RUL is at most LATE_RUL."""
    sse, n = [], []
    for truth, pred in zip(rul, predictions, strict=True):
        late = truth <= LATE_RUL
        sse.append(float(((pred[late] - truth[late]) ** 2).sum()))
        n.append(int(late.sum()))
    return np.array(sse), np.array(n)


def last_cycle_squared_errors(
    scores: Sequence[np.ndarray], rul: Sequence[np.ndarray], cap: int
) -> np.ndarray:
    """Squared error at each engine's last observed cycle against the truth capped at cap."""
    last = np.array([s[-1] for s in scores])
    return (last - np.minimum([r[-1] for r in rul], cap)) ** 2


@dataclass(frozen=True)
class Tuned:
    theta: float
    cost_rate: float


def tune_r1(
    scores: Sequence[np.ndarray], lead: int, cost_ratio: float, levels: int
) -> Tuned:
    """The threshold with the lowest pooled cost rate, as policies.tune picks it for R1."""
    t = trial(scores, 1.0, levels)
    rates = cost_rate(outcomes(t.alarms, lives_of(scores), lead), cost_ratio)
    i = median_minimiser(rates)
    return Tuned(float(t.candidates[i]), float(rates[i]))


def legacy_booking(
    val: Sequence[tuple[np.ndarray, np.ndarray]],
    test: Sequence[tuple[np.ndarray, np.ndarray]],
    costs: CostModel,
    thresholds: np.ndarray,
) -> tuple[float, np.ndarray]:
    """The engineering project's protocol (scripts/run_decision.py) for one fit.

    The threshold is tuned on validation histories cut at every cycle (replay_snapshots),
    the first of the cheapest as np.argmin picks it, and applied to the truncated test
    engines, which keep their last booking (replay_alarms). Returns the threshold and each
    test engine's cost: the failure cost if it fails, otherwise the cycles thrown away.
    """
    on_val = replay_snapshots(val, thresholds)
    best = int(np.argmin(on_val.cost(costs)))
    on_test = replay_alarms(test, thresholds[[best]], run_to_failure=False)
    per_engine = (
        costs.failure * on_test.failed[0] + costs.per_cycle * on_test.discarded[0]
    )
    return float(thresholds[best]), per_engine


def fit_quantities(
    fit: pd.DataFrame,
    alphas: Sequence[float],
    lead: int,
    cost_ratio: float,
    levels: int,
    cap: int,
    costs: CostModel,
    thresholds: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Every per-engine quantity X1-X3 need from one fit's predictions.

    official: isotonic late-phase errors, the legacy booking cost and the last-cycle squared
    error of each EMA, per official test engine. outer: R1 outcomes of each EMA on outer_test,
    tuned on inner_val at (lead, cost_ratio). policies: each alpha's threshold and in-sample
    cost rate.
    """
    parts = splits_of(fit)
    val, outer = parts["inner_val"], parts["outer_test"]
    official = parts["official_test"]
    iso = recalibrate(
        np.concatenate(val.predictions),
        np.minimum(np.concatenate(val.rul), cap),
        np.concatenate(official.predictions),
    )
    sizes = np.cumsum([len(p) for p in official.predictions])[:-1]
    sse_iso, n_late = late_squared_errors(official.rul, np.split(iso, sizes))
    _, legacy = legacy_booking(val.booking(), official.booking(), costs, thresholds)
    per_official = pd.DataFrame(
        {
            "unit_id": official.units,
            "sse_late_iso": sse_iso,
            "n_late": n_late,
            "legacy_cost": legacy,
        }
    )
    policies, rows = [], []
    for alpha in alphas:
        tuned = tune_r1(ema_all(val.predictions, alpha), lead, cost_ratio, levels)
        scores = ema_all(outer.predictions, alpha)
        found = outcomes(
            alarm_times(scores, np.array([tuned.theta])), outer.lives, lead
        )
        policies.append(
            {"alpha": alpha, "theta": tuned.theta, "tuned_cost_rate": tuned.cost_rate}
        )
        rows.append(
            pd.DataFrame(
                {
                    "alpha": alpha,
                    "unit_id": outer.units,
                    "failed": found.failed[0],
                    "used": found.used[0],
                }
            )
        )
        per_official[f"sq_last_ema_{alpha:g}"] = last_cycle_squared_errors(
            ema_all(official.predictions, alpha), official.rul, cap
        )
    return {
        "official": per_official,
        "outer": pd.concat(rows, ignore_index=True),
        "policies": pd.DataFrame(policies),
    }
