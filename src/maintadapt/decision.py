"""Cost model and alarm policy that turn RUL predictions into maintenance decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    failure: float
    per_cycle: float

    def __post_init__(self) -> None:
        if self.failure <= 0 or self.per_cycle <= 0:
            raise ValueError("failure and per-cycle costs must both be positive")


@dataclass(frozen=True, eq=False)
class Outcomes:
    failed: np.ndarray
    discarded: np.ndarray

    def cost(self, costs: CostModel) -> np.ndarray:
        """Expected cost per engine for every threshold row."""
        # linear in the outcomes, so failure shares averaged over snapshots price correctly
        per_engine = costs.failure * self.failed + costs.per_cycle * self.discarded
        return per_engine.mean(axis=-1)


def replay_alarms(
    engines: Sequence[tuple[np.ndarray, np.ndarray]],
    thresholds: np.ndarray,
    *,
    run_to_failure: bool,
) -> Outcomes:
    """Service each engine at the first cycle its predicted RUL reaches the threshold."""
    failed = np.zeros((len(thresholds), len(engines)), dtype=bool)
    discarded = np.zeros((len(thresholds), len(engines)))
    for j, (predicted, true_rul) in enumerate(engines):
        # the running minimum only falls, so the first crossing is a sorted search
        crossing = -np.minimum.accumulate(predicted)
        first = np.searchsorted(crossing, -thresholds, side="left")
        alarmed = first < len(predicted)
        discarded[alarmed, j] = true_rul[first[alarmed]]
        if run_to_failure:
            failed[~alarmed, j] = True
            continue
        # a history cut before failure keeps its last booking, due once that prediction hits the threshold
        delay = predicted[-1] - thresholds[~alarmed]
        late = delay > true_rul[-1]
        failed[~alarmed, j] = late
        discarded[~alarmed, j] = np.where(late, 0.0, true_rul[-1] - delay)
    return Outcomes(failed, discarded)


def replay_snapshots(
    engines: Sequence[tuple[np.ndarray, np.ndarray]], thresholds: np.ndarray
) -> Outcomes:
    """Mean outcome per engine when its run-to-failure history is cut at every cycle in turn."""
    failed = np.zeros((len(thresholds), len(engines)))
    discarded = np.zeros((len(thresholds), len(engines)))
    for j, (predicted, true_rul) in enumerate(engines):
        crossing = -np.minimum.accumulate(predicted)
        first = np.searchsorted(crossing, -thresholds, side="left")
        cut = np.arange(len(predicted))
        alarmed = first[:, None] <= cut[None, :]
        delay = predicted[None, :] - thresholds[:, None]
        late = ~alarmed & (delay > true_rul[None, :])
        at_alarm = true_rul[np.minimum(first, len(predicted) - 1)][:, None]
        kept = np.where(alarmed, at_alarm, true_rul[None, :] - delay)
        failed[:, j] = late.mean(axis=1)
        discarded[:, j] = np.where(late, 0.0, kept).mean(axis=1)
    return Outcomes(failed, discarded)


def fixed_schedule(lifetimes: np.ndarray, ages: np.ndarray) -> Outcomes:
    """Maintain every engine after a fixed number of cycles; shorter-lived engines fail first."""
    life = np.asarray(lifetimes, dtype=float)[None, :]
    age = np.asarray(ages, dtype=float)[:, None]
    failed = life < age
    return Outcomes(failed, np.where(failed, 0.0, life - age))
