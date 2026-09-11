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
        per_engine = np.where(
            self.failed, costs.failure, costs.per_cycle * self.discarded
        )
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
        # a history cut before failure keeps its last booking, due when that prediction reaches the threshold
        delay = predicted[-1] - thresholds[~alarmed]
        late = delay > true_rul[-1]
        failed[~alarmed, j] = late
        discarded[~alarmed, j] = np.where(late, 0.0, true_rul[-1] - delay)
    return Outcomes(failed, discarded)


def fixed_schedule(lifetimes: np.ndarray, ages: np.ndarray) -> Outcomes:
    """Maintain every engine after a fixed number of cycles; shorter-lived engines fail first."""
    life = np.asarray(lifetimes, dtype=float)[None, :]
    age = np.asarray(ages, dtype=float)[:, None]
    failed = life < age
    return Outcomes(failed, np.where(failed, 0.0, life - age))
