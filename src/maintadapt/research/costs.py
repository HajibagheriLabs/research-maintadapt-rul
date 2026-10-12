"""Renewal-reward cost rates, the reference policies and the savings share."""

from __future__ import annotations

import numpy as np

from maintadapt.research.policies import (
    NO_ALARM,
    Outcomes,
    median_minimiser,
    outcomes,
)


def engine_costs(found: Outcomes, cost_ratio: float) -> np.ndarray:
    """Cost of each engine: 1 for preventive maintenance, K for a failure."""
    if cost_ratio <= 1.0:
        raise ValueError(
            f"a failure must cost more than maintenance, got K = {cost_ratio}"
        )
    return np.where(found.failed, float(cost_ratio), 1.0)


def cost_rate(found: Outcomes, cost_ratio: float) -> np.ndarray:
    """Sum of costs over sum of cycles used, pooled over engines (the last axis)."""
    return engine_costs(found, cost_ratio).sum(axis=-1) / found.used.sum(axis=-1)


def perfect_rate(lives: np.ndarray) -> float:
    """CR*: every engine maintained preventively at the end of its life, N / sum of T."""
    lives = np.asarray(lives)
    return len(lives) / float(lives.sum())


def age_outcomes(age: float, lives: np.ndarray) -> Outcomes:
    """Maintenance at a fixed age A: preventive when A <= T, otherwise a failure at T."""
    alarm = NO_ALARM if np.isinf(age) else np.int64(age)
    return outcomes(np.array([[alarm]]), np.asarray(lives)[None, :], 0)


def tune_age(lifetimes: np.ndarray, cost_ratio: float) -> float:
    """A* minimising the cost rate of age replacement on the given lifetimes.

    Between two consecutive lifetimes the failures are fixed and the cycles used grow with A,
    so the optimum is a lifetime itself, or never maintaining (A = inf). Ties go to the median
    minimiser.
    """
    lifetimes = np.asarray(lifetimes)
    candidates = np.concatenate([np.unique(lifetimes).astype(float), [np.inf]])
    rates = np.array(
        [cost_rate(age_outcomes(a, lifetimes), cost_ratio)[0] for a in candidates]
    )
    return float(candidates[median_minimiser(rates)])


def savings_share(
    rate: np.ndarray | float, age_rate: np.ndarray | float, perfect: np.ndarray | float
) -> np.ndarray | float:
    """S = (CR_age - CR) / (CR_age - CR*): 0 matches age replacement, 1 perfect prognostics."""
    return (np.asarray(age_rate) - rate) / (np.asarray(age_rate) - perfect)


def regret(found: Outcomes, failure_cost: float) -> np.ndarray:
    """Per-engine regret in cycles: C_f for a failure plus the life thrown away."""
    return failure_cost * found.failed + found.waste
