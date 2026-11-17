"""Rules R3 and R4 of paper 2, on the predictive quantiles of the probabilistic families.

R3 alarms at the first cycle whose lower quantile falls to a tuned threshold: rule R1 applied to
a quantile instead of the point prediction, with the same candidates and tie rule. R4 needs no
tuning: it alarms at the first cycle with P(RUL <= L) >= p*, where p* follows from the cost
model (DECISIONS.md, "Paper 2: the plug-in probability of rule R4"). Both reuse the frozen
alarm and outcome code of policies.py unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from maintadapt.research.policies import (
    Outcomes,
    TunedPolicy,
    alarm_times,
    lives_of,
    outcomes,
    tune,
)
from maintadapt.research.probabilistic import LEVELS, cdf_at

R3_LEVELS = (0.05, 0.1, 0.2)


def quantile_scores(
    quantiles: Sequence[np.ndarray], level: float, levels: Sequence[float] = LEVELS
) -> list[np.ndarray]:
    """Each engine's trajectory of one stored quantile; quantiles are cycles x levels."""
    j = list(levels).index(level)
    return [np.asarray(q, dtype=float)[:, j] for q in quantiles]


def tune_r3(
    quantiles: Sequence[np.ndarray],
    level: float,
    lead: int,
    cost_ratios: Sequence[float],
    threshold_levels: int,
) -> dict[float, TunedPolicy]:
    """R3 thresholds tuned on complete trajectories, per cost ratio."""
    return tune(
        quantile_scores(quantiles, level),
        "R1",
        [1.0],
        lead,
        cost_ratios,
        threshold_levels,
    )


def apply_r3(
    quantiles: Sequence[np.ndarray], level: float, policy: TunedPolicy, lead: int
) -> Outcomes:
    scores = quantile_scores(quantiles, level)
    return outcomes(
        alarm_times(scores, np.array([policy.theta])), lives_of(scores), lead
    )


def plugin_probability(cost_ratio: float, lead: int, mean_life: float) -> float:
    """p* = (L + 1) g / (K - 1) with g = 1 / mean life, the cost rate of perfect prognostics.

    Derived in DECISIONS.md: waiting one more cycle gains a cycle worth g and adds the chance
    of failing in that cycle times K - 1; spreading P(RUL <= L) evenly over the L + 1 integer
    values 0..L turns the comparison into a bound on P(RUL <= L). Capped at 1.
    """
    if cost_ratio <= 1.0:
        raise ValueError(
            f"a failure must cost more than maintenance, got K = {cost_ratio}"
        )
    if lead < 0 or mean_life <= 0:
        raise ValueError(
            f"need L >= 0 and a positive mean life, got {lead}, {mean_life}"
        )
    return min(1.0, (lead + 1) / ((cost_ratio - 1.0) * mean_life))


def failure_probabilities(
    quantiles: Sequence[np.ndarray], lead: int, levels: Sequence[float] = LEVELS
) -> list[np.ndarray]:
    """Each engine's trajectory of P(RUL_t <= L) read from its stored quantiles."""
    return [cdf_at(np.asarray(q, dtype=float), float(lead), levels) for q in quantiles]


def apply_r4(quantiles: Sequence[np.ndarray], lead: int, p_star: float) -> Outcomes:
    """R4: alarm at the first cycle with P(RUL_t <= L) >= p*, maintenance L cycles later."""
    probabilities = failure_probabilities(quantiles, lead)
    # alarm_times fires on score <= theta; -P <= -p* is P >= p*
    scores = [-p for p in probabilities]
    return outcomes(alarm_times(scores, np.array([-p_star])), lives_of(scores), lead)


def scarce_tuning_sets(
    units: Sequence[int], failures: int, draws: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """draws subsets of `failures` run-to-failure engines, each engine one observed failure."""
    units = np.asarray(units)
    if failures > len(units):
        raise ValueError(f"{failures} failures requested from {len(units)} engines")
    return [
        np.sort(rng.choice(units, size=failures, replace=False)) for _ in range(draws)
    ]
