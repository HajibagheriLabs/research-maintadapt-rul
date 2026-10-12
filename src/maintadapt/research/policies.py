"""Alarm policies on complete run-to-failure trajectories: scores, alarms, outcomes, tuning.

An engine is a 1-D array of predictions at cycles 1..T, so its life T is the array's length.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

RULES = ("R1", "R2")
# "no alarm": larger than any life plus lead time, so the engine fails without a special case
NO_ALARM = np.int64(2**40)


@dataclass(frozen=True, eq=False)
class Outcomes:
    """One row per threshold, one column per engine."""

    failed: np.ndarray
    # cycles the engine ran: the maintenance cycle m, or its life T when it failed
    used: np.ndarray
    # cycles of life thrown away by preventive maintenance, T - m; zero after a failure
    waste: np.ndarray


@dataclass(frozen=True)
class TunedPolicy:
    theta: float
    alpha: float
    # cost rate on the engines the policy was tuned on
    cost_rate: float


def ema(predictions: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA: s_1 = y_1, s_t = alpha * y_t + (1 - alpha) * s_(t-1)."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
    values = np.asarray(predictions, dtype=float)
    if alpha == 1.0:
        return values
    return pd.Series(values).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def scores(
    engines: Sequence[np.ndarray], rule: str, alpha: float = 1.0
) -> list[np.ndarray]:
    """R1 scores each cycle by its prediction; R2 by the causal EMA of the predictions."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    if rule == "R1":
        return [np.asarray(e, dtype=float) for e in engines]
    return [ema(e, alpha) for e in engines]


def alarm_times(scores: Sequence[np.ndarray], thresholds: np.ndarray) -> np.ndarray:
    """First cycle (1-based) with score <= theta, per threshold and engine; NO_ALARM if none."""
    thresholds = np.asarray(thresholds, dtype=float)
    alarms = np.empty((len(thresholds), len(scores)), dtype=np.int64)
    for j, s in enumerate(scores):
        # the first cycle at or below theta is the first where the running minimum is
        falling = -np.minimum.accumulate(np.asarray(s, dtype=float))
        first = np.searchsorted(falling, -thresholds, side="left")
        alarms[:, j] = np.where(first < len(s), first + 1, NO_ALARM)
    return alarms


def outcomes(alarms: np.ndarray, lives: np.ndarray, lead: int) -> Outcomes:
    """Maintenance at m = a + L is preventive when m <= T; otherwise the engine fails at T."""
    if lead < 0:
        raise ValueError(f"lead time must be non-negative, got {lead}")
    lives = np.asarray(lives, dtype=np.int64)
    m = np.asarray(alarms, dtype=np.int64) + lead
    preventive = m <= lives
    return Outcomes(
        failed=~preventive,
        used=np.where(preventive, m, lives),
        waste=np.where(preventive, lives - m, 0),
    )


def lives_of(engines: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([len(e) for e in engines], dtype=np.int64)


def threshold_candidates(scores: Sequence[np.ndarray], levels: int) -> np.ndarray:
    """Inverted-CDF quantiles of the pooled scores at equally spaced levels, plus -inf, +inf.

    Every quantile is one of the scores, so a strictly increasing transform of the scores
    transforms the candidates with them and leaves every alarm unchanged.
    """
    pooled = np.concatenate([np.asarray(s, dtype=float) for s in scores])
    quantiles = np.quantile(
        pooled, np.linspace(0.0, 1.0, levels), method="inverted_cdf"
    )
    return np.concatenate([[-np.inf], quantiles, [np.inf]])


def median_minimiser(values: np.ndarray) -> int:
    """Index of the median of the minimising entries, the lower one for an even count."""
    best = np.flatnonzero(values == values.min())
    return int(best[(len(best) - 1) // 2])


def apply_policy(
    engines: Sequence[np.ndarray], rule: str, policy: TunedPolicy, lead: int
) -> Outcomes:
    """Outcomes of a tuned policy on engines it may never have seen; one row."""
    s = scores(engines, rule, policy.alpha)
    return outcomes(alarm_times(s, np.array([policy.theta])), lives_of(engines), lead)


def tune(
    engines: Sequence[np.ndarray],
    rule: str,
    alphas: Sequence[float],
    lead: int,
    cost_ratios: Sequence[float],
    levels: int,
) -> dict[float, TunedPolicy]:
    """The threshold (and for R2 the alpha) with the lowest pooled cost rate, per cost ratio.

    Outcomes are computed once per alpha for every candidate threshold. Ties between
    thresholds go to the median minimiser; ties between alphas to the larger alpha, the
    lighter smoothing.
    """
    # imported here: costs builds on this module's outcomes
    from maintadapt.research.costs import cost_rate

    if rule not in RULES or (rule == "R2" and not alphas):
        raise ValueError(f"tuning needs a rule in {RULES} and, for R2, some alphas")
    lives = lives_of(engines)
    trials = []
    for alpha in sorted([1.0] if rule == "R1" else set(alphas), reverse=True):
        s = scores(engines, rule, alpha)
        candidates = threshold_candidates(s, levels)
        trials.append(
            (alpha, candidates, outcomes(alarm_times(s, candidates), lives, lead))
        )
    tuned = {}
    for k in cost_ratios:
        best: TunedPolicy | None = None
        for alpha, candidates, found in trials:
            rates = cost_rate(found, k)
            i = median_minimiser(rates)
            if best is None or rates[i] < best.cost_rate:
                best = TunedPolicy(float(candidates[i]), alpha, float(rates[i]))
        tuned[k] = best
    return tuned
