"""Engine bootstrap, intervals, rank correlation, pairwise tests and multiplicity control."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def bootstrap_counts(n: int, n_boot: int, seed: int | Sequence[int]) -> np.ndarray:
    """B x N multinomial counts: how often each of N engines enters each replicate.

    A pooled sum over a resample is then counts @ per-engine values, for every replicate at
    once. One matrix serves every configuration and repeat that shares these engines.
    """
    if n < 1 or n_boot < 1:
        raise ValueError(f"need engines and replicates, got N = {n}, B = {n_boot}")
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)


def percentile_interval(samples: np.ndarray, ci: float, axis: int = 0) -> np.ndarray:
    """Lower and upper percentile bounds of the central ci interval, stacked on axis 0."""
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must lie in (0, 1), got {ci}")
    tail = (1.0 - ci) / 2.0
    return np.quantile(np.asarray(samples), [tail, 1.0 - tail], axis=axis)


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b, which corrects for ties in either ranking."""
    return float(kendalltau(np.asarray(x), np.asarray(y), variant="b").statistic)


def bootstrap_p_value(differences: np.ndarray) -> float:
    """Two-sided p-value of a zero difference from its bootstrap distribution.

    Twice the smaller tail share, each counted with one added to numerator and denominator,
    so that B replicates never report a p-value below 2 / (B + 1).
    """
    d = np.asarray(differences, dtype=float)
    below = (np.sum(d <= 0) + 1) / (len(d) + 1)
    above = (np.sum(d >= 0) + 1) / (len(d) + 1)
    return float(min(1.0, 2.0 * min(below, above)))


def pairwise_differences(
    estimates: pd.Series, samples: pd.DataFrame, ci: float
) -> pd.DataFrame:
    """Every pair a < b of items: estimate a - b, its percentile interval and p-value.

    samples holds one bootstrap replicate per row and one item per column.
    """
    rows = []
    for a, b in itertools.combinations(estimates.index, 2):
        d = samples[a].to_numpy() - samples[b].to_numpy()
        lo, hi = percentile_interval(d, ci)
        rows.append(
            {
                "a": a,
                "b": b,
                "difference": estimates[a] - estimates[b],
                "lo": lo,
                "hi": hi,
                "p_value": bootstrap_p_value(d),
            }
        )
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg adjusted p-values, and which hypotheses are rejected at level q."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    # step-up: each adjusted value is the smallest scaled value at or after its rank
    adjusted_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted <= q


def choose(
    validation: pd.DataFrame, metric: str, by: Sequence[str] = ("repeat", "fold")
) -> pd.DataFrame:
    """The configuration with the lowest validation metric in each (repeat, fold).

    Ties go to the configuration listed first, which keeps the choice deterministic.
    """
    order = validation.reset_index(drop=True)
    best = order.loc[order.groupby(list(by), sort=True)[metric].idxmin()]
    return best.reset_index(drop=True)


def selection_loss(
    s_by_cost: np.ndarray | float, s_by_rmse: np.ndarray | float
) -> np.ndarray | float:
    """S of the models chosen by validation cost minus S of those chosen by validation RMSE.

    Positive values mean that choosing by RMSE loses maintenance value.
    """
    return np.asarray(s_by_cost) - np.asarray(s_by_rmse)
