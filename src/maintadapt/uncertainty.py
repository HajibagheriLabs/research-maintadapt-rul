"""Prediction intervals and their empirical coverage."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from maintadapt.baselines import fit_xgboost
from maintadapt.config import XGBoostConfig


def interval_quantiles(levels: list[float]) -> list[float]:
    """Sorted lower and upper quantiles of every central interval level, plus the median."""
    tails = {round((1 - level) / 2, 6) for level in levels}
    return sorted(tails | {0.5} | {round(1 - t, 6) for t in tails})


def fit_quantile_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    params: XGBoostConfig,
    seed: int,
    quantiles: list[float],
) -> XGBRegressor:
    """One booster that predicts every requested quantile of RUL."""
    return fit_xgboost(
        x_train,
        y_train,
        x_val,
        y_val,
        params,
        seed,
        objective="reg:quantileerror",
        quantile_alpha=np.asarray(quantiles),
    )


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def conformal_margin(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, level: float
) -> float:
    """Split-conformal widening that makes [lower - m, upper + m] cover `level` on calibration data."""
    scores = np.sort(np.maximum(lower - y_true, y_true - upper))
    rank = math.ceil((len(scores) + 1) * level)
    if rank > len(scores):
        raise ValueError(
            f"{len(scores)} calibration points cannot certify level {level}"
        )
    return float(scores[rank - 1])


def sample_interval(draws: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Central interval of Monte Carlo draws with shape (samples, n)."""
    tail = (1 - level) / 2
    return np.quantile(draws, tail, axis=0), np.quantile(draws, 1 - tail, axis=0)


# ---------------------------------------------------------------- calibration for paper 2
#
# The legacy conformal_margin scores every validation cycle alike. On run-to-failure engines most
# cycles sit at the cap, where a clipped interval covers trivially, so the margin collapses
# (docs/legacy/ERRATA.md, a). The functions below score calibration engines at uniformly drawn
# cut points instead, as the official test engines are cut, and give every engine the same total
# weight, so long engines do not dominate. The finite-sample correction counts engines, the
# exchangeable unit, not cycles.

LATE_RUL = 50
TRUE_RUL_EDGES = (0, 25, 50, 75, 100, 125, math.inf)


def cut_points(
    lives: pd.Series, per_engine: int, rng: np.random.Generator
) -> pd.DataFrame:
    """per_engine cycles per engine, uniform on 1..T with replacement; lives is indexed by unit."""
    rows = [
        pd.DataFrame({"unit_id": unit, "cycle": rng.integers(1, life + 1, per_engine)})
        for unit, life in lives.items()
    ]
    return pd.concat(rows, ignore_index=True)


def engine_weights(units: np.ndarray) -> np.ndarray:
    """Weights summing to 1 that give every engine the same total, shared by its points."""
    units = np.asarray(units)
    _, inverse, counts = np.unique(units, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inverse])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """The smallest value whose cumulative weight reaches level (inverted CDF)."""
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    cumulative /= cumulative[-1]
    at = min(np.searchsorted(cumulative, level - 1e-12, side="left"), len(values) - 1)
    return float(np.asarray(values)[order][at])


def conformal_level(level: float, n_engines: int) -> float:
    """level * (1 + 1/E), the split-conformal correction with E exchangeable engines."""
    corrected = level * (1.0 + 1.0 / n_engines)
    if corrected > 1.0:
        raise ValueError(
            f"{n_engines} calibration engines cannot certify level {level}"
        )
    return corrected


def interval_scores(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Conformity scores of an interval: how far the truth lies outside it (negative inside)."""
    return np.maximum(np.asarray(lower) - y_true, np.asarray(y_true) - upper)


def matched_margin(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    units: np.ndarray,
    level: float,
) -> float:
    """Conformal margin from cut-point scores, every engine weighted equally.

    With quantile-regression bounds this is conformalised quantile regression (Romano et al.,
    2019); with a point prediction as both bounds it is the absolute-residual margin.
    """
    n_engines = len(np.unique(units))
    return weighted_quantile(
        interval_scores(y_true, lower, upper),
        engine_weights(units),
        conformal_level(level, n_engines),
    )


def rul_bins(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """Index of the bin [edges[i], edges[i+1]) that holds each value."""
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def mondrian_margins(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    point: np.ndarray,
    units: np.ndarray,
    edges: tuple[float, ...],
    level: float,
    min_engines: int,
) -> np.ndarray:
    """One matched margin per predicted-RUL bin; a bin with too few engines takes the pooled one.

    Binning by the prediction, which is known at test time, gives coverage conditional on the
    predicted RUL, where a single margin can over-cover early life and under-cover late life.
    """
    pooled = matched_margin(y_true, lower, upper, units, level)
    bins = rul_bins(point, edges)
    margins = np.full(len(edges) - 1, pooled)
    for b in range(len(edges) - 1):
        keep = bins == b
        if len(np.unique(units[keep])) >= min_engines:
            margins[b] = matched_margin(
                y_true[keep], lower[keep], upper[keep], units[keep], level
            )
    return margins


def quantile_shifts(
    y_true: np.ndarray, quantiles: np.ndarray, levels: np.ndarray, units: np.ndarray
) -> np.ndarray:
    """One-sided conformal shift per level, so that y <= q + shift with the level's share.

    quantiles is points x levels. Each column is calibrated on its own (a conformalised
    quantile per level), with engines weighted equally; adding the shifts and sorting each row
    gives a monotone, calibrated quantile function.
    """
    weights = engine_weights(units)
    return np.array(
        [
            weighted_quantile(y_true - quantiles[:, j], weights, level)
            for j, level in enumerate(levels)
        ]
    )


def coverage_by_true_rul(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    edges: tuple[float, ...] = TRUE_RUL_EDGES,
) -> pd.DataFrame:
    """Coverage and mean width per true-RUL bin, plus the late phase (true RUL <= 50)."""
    y_true, lower, upper = (np.asarray(a, dtype=float) for a in (y_true, lower, upper))
    inside = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    bins = rul_bins(y_true, edges)
    rows = [
        {
            "bin": f"[{edges[b]:g}, {edges[b + 1]:g})",
            "n": int((bins == b).sum()),
            "coverage": float(inside[bins == b].mean())
            if (bins == b).any()
            else np.nan,
            "width": float(width[bins == b].mean()) if (bins == b).any() else np.nan,
        }
        for b in range(len(edges) - 1)
    ]
    late = y_true <= LATE_RUL
    rows.append(
        {
            "bin": f"true RUL <= {LATE_RUL}",
            "n": int(late.sum()),
            "coverage": float(inside[late].mean()) if late.any() else np.nan,
            "width": float(width[late].mean()) if late.any() else np.nan,
        }
    )
    return pd.DataFrame(rows)
