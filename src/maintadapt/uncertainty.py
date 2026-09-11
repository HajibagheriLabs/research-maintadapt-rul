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
