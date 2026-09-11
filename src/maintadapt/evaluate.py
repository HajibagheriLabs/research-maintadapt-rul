"""RMSE, the asymmetric NASA scoring function and aggregation across seeds."""

import numpy as np
import pandas as pd

# PHM08 constants: an early prediction by d cycles costs exp(d/13) - 1, a late one exp(d/10) - 1
EARLY_SCALE = 13.0
LATE_SCALE = 10.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    t, p = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Summed PHM08 score; late predictions (predicted RUL above the truth) cost more."""
    t, p = _pair(y_true, y_pred)
    error = p - t
    penalty = np.where(
        error < 0, np.exp(-error / EARLY_SCALE), np.exp(error / LATE_SCALE)
    )
    return float(np.sum(penalty - 1.0))


def summarise(runs: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean and sample standard deviation of each metric across seeds, one row per group."""
    grouped = runs.groupby(by, sort=False)
    table = grouped[metrics].agg(["mean", "std"])
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
    table["n_seeds"] = grouped["seed"].nunique()
    return table.reset_index()


def _pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if t.ndim != 1 or t.shape != p.shape or t.size == 0:
        raise ValueError(
            f"expected two equal-length 1-D arrays, got {t.shape} and {p.shape}"
        )
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError("targets and predictions must be finite")
    return t, p
