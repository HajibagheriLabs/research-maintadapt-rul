"""Classical baselines: ridge regression and gradient boosting on rolling features."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from maintadapt.config import XGBoostConfig
from maintadapt.evaluate import rmse


def fit_ridge(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    alphas: list[float],
) -> tuple[Pipeline, float]:
    """Fit ridge for every penalty and keep the one with the lowest validation RMSE."""
    best: tuple[Pipeline, float, float] | None = None
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
            x_train, y_train
        )
        score = rmse(y_val.to_numpy(), model.predict(x_val))
        if best is None or score < best[1]:
            best = (model, score, alpha)
    if best is None:
        raise ValueError("alphas must not be empty")
    return best[0], best[2]


def fit_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    params: XGBoostConfig,
    seed: int,
    **objective: object,
) -> XGBRegressor:
    """Gradient boosting with early stopping on validation units."""
    model = XGBRegressor(
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        max_depth=params.max_depth,
        min_child_weight=params.min_child_weight,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        early_stopping_rounds=params.early_stopping_rounds,
        tree_method="hist",
        random_state=seed,
        **objective,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model
