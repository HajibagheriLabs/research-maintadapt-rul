"""One interface over the ten model families: fit on a fold, predict every cycle."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from maintadapt.baselines import fit_ridge, fit_xgboost
from maintadapt.config import TrainingConfig, XGBoostConfig
from maintadapt.evaluate import rmse
from maintadapt.experiment import seed_everything
from maintadapt.research.config import ResearchTrainingConfig
from maintadapt.research.fold_data import PREDICTED, FoldData
from maintadapt.research.models.networks import (
    CNNRegressor,
    MLPRegressor,
    TransformerRegressor,
)
from maintadapt.research.models.spaces import FAMILIES
from maintadapt.sequence import LSTMRegressor, TCNRegressor, predict, train_model

CPU_FAMILIES = ("ridge", "knn", "forest", "xgboost", "xgboost_raw")
NEURAL_FAMILIES = ("mlp", "lstm", "tcn", "cnn", "transformer")

# builds a model's inputs for one split of a fold
Inputs = Callable[[FoldData, str], Any]
Predictor = Callable[[Any], np.ndarray]


@dataclass(frozen=True, eq=False)
class Fitted:
    """A trained model with the inputs it reads; it predicts any split of any FoldData."""

    family: str
    inputs: Inputs
    model: Predictor
    best_iteration: int | None
    n_inputs: int

    def predict(self, data: FoldData, split: str) -> np.ndarray:
        """One prediction per row of the split's frame, in frame order, clipped to [0, cap]."""
        if data.frame(split).empty:
            return np.empty(0)
        values = np.asarray(self.model(self.inputs(data, split)), dtype=float)
        return np.clip(values, 0.0, data.cap)


@dataclass(frozen=True, eq=False)
class FitOutput:
    # inner_val, outer_test and official_test, one prediction per cycle in frame order
    predictions: dict[str, np.ndarray]
    # over every inner_val cycle against capped truth
    val_rmse: float
    # wall clock of training and prediction
    fit_seconds: float
    # boosting round or epoch kept by early stopping; None for families without one
    best_iteration: int | None
    n_inputs: int


@dataclass(frozen=True)
class _Run:
    seed: int
    device: torch.device
    training: ResearchTrainingConfig
    threads: int


_Trained = tuple[Inputs, Predictor, int | None]


def _rolling(data: FoldData, split: str) -> Any:
    return data.rolling(split)


def _raw(data: FoldData, split: str) -> Any:
    return data.raw(split)


def _ridge(params: dict[str, Any], data: FoldData, run: _Run) -> _Trained:
    model, _ = fit_ridge(
        data.rolling("inner_fit"),
        data.target("inner_fit"),
        data.rolling("inner_val"),
        data.target("inner_val"),
        [params["alpha"]],
    )
    return _rolling, model.predict, None


def _knn(params: dict[str, Any], data: FoldData, run: _Run) -> _Trained:
    model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            n_jobs=run.threads,
        ),
    ).fit(data.rolling("inner_fit"), data.target("inner_fit"))
    return _rolling, model.predict, None


def _forest(params: dict[str, Any], data: FoldData, run: _Run) -> _Trained:
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_features=params["max_features"],
        min_samples_leaf=params["min_samples_leaf"],
        max_depth=params["max_depth"],
        max_samples=params["max_samples"],
        random_state=run.seed,
        n_jobs=run.threads,
    ).fit(data.rolling("inner_fit"), data.target("inner_fit"))
    # threads add tree predictions in the order they finish, which moves the last bits
    # between identical calls; one thread sums the trees in a fixed order
    model.set_params(n_jobs=1)
    return _rolling, model.predict, None


def _xgboost(inputs: Inputs) -> Callable[[dict[str, Any], FoldData, _Run], _Trained]:
    def trainer(params: dict[str, Any], data: FoldData, run: _Run) -> _Trained:
        settings = XGBoostConfig(
            **{f.name: params[f.name] for f in fields(XGBoostConfig)}
        )
        model = fit_xgboost(
            inputs(data, "inner_fit"),
            data.target("inner_fit"),
            inputs(data, "inner_val"),
            data.target("inner_val"),
            settings,
            run.seed,
            reg_lambda=params["reg_lambda"],
            n_jobs=run.threads,
        )
        # predict() then stops at the best round as well
        return inputs, model.predict, int(model.best_iteration)

    return trainer


def build_network(family: str, n_inputs: int, params: dict[str, Any]) -> nn.Module:
    """The untrained network of a neural configuration."""
    p = params
    if family == "mlp":
        return MLPRegressor(n_inputs, p["hidden"], p["dropout"])
    if family == "lstm":
        return LSTMRegressor(n_inputs, p["hidden"], p["layers"], p["dropout"])
    if family == "tcn":
        return TCNRegressor(
            n_inputs, p["channels"], p["kernel_size"], p["dilations"], p["dropout"]
        )
    if family == "cnn":
        return CNNRegressor(
            n_inputs,
            p["window"],
            p["filters"],
            p["kernel_size"],
            p["conv_layers"],
            p["fc"],
            p["dropout"],
        )
    if family == "transformer":
        return TransformerRegressor(
            n_inputs,
            p["window"],
            p["d_model"],
            p["heads"],
            p["layers"],
            p["ff"],
            p["dropout"],
        )
    raise ValueError(f"{family!r} is not a neural family; expected {NEURAL_FAMILIES}")


def _train_network(
    family: str,
    inputs: Inputs,
    params: dict[str, Any],
    data: FoldData,
    run: _Run,
) -> _Trained:
    x_fit = inputs(data, "inner_fit")
    model = build_network(family, x_fit.shape[-1], params)
    cfg = TrainingConfig(
        batch_size=params["batch_size"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        max_epochs=run.training.max_epochs,
        patience=run.training.patience,
        grad_clip=run.training.grad_clip,
    )
    result = train_model(
        model,
        (x_fit, data.target("inner_fit").to_numpy()),
        (inputs(data, "inner_val"), data.target("inner_val").to_numpy()),
        cfg,
        data.cap,
        run.device,
        None,
    )

    def model_predict(x: np.ndarray) -> np.ndarray:
        return predict(result.model, x, data.cap, run.device, cfg.batch_size)

    return inputs, model_predict, result.best_epoch


def _mlp(params: dict[str, Any], data: FoldData, run: _Run) -> _Trained:
    scaler = StandardScaler().fit(data.rolling("inner_fit"))

    def inputs(fold: FoldData, split: str) -> np.ndarray:
        return scaler.transform(fold.rolling(split)).astype(np.float32)

    return _train_network("mlp", inputs, params, data, run)


def _sequence(family: str) -> Callable[[dict[str, Any], FoldData, _Run], _Trained]:
    def trainer(params: dict[str, Any], data: FoldData, run: _Run) -> _Trained:
        length = params["window"]

        def inputs(fold: FoldData, split: str) -> np.ndarray:
            return fold.windows(split, length)

        return _train_network(family, inputs, params, data, run)

    return trainer


TRAINERS: dict[str, Callable[[dict[str, Any], FoldData, _Run], _Trained]] = {
    "ridge": _ridge,
    "knn": _knn,
    "forest": _forest,
    "xgboost": _xgboost(_rolling),
    "xgboost_raw": _xgboost(_raw),
    "mlp": _mlp,
    "lstm": _sequence("lstm"),
    "tcn": _sequence("tcn"),
    "cnn": _sequence("cnn"),
    "transformer": _sequence("transformer"),
}


def check_params(family: str, params: dict[str, Any]) -> None:
    """Raise unless params set exactly the keys of the family's configurations."""
    specs = {f.name: f for f in FAMILIES}
    if family not in specs or family not in TRAINERS:
        raise ValueError(f"unknown family {family!r}; expected one of {list(TRAINERS)}")
    spec = specs[family]
    expected = set(spec.reference) | set(spec.shared)
    if set(params) != expected:
        raise ValueError(
            f"{family}: unknown keys {sorted(set(params) - expected)}, "
            f"missing keys {sorted(expected - set(params))}"
        )


def train(
    family: str,
    params: dict[str, Any],
    data: FoldData,
    seed: int,
    device: torch.device,
    training: ResearchTrainingConfig,
) -> Fitted:
    """Fit one configuration on inner_fit; inner_val serves early stopping only."""
    check_params(family, params)
    neural = family in NEURAL_FAMILIES
    if neural and device.type == "cuda":
        threads = training.gpu_torch_threads
    else:
        threads = training.resolved_cpu_jobs()
    if neural:
        torch.set_num_threads(threads)
    seed_everything(seed)
    inputs, model, best = TRAINERS[family](
        params, data, _Run(seed, device, training, threads)
    )
    n_inputs = int(inputs(data, "inner_fit").shape[-1])
    return Fitted(family, inputs, model, best, n_inputs)


def fit(
    family: str,
    params: dict[str, Any],
    data: FoldData,
    seed: int,
    device: torch.device,
    training: ResearchTrainingConfig,
) -> FitOutput:
    """Train one configuration and predict every cycle of inner_val, outer_test and official_test."""
    start = time.perf_counter()
    fitted = train(family, params, data, seed, device, training)
    predictions = {split: fitted.predict(data, split) for split in PREDICTED}
    seconds = time.perf_counter() - start
    return FitOutput(
        predictions=predictions,
        val_rmse=rmse(data.target("inner_val").to_numpy(), predictions["inner_val"]),
        fit_seconds=seconds,
        best_iteration=fitted.best_iteration,
        n_inputs=fitted.n_inputs,
    )
