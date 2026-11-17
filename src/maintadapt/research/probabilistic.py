"""Probabilistic families for paper 2: predictive quantiles stored with every prediction.

Not part of the pre-registered study of PREREGISTRATION.md; the fits go to an archive of their
own (runs/research/archive_prob) and never touch the confirmatory one. Three families, each
built on a configuration of the frozen pool so that the point model is known:
- xgboost_quantile: the legacy quantile XGBoost, one booster for every level, on the settings
  of xgboost-00;
- mc_dropout: cnn-00 trained once and sampled with dropout left on;
- ensemble: five independently seeded fits of cnn-00, read as a normal distribution with the
  members' mean and standard deviation.
Every family stores the same grid of levels, so the rules of paper 2 read any of them alike.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from maintadapt.baselines import fit_xgboost
from maintadapt.config import TrainingConfig, XGBoostConfig
from maintadapt.experiment import seed_everything
from maintadapt.research.archive import SCHEMA
from maintadapt.research.config import ResearchTrainingConfig
from maintadapt.research.fold_data import PREDICTED, FoldData
from maintadapt.research.models.registry import build_network
from maintadapt.sequence import predict, predict_mc_dropout, train_model

# dense in the lower tail, where rule R4 reads P(RUL <= L) at small probabilities
LEVELS = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    *(round(0.05 * k, 2) for k in range(1, 20)),
    0.975,
    0.99,
)
# family -> the pool configuration it is built on
BASES = {"xgboost_quantile": "xgboost-00", "mc_dropout": "cnn-00", "ensemble": "cnn-00"}
MC_SAMPLES = 100
ENSEMBLE_MEMBERS = 5
# a member spread of zero would make every quantile the mean
MIN_SD = 1e-6


def column(level: float) -> str:
    """Archive column of a level, in ten-thousandths: 0.025 -> q0250."""
    return f"q{round(level * 10000):04d}"


QUANTILE_COLUMNS = tuple(column(level) for level in LEVELS)


def monotone(quantiles: np.ndarray, cap: float) -> np.ndarray:
    """Quantiles clipped to [0, cap] and sorted along each row, so they never cross."""
    return np.sort(np.clip(quantiles, 0.0, cap), axis=1)


def quantiles_from_draws(
    draws: np.ndarray, levels: Sequence[float] = LEVELS
) -> np.ndarray:
    """samples x n Monte Carlo draws -> n x levels empirical quantiles."""
    return np.quantile(draws, np.asarray(levels), axis=0).T


def gaussian_quantiles(
    mean: np.ndarray, sd: np.ndarray, levels: Sequence[float] = LEVELS
) -> np.ndarray:
    """n x levels quantiles of normal distributions."""
    z = norm.ppf(np.asarray(levels))
    return np.asarray(mean)[:, None] + np.maximum(np.asarray(sd), MIN_SD)[:, None] * z


def cdf_at(
    quantiles: np.ndarray, x: np.ndarray | float, levels: Sequence[float] = LEVELS
) -> np.ndarray:
    """P(RUL <= x) per row, by linear interpolation between stored quantiles.

    Below the lowest stored quantile the probability is 0 and at or above the highest it is
    1, so probabilities under the lowest level (0.001) cannot be told apart.
    """
    q = np.asarray(quantiles, dtype=float)
    a = np.asarray(levels, dtype=float)
    x = np.broadcast_to(np.asarray(x, dtype=float), (len(q),))
    # quantiles at or below x; with ties this counts every tied level
    count = (q <= x[:, None]).sum(axis=1)
    out = np.where(count == 0, 0.0, 1.0)
    inside = (count > 0) & (count < len(a))
    rows = np.flatnonzero(inside)
    lo = count[inside] - 1
    q_lo, q_hi = q[rows, lo], q[rows, lo + 1]
    share = np.where(
        q_hi > q_lo, (x[inside] - q_lo) / np.where(q_hi > q_lo, q_hi - q_lo, 1), 0
    )
    out[inside] = a[lo] + (a[lo + 1] - a[lo]) * share
    return out


@dataclass(frozen=True, eq=False)
class ProbOutput:
    """Point predictions, quantiles and any extra columns of every predicted split."""

    point: dict[str, np.ndarray]
    quantiles: dict[str, np.ndarray]
    extra: dict[str, dict[str, np.ndarray]]
    val_rmse: float
    fit_seconds: float
    best_iteration: int | None
    n_inputs: int


def _xgboost_settings(params: dict[str, Any]) -> XGBoostConfig:
    return XGBoostConfig(**{f.name: params[f.name] for f in fields(XGBoostConfig)})


def _quantile_xgboost(
    params: dict[str, Any],
    data: FoldData,
    seed: int,
    device: torch.device,
    threads: int,
) -> tuple[dict, dict, dict, int, int]:
    model = fit_xgboost(
        data.rolling("inner_fit"),
        data.target("inner_fit"),
        data.rolling("inner_val"),
        data.target("inner_val"),
        _xgboost_settings(params),
        seed,
        objective="reg:quantileerror",
        quantile_alpha=np.asarray(LEVELS),
        reg_lambda=params["reg_lambda"],
        n_jobs=threads,
        device="cuda" if device.type == "cuda" else "cpu",
    )
    # host inputs reach a GPU booster through a DMatrix; XGBoost warns about the copy
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*mismatched devices.*")
        quantiles = {
            split: monotone(model.predict(data.rolling(split)), data.cap)
            if len(data.frame(split))
            else np.empty((0, len(LEVELS)))
            for split in PREDICTED
        }
    median = LEVELS.index(0.5)
    point = {split: q[:, median] for split, q in quantiles.items()}
    n_inputs = data.rolling("inner_fit").shape[1]
    return point, quantiles, {}, int(model.best_iteration), n_inputs


def _train_cnn(
    params: dict[str, Any],
    data: FoldData,
    seed: int,
    device: torch.device,
    training: ResearchTrainingConfig,
) -> tuple[torch.nn.Module, int, Callable[[str], np.ndarray]]:
    seed_everything(seed)

    def inputs(split: str) -> np.ndarray:
        return data.windows(split, params["window"])

    x_fit = inputs("inner_fit")
    model = build_network("cnn", x_fit.shape[-1], params)
    cfg = TrainingConfig(
        batch_size=params["batch_size"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        max_epochs=training.max_epochs,
        patience=training.patience,
        grad_clip=training.grad_clip,
    )
    result = train_model(
        model,
        (x_fit, data.target("inner_fit").to_numpy()),
        (inputs("inner_val"), data.target("inner_val").to_numpy()),
        cfg,
        data.cap,
        device,
        None,
    )
    return result.model, result.best_epoch, inputs


def _mc_dropout(
    params: dict[str, Any],
    data: FoldData,
    seed: int,
    device: torch.device,
    training: ResearchTrainingConfig,
) -> tuple[dict, dict, dict, int, int]:
    model, best, inputs = _train_cnn(params, data, seed, device, training)
    point, quantiles = {}, {}
    # the dropout masks are drawn from the fit's seed, so a rerun gives the same draws
    torch.manual_seed(seed)
    for split in PREDICTED:
        x = inputs(split)
        if len(x) == 0:
            point[split], quantiles[split] = np.empty(0), np.empty((0, len(LEVELS)))
            continue
        point[split] = predict(model, x, data.cap, device, params["batch_size"])
        draws = predict_mc_dropout(
            model, x, data.cap, device, params["batch_size"], MC_SAMPLES
        )
        quantiles[split] = monotone(quantiles_from_draws(draws), data.cap)
    return point, quantiles, {}, best, int(inputs("inner_fit").shape[-1])


def _ensemble(
    params: dict[str, Any],
    data: FoldData,
    seed: int,
    device: torch.device,
    training: ResearchTrainingConfig,
) -> tuple[dict, dict, dict, int, int]:
    members, best = [], []
    for k in range(ENSEMBLE_MEMBERS):
        model, epoch, inputs = _train_cnn(
            params, data, member_seed(seed, k), device, training
        )
        best.append(epoch)
        members.append(
            {
                split: predict(
                    model, inputs(split), data.cap, device, params["batch_size"]
                )
                if len(data.frame(split))
                else np.empty(0)
                for split in PREDICTED
            }
        )
    point, quantiles, extra = {}, {}, {}
    for split in PREDICTED:
        stack = np.stack([m[split] for m in members])
        point[split] = stack.mean(axis=0)
        sd = stack.std(axis=0, ddof=1)
        quantiles[split] = monotone(gaussian_quantiles(point[split], sd), data.cap)
        extra[split] = {f"member{k}": stack[k] for k in range(ENSEMBLE_MEMBERS)}
    return (
        point,
        quantiles,
        extra,
        int(np.median(best)),
        int(inputs("inner_fit").shape[-1]),
    )


def member_seed(seed: int, member: int) -> int:
    """Seed of ensemble member k: distinct for every member, 31 bits like fit_seed."""
    return int(np.random.default_rng([seed, member]).integers(0, 2**31 - 1))


TRAINERS = {
    "xgboost_quantile": _quantile_xgboost,
    "mc_dropout": _mc_dropout,
    "ensemble": _ensemble,
}


def fit_probabilistic(
    family: str,
    params: dict[str, Any],
    data: FoldData,
    seed: int,
    device: torch.device,
    training: ResearchTrainingConfig,
) -> ProbOutput:
    """Fit one probabilistic family on inner_fit and predict every predicted split."""
    if family not in TRAINERS:
        raise ValueError(f"unknown family {family!r}; expected one of {list(TRAINERS)}")
    threads = (
        training.gpu_threads if device.type == "cuda" else training.resolved_cpu_jobs()
    )
    torch.set_num_threads(threads)
    start = time.perf_counter()
    with threadpool_limits(limits=threads):
        if family == "xgboost_quantile":
            seed_everything(seed)
            result = _quantile_xgboost(params, data, seed, device, threads)
        else:
            result = TRAINERS[family](params, data, seed, device, training)
    point, quantiles, extra, best, n_inputs = result
    truth = data.target("inner_val").to_numpy()
    val_rmse = float(np.sqrt(np.mean((point["inner_val"] - truth) ** 2)))
    return ProbOutput(
        point=point,
        quantiles=quantiles,
        extra=extra,
        val_rmse=val_rmse,
        fit_seconds=time.perf_counter() - start,
        best_iteration=best,
        n_inputs=n_inputs,
    )


def probabilistic_table(data: FoldData, out: ProbOutput) -> pa.Table:
    """The archive's columns, then one float32 column per level and any extra columns."""
    parts = []
    for split in PREDICTED:
        frame = data.frame(split)
        columns = {
            "unit_id": frame["unit_id"].to_numpy(np.int32),
            "cycle": frame["cycle"].to_numpy(np.int32),
            "rul": frame["rul"].to_numpy(np.int32),
            "split": split,
            "pred": np.asarray(out.point[split], dtype=np.float32),
        }
        q = np.asarray(out.quantiles[split], dtype=np.float32).reshape(len(frame), -1)
        columns |= {name: q[:, j] for j, name in enumerate(QUANTILE_COLUMNS)}
        columns |= {
            name: np.asarray(v, dtype=np.float32)
            for name, v in out.extra.get(split, {}).items()
        }
        parts.append(pd.DataFrame(columns))
    table = pd.concat(parts, ignore_index=True)
    table["split"] = pd.Categorical(table["split"], categories=PREDICTED)
    extra_names = [
        c for c in table.columns if c not in SCHEMA.names and c not in QUANTILE_COLUMNS
    ]
    schema = pa.schema(
        [
            *SCHEMA,
            *[(name, pa.float32()) for name in QUANTILE_COLUMNS],
            *[(name, pa.float32()) for name in extra_names],
        ]
    )
    return pa.Table.from_pandas(table, schema=schema, preserve_index=False)
