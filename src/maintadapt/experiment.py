"""Seeding, run directories and MLflow logging shared by the experiment scripts."""

from __future__ import annotations

import random
from pathlib import Path

import mlflow
import numpy as np
import torch

from maintadapt.config import Config, save_config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN would otherwise choose kernels by timing, which varies results between runs
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def start_run_dir(cfg: Config, name: str) -> Path:
    """Create runs/<name> holding the exact config used, and log to MLflow experiment `name`."""
    path = Path(cfg.paths.runs) / name
    path.mkdir(parents=True, exist_ok=True)
    save_config(cfg, path / "config.yaml")
    store = Path(cfg.paths.mlruns).resolve()
    store.mkdir(parents=True, exist_ok=True)
    # MLflow 3 freezes its plain-file backend, so runs go to SQLite inside the same folder
    mlflow.set_tracking_uri(f"sqlite:///{(store / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment(name)
    return path


def log_run(run_name: str, params: dict, metrics: dict) -> None:
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
