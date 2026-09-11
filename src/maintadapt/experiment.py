"""Seeding, run directories, resumable progress and MLflow logging for experiment scripts."""

from __future__ import annotations

import ctypes
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from maintadapt.config import Config, load_config, save_config

PROGRESS = "progress.csv"


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
    snapshot, progress = path / "config.yaml", path / PROGRESS
    # rows recorded under another config are stale, so only an unchanged experiment resumes
    if progress.exists() and (not snapshot.exists() or load_config(snapshot) != cfg):
        progress.unlink()
    save_config(cfg, snapshot)
    store = Path(cfg.paths.mlruns).resolve()
    store.mkdir(parents=True, exist_ok=True)
    # MLflow 3 freezes its plain-file backend, so runs go to SQLite inside the same folder
    mlflow.set_tracking_uri(f"sqlite:///{(store / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment(name)
    return path


def finished_runs(run_dir: Path) -> list[dict]:
    """Rows already recorded in this run directory, so an interrupted experiment resumes."""
    progress = run_dir / PROGRESS
    return pd.read_csv(progress).to_dict("records") if progress.exists() else []


def record_run(
    run_dir: Path, run_name: str, row: dict, params: dict, metrics: dict
) -> None:
    """Persist one finished run to the progress file and to MLflow."""
    progress = run_dir / PROGRESS
    frame = pd.DataFrame([row])
    if progress.exists():
        frame = pd.concat([pd.read_csv(progress), frame], ignore_index=True)
    frame.to_csv(progress, index=False)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


def exit_without_teardown() -> None:
    """End a GPU script once every result is on disk and every MLflow run is closed."""
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32":
        # CUDA libraries crash while unloading after LSTM training (0xC0000409), even via _exit
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess(ctypes.c_void_p(kernel32.GetCurrentProcess()), 0)
