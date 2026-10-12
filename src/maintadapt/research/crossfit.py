"""Cross-fitting: which fits to run, their seeds, one retry, and their archive records."""

from __future__ import annotations

import hashlib
import itertools
import os
import platform
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from maintadapt.research.archive import (
    FAILED,
    OK,
    FitKey,
    crossfit_key,
    fit_status,
    prediction_table,
    write_fit,
)
from maintadapt.research.fold_data import FoldData
from maintadapt.research.models.registry import (
    CONTROL_FAMILIES,
    CPU_FAMILIES,
    GPU_FAMILIES,
    FitOutput,
)
from maintadapt.research.models.spaces import ModelConfig, config_hash

# the controls take no time, so they ride with the lighter CPU stream
STREAMS = {
    "cpu": (*CPU_FAMILIES, *CONTROL_FAMILIES),
    "gpu": GPU_FAMILIES,
    "all": (*CPU_FAMILIES, *GPU_FAMILIES, *CONTROL_FAMILIES),
}
# PREREGISTRATION.md section 9: one retry with seed + 1000
RETRY_OFFSET = 1000
PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "torch", "pyarrow")


@dataclass(frozen=True)
class Job:
    fleet: str
    repeat: int
    fold: int
    config: ModelConfig

    @property
    def key(self) -> FitKey:
        c = self.config
        return crossfit_key(self.fleet, c.family, c.id, self.repeat, self.fold)


def fit_seed(fleet: str, repeat: int, fold: int, config_id: str) -> int:
    """The first 31 bits of sha256("fleet|repeat|fold|config_id")."""
    digest = hashlib.sha256(f"{fleet}|{repeat}|{fold}|{config_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") >> 1


def control_configs() -> list[ModelConfig]:
    return [
        ModelConfig(f"{name}-00", name, config_hash(name, {}), {})
        for name in CONTROL_FAMILIES
    ]


def config_index(config: ModelConfig) -> int:
    return int(config.id.rsplit("-", 1)[1])


def select_configs(
    pool: Sequence[ModelConfig],
    families: Iterable[str],
    indices: Iterable[int] | None = None,
    limit_per_family: Callable[[int], int] | None = None,
) -> list[ModelConfig]:
    """Pool configurations of the given families, in pool order.

    indices keeps only those configuration numbers; limit_per_family maps a family's size to
    the number of leading configurations kept, which is how dropped configurations leave.
    """
    wanted, keep = set(families), None if indices is None else set(indices)
    sizes = Counter(c.family for c in pool)
    chosen = []
    for config in pool:
        i = config_index(config)
        if config.family not in wanted or (keep is not None and i not in keep):
            continue
        if limit_per_family is not None and i >= limit_per_family(sizes[config.family]):
            continue
        chosen.append(config)
    return chosen


def plan(
    configs: Sequence[ModelConfig],
    fleets: Iterable[str],
    repeats: Iterable[int],
    folds: Iterable[int],
) -> list[Job]:
    """Jobs in the order fleet, repeat, fold, family, configuration."""
    return [
        Job(fleet, repeat, fold, config)
        for fleet, repeat, fold in itertools.product(fleets, repeats, folds)
        for config in configs
    ]


def git_commit(cwd: str | Path = ".") -> str:
    """HEAD of the working copy, marked +dirty when tracked files have changed."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return "unknown"
    return (head or "unknown") + ("+dirty" if changed else "")


def environment(device: torch.device) -> dict[str, Any]:
    """Software and hardware stored with every fit (PREREGISTRATION.md section 11)."""
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "hardware": {
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "gpu": gpu,
            "os": platform.platform(),
        },
        "git_commit": git_commit(),
    }


def _release_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_fit(
    root: str | Path,
    key: FitKey,
    identity: dict[str, Any],
    data: FoldData,
    fit_once: Callable[[int], FitOutput],
    seed: int,
    env: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Fit with seed, retry once with seed + 1000, then record success or failure."""
    started = datetime.now(UTC).isoformat(timespec="seconds")
    start = time.perf_counter()
    errors, out, table, used = [], None, None, seed
    for used in (seed, seed + RETRY_OFFSET):
        try:
            out = fit_once(used)
            bad = [s for s, p in out.predictions.items() if not np.isfinite(p).all()]
            if not bad and np.isfinite(out.val_rmse):
                table = prediction_table(data, out.predictions)
                break
            errors.append(f"seed {used}: non-finite predictions in {bad or 'val_rmse'}")
        # section 9: a failing fit is recorded and the run moves on; Ctrl-C still stops it
        except Exception as exc:  # noqa: BLE001
            errors.append(f"seed {used}: {type(exc).__name__}: {exc}")
            _release_gpu()
        out = None
    record = {
        **identity,
        "seed": used,
        "status": OK if table is not None else FAILED,
        "attempts": len(errors) + (table is not None),
        "error": "; ".join(errors) or None,
        "fit_seconds": None if out is None else out.fit_seconds,
        "wall_seconds": None,
        "best_iteration_or_epoch": None if out is None else out.best_iteration,
        "val_rmse": None if out is None else out.val_rmse,
        "n_inputs": None if out is None else out.n_inputs,
        "n_rows": None if table is None else table.num_rows,
        "device": device,
        **env,
        "started_at": started,
    }
    record["wall_seconds"] = time.perf_counter() - start
    write_fit(root, key, table, record)
    return record


def describe(job: Job, record: dict[str, Any]) -> str:
    """One progress line: ids, seconds, validation RMSE and status, never a test metric."""
    val = record["val_rmse"]
    text = (
        f"{job.fleet} r{job.repeat} f{job.fold} {job.config.id:<18} seed {record['seed']:>10}"
        f" {record['wall_seconds']:8.1f}s val_rmse "
        f"{'-' if val is None else f'{val:7.3f}'} {record['status']}"
    )
    if record["attempts"] > 1:
        text += f" (attempts {record['attempts']}: {record['error']})"
    return text


def run_jobs(
    jobs: Sequence[Job],
    root: str | Path,
    build_data: Callable[[str, int, int], FoldData],
    fit_job: Callable[[Job, FoldData, int], FitOutput],
    env: dict[str, Any],
    device_for: Callable[[str], str],
    *,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> Counter[str]:
    """Run every pending job; FoldData is built once per (fleet, repeat, fold) with work left."""
    counts: Counter[str] = Counter()
    groups = itertools.groupby(jobs, key=lambda j: (j.fleet, j.repeat, j.fold))
    for (fleet, repeat, fold), group in groups:
        group = list(group)
        pending = [j for j in group if fit_status(root, j.key) is None]
        counts["skipped"] += len(group) - len(pending)
        if not pending:
            continue
        if limit is not None and counts["ran"] >= limit:
            counts["left"] += len(pending)
            continue
        data = build_data(fleet, repeat, fold)
        for job in pending:
            if limit is not None and counts["ran"] >= limit:
                counts["left"] += 1
                continue
            c = job.config
            identity = {
                "fleet": job.fleet,
                "family": c.family,
                "config_id": c.id,
                "config_hash": c.hash,
                "repeat": job.repeat,
                "fold": job.fold,
            }
            record = run_fit(
                root,
                job.key,
                identity,
                data,
                lambda seed, job=job, data=data: fit_job(job, data, seed),
                fit_seed(job.fleet, job.repeat, job.fold, c.id),
                env,
                device_for(c.family),
            )
            counts["ran"] += 1
            counts[record["status"]] += 1
            log(describe(job, record))
    return counts
