"""A small synthetic C-MAPSS-like fleet shared by the research tests."""

from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from maintadapt.data import SENSORS, Subset, add_rul
from maintadapt.research.config import (
    FoldsConfig,
    ResearchTrainingConfig,
    load_research_config,
)
from maintadapt.research.fold_data import FoldData, build_fold_data
from maintadapt.research.folds import Fold, fleet_folds, lifetimes

ROOT = Path(__file__).resolve().parents[1]
CFG = load_research_config(ROOT / "configs" / "research.yaml")
DATA_CFG = replace(CFG.data, fleets=["FD001"], windows=[5, 10])
# short training keeps every neural family to a few seconds on a CPU
TRAINING = ResearchTrainingConfig(
    device="cpu",
    max_epochs=3,
    patience=2,
    grad_clip=1.0,
    batch_size=256,
    cpu_jobs=2,
    gpu_torch_threads=1,
)
FOLD = Fold(
    inner_fit=tuple(range(1, 13)),
    inner_val=tuple(range(13, 17)),
    outer_test=tuple(range(17, 21)),
)
CONSTANT = ["sensor_1", "sensor_5", "sensor_10"]


def engine(unit: int, life: int, rng: np.random.Generator) -> pd.DataFrame:
    cycle = np.arange(1, life + 1)
    # wear grows slowly, then fast near failure, as in C-MAPSS degradation
    wear = np.exp(3.0 * cycle / life) - 1.0
    frame = pd.DataFrame({"unit_id": unit, "cycle": cycle})
    frame["setting_1"] = rng.normal(0.0, 0.002, life)
    frame["setting_2"] = rng.normal(0.0, 0.0002, life)
    frame["setting_3"] = 100.0
    for k, sensor in enumerate(SENSORS):
        trend = (k % 5 - 2) * 0.2 * wear
        frame[sensor] = 1.0 if sensor in CONSTANT else 10.0 + trend
        if sensor not in CONSTANT:
            frame[sensor] += rng.normal(0.0, 0.5, life)
    return frame


@cache
def subset() -> Subset:
    """20 run-to-failure engines and 6 truncated official test engines."""
    rng = np.random.default_rng(0)
    train = pd.concat(
        [engine(u, int(rng.integers(60, 160)), rng) for u in range(1, 21)],
        ignore_index=True,
    )
    tests, final = [], {}
    for unit in range(1, 7):
        full = engine(unit, int(rng.integers(60, 160)), rng)
        seen = int(len(full) * rng.uniform(0.3, 0.9))
        tests.append(full.iloc[:seen])
        final[unit] = len(full) - seen
    test = add_rul(pd.concat(tests, ignore_index=True), 125, pd.Series(final))
    return Subset("FD001", add_rul(train, 125), test)


def fold_data(
    fleet: Subset, fold: Fold = FOLD, repeat: int = 0, index: int = 0
) -> FoldData:
    return build_fold_data(
        fleet, fold, DATA_CFG, CFG.folds.seed, repeat=repeat, index=index
    )


@cache
def repeats(n_repeats: int = 2) -> list[list[Fold]]:
    """Folds of the synthetic fleet by the registered block rule."""
    cfg = FoldsConfig(n_outer=5, n_repeats=n_repeats, inner_val_fraction=0.25, seed=1)
    return fleet_folds(lifetimes(subset().train), cfg, "FD001")
