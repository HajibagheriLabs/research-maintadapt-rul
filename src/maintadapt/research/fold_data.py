"""Prepared data of one (fleet, repeat, fold), with every statistic fitted on inner_fit engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from maintadapt.data import OperatingConditions, Subset
from maintadapt.features import Normaliser, build_features, prepare_subset
from maintadapt.research.config import FleetDataConfig
from maintadapt.research.folds import Fold
from maintadapt.sequence import make_windows

SPLITS = ("inner_fit", "inner_val", "outer_test", "official_test")
# every prediction file covers these splits; inner_fit engines are never predicted
PREDICTED = ("inner_val", "outer_test", "official_test")


@dataclass(frozen=True, eq=False)
class FoldData:
    fleet: str
    repeat: int
    fold: int
    # normalised frames sorted by unit_id then cycle, with rul and rul_capped
    frames: dict[str, pd.DataFrame]
    sensors: list[str]
    cap: int
    # trailing windows of the rolling features, not sequence window lengths
    feature_windows: list[int]
    include_cycle: bool
    conditions: OperatingConditions
    screening: pd.DataFrame
    normaliser: Normaliser
    # not an init field, so dataclasses.replace starts a copy with an empty cache
    _cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )

    def frame(self, split: str) -> pd.DataFrame:
        if split not in self.frames:
            raise KeyError(f"unknown split {split!r}; expected one of {SPLITS}")
        return self.frames[split]

    def target(self, split: str) -> pd.Series:
        """Capped RUL, the training target and the truth of validation RMSE."""
        return self.frame(split)["rul_capped"].astype(float)

    def rolling(self, split: str) -> pd.DataFrame:
        """Current readings, their trailing statistics and the cycle, one row per cycle."""
        key = ("rolling", split)
        if key not in self._cache:
            self._cache[key] = build_features(
                self.frame(split),
                self.sensors,
                self.feature_windows,
                self.include_cycle,
            )
        return self._cache[key]

    def raw(self, split: str) -> pd.DataFrame:
        """The cycle and the current readings only."""
        return self.frame(split)[["cycle", *self.sensors]].astype(float)

    def windows(self, split: str, length: int) -> np.ndarray:
        """Windows of `length` cycles of normalised readings ending at every row."""
        key = ("windows", split, length)
        if key not in self._cache:
            self._cache[key] = make_windows(self.frame(split), self.sensors, length)
        return self._cache[key]


def build_fold_data(
    subset: Subset,
    fold: Fold,
    cfg: FleetDataConfig,
    seed: int,
    *,
    repeat: int,
    index: int,
) -> FoldData:
    """Fit regimes, screening and normalisation on inner_fit; apply them to every split."""
    prepared = prepare_subset(
        subset,
        {"train": list(fold.inner_fit), "val": [*fold.inner_val, *fold.outer_test]},
        cfg.n_conditions[subset.name],
        cfg.max_modal_share,
        seed,
        condition_aware=True,
    )
    held_out = prepared.val
    frames = {
        "inner_fit": prepared.train,
        "inner_val": held_out[held_out["unit_id"].isin(fold.inner_val)],
        "outer_test": held_out[held_out["unit_id"].isin(fold.outer_test)],
        "official_test": prepared.test,
    }
    expected = {
        "inner_fit": fold.inner_fit,
        "inner_val": fold.inner_val,
        "outer_test": fold.outer_test,
    }
    for split, units in expected.items():
        missing = sorted(set(units) - set(frames[split]["unit_id"]))
        if missing:
            raise ValueError(f"{subset.name}: {split} engines {missing} have no rows")
    return FoldData(
        fleet=subset.name,
        repeat=repeat,
        fold=index,
        frames={
            split: frame.sort_values(["unit_id", "cycle"], ignore_index=True)
            for split, frame in frames.items()
        },
        sensors=prepared.sensors,
        cap=cfg.cap,
        feature_windows=list(cfg.windows),
        include_cycle=cfg.include_cycle,
        conditions=prepared.conditions,
        screening=prepared.screening,
        normaliser=prepared.normaliser,
    )
