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
    # C-MAPSS regimes; None for N-CMAPSS, whose flight class takes their place
    conditions: OperatingConditions | None
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


# N-CMAPSS flight classes: short, medium and long flights
FLIGHT_CLASSES = (1, 2, 3)


def build_ncmapss_fold_data(
    subset: Subset,
    fold: Fold,
    features: list[str],
    cfg: FleetDataConfig,
    *,
    repeat: int,
    index: int,
) -> FoldData:
    """N-CMAPSS: z-score the per-flight features within flight class on inner_fit units.

    A zero standard deviation counts as 1, so a constant feature centres to zero; a flight
    class that inner_fit never flies stops the fold. The flight class enters as indicators.
    There is no official test split: its frame is empty and no fit predicts it.
    """
    frame = subset.train
    fitted = frame[frame["unit_id"].isin(fold.inner_fit)]
    unseen = sorted(set(frame["fc"]) - set(fitted["fc"]))
    if unseen:
        raise ValueError(
            f"{subset.name}: inner_fit never flies flight classes {unseen}"
        )
    grouped = fitted.groupby("fc")[features]
    std = grouped.std(ddof=0)
    normaliser = Normaliser(
        list(features), "fc", grouped.mean(), std.mask(std == 0, 1.0)
    )
    indicators = {f"fc_{c}": (frame["fc"] == c).astype(float) for c in FLIGHT_CLASSES}
    scaled = normaliser.transform(frame).assign(**indicators)
    members = {
        "inner_fit": fold.inner_fit,
        "inner_val": fold.inner_val,
        "outer_test": fold.outer_test,
    }
    frames = {}
    for split, units in members.items():
        rows = scaled[scaled["unit_id"].isin(units)]
        missing = sorted(set(units) - set(rows["unit_id"]))
        if missing:
            raise ValueError(f"{subset.name}: {split} units {missing} have no rows")
        frames[split] = rows.sort_values(["unit_id", "cycle"], ignore_index=True)
    frames["official_test"] = scaled.iloc[:0].reset_index(drop=True)
    return FoldData(
        fleet=subset.name,
        repeat=repeat,
        fold=index,
        frames=frames,
        sensors=[*features, *indicators],
        cap=cfg.cap,
        feature_windows=list(cfg.windows),
        include_cycle=cfg.include_cycle,
        conditions=None,
        screening=pd.DataFrame(),
        normaliser=normaliser,
    )
