"""Operating-condition-aware normalisation and backward-looking rolling features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from maintadapt.data import OperatingConditions, Subset, fit_conditions, screen_sensors


@dataclass(frozen=True, eq=False)
class Normaliser:
    columns: list[str]
    by: str | None
    mean: pd.DataFrame
    std: pd.DataFrame

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        keys = frame[self.by].to_numpy() if self.by else np.zeros(len(frame), dtype=int)
        unseen = set(np.unique(keys)) - set(self.mean.index)
        if unseen:
            raise ValueError(
                f"groups {sorted(unseen)} never occur in the training units"
            )
        centred = frame[self.columns].to_numpy() - self.mean.loc[keys].to_numpy()
        scaled = centred / self.std.loc[keys].to_numpy()
        return frame.assign(**dict(zip(self.columns, scaled.T, strict=True)))


@dataclass(frozen=True, eq=False)
class PreparedSubset:
    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    sensors: list[str]
    screening: pd.DataFrame
    conditions: OperatingConditions
    normaliser: Normaliser


def fit_normaliser(
    frame: pd.DataFrame,
    train_units: Sequence[int],
    columns: list[str],
    by: str | None = None,
) -> Normaliser:
    """Fit z-score statistics on the rows of train_units alone, per group of `by` if given."""
    rows = frame[frame["unit_id"].isin(train_units)]
    if rows.empty:
        raise ValueError("none of train_units appear in the frame")
    keys = rows[by].to_numpy() if by else np.zeros(len(rows), dtype=int)
    grouped = rows[columns].groupby(keys)
    mean, std = grouped.mean(), grouped.std(ddof=0)
    flat = std.columns[(std == 0).any()].tolist()
    if flat:
        raise ValueError(
            f"{flat} have zero variance within a group; screen sensors first"
        )
    return Normaliser(list(columns), by, mean, std)


def prepare_subset(
    subset: Subset,
    split: dict[str, list[int]],
    n_conditions: int,
    max_modal_share: float,
    seed: int,
    *,
    condition_aware: bool,
    sensors: list[str] | None = None,
) -> PreparedSubset:
    """Fit regimes, sensor screening and normalisation on training units; apply everywhere."""
    is_train = subset.train["unit_id"].isin(split["train"])
    conditions = fit_conditions(subset.train[is_train], n_conditions, seed)
    fleet = subset.train.assign(condition=conditions.assign(subset.train))
    test = subset.test.assign(condition=conditions.assign(subset.test))
    train_rows = fleet[is_train]
    screening = screen_sensors(
        train_rows, train_rows["condition"].to_numpy(), max_modal_share
    )
    kept = screening.loc[~screening["dropped"], "sensor"].tolist()
    # a network shared across subsets fixes its inputs, and each one must survive screening here
    if sensors is not None and not set(sensors) <= set(kept):
        dropped = sorted(set(sensors) - set(kept))
        raise ValueError(f"{subset.name}: requested sensors {dropped} fail screening")
    sensors = kept if sensors is None else list(sensors)
    by = "condition" if condition_aware else None
    normaliser = fit_normaliser(fleet, split["train"], sensors, by=by)
    fleet = normaliser.transform(fleet)
    return PreparedSubset(
        name=subset.name,
        train=fleet[is_train].reset_index(drop=True),
        val=fleet[fleet["unit_id"].isin(split["val"])].reset_index(drop=True),
        test=normaliser.transform(test),
        sensors=sensors,
        screening=screening,
        conditions=conditions,
        normaliser=normaliser,
    )


def rolling_features(
    frame: pd.DataFrame, sensors: list[str], windows: list[int]
) -> pd.DataFrame:
    """Trailing mean, std and least-squares slope per sensor and window, never looking ahead."""
    order = frame[["unit_id", "cycle"]]
    if not order.equals(order.sort_values(["unit_id", "cycle"])):
        raise ValueError("frame must be sorted by unit_id then cycle")
    x = frame[sensors].to_numpy(dtype=float)
    t = frame["cycle"].to_numpy(dtype=float)[:, None]
    unit = frame["unit_id"].to_numpy()
    row = np.arange(len(frame))
    position = frame.groupby("unit_id", sort=False).cumcount().to_numpy()
    # running sums restart for every engine, so no engine's history can touch another's
    running = [
        pd.DataFrame(v).groupby(unit, sort=False).cumsum().to_numpy()
        for v in (x, x * x, t, t * t, t * x)
    ]
    columns = {}
    for window in windows:
        n = np.minimum(position + 1, window)[:, None].astype(float)
        trimmed = (position >= window)[:, None]
        before = np.maximum(row - window, 0)
        s_x, s_xx, s_t, s_tt, s_tx = (
            r - np.where(trimmed, r[before], 0.0) for r in running
        )
        mean = s_x / n
        std = np.sqrt(np.clip(s_xx / n - mean**2, 0.0, None))
        denom = n * s_tt - s_t**2
        # a single observation has no trend yet, so its slope is reported as flat
        slope = np.divide(
            n * s_tx - s_t * s_x, denom, out=np.zeros_like(x), where=denom > 0
        )
        for j, sensor in enumerate(sensors):
            columns[f"{sensor}_mean_{window}"] = mean[:, j]
            columns[f"{sensor}_std_{window}"] = std[:, j]
            columns[f"{sensor}_slope_{window}"] = slope[:, j]
    return pd.DataFrame(columns, index=frame.index)


def build_features(
    frame: pd.DataFrame, sensors: list[str], windows: list[int], include_cycle: bool
) -> pd.DataFrame:
    """Model inputs per row: current readings, their rolling statistics and optionally age."""
    parts = [frame[sensors], rolling_features(frame, sensors, windows)]
    if include_cycle:
        parts.insert(0, frame[["cycle"]].astype(float))
    return pd.concat(parts, axis=1)


def last_cycle(frame: pd.DataFrame) -> pd.Series:
    """Boolean mask of each unit's final observed row, where test predictions are scored."""
    return frame["cycle"] == frame.groupby("unit_id")["cycle"].transform("max")
