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
    sensors = screening.loc[~screening["dropped"], "sensor"].tolist()
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
