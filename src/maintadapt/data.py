"""Parsing, remaining-useful-life labels, unit-level splits and sensor screening."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SETTINGS = [f"setting_{i}" for i in range(1, 4)]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["unit_id", "cycle", *SETTINGS, *SENSORS]


@dataclass(frozen=True, eq=False)
class Subset:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True, eq=False)
class OperatingConditions:
    mean: np.ndarray
    scale: np.ndarray
    centers: np.ndarray

    def assign(self, frame: pd.DataFrame) -> np.ndarray:
        z = (frame[SETTINGS].to_numpy() - self.mean) / self.scale
        return ((z[:, None, :] - self.centers[None]) ** 2).sum(axis=2).argmin(axis=1)

    def centers_in_setting_units(self) -> np.ndarray:
        return self.centers * self.scale + self.mean


def read_trajectories(path: str | Path) -> pd.DataFrame:
    """Parse one whitespace-separated C-MAPSS trajectory file into a tidy frame."""
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path}: expected {len(COLUMNS)} columns, found {frame.shape[1]}"
        )
    frame.columns = COLUMNS
    dtypes = {c: "int64" for c in COLUMNS[:2]} | {c: "float64" for c in COLUMNS[2:]}
    frame = frame.astype(dtypes).sort_values(["unit_id", "cycle"], ignore_index=True)
    first = frame.groupby("unit_id")["cycle"].transform("min")
    step = frame.groupby("unit_id")["cycle"].diff().fillna(1)
    if (first != 1).any() or (step != 1).any():
        raise ValueError(
            f"{path}: every unit must run cycles 1, 2, 3, ... without gaps"
        )
    return frame


def read_final_rul(path: str | Path) -> pd.Series:
    """Ground-truth RUL at the last observed cycle of each test unit, indexed by unit_id."""
    values = pd.read_csv(path, sep=r"\s+", header=None)
    if values.shape[1] != 1:
        raise ValueError(f"{path}: expected one RUL value per line")
    index = pd.RangeIndex(1, len(values) + 1, name="unit_id")
    return pd.Series(values[0].to_numpy(dtype="int64"), index=index, name="rul")


def add_rul(
    frame: pd.DataFrame, cap: int, final_rul: pd.Series | None = None
) -> pd.DataFrame:
    """Attach true RUL and the capped target; truncated test units are offset by final_rul."""
    rul = frame.groupby("unit_id")["cycle"].transform("max") - frame["cycle"]
    if final_rul is not None:
        offset = frame["unit_id"].map(final_rul)
        if offset.isna().any():
            raise ValueError("final_rul has no entry for some units in the frame")
        rul = rul + offset.astype("int64")
    return frame.assign(
        rul=rul.astype("int64"), rul_capped=rul.clip(upper=cap).astype("int64")
    )


def load_subset(raw_dir: str | Path, name: str, cap: int) -> Subset:
    paths = {
        kind: Path(raw_dir) / f"{kind}_{name}.txt" for kind in ("train", "test", "RUL")
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing C-MAPSS files {missing}; see Data in the README"
        )
    test = read_trajectories(paths["test"])
    final_rul = read_final_rul(paths["RUL"])
    if set(final_rul.index) != set(test["unit_id"]):
        raise ValueError(
            f"{paths['RUL']} does not list exactly the units in {paths['test']}"
        )
    train = add_rul(read_trajectories(paths["train"]), cap)
    return Subset(name, train, add_rul(test, cap, final_rul))


def make_splits(
    units: dict[str, Sequence[int]], val_fraction: float, seed: int
) -> dict[str, dict[str, list[int]]]:
    """Assign whole engines to train or validation, reproducibly and per subset."""
    splits = {}
    for name, ids in units.items():
        pool = np.unique(np.asarray(ids))
        rng = np.random.default_rng([seed, int(name[2:])])
        n_val = max(1, round(val_fraction * len(pool)))
        val = np.sort(rng.choice(pool, size=n_val, replace=False))
        splits[name] = {"train": np.setdiff1d(pool, val).tolist(), "val": val.tolist()}
    return splits


def save_splits(
    splits: dict[str, dict[str, list[int]]],
    path: str | Path,
    seed: int,
    val_fraction: float,
) -> None:
    # one line per id list keeps the committed file short enough to review in a diff
    blocks = [
        f'    "{name}": {{\n'
        f'      "train": {json.dumps(parts["train"])},\n'
        f'      "val": {json.dumps(parts["val"])}\n'
        "    }"
        for name, parts in splits.items()
    ]
    header = f'{{\n  "split_seed": {seed},\n  "val_fraction": {val_fraction},\n'
    text = header + '  "subsets": {\n' + ",\n".join(blocks) + "\n  }\n}\n"
    Path(path).write_text(text, encoding="utf-8")


def load_splits(
    path: str | Path, seed: int, val_fraction: float
) -> dict[str, dict[str, list[int]]]:
    """Read the committed splits, refusing a stale file or one with overlapping units."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw["split_seed"] != seed or raw["val_fraction"] != val_fraction:
        raise ValueError(
            f"{path} was written with other split settings; rerun `make data`"
        )
    for name, parts in raw["subsets"].items():
        if set(parts["train"]) & set(parts["val"]):
            raise ValueError(f"{path}: {name} places a unit in both train and val")
    return raw["subsets"]


def fit_conditions(
    frame: pd.DataFrame, n_conditions: int, seed: int
) -> OperatingConditions:
    """Cluster the operating settings of training units into flight regimes."""
    settings = frame[SETTINGS].to_numpy()
    mean, std = settings.mean(axis=0), settings.std(axis=0)
    # a setting held fixed across the fleet (setting_3 in FD001) would divide by zero
    scale = np.where(std > 0, std, 1.0)
    kmeans = KMeans(n_clusters=n_conditions, n_init=10, random_state=seed)
    centers = kmeans.fit((settings - mean) / scale).cluster_centers_
    raw = centers * scale + mean
    # order regimes by altitude, then throttle, so condition ids are stable across reruns
    order = np.lexsort((raw[:, 1], raw[:, 2], raw[:, 0]))
    return OperatingConditions(mean, scale, centers[order])


def screen_sensors(
    frame: pd.DataFrame, conditions: np.ndarray, max_modal_share: float
) -> pd.DataFrame:
    """Flag sensors that are constant or near-constant within any operating condition."""
    n_conditions = len(np.unique(conditions))
    rows = []
    for sensor in SENSORS:
        grouped = frame[sensor].groupby(conditions)
        share = grouped.agg(lambda x: x.value_counts(normalize=True).iloc[0])
        frozen = int((share == 1.0).sum())
        if frozen == n_conditions:
            reason = "constant in every operating condition"
        elif frozen:
            reason = f"constant in {frozen} of {n_conditions} operating conditions"
        elif share.max() >= max_modal_share:
            reason = f"one value covers {share.max():.1%} of readings in some condition"
        else:
            reason = ""
        rows.append(
            {
                "sensor": sensor,
                "min_modal_share": share.min(),
                "max_modal_share": share.max(),
                "dropped": bool(reason),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)
