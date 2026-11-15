"""N-CMAPSS (NASA PCoE data set 17): reading, inspection, the cap and the per-flight frame.

Each HDF5 file holds one sub-dataset sampled at 1 Hz: W (scenario descriptors), X_s (physical
sensors), X_v (virtual sensors), T (health parameters), Y (RUL in flights) and A (unit, cycle,
flight class Fc and health state hs), each with a _dev and a _test suffix, plus <group>_var
with the column names. A file is read one unit at a time through slices of its contiguous rows.

Only W and X_s ever become model inputs (PREREGISTRATION.md section 8): read_inputs refuses any
other group, and the per-flight frame is built from it alone. X_v is never read. T and the health
state are read only to describe the data and to compute the cap.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from maintadapt.data import Subset

FLEET = "NCMAPSS"
GROUPS = ("W", "X_s", "X_v", "T", "Y", "A")
PARTS = ("dev", "test")
# the only groups a model may read
INPUT_GROUPS = ("W", "X_s")
A_COLUMNS = ("unit", "cycle", "Fc", "hs")
FILE_PATTERN = "N-CMAPSS_DS*.h5"
# unit_id = UNIT_BASE * (position of the file among all files) + unit within its file
UNIT_BASE = 100
CAP_STEP = 5


class LeakageError(ValueError):
    """A group other than W and X_s was requested as a model input."""


@dataclass(frozen=True)
class SubDataset:
    path: Path
    # 1-based position among every file in the folder, readable or not, so ids stay fixed
    number: int

    @property
    def name(self) -> str:
        """DS01-005 for N-CMAPSS_DS01-005.h5."""
        return self.path.stem.removeprefix("N-CMAPSS_")


@dataclass(frozen=True)
class UnitBlock:
    part: str
    unit: int
    start: int
    stop: int


def sub_datasets(raw_dir: str | Path) -> list[SubDataset]:
    """Every N-CMAPSS file in raw_dir, in name order."""
    paths = sorted(Path(raw_dir).glob(FILE_PATTERN))
    if not paths:
        raise FileNotFoundError(f"no {FILE_PATTERN} files in {raw_dir}")
    return [SubDataset(p, i + 1) for i, p in enumerate(paths)]


def open_failure(path: str | Path) -> str | None:
    """Why the file cannot be opened or lacks a group, or None when it is readable."""
    try:
        with h5py.File(path, "r") as h:
            missing = [
                f"{g}_{p}" for g in GROUPS for p in PARTS if f"{g}_{p}" not in h
            ] + [f"{g}_var" for g in GROUPS if g != "Y" and f"{g}_var" not in h]
            if missing:
                return f"missing groups {missing}"
            for part in PARTS:
                rows = {h[f"{g}_{part}"].shape[0] for g in GROUPS}
                if len(rows) != 1:
                    return (
                        f"the groups of {part} disagree on their rows: {sorted(rows)}"
                    )
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def names(h: h5py.File, group: str) -> list[str]:
    """Column names of a group, from <group>_var."""
    return [
        v.decode() if isinstance(v, bytes) else str(v) for v in h[f"{group}_var"][()]
    ]


def unit_blocks(h: h5py.File, part: str) -> list[UnitBlock]:
    """Each unit's rows, which the files store as one contiguous block per unit."""
    units = h[f"A_{part}"][:, 0]
    if len(units) == 0:
        return []
    change = np.flatnonzero(np.diff(units) != 0) + 1
    starts, stops = np.r_[0, change], np.r_[change, len(units)]
    ids = units[starts]
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"A_{part}: a unit's rows are not contiguous")
    return [
        UnitBlock(part, int(u), int(s), int(e))
        for u, s, e in zip(ids, starts, stops, strict=True)
    ]


def check_inputs(groups: Sequence[str]) -> None:
    """The leakage guard: model inputs come from W and X_s only."""
    refused = [g for g in groups if g not in INPUT_GROUPS]
    if refused:
        raise LeakageError(
            f"{refused} may never be model inputs; only {list(INPUT_GROUPS)} may"
        )


def read_inputs(
    h: h5py.File, block: UnitBlock, groups: Sequence[str] = INPUT_GROUPS
) -> dict[str, np.ndarray]:
    """The rows of one unit for the input groups; any other group raises LeakageError."""
    check_inputs(groups)
    return {g: h[f"{g}_{block.part}"][block.start : block.stop] for g in groups}


def read_labels(h: h5py.File, block: UnitBlock) -> tuple[np.ndarray, np.ndarray]:
    """A (unit, cycle, Fc, hs) and Y of one unit: identifiers, labels and the health state."""
    a = h[f"A_{block.part}"][block.start : block.stop]
    y = h[f"Y_{block.part}"][block.start : block.stop, 0]
    return a, y


def flight_starts(cycles: np.ndarray) -> np.ndarray:
    """Row index where each flight of a unit begins; the flights must run 1..T in order."""
    starts = np.r_[0, np.flatnonzero(np.diff(cycles) != 0) + 1]
    flights = cycles[starts]
    if not np.array_equal(flights, np.arange(1, len(flights) + 1)):
        raise ValueError("a unit's flights must run 1..T, each in one block of rows")
    return starts


@dataclass(frozen=True)
class UnitSummary:
    """What the inspection reports about one unit, from A, Y and the ends of T."""

    flights: int
    flight_classes: tuple[int, ...]
    # RUL is T - t at every flight and constant within each flight
    rul_is_life_minus_cycle: bool
    # health state 1 until some flight, 0 from then on
    health_state_monotone: bool
    # first flight with health state 0, and the RUL there; None if the unit stays healthy
    onset_flight: int | None
    onset_rul: int | None


def summarise(a: np.ndarray, y: np.ndarray) -> UnitSummary:
    cycles = a[:, 1].astype(np.int64)
    starts = flight_starts(cycles)
    life = len(starts)
    per_flight = y[starts]
    constant = bool(
        np.array_equal(np.repeat(per_flight, np.diff(np.r_[starts, len(y)])), y)
    )
    rul_ok = constant and np.array_equal(per_flight, life - np.arange(1, life + 1))
    hs = a[:, 3]
    monotone = bool(np.isin(hs, (0.0, 1.0)).all() and (np.diff(hs) <= 0).all())
    abnormal = np.flatnonzero(hs[starts] == 0)
    onset = None if not len(abnormal) else int(abnormal[0])
    return UnitSummary(
        flights=life,
        flight_classes=tuple(int(v) for v in np.unique(a[:, 2])),
        rul_is_life_minus_cycle=bool(rul_ok),
        health_state_monotone=monotone,
        onset_flight=None if onset is None else onset + 1,
        onset_rul=None if onset is None else int(per_flight[onset]),
    )


def health_changes(h: h5py.File, block: UnitBlock, tol: float = 1e-9) -> list[str]:
    """Health parameters (T) that differ between a unit's first and last sample.

    For the inspection report only: which components degrade is the unit's failure mode.
    """
    t = h[f"T_{block.part}"]
    change = np.abs(t[block.stop - 1] - t[block.start])
    return [n for n, d in zip(names(h, "T"), change, strict=True) if d > tol]


def cap_from_onsets(onset_rul: Sequence[int]) -> int:
    """Section 8: the median RUL at the first abnormal health state, rounded to 5 cycles."""
    values = np.asarray(onset_rul, dtype=float)
    if len(values) == 0 or np.isnan(values).any():
        raise ValueError("every unit needs an onset RUL")
    # round half up, so a median of 62.5 gives 65 on every platform
    return int(CAP_STEP * np.floor(np.median(values) / CAP_STEP + 0.5))


def flight_frame(
    inputs: dict[str, np.ndarray],
    columns: dict[str, list[str]],
    a: np.ndarray,
) -> pd.DataFrame:
    """One row per flight: mean and standard deviation of each W and X_s variable over the
    flight's 1 Hz samples, the flight's duration in samples and its flight class."""
    check_inputs(list(inputs))
    cycles = a[:, 1].astype(np.int64)
    starts = flight_starts(cycles)
    counts = np.diff(np.r_[starts, len(cycles)])
    out = {
        "cycle": cycles[starts],
        "fc": a[starts, 2].astype(np.int64),
        "duration": counts.astype(float),
    }
    for group in INPUT_GROUPS:
        values = np.asarray(inputs[group], dtype=float)
        sums = np.add.reduceat(values, starts, axis=0)
        mean = sums / counts[:, None]
        # population variance about each flight's own mean, so a flat signal gives exactly 0
        centred = values - np.repeat(mean, counts, axis=0)
        var = np.add.reduceat(centred**2, starts, axis=0) / counts[:, None]
        for j, name in enumerate(columns[group]):
            out[f"mean_{name}"] = mean[:, j]
            out[f"std_{name}"] = np.sqrt(var[:, j])
    return pd.DataFrame(out)


def feature_columns(columns: dict[str, list[str]]) -> list[str]:
    """The per-flight inputs in frame order: duration, then mean and std of each variable."""
    names_ = ["duration"]
    for group in INPUT_GROUPS:
        for name in columns[group]:
            names_ += [f"mean_{name}", f"std_{name}"]
    return names_


def unit_id(sub: SubDataset, unit: int) -> int:
    if not 0 < unit < UNIT_BASE:
        raise ValueError(f"unit {unit} does not fit below {UNIT_BASE}")
    return UNIT_BASE * sub.number + unit


def units(sub: SubDataset) -> Iterator[tuple[h5py.File, UnitBlock]]:
    """Every unit of a readable file, development units first, one block at a time."""
    with h5py.File(sub.path, "r") as h:
        for part in PARTS:
            for block in unit_blocks(h, part):
                yield h, block


@dataclass(frozen=True, eq=False)
class Frame:
    """The per-flight frame of every readable unit, and the files left out."""

    flights: pd.DataFrame
    features: list[str]
    excluded: list[tuple[str, str]]


def build_frame(raw_dir: str | Path) -> Frame:
    """One row per unit and flight from W and X_s, unit by unit; unreadable files are skipped.

    Besides the features the rows carry identifiers (unit_id, sub_dataset, part, unit, cycle),
    the flight class and the true RUL, which Y gives as T - t (checked here).
    """
    parts, excluded, columns = [], [], None
    for sub in sub_datasets(raw_dir):
        why = open_failure(sub.path)
        if why is not None:
            excluded.append((sub.path.name, why))
            continue
        with h5py.File(sub.path, "r") as h:
            found = {g: names(h, g) for g in INPUT_GROUPS}
            if columns is not None and found != columns:
                raise ValueError(
                    f"{sub.name}: W or X_s columns differ from other files"
                )
            columns = found
            for part in PARTS:
                for block in unit_blocks(h, part):
                    a, y = read_labels(h, block)
                    if not summarise(a, y).rul_is_life_minus_cycle:
                        raise ValueError(
                            f"{sub.name} unit {block.unit}: Y is not T - t"
                        )
                    flights = flight_frame(read_inputs(h, block), columns, a)
                    life = len(flights)
                    parts.append(
                        flights.assign(
                            unit_id=unit_id(sub, block.unit),
                            sub_dataset=sub.name,
                            part=part,
                            unit=block.unit,
                            rul=life - flights["cycle"],
                        )
                    )
    if columns is None:
        raise ValueError(f"no readable N-CMAPSS file in {raw_dir}")
    frame = pd.concat(parts, ignore_index=True)
    ids = ["unit_id", "sub_dataset", "part", "unit", "cycle", "fc", "rul"]
    features = feature_columns(columns)
    frame = frame[ids + features].sort_values(["unit_id", "cycle"], ignore_index=True)
    return Frame(frame, features, excluded)


def frame_features(flights: pd.DataFrame) -> list[str]:
    """The per-flight inputs of a stored frame, in its column order."""
    return [
        c for c in flights.columns if c == "duration" or c.startswith(("mean_", "std_"))
    ]


def load_subset(path: str | Path, cap: int) -> tuple[Subset, list[str]]:
    """The stored frame as a Subset whose train holds every unit and whose test is empty."""
    flights = pd.read_parquet(path).sort_values(["unit_id", "cycle"], ignore_index=True)
    train = flights.assign(rul_capped=flights["rul"].clip(upper=cap).astype("int64"))
    return Subset(FLEET, train, train.iloc[:0]), frame_features(flights)
