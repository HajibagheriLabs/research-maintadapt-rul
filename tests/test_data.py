from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maintadapt.config import load_config
from maintadapt.data import (
    COLUMNS,
    SENSORS,
    SETTINGS,
    Subset,
    add_rul,
    load_splits,
    make_splits,
    read_trajectories,
)
from maintadapt.features import prepare_subset

ROOT = Path(__file__).resolve().parents[1]
TRAIN_UNITS = {"FD001": 100, "FD002": 260, "FD003": 100, "FD004": 249}


def _fleet(
    unit_ids: list[int], seed: int, sensor_scale: float, altitude_shift: float = 0.0
):
    rng = np.random.default_rng(seed)
    frames = []
    for unit in unit_ids:
        n = int(rng.integers(30, 60))
        frame = pd.DataFrame({"unit_id": unit, "cycle": np.arange(1, n + 1)})
        regime = rng.integers(0, 2, size=n)
        frame["setting_1"] = np.where(regime, 20.0, 0.0) + altitude_shift
        frame["setting_2"] = np.where(regime, 0.7, 0.0)
        frame["setting_3"] = 100.0
        frame[SENSORS] = rng.normal(
            10 * sensor_scale, sensor_scale, size=(n, len(SENSORS))
        )
        frames.append(frame)
    return add_rul(pd.concat(frames, ignore_index=True), cap=125)


def test_committed_splits_are_disjoint_complete_and_reproducible() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    splits = load_splits(
        ROOT / cfg.paths.splits, cfg.data.split_seed, cfg.data.val_fraction
    )
    for name, n_units in TRAIN_UNITS.items():
        train, val = set(splits[name]["train"]), set(splits[name]["val"])
        assert not train & val
        assert train | val == set(range(1, n_units + 1))
    units = {name: range(1, n + 1) for name, n in TRAIN_UNITS.items()}
    assert make_splits(units, cfg.data.val_fraction, cfg.data.split_seed) == splits


def test_rul_cap_matches_hand_checked_values() -> None:
    engine = pd.DataFrame({"unit_id": 1, "cycle": np.arange(1, 201)})
    labelled = add_rul(engine, cap=125).set_index("cycle")
    # the engine fails after cycle 200, so cycle c has 200 - c cycles left
    assert labelled.loc[1, ["rul", "rul_capped"]].tolist() == [199, 125]
    assert labelled.loc[75, ["rul", "rul_capped"]].tolist() == [125, 125]
    assert labelled.loc[76, ["rul", "rul_capped"]].tolist() == [124, 124]
    assert labelled.loc[200, ["rul", "rul_capped"]].tolist() == [0, 0]

    truncated = pd.DataFrame({"unit_id": 7, "cycle": np.arange(1, 51)})
    labelled = add_rul(truncated, cap=125, final_rul=pd.Series({7: 10})).set_index(
        "cycle"
    )
    # observed until cycle 50 with 10 cycles to go, so cycle 1 had 59 left
    assert labelled.loc[1, "rul"] == 59
    assert labelled.loc[50, "rul"] == 10


def test_parser_returns_tidy_typed_frame_and_rejects_gaps(tmp_path: Path) -> None:
    def write(rows: list[list[float]]) -> Path:
        path = tmp_path / "train_FD001.txt"
        # the raw files pad every line with trailing spaces
        path.write_text(
            "".join(" ".join(map(str, r)) + "  \n" for r in rows), encoding="utf-8"
        )
        return path

    rows = [[1, c, 0.0, 0.0, 100.0, *[float(c)] * 21] for c in (1, 2, 3)]
    rows.append([2, 1, 0.0, 0.0, 100.0, *[0.5] * 21])
    frame = read_trajectories(write(rows))
    assert list(frame.columns) == COLUMNS
    assert frame.shape == (4, 26)
    assert frame["unit_id"].dtype == np.int64 and frame["cycle"].dtype == np.int64
    assert (frame[SETTINGS + SENSORS].dtypes == np.float64).all()

    rows[2][1] = 4
    with pytest.raises(ValueError, match="without gaps"):
        read_trajectories(write(rows))


def test_training_statistics_ignore_appended_validation_units() -> None:
    train = _fleet([1, 2, 3, 4], seed=0, sensor_scale=1.0)
    # validation engines are wildly different, so any leak into the fit would show
    val = _fleet([5, 6], seed=1, sensor_scale=1000.0, altitude_shift=5.0)
    test = _fleet([1, 2], seed=2, sensor_scale=1.0)
    split = {"train": [1, 2, 3, 4], "val": [5, 6]}
    kwargs = {
        "n_conditions": 2,
        "max_modal_share": 0.95,
        "seed": 0,
        "condition_aware": True,
    }

    alone = prepare_subset(Subset("FD002", train, test), split, **kwargs)
    fleet = pd.concat([train, val], ignore_index=True)
    appended = prepare_subset(Subset("FD002", fleet, test), split, **kwargs)

    pd.testing.assert_frame_equal(alone.normaliser.mean, appended.normaliser.mean)
    pd.testing.assert_frame_equal(alone.normaliser.std, appended.normaliser.std)
    np.testing.assert_array_equal(alone.conditions.centers, appended.conditions.centers)
    pd.testing.assert_frame_equal(alone.train, appended.train)
    pd.testing.assert_frame_equal(alone.test, appended.test)
    assert set(appended.val["unit_id"]) == {5, 6}
