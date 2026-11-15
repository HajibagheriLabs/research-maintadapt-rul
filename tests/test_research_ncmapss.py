from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from synthetic_fleet import CFG
from test_research_analysis import _inputs

from maintadapt.data import Subset
from maintadapt.research.analysis import decision_loss
from maintadapt.research.fold_data import build_ncmapss_fold_data
from maintadapt.research.folds import Fold
from maintadapt.research.ncmapss import (
    INPUT_GROUPS,
    LeakageError,
    SubDataset,
    build_frame,
    cap_from_onsets,
    check_inputs,
    flight_frame,
    open_failure,
    read_inputs,
    read_labels,
    sub_datasets,
    summarise,
    unit_blocks,
    unit_id,
)
from maintadapt.research.ncmapss_analysis import (
    NcmapssInputs,
    analyse_ncmapss,
    jitter,
    leaderboard_rmse,
)

W_VAR = ["alt", "Mach", "TRA", "T2"]
XS_VAR = ["T24", "Nf"]
OTHER = {"X_v": ["T40", "SmFan"], "T": ["HPT_eff_mod", "LPT_eff_mod"]}
# a value no input can ever take: if it reaches the frame, a forbidden group leaked
POISON = 1e12


def _unit(unit: int, durations: list[int], onset: int, fc: int, rng) -> dict:
    """1 Hz rows of one unit: flights of the given durations, abnormal from flight `onset`."""
    life = len(durations)
    cycle = np.repeat(np.arange(1, life + 1), durations)
    n = len(cycle)
    return {
        "W": rng.normal(size=(n, len(W_VAR))),
        "X_s": rng.normal(size=(n, len(XS_VAR))),
        "X_v": np.full((n, 2), POISON),
        "T": np.c_[np.linspace(0, -0.1, n), np.zeros(n)],
        "Y": (life - cycle)[:, None],
        "A": np.c_[
            np.full(n, unit), cycle, np.full(n, fc), (cycle < onset).astype(float)
        ],
    }


def _write(path: Path, dev: list[dict], test: list[dict]) -> Path:
    with h5py.File(path, "w") as h:
        for part, units in (("dev", dev), ("test", test)):
            for group in ("W", "X_s", "X_v", "T", "Y", "A"):
                h[f"{group}_{part}"] = np.concatenate([u[group] for u in units])
        h["W_var"] = np.array(W_VAR, dtype="S")
        h["X_s_var"] = np.array(XS_VAR, dtype="S")
        for group, cols in OTHER.items():
            h[f"{group}_var"] = np.array(cols, dtype="S")
        h["A_var"] = np.array(["unit", "cycle", "Fc", "hs"], dtype="S")
    return path


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    _write(
        tmp_path / "N-CMAPSS_DS01-005.h5",
        [_unit(1, [3, 2, 4, 3], 3, 1, rng), _unit(2, [2, 2, 2], 2, 3, rng)],
        [_unit(7, [5, 1, 2, 2, 3], 4, 2, rng)],
    )
    _write(
        tmp_path / "N-CMAPSS_DS02-006.h5",
        [_unit(2, [2, 3], 2, 1, rng)],
        [_unit(11, [4, 4, 4], 2, 1, rng)],
    )
    return tmp_path


def test_only_w_and_x_s_may_be_model_inputs(raw: Path) -> None:
    assert INPUT_GROUPS == ("W", "X_s")
    for forbidden in ("X_v", "T", "Y", "A"):
        with pytest.raises(LeakageError):
            check_inputs(["W", forbidden])
    with h5py.File(raw / "N-CMAPSS_DS01-005.h5", "r") as h:
        block = unit_blocks(h, "dev")[0]
        assert set(read_inputs(h, block)) == {"W", "X_s"}
        with pytest.raises(LeakageError):
            read_inputs(h, block, ["X_s", "X_v"])
        a, _ = read_labels(h, block)
        with pytest.raises(LeakageError):
            flight_frame({"X_v": np.zeros((12, 2))}, {"X_v": OTHER["X_v"]}, a)


def test_the_frame_never_carries_virtual_sensors_health_parameters_or_state(
    raw: Path,
) -> None:
    frame = build_frame(raw)
    columns = set(frame.flights.columns)
    forbidden = {c for cols in OTHER.values() for c in cols} | {"hs"}
    assert not {c for c in columns for f in forbidden if f in c}
    # the virtual sensors hold a poison value: none of it reaches any column
    numeric = frame.flights[frame.features].to_numpy()
    assert np.abs(numeric).max() < 1e6
    assert frame.features == [
        "duration",
        *[f"{s}_{v}" for v in W_VAR + XS_VAR for s in ("mean", "std")],
    ]


def test_flight_aggregates_by_hand() -> None:
    a = np.c_[np.ones(5), [1, 1, 2, 2, 2], np.full(5, 2), np.ones(5)]
    w = np.c_[[1.0, 3.0, 2.0, 2.0, 5.0], np.zeros((5, 3))]
    xs = np.c_[[0.0, 0.0, 1.0, 2.0, 3.0], np.ones(5)]
    out = flight_frame({"W": w, "X_s": xs}, {"W": W_VAR, "X_s": XS_VAR}, a)
    assert out["cycle"].tolist() == [1, 2] and out["fc"].tolist() == [2, 2]
    assert out["duration"].tolist() == [2.0, 3.0]
    assert out["mean_alt"].tolist() == [2.0, 3.0]
    # population deviations: (1, 3) about 2 gives 1; (2, 2, 5) about 3 gives sqrt(2)
    assert out["std_alt"].tolist() == pytest.approx([1.0, np.sqrt(2.0)])
    assert out["mean_T24"].tolist() == [0.0, 2.0]
    assert out["std_Nf"].tolist() == [0.0, 0.0]


def test_units_are_read_as_contiguous_blocks_and_checked(raw: Path) -> None:
    with h5py.File(raw / "N-CMAPSS_DS01-005.h5", "r") as h:
        blocks = unit_blocks(h, "dev")
        assert [(b.unit, b.start, b.stop) for b in blocks] == [(1, 0, 12), (2, 12, 18)]
        s = summarise(*read_labels(h, blocks[0]))
    assert (s.flights, s.flight_classes) == (4, (1,))
    assert s.rul_is_life_minus_cycle and s.health_state_monotone
    # healthy in flights 1 and 2, abnormal from flight 3, where RUL = 4 - 3 = 1
    assert (s.onset_flight, s.onset_rul) == (3, 1)


def test_the_cap_is_the_median_onset_rul_rounded_to_five() -> None:
    assert cap_from_onsets([40, 50, 60]) == 50
    assert cap_from_onsets([47, 48]) == 50
    assert cap_from_onsets([46, 47]) == 45
    # 62.5 is half way: it rounds up
    assert cap_from_onsets([60, 65]) == 65
    with pytest.raises(ValueError, match="onset"):
        cap_from_onsets([])


def test_units_get_ids_that_never_collide_across_files(raw: Path) -> None:
    frame = build_frame(raw).flights
    ids = frame.groupby(["sub_dataset", "unit"])["unit_id"].first().to_dict()
    assert ids == {
        ("DS01-005", 1): 101,
        ("DS01-005", 2): 102,
        ("DS01-005", 7): 107,
        ("DS02-006", 2): 202,
        ("DS02-006", 11): 211,
    }
    life = frame.groupby("unit_id")["cycle"].transform("max")
    assert (frame["rul"] == life - frame["cycle"]).all()
    with pytest.raises(ValueError, match="below"):
        unit_id(SubDataset(raw / "x.h5", 1), 100)


def test_a_truncated_file_is_reported_and_left_out(raw: Path) -> None:
    broken = raw / "N-CMAPSS_DS03-012.h5"
    data = (raw / "N-CMAPSS_DS02-006.h5").read_bytes()
    broken.write_bytes(data[:-32])
    assert "truncated" in open_failure(broken)
    assert open_failure(raw / "N-CMAPSS_DS02-006.h5") is None
    frame = build_frame(raw)
    assert [name for name, _ in frame.excluded] == ["N-CMAPSS_DS03-012.h5"]
    assert set(frame.flights["sub_dataset"]) == {"DS01-005", "DS02-006"}
    assert [s.number for s in sub_datasets(raw)] == [1, 2, 3]


# the per-flight frame inside the pipeline


def _flights() -> pd.DataFrame:
    rows = []
    for unit, fc, life in ((1, 1, 4), (2, 1, 3), (3, 2, 3), (4, 2, 2), (5, 1, 3)):
        for cycle in range(1, life + 1):
            rows.append(
                {
                    "unit_id": unit,
                    "cycle": cycle,
                    "fc": fc,
                    "rul": life - cycle,
                    "rul_capped": min(life - cycle, 2),
                    "duration": 10.0 * fc,
                    "mean_alt": float(unit + cycle),
                }
            )
    return pd.DataFrame(rows)


def test_features_are_standardised_within_flight_class_on_inner_fit_only() -> None:
    flights = _flights()
    subset = Subset("NCMAPSS", flights, flights.iloc[:0])
    fold = Fold(inner_fit=(1, 3), inner_val=(2,), outer_test=(4, 5))
    cfg = replace(CFG.data, cap=2)
    data = build_ncmapss_fold_data(
        subset, fold, ["duration", "mean_alt"], cfg, repeat=0, index=0
    )
    assert data.sensors == ["duration", "mean_alt", "fc_1", "fc_2", "fc_3"]
    # class 1 on inner_fit is unit 1 alone: mean_alt 2..5, mean 3.5, sd sqrt(1.25)
    val = data.frame("inner_val")
    assert val["mean_alt"].tolist() == pytest.approx(
        [(v - 3.5) / np.sqrt(1.25) for v in (3.0, 4.0, 5.0)]
    )
    # duration never varies within a class: it centres to zero instead of failing
    assert (data.frame("outer_test")["duration"] == 0).all()
    assert data.frame("outer_test")["fc_2"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert data.frame("official_test").empty and data.cap == 2
    with pytest.raises(ValueError, match="never flies"):
        build_ncmapss_fold_data(
            subset,
            Fold(inner_fit=(1, 2), inner_val=(3,), outer_test=(4, 5)),
            ["duration", "mean_alt"],
            cfg,
            repeat=0,
            index=0,
        )


def test_ncmapss_accuracy_metrics_pool_each_fits_own_units_by_hand() -> None:
    members = np.array([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    inputs = NcmapssInputs(
        base=None,
        sse=np.array([[[4.0, 4.0, 9.0, 27.0]]]),
        rows=np.array([[[1.0, 1.0, 1.0, 3.0]]]),
        jit_sum=np.array([[[10.0, 99.0, 6.0, 30.0]]]),
        jit_n=np.array([[[5.0, 2.0, 6.0, 10.0]]]),
        members=members,
    )
    once = np.ones((1, 4))
    # fold 0: sqrt(8 / 2) = 2; fold 1: sqrt(36 / 4) = 3
    assert leaderboard_rmse(inputs, once).tolist() == [[2.5]]
    # unit 2 has two steps only; fold 0 is unit 1 alone (2), fold 1 averages 1 and 3
    assert jitter(inputs, once).tolist() == [[2.0]]
    # drawing unit 4 three times and unit 3 never: fold 1 becomes sqrt(81 / 9) = 3
    assert leaderboard_rmse(inputs, np.array([[1.0, 1.0, 0.0, 3.0]])).tolist() == [
        [2.5]
    ]


def test_ncmapss_endpoints_follow_the_confirmatory_formulas() -> None:
    base = _inputs([1.0, 2.0, 3.0], [0, 3, 6])
    n = base.cost.shape[-1]
    members = np.stack([base.fold_of == k for k in range(3)], axis=1).astype(float)
    rows = np.ones((3, 2, n))
    inputs = NcmapssInputs(
        base=base,
        sse=np.array([1.0, 4.0, 9.0])[:, None, None] * rows,
        rows=rows,
        jit_sum=np.array([3.0, 2.0, 1.0])[:, None, None] * 6 * rows,
        jit_n=6 * rows,
        members=members,
    )
    r = analyse_ncmapss(inputs, 50, 0.95, [2026, 5])
    # D rises with failures, RMSE with the configuration, jitter falls: tau-b 1 and -1
    assert r.e1.point == pytest.approx(1.0) and r.tau_jitter.point == pytest.approx(
        -1.0
    )
    assert r.e3.point == pytest.approx(-2.0)
    assert r.points["decision_loss"].tolist() == pytest.approx(
        decision_loss(base, np.ones((1, n)))[0].tolist()
    )
