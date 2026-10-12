import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from synthetic_fleet import fold_data, repeats, subset

from maintadapt.research.archive import (
    FAILED,
    OK,
    crossfit_key,
    fit_status,
    load,
    prediction_table,
    validate_fleet,
    write_fit,
)
from maintadapt.research.fold_data import PREDICTED

CAP = 125
HASHES = {"ridge-00": "aaaa", "ridge-01": "bbbb"}


def _record(config_id: str, repeat: int, fold: int, status: str = OK) -> dict:
    return {
        "fleet": "FD001",
        "family": "ridge",
        "config_id": config_id,
        "config_hash": HASHES[config_id],
        "repeat": repeat,
        "fold": fold,
        "status": status,
        "n_rows": None,
    }


def _write(root: Path, config_id: str, repeat: int, fold: int) -> Path:
    """A fit whose predictions are the capped truth, written like the runner does."""
    data = fold_data(subset(), repeats()[repeat][fold], repeat, fold)
    predictions = {s: data.target(s).to_numpy() for s in PREDICTED}
    table = prediction_table(data, predictions)
    key = crossfit_key("FD001", "ridge", config_id, repeat, fold)
    write_fit(
        root, key, table, {**_record(config_id, repeat, fold), "n_rows": table.num_rows}
    )
    return key.parquet(root)


def _validate(root: Path) -> list[str]:
    problems, _ = validate_fleet(root, "FD001", subset(), repeats(), CAP, HASHES)
    return problems


def test_fits_round_trip_with_their_types_and_load_filters(tmp_path: Path) -> None:
    _write(tmp_path, "ridge-00", 0, 0)
    _write(tmp_path, "ridge-01", 0, 1)
    key = crossfit_key("FD001", "ridge", "ridge-01", 1, 0)
    write_fit(tmp_path, key, None, _record("ridge-01", 1, 0, FAILED))

    every = load(tmp_path)
    types = every.dtypes.astype(str).to_dict()
    assert {k: types[k] for k in ("unit_id", "cycle", "rul", "pred", "repeat")} == {
        "unit_id": "int32",
        "cycle": "int32",
        "rul": "int32",
        "pred": "float32",
        "repeat": "int32",
    }
    assert str(every["split"].dtype) == "category"
    # the failed fit has no predictions and is never loaded
    assert set(zip(every["config_id"], every["fold"], strict=True)) == {
        ("ridge-00", 0),
        ("ridge-01", 1),
    }
    data = fold_data(subset(), repeats()[0][0])
    outer = load(tmp_path, config_ids=["ridge-00"], splits=["outer_test"])
    assert set(outer["split"].astype(str)) == {"outer_test"}
    assert len(outer) == len(data.frame("outer_test"))
    narrow = load(tmp_path, folds=[1], columns=["unit_id", "pred", "fold"])
    assert list(narrow.columns) == ["unit_id", "pred", "fold"]
    assert (narrow["fold"] == 1).all()
    assert load(tmp_path, families=["knn"]).empty


def test_the_sidecar_marks_a_fit_finished_and_writes_leave_no_temporary_files(
    tmp_path: Path,
) -> None:
    key = crossfit_key("FD001", "ridge", "ridge-00", 0, 0)
    assert fit_status(tmp_path, key) is None
    parquet = _write(tmp_path, "ridge-00", 0, 0)
    assert fit_status(tmp_path, key) == OK and parquet.exists()
    assert not list(tmp_path.rglob("*.tmp"))
    # a failure recorded later removes the stale predictions with it
    write_fit(tmp_path, key, None, _record("ridge-00", 0, 0, FAILED))
    assert fit_status(tmp_path, key) == FAILED and not parquet.exists()
    with pytest.raises(ValueError, match="prediction table"):
        write_fit(tmp_path, key, None, _record("ridge-00", 0, 0))


def test_prediction_table_refuses_a_prediction_count_that_differs_from_the_rows() -> (
    None
):
    data = fold_data(subset())
    predictions = {s: data.target(s).to_numpy() for s in PREDICTED}
    predictions["outer_test"] = predictions["outer_test"][:-1]
    with pytest.raises(ValueError, match="outer_test"):
        prediction_table(data, predictions)


def test_validate_accepts_complete_fits(tmp_path: Path) -> None:
    _write(tmp_path, "ridge-00", 0, 0)
    _write(tmp_path, "ridge-01", 1, 4)
    problems, counts = validate_fleet(
        tmp_path, "FD001", subset(), repeats(), CAP, HASHES
    )
    assert problems == []
    assert counts["ok"] == 2


def _tamper(parquet: Path, change) -> None:
    table = pq.read_table(parquet).to_pandas()
    pq.write_table(pa.Table.from_pandas(change(table), preserve_index=False), parquet)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        # cycle 3 of the first official test engine vanishes: cycles are no longer 1..T
        (
            lambda t: t.drop(
                t.index[(t["split"] == "official_test") & (t["cycle"] == 3)][:1]
            ),
            "cycles or truth differ",
        ),
        (lambda t: t.drop(t.index[t["unit_id"] == t["unit_id"].iloc[0]]), "missing"),
        (lambda t: t.drop(t.index[t["split"] == "inner_val"][1:]), "missing"),
        (lambda t: t.iloc[np.r_[0, np.arange(len(t))]], "duplicated"),
        (
            lambda t: t.assign(pred=np.where(t.index == 5, np.nan, t["pred"])),
            "non-finite",
        ),
        (lambda t: t.assign(pred=t["pred"] + 200), "outside"),
        (lambda t: t.assign(rul=t["rul"] + 1), "cycles or truth differ"),
    ],
)
def test_validate_reports_each_kind_of_damage(
    tmp_path: Path, change, message: str
) -> None:
    parquet = _write(tmp_path, "ridge-00", 0, 2)
    _tamper(parquet, change)
    problems = _validate(tmp_path)
    assert any(message in p for p in problems), problems


def test_validate_reports_foreign_configurations_and_orphan_files(
    tmp_path: Path,
) -> None:
    parquet = _write(tmp_path, "ridge-00", 0, 0)
    sidecar = parquet.with_suffix(".json")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar.write_text(json.dumps({**record, "config_hash": "zzzz"}), encoding="utf-8")
    assert any("frozen pool" in p for p in _validate(tmp_path))
    sidecar.unlink()
    assert any("without a sidecar" in p for p in _validate(tmp_path))
