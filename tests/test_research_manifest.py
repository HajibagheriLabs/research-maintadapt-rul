import json
from pathlib import Path

import pytest

from maintadapt.research.config import load_research_config
from maintadapt.research.manifest import (
    build_manifest,
    compare_manifests,
    file_entry,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = load_research_config(ROOT / "configs" / "research.yaml")
HAS_DATA = (ROOT / CFG.data.raw_dir / "train_FD001.txt").is_file()


def _rows(value: float) -> list[str]:
    readings = " ".join(["1.5"] * 3 + [str(value)] * 21)
    return [f"{u} {c} {readings}" for u in (1, 2) for c in (1, 2, 3)]


def _write_fleet(root: Path, value: float, newline: str = "\n", pad: str = "") -> None:
    root.mkdir(parents=True, exist_ok=True)
    for kind in ("train", "test"):
        text = newline.join(line + pad for line in _rows(value)) + newline
        (root / f"{kind}_FD001.txt").write_bytes(text.encode())
    (root / "RUL_FD001.txt").write_bytes(f"7{newline}12{newline}".encode())


def test_entries_count_rows_and_units_and_hash_parsed_values(tmp_path: Path) -> None:
    _write_fleet(tmp_path / "lf", 2.0)
    entry = file_entry(tmp_path / "lf" / "train_FD001.txt")
    assert (entry["rows"], entry["units"]) == (6, 2)
    assert len(entry["sha256"]) == 64
    assert file_entry(tmp_path / "lf" / "RUL_FD001.txt")["rows"] == 2


def test_line_endings_and_whitespace_do_not_change_the_manifest(tmp_path: Path) -> None:
    _write_fleet(tmp_path / "lf", 2.0)
    _write_fleet(tmp_path / "crlf", 2.0, newline="\r\n", pad="  ")
    lf = build_manifest(tmp_path / "lf", ["FD001"])
    assert compare_manifests(lf, build_manifest(tmp_path / "crlf", ["FD001"])) == []


def test_a_changed_or_missing_value_is_reported(tmp_path: Path) -> None:
    _write_fleet(tmp_path / "a", 2.0)
    _write_fleet(tmp_path / "b", 2.5)
    recorded = build_manifest(tmp_path / "a", ["FD001"])
    problems = compare_manifests(recorded, build_manifest(tmp_path / "b", ["FD001"]))
    assert [p.split(" is ")[0] for p in problems] == [
        "test_FD001.txt: sha256",
        "train_FD001.txt: sha256",
    ]
    del recorded["files"]["RUL_FD001.txt"]
    found = build_manifest(tmp_path / "a", ["FD001"])
    assert compare_manifests(recorded, found) == [
        "RUL_FD001.txt: not in the recorded manifest"
    ]


@pytest.mark.skipif(not HAS_DATA, reason="C-MAPSS files are not in data/raw/CMAPSS")
def test_raw_data_match_the_committed_manifest() -> None:
    recorded = json.loads((ROOT / CFG.paths.manifest).read_text(encoding="utf-8"))
    found = build_manifest(ROOT / CFG.data.raw_dir, CFG.data.fleets)
    assert compare_manifests(recorded, found) == []
    rows = {name: entry["rows"] for name, entry in recorded["files"].items()}
    assert rows["train_FD001.txt"] == 20631 and rows["RUL_FD004.txt"] == 248
