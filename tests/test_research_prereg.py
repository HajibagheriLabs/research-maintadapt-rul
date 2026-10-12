import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREREG = (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
FROZEN = (
    "configs/research_folds.json",
    "configs/research_pool.yaml",
    "docs/research/data_manifest.json",
)


@pytest.mark.parametrize("name", FROZEN)
def test_frozen_files_match_the_registered_sha256(name: str) -> None:
    # .gitattributes checks text out with LF everywhere, so the bytes hash the same on any OS
    registered = re.search(rf"{re.escape(name)}, SHA-256 ([0-9a-f]{{64}})", PREREG)
    assert registered is not None, f"PREREGISTRATION.md registers no hash for {name}"
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert actual == registered.group(1)


def test_no_fill_marker_is_left_for_the_code_to_fill() -> None:
    assert "[[TO FILL" not in PREREG
