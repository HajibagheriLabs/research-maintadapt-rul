import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maintadapt.data import load_subset
from maintadapt.research.config import FoldsConfig, load_research_config
from maintadapt.research.folds import (
    assign_blocks,
    fleet_folds,
    lifetimes,
    load_folds,
    make_folds,
    render_folds,
    save_folds,
    sort_by_lifetime,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = load_research_config(ROOT / "configs" / "research.yaml")
HAS_DATA = (ROOT / CFG.data.raw_dir / "train_FD001.txt").is_file()
TOY_CFG = FoldsConfig(n_outer=5, n_repeats=3, inner_val_fraction=0.25, seed=7)


def _toy_lives(n: int = 23) -> pd.Series:
    # repeated lifetimes exercise the unit-id tie break
    rng = np.random.default_rng(0)
    lives = rng.integers(128, 140, size=n)
    return pd.Series(lives, index=pd.Index(np.arange(1, n + 1) * 3, name="unit_id"))


def test_lifetime_order_breaks_ties_by_unit_id() -> None:
    lives = pd.Series([200, 150, 200, 150], index=[4, 9, 2, 1])
    assert sort_by_lifetime(lives).tolist() == [1, 9, 2, 4]
    frame = pd.DataFrame({"unit_id": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})
    assert lifetimes(frame).to_dict() == {1: 3, 2: 2}


def test_blocks_permute_the_labels_and_a_short_last_block_repeats_none() -> None:
    rng = np.random.default_rng(1)
    labels = assign_blocks(13, range(5), rng)
    for start in (0, 5):
        assert sorted(labels[start : start + 5]) == [0, 1, 2, 3, 4]
    assert len(set(labels[10:])) == 3


def test_toy_folds_follow_the_block_rule_and_partition_every_repeat() -> None:
    lives = _toy_lives()
    order = sort_by_lifetime(lives)
    repeats = fleet_folds(lives, TOY_CFG, "FD002")
    assert len(repeats) == 3 and all(len(r) == 5 for r in repeats)
    for repeat in repeats:
        fold_of = {u: k for k, f in enumerate(repeat) for u in f.outer_test}
        # every engine is tested exactly once per repeat
        assert sorted(fold_of) == sorted(lives.index)
        assert sum(len(f.outer_test) for f in repeat) == len(lives)
        sizes = [len(f.outer_test) for f in repeat]
        assert max(sizes) - min(sizes) <= 1
        # consecutive lifetime-sorted blocks of five spread over distinct folds
        labels = [fold_of[u] for u in order]
        for start in range(0, len(order), 5):
            block = labels[start : start + 5]
            assert len(set(block)) == len(block)
        for k, fold in enumerate(repeat):
            parts = [set(fold.inner_fit), set(fold.inner_val), set(fold.outer_test)]
            assert sum(map(len, parts)) == len(lives)
            assert set().union(*parts) == set(lives.index)
            train = [u for u in order if fold_of[u] != k]
            n_val = len(fold.inner_val)
            assert abs(n_val - 0.25 * len(train)) <= 1
            # one inner_val engine per full block of four outer-train engines
            for start in range(0, len(train), 4):
                block = train[start : start + 4]
                picked = sum(u in fold.inner_val for u in block)
                assert picked == 1 if len(block) == 4 else picked <= 1
    assert repeats[0][0].outer_test != repeats[1][0].outer_test


def test_regeneration_is_byte_identical_and_round_trips(tmp_path: Path) -> None:
    lives = {"FD001": _toy_lives(23), "FD003": _toy_lives(31)}
    text = render_folds(make_folds(lives, TOY_CFG), TOY_CFG)
    assert render_folds(make_folds(lives, TOY_CFG), TOY_CFG) == text
    path = tmp_path / "folds.json"
    save_folds(make_folds(lives, TOY_CFG), TOY_CFG, path)
    assert path.read_bytes() == text.encode("utf-8")
    loaded = load_folds(path, TOY_CFG, ["FD001", "FD003"])
    assert loaded == make_folds(lives, TOY_CFG)


def test_loader_refuses_other_settings_and_broken_folds(tmp_path: Path) -> None:
    path = tmp_path / "folds.json"
    save_folds(make_folds({"FD001": _toy_lives()}, TOY_CFG), TOY_CFG, path)
    other = FoldsConfig(n_outer=5, n_repeats=3, inner_val_fraction=0.25, seed=8)
    with pytest.raises(ValueError, match="other fold settings"):
        load_folds(path, other, ["FD001"])

    raw = json.loads(path.read_text(encoding="utf-8"))
    fold = raw["fleets"]["FD001"][1][2]
    fold["inner_fit"].append(fold["outer_test"][0])
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        load_folds(path, TOY_CFG, ["FD001"])


def test_committed_folds_load_under_the_research_config() -> None:
    folds = load_folds(ROOT / CFG.paths.folds, CFG.folds, CFG.data.fleets)
    expected = {"FD001": 100, "FD002": 260, "FD003": 100, "FD004": 249}
    assert {fleet: len(r[0][0].units()) for fleet, r in folds.items()} == expected


@pytest.mark.skipif(not HAS_DATA, reason="C-MAPSS files are not in data/raw/CMAPSS")
def test_committed_folds_regenerate_from_the_raw_data() -> None:
    lives = {
        fleet: lifetimes(
            load_subset(ROOT / CFG.data.raw_dir, fleet, CFG.data.cap).train
        )
        for fleet in CFG.data.fleets
    }
    committed = (ROOT / CFG.paths.folds).read_bytes()
    assert render_folds(make_folds(lives, CFG.folds), CFG.folds).encode() == committed
