import re
from pathlib import Path

import pytest
import yaml

from maintadapt.config import load_config
from maintadapt.research.config import load_research_config
from maintadapt.research.models.spaces import (
    FAMILIES,
    FAMILY_NAMES,
    Uniform,
    build_pool,
    config_hash,
    load_pool,
    render_pool,
    round_sig,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = load_research_config(ROOT / "configs" / "research.yaml")
LEGACY = load_config(ROOT / "configs" / "default.yaml")
POOL = build_pool(CFG.pool.seed, CFG.pool.n_configs)
BY_ID = {c.id: c.params for c in POOL}


def test_pool_is_deterministic_and_matches_the_committed_file() -> None:
    again = build_pool(CFG.pool.seed, CFG.pool.n_configs)
    assert [(c.id, c.hash, c.params) for c in again] == [
        (c.id, c.hash, c.params) for c in POOL
    ]
    committed = (ROOT / CFG.paths.pool).read_bytes()
    assert render_pool(CFG.pool.seed, CFG.pool.n_configs).encode() == committed
    loaded = load_pool(ROOT / CFG.paths.pool, CFG.pool.seed, CFG.pool.n_configs)
    assert [(c.id, c.params) for c in loaded] == [(c.id, c.params) for c in POOL]


def test_ids_and_hashes_are_unique_and_counts_match_the_config() -> None:
    assert len(POOL) == 113
    assert len({c.id for c in POOL}) == len({c.hash for c in POOL}) == 113
    for name in FAMILY_NAMES:
        ids = [c.id for c in POOL if c.family == name]
        assert ids == [f"{name}-{i:02d}" for i in range(CFG.pool.n_configs[name])]
    for c in POOL:
        assert c.hash == config_hash(c.family, c.params)


def test_every_value_lies_inside_its_space_and_shared_settings_apply() -> None:
    for family in FAMILIES:
        configs = [c.params for c in POOL if c.family == family.name]
        keys = set(configs[0])
        # no duplicates: the content hash differs for every configuration
        assert len({config_hash(family.name, p) for p in configs}) == len(configs)
        for i, params in enumerate(configs):
            assert set(params) == keys
            for key, dist in family.space.items():
                assert dist.contains(params[key]), (family.name, i, key)
                if isinstance(dist, Uniform):
                    assert params[key] == round_sig(params[key])
            for key, value in family.shared.items():
                assert params[key] == value
            fixed = family.reference if i == 0 else family.others
            for key in family.others:
                assert params[key] == fixed[key]


def test_grid_families_list_the_remaining_grid_points_in_order() -> None:
    assert [BY_ID[f"ridge-{i:02d}"]["alpha"] for i in range(5)] == [
        1.0,
        0.1,
        10.0,
        100.0,
        1000.0,
    ]
    knn = [tuple(BY_ID[f"knn-{i:02d}"].values()) for i in range(12)]
    assert knn[0] == (30, "uniform")
    grid = [(k, w) for k in (5, 10, 20, 30, 50, 100) for w in ("uniform", "distance")]
    assert knn[1:] == [g for g in grid if g != (30, "uniform")]


def test_reference_configurations_equal_the_legacy_settings() -> None:
    xgb = LEGACY.baselines.xgboost
    legacy_xgb = {
        "max_depth": xgb.max_depth,
        "learning_rate": xgb.learning_rate,
        "min_child_weight": xgb.min_child_weight,
        "subsample": xgb.subsample,
        "colsample_bytree": xgb.colsample_bytree,
        "n_estimators": xgb.n_estimators,
        "early_stopping_rounds": xgb.early_stopping_rounds,
        # the legacy model left reg_lambda at XGBoost's default
        "reg_lambda": 1.0,
    }
    assert BY_ID["xgboost-00"] == BY_ID["xgboost_raw-00"] == legacy_xgb

    seq, train = LEGACY.sequence, LEGACY.sequence.training
    shared = {
        "dropout": seq.dropout,
        "lr": train.lr,
        "window": seq.window,
        "batch_size": train.batch_size,
        "weight_decay": train.weight_decay,
    }
    assert BY_ID["lstm-00"] == {
        "hidden": seq.lstm.hidden,
        "layers": seq.lstm.layers,
        **shared,
    }
    assert BY_ID["tcn-00"] == {
        "channels": seq.tcn.channels,
        "kernel_size": seq.tcn.kernel_size,
        "dilations": seq.tcn.dilations,
        **shared,
    }


def test_other_reference_configurations_follow_the_registered_table() -> None:
    assert BY_ID["ridge-00"] == {"alpha": 1.0}
    assert BY_ID["knn-00"] == {"n_neighbors": 30, "weights": "uniform"}
    assert BY_ID["forest-00"] == {
        "max_features": 0.3,
        "min_samples_leaf": 5,
        "max_depth": None,
        "n_estimators": 200,
        "max_samples": 0.5,
    }
    assert BY_ID["mlp-00"] == {
        "hidden": [128, 64],
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
    }
    assert BY_ID["cnn-00"] == {
        "filters": 10,
        "kernel_size": 10,
        "conv_layers": 4,
        "fc": 100,
        "dropout": 0.5,
        "lr": 1e-3,
        "window": 30,
        "batch_size": 512,
        "weight_decay": 1e-4,
    }
    assert BY_ID["transformer-00"] == {
        "d_model": 64,
        "heads": 4,
        "layers": 2,
        "ff": 128,
        "dropout": 0.1,
        "lr": 1e-3,
        "window": 30,
        "batch_size": 256,
        "weight_decay": 1e-4,
    }
    for c in POOL:
        if c.family == "transformer":
            assert c.params["d_model"] % c.params["heads"] == 0


def test_loader_refuses_edited_or_mismatched_pools(tmp_path: Path) -> None:
    text = (ROOT / CFG.paths.pool).read_text(encoding="utf-8")
    path = tmp_path / "pool.yaml"
    path.write_text(text.replace("alpha: 1000.0", "alpha: 999.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="ridge-04 does not match its hash"):
        load_pool(path, CFG.pool.seed, CFG.pool.n_configs)
    with pytest.raises(ValueError, match="other pool settings"):
        load_pool(ROOT / CFG.paths.pool, CFG.pool.seed + 1, CFG.pool.n_configs)


def test_committed_pool_uses_no_yaml_aliases_and_one_line_per_config() -> None:
    text = (ROOT / CFG.paths.pool).read_text(encoding="utf-8")
    assert not re.search(r"[&*]id\d", text)
    assert len(re.findall(r"^    - \{id: ", text, flags=re.MULTILINE)) == 113
    assert yaml.safe_load(text)["total"] == 113
