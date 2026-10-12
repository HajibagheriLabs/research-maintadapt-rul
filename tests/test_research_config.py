import os
from pathlib import Path

import pytest
import yaml

from maintadapt.research.config import fleet_number, load_research_config

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "configs" / "research.yaml"


def _raw() -> dict:
    return yaml.safe_load(RESEARCH.read_text(encoding="utf-8"))


def _load(tmp_path: Path, raw: dict):
    path = tmp_path / "research.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_research_config(path)


def test_committed_research_config_holds_the_registered_values() -> None:
    cfg = load_research_config(RESEARCH)
    assert cfg.data.fleets == ["FD001", "FD002", "FD003", "FD004"]
    assert [fleet_number(f) for f in cfg.data.fleets] == [1, 2, 3, 4]
    assert cfg.data.cap == 125
    assert cfg.data.n_conditions == {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
    assert (cfg.folds.n_outer, cfg.folds.n_repeats, cfg.folds.inner_block) == (5, 3, 4)
    assert cfg.pool.total == 113
    assert cfg.decision.cost_ratios == [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    p = cfg.decision.primary
    assert (p.cost_ratio, p.lead_time, p.rule, p.tuning) == (
        10.0,
        10,
        "R1",
        "validation",
    )
    assert (cfg.stats.n_boot, cfg.stats.ci, cfg.stats.seed) == (2000, 0.95, 2026)


def test_cpu_jobs_defaults_to_the_cpu_count_minus_two(tmp_path: Path) -> None:
    raw = _raw()
    assert raw["training"]["cpu_jobs"] is None
    expected = max(1, (os.cpu_count() or 1) - 2)
    assert _load(tmp_path, raw).training.resolved_cpu_jobs() == expected

    raw["training"]["cpu_jobs"] = 3
    assert _load(tmp_path, raw).training.resolved_cpu_jobs() == 3
    # an optional integer still refuses a string or a stray boolean
    for bad in ("4", True):
        raw["training"]["cpu_jobs"] = bad
        with pytest.raises(TypeError, match="cpu_jobs"):
            _load(tmp_path, raw)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("folds", "inner_val_fraction", 0.3, "inner_val_fraction"),
        ("folds", "n_outer", 1, "n_outer"),
        ("data", "fleets", ["FD001", "FD005"], "fleets"),
        ("decision", "cost_ratios", [1.0, 10.0], "cost_ratios"),
        ("decision", "ema_alphas", [0.5, 0.3], "ema_alphas"),
        (
            "decision",
            "primary",
            {"cost_ratio": 7, "lead_time": 10, "rule": "R1", "tuning": "validation"},
            "primary",
        ),
        ("pool", "n_configs", {"ridge": 5}, "n_configs"),
        ("stats", "ci", 95, "ci"),
        ("training", "device", "gpu", "device"),
        ("data", "unexpected", 1, "unknown keys"),
    ],
)
def test_bad_research_values_raise_clear_errors(
    tmp_path: Path, section: str, key: str, value: object, message: str
) -> None:
    raw = _raw()
    raw[section][key] = value
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, raw)


def test_unknown_fleet_has_no_number() -> None:
    with pytest.raises(ValueError, match="unknown fleet"):
        fleet_number("NCMAPSS")
