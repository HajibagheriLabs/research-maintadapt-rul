from pathlib import Path

import pytest
import yaml

from maintadapt.config import load_config

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_default_config_loads_and_bad_values_raise(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT)
    assert cfg.data.rul_cap > 0
    assert len(cfg.seeds) >= 5

    raw = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    raw["data"]["val_fraction"] = 1.5
    with pytest.raises(ValueError, match="val_fraction"):
        load_config(_write(tmp_path, raw))

    raw["data"]["val_fraction"] = 0.2
    raw["data"]["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(_write(tmp_path, raw))
