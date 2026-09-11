"""Experiment configuration: YAML on disk, typed dataclasses in code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

SUBSETS = ("FD001", "FD002", "FD003", "FD004")


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: str
    splits: str
    reports: str
    runs: str
    mlruns: str


@dataclass(frozen=True)
class DataConfig:
    subsets: list[str]
    rul_cap: int
    val_fraction: float
    split_seed: int
    n_conditions: dict[str, int]
    max_modal_share: float

    def __post_init__(self) -> None:
        if not self.subsets or not set(self.subsets) <= set(SUBSETS):
            raise ValueError(
                f"data.subsets must be drawn from {SUBSETS}, got {self.subsets}"
            )
        if self.rul_cap <= 0:
            raise ValueError(f"data.rul_cap must be positive, got {self.rul_cap}")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(
                f"data.val_fraction must lie in (0, 1), got {self.val_fraction}"
            )
        missing = sorted(set(self.subsets) - set(self.n_conditions))
        if missing or min(self.n_conditions.values()) < 1:
            raise ValueError(
                f"data.n_conditions needs a positive count per subset, missing {missing}"
            )
        if not 0.5 < self.max_modal_share <= 1.0:
            raise ValueError(
                f"data.max_modal_share must lie in (0.5, 1], got {self.max_modal_share}"
            )


@dataclass(frozen=True)
class Config:
    seeds: list[int]
    paths: PathsConfig
    data: DataConfig

    def __post_init__(self) -> None:
        if len(self.seeds) < 5 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError(
                f"seeds must hold at least five distinct values, got {self.seeds}"
            )


def load_config(path: str | Path) -> Config:
    """Load a YAML config; unknown, missing or mistyped keys raise."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _build(Config, raw, "config")


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(asdict(cfg), fh, sort_keys=False)


def _build(cls: type, raw: Any, where: str) -> Any:
    if not isinstance(raw, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(raw).__name__}")
    names = [f.name for f in fields(cls)]
    unknown, missing = sorted(set(raw) - set(names)), sorted(set(names) - set(raw))
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {unknown}, missing keys {missing}")
    hints = get_type_hints(cls)
    kwargs = {}
    for name in names:
        hint, value, key = hints[name], raw[name], f"{where}.{name}"
        kwargs[name] = (
            _build(hint, value, key) if is_dataclass(hint) else _check(value, hint, key)
        )
    return cls(**kwargs)


def _check(value: Any, hint: Any, where: str) -> Any:
    origin = get_origin(hint)
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"{where}: expected a list, got {type(value).__name__}")
        (item,) = get_args(hint)
        return [_check(v, item, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected a mapping, got {type(value).__name__}")
        key_hint, value_hint = get_args(hint)
        return {
            _check(k, key_hint, where): _check(v, value_hint, f"{where}.{k}")
            for k, v in value.items()
        }
    # bool subclasses int, so a stray `true` must not pass as a number
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if hint is float and is_number:
        return float(value)
    if hint is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if hint in (str, bool) and isinstance(value, hint):
        return value
    raise TypeError(f"{where}: expected {hint.__name__}, got {type(value).__name__}")
