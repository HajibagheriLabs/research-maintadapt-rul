"""Hyperparameter spaces of the ten model families and the frozen pool drawn from them."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

MAX_DRAWS = 10_000


def round_sig(value: float, digits: int = 3) -> float:
    return float(f"{value:.{digits}g}")


def _same(a: Any, b: Any) -> bool:
    # bool is an int, and 1 == 1.0, so equal-looking values of another type must not match
    return type(a) is type(b) and a == b


def _seen(params: dict[str, Any], found: list[dict[str, Any]]) -> bool:
    return any(params == other for other in found)


class _PoolDumper(yaml.SafeDumper):
    # a list that appears twice would otherwise be written once and aliased
    def ignore_aliases(self, data: Any) -> bool:
        return True


class _FlowMap(dict):
    pass


class _FlowList(list):
    pass


def _flow(value: Any) -> Any:
    """Mark a value for one-line flow style, so each configuration takes one line."""
    if isinstance(value, dict):
        return _FlowMap({k: _flow(v) for k, v in value.items()})
    if isinstance(value, list):
        return _FlowList(_flow(v) for v in value)
    return value


_PoolDumper.add_representer(
    _FlowMap,
    lambda d, v: d.represent_mapping("tag:yaml.org,2002:map", v, flow_style=True),
)
_PoolDumper.add_representer(
    _FlowList,
    lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True),
)


@dataclass(frozen=True, eq=False)
class Choice:
    options: list[Any]

    def draw(self, rng: np.random.Generator) -> Any:
        return self.options[int(rng.integers(len(self.options)))]

    def contains(self, value: Any) -> bool:
        return any(_same(value, option) for option in self.options)

    def describe(self) -> list[Any]:
        return list(self.options)


@dataclass(frozen=True, eq=False)
class Uniform:
    low: float
    high: float
    log: bool = False

    def draw(self, rng: np.random.Generator) -> float:
        if self.log:
            value = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        else:
            value = rng.uniform(self.low, self.high)
        return round_sig(value)

    def contains(self, value: Any) -> bool:
        return isinstance(value, float) and self.low <= value <= self.high

    def describe(self) -> str:
        kind = "log-uniform" if self.log else "uniform"
        return f"{kind} [{self.low:g}, {self.high:g}]"


@dataclass(frozen=True, eq=False)
class Family:
    """A model family: its inputs, reference configuration (config 0) and search space."""

    name: str
    inputs: str
    space: dict[str, Choice | Uniform]
    reference: dict[str, Any]
    # "all:" settings, shared by every configuration of the family, the reference included
    shared: dict[str, Any] = field(default_factory=dict)
    # fixed settings of the non-reference configurations only
    others: dict[str, Any] = field(default_factory=dict)
    # grid families list their remaining grid points in order instead of sampling
    grid: bool = False

    def __post_init__(self) -> None:
        if set(self.space) - set(self.reference):
            raise ValueError(f"{self.name}: the reference must set every space key")
        extra = set(self.reference) - set(self.space)
        if extra != set(self.others) or set(self.shared) & set(self.reference):
            raise ValueError(
                f"{self.name}: settings outside the space must be fixed for every config"
            )
        bad = [k for k in self.space if not self.space[k].contains(self.reference[k])]
        if bad:
            raise ValueError(
                f"{self.name}: reference values {bad} lie outside the space"
            )

    def complete(self, drawn: dict[str, Any], *, reference: bool) -> dict[str, Any]:
        fixed = (
            {k: self.reference[k] for k in self.others} if reference else self.others
        )
        params = {**{k: drawn[k] for k in self.space}, **fixed, **self.shared}
        # a copy, so that editing one configuration's list never edits the space
        return copy.deepcopy(params)


@dataclass(frozen=True, eq=False)
class ModelConfig:
    id: str
    family: str
    hash: str
    params: dict[str, Any]


LR = Uniform(3e-4, 3e-3, log=True)
WINDOW = Choice([20, 30, 40])
NEURAL = {"batch_size": 256, "weight_decay": 1e-4}
# the legacy XGBoost settings of configs/default.yaml; reg_lambda was XGBoost's default of 1
LEGACY_XGBOOST = {
    "max_depth": 5,
    "learning_rate": 0.03,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
}
XGBOOST_SPACE = {
    "max_depth": Choice([3, 4, 5, 6, 8]),
    "learning_rate": Uniform(0.01, 0.1, log=True),
    "min_child_weight": Choice([1.0, 3.0, 5.0, 10.0, 20.0]),
    "subsample": Uniform(0.6, 1.0),
    "colsample_bytree": Uniform(0.3, 1.0),
    "reg_lambda": Uniform(0.1, 10.0, log=True),
}

FAMILIES: tuple[Family, ...] = (
    Family(
        "ridge",
        "rolling features, scaled",
        {"alpha": Choice([0.1, 1.0, 10.0, 100.0, 1000.0])},
        {"alpha": 1.0},
        grid=True,
    ),
    Family(
        "knn",
        "rolling features, scaled",
        {
            "n_neighbors": Choice([5, 10, 20, 30, 50, 100]),
            "weights": Choice(["uniform", "distance"]),
        },
        {"n_neighbors": 30, "weights": "uniform"},
        grid=True,
    ),
    Family(
        "forest",
        "rolling features",
        {
            "max_features": Choice([0.1, 0.2, 0.3, 0.5]),
            "min_samples_leaf": Choice([1, 5, 10, 20]),
            "max_depth": Choice([None, 12, 20]),
        },
        {"max_features": 0.3, "min_samples_leaf": 5, "max_depth": None},
        shared={"n_estimators": 200, "max_samples": 0.5},
    ),
    Family(
        "xgboost",
        "rolling features",
        XGBOOST_SPACE,
        LEGACY_XGBOOST,
        shared={"early_stopping_rounds": 100},
        others={"n_estimators": 3000},
    ),
    Family(
        "xgboost_raw",
        "cycle and current readings",
        XGBOOST_SPACE,
        LEGACY_XGBOOST,
        shared={"early_stopping_rounds": 100},
        others={"n_estimators": 3000},
    ),
    Family(
        "mlp",
        "rolling features, scaled",
        {
            "hidden": Choice([[64], [128, 64], [256, 128], [256, 128, 64]]),
            "dropout": Choice([0.0, 0.1, 0.2]),
            "lr": LR,
            "weight_decay": Choice([0.0, 1e-5, 1e-4]),
        },
        {"hidden": [128, 64], "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
        shared={"batch_size": 256},
    ),
    Family(
        "lstm",
        "windows",
        {
            "hidden": Choice([32, 64, 128]),
            "layers": Choice([1, 2, 3]),
            "dropout": Choice([0.1, 0.2, 0.3, 0.5]),
            "lr": LR,
            "window": WINDOW,
        },
        {"hidden": 64, "layers": 2, "dropout": 0.2, "lr": 1e-3, "window": 30},
        shared=NEURAL,
    ),
    Family(
        "tcn",
        "windows",
        {
            "channels": Choice([32, 64, 128]),
            "kernel_size": Choice([2, 3, 5]),
            # three, four or five residual blocks
            "dilations": Choice([[1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8, 16]]),
            "dropout": Choice([0.1, 0.2, 0.3]),
            "lr": LR,
            "window": WINDOW,
        },
        {
            "channels": 64,
            "kernel_size": 3,
            "dilations": [1, 2, 4, 8],
            "dropout": 0.2,
            "lr": 1e-3,
            "window": 30,
        },
        shared=NEURAL,
    ),
    Family(
        "cnn",
        "windows",
        {
            "filters": Choice([10, 16, 32]),
            "kernel_size": Choice([5, 10, 15]),
            "conv_layers": Choice([3, 4, 5]),
            "fc": Choice([50, 100, 200]),
            "dropout": Choice([0.2, 0.5]),
            "lr": LR,
            "window": WINDOW,
        },
        # Li, Ding and Sun (2018); the final 1-filter convolution of kernel 3 is fixed
        {
            "filters": 10,
            "kernel_size": 10,
            "conv_layers": 4,
            "fc": 100,
            "dropout": 0.5,
            "lr": 1e-3,
            "window": 30,
        },
        shared={**NEURAL, "batch_size": 512},
    ),
    Family(
        "transformer",
        "windows",
        {
            "d_model": Choice([32, 64, 128]),
            "heads": Choice([2, 4]),
            "layers": Choice([1, 2, 3]),
            "ff": Choice([64, 128, 256]),
            "dropout": Choice([0.1, 0.2]),
            "lr": LR,
            "window": WINDOW,
        },
        {
            "d_model": 64,
            "heads": 4,
            "layers": 2,
            "ff": 128,
            "dropout": 0.1,
            "lr": 1e-3,
            "window": 30,
        },
        shared=NEURAL,
    ),
)
FAMILY_NAMES = tuple(f.name for f in FAMILIES)


def config_hash(family: str, params: dict[str, Any]) -> str:
    """Content hash of a configuration, independent of its id and key order."""
    text = json.dumps(
        {"family": family, "params": params}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def family_configs(family: Family, n: int, seed: int) -> list[dict[str, Any]]:
    """Parameters of the family's n configurations, the reference first."""
    index = FAMILY_NAMES.index(family.name)
    reference = family.complete(family.reference, reference=True)
    found = [reference]
    if family.grid:
        keys = list(family.space)
        for values in itertools.product(*(family.space[k].options for k in keys)):
            params = family.complete(
                dict(zip(keys, values, strict=True)), reference=False
            )
            if not _seen(params, found):
                found.append(params)
        if len(found) < n:
            raise ValueError(f"{family.name}: the grid holds only {len(found)} points")
        return found[:n]
    rng = np.random.default_rng([seed, index])
    for _ in range(MAX_DRAWS):
        if len(found) == n:
            return found
        drawn = {k: dist.draw(rng) for k, dist in family.space.items()}
        params = family.complete(drawn, reference=False)
        if not _seen(params, found):
            found.append(params)
    raise ValueError(f"{family.name}: {MAX_DRAWS} draws gave fewer than {n} configs")


def build_pool(seed: int, counts: dict[str, int]) -> list[ModelConfig]:
    if set(counts) != set(FAMILY_NAMES):
        raise ValueError(f"pool counts must name exactly the families {FAMILY_NAMES}")
    pool = []
    for family in FAMILIES:
        for i, params in enumerate(family_configs(family, counts[family.name], seed)):
            pool.append(
                ModelConfig(
                    f"{family.name}-{i:02d}",
                    family.name,
                    config_hash(family.name, params),
                    params,
                )
            )
    return pool


def render_pool(seed: int, counts: dict[str, int]) -> str:
    pool = build_pool(seed, counts)
    families = {}
    for index, family in enumerate(FAMILIES):
        entry: dict[str, Any] = {
            "index": index,
            "inputs": family.inputs,
            "n_configs": counts[family.name],
            "all": _flow(family.shared),
            "space": {k: _flow(d.describe()) for k, d in family.space.items()},
        }
        if family.others:
            entry["other_configs"] = _flow(family.others)
        entry["configs"] = [
            _flow({"id": c.id, "hash": c.hash, "params": c.params})
            for c in pool
            if c.family == family.name
        ]
        families[family.name] = entry
    body = yaml.dump(
        {"seed": seed, "total": len(pool), "families": families},
        Dumper=_PoolDumper,
        sort_keys=False,
        width=1000,
        allow_unicode=True,
    )
    header = (
        "# Frozen model pool, written by scripts/research/make_pool.py from\n"
        "# src/maintadapt/research/models/spaces.py. Do not edit by hand.\n"
        "# Configuration 0 of every family is its reference configuration.\n"
    )
    return header + body


def save_pool(seed: int, counts: dict[str, int], path: str | Path) -> None:
    Path(path).write_text(render_pool(seed, counts), encoding="utf-8", newline="\n")


def load_pool(path: str | Path, seed: int, counts: dict[str, int]) -> list[ModelConfig]:
    """Read the frozen pool, refusing one written with other settings or edited by hand."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    written = {name: entry["n_configs"] for name, entry in raw["families"].items()}
    if raw["seed"] != seed or written != counts:
        raise ValueError(
            f"{path} was written with other pool settings; rerun make_pool"
        )
    pool = []
    for name, entry in raw["families"].items():
        for i, item in enumerate(entry["configs"]):
            if item["id"] != f"{name}-{i:02d}":
                raise ValueError(f"{path}: {name} config {i} has id {item['id']}")
            if config_hash(name, item["params"]) != item["hash"]:
                raise ValueError(f"{path}: {item['id']} does not match its hash")
            pool.append(ModelConfig(item["id"], name, item["hash"], item["params"]))
    return pool
