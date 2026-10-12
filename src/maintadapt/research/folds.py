"""Lifetime-stratified, repeated outer and inner folds over run-to-failure engines."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from maintadapt.research.config import FoldsConfig, fleet_number

SPLITS = ("outer_test", "inner_val", "inner_fit")


@dataclass(frozen=True)
class Fold:
    inner_fit: tuple[int, ...]
    inner_val: tuple[int, ...]
    outer_test: tuple[int, ...]

    def units(self) -> set[int]:
        return {*self.inner_fit, *self.inner_val, *self.outer_test}


def lifetimes(train: pd.DataFrame) -> pd.Series:
    """Engine life T, the last recorded cycle, indexed by unit_id."""
    return train.groupby("unit_id")["cycle"].max()


def sort_by_lifetime(lives: pd.Series) -> np.ndarray:
    """Unit ids ordered by lifetime, ties broken by unit id."""
    ids = lives.index.to_numpy()
    return ids[np.lexsort((ids, lives.to_numpy()))]


def assign_blocks(
    n: int, labels: Sequence[int], rng: np.random.Generator
) -> np.ndarray:
    """Label n sorted items in consecutive blocks, each block a random permutation of labels."""
    size = len(labels)
    out = np.empty(n, dtype=int)
    for start in range(0, n, size):
        stop = min(start + size, n)
        # a short last block takes the head of its permutation, so no label repeats in it
        out[start:stop] = rng.permutation(np.asarray(labels))[: stop - start]
    return out


def fleet_folds(lives: pd.Series, cfg: FoldsConfig, fleet: str) -> list[list[Fold]]:
    """Folds of one fleet per repeat; one generator per repeat, outer blocks drawn first."""
    order = sort_by_lifetime(lives)
    inner_labels = [1] + [0] * (cfg.inner_block - 1)
    repeats = []
    for repeat in range(cfg.n_repeats):
        rng = np.random.default_rng([cfg.seed, repeat, fleet_number(fleet)])
        outer = assign_blocks(len(order), range(cfg.n_outer), rng)
        folds = []
        for fold in range(cfg.n_outer):
            # a subsequence of the sorted order is still sorted by lifetime
            train = order[outer != fold]
            is_val = assign_blocks(len(train), inner_labels, rng).astype(bool)
            folds.append(
                Fold(
                    inner_fit=_ids(train[~is_val]),
                    inner_val=_ids(train[is_val]),
                    outer_test=_ids(order[outer == fold]),
                )
            )
        repeats.append(folds)
    return repeats


def make_folds(
    lives: dict[str, pd.Series], cfg: FoldsConfig
) -> dict[str, list[list[Fold]]]:
    return {fleet: fleet_folds(series, cfg, fleet) for fleet, series in lives.items()}


def render_folds(folds: dict[str, list[list[Fold]]], cfg: FoldsConfig) -> str:
    # one line per id list keeps the committed file short enough to review in a diff
    lines = ["{", f'  "settings": {_settings(cfg, list(folds))},', '  "fleets": {']
    fleet_blocks = []
    for fleet, repeats in folds.items():
        repeat_blocks = []
        for repeat in repeats:
            fold_blocks = []
            for fold in repeat:
                lists = [
                    f'          "{name}": {json.dumps(list(getattr(fold, name)))}'
                    for name in SPLITS
                ]
                fold_blocks.append("        {\n" + ",\n".join(lists) + "\n        }")
            repeat_blocks.append("      [\n" + ",\n".join(fold_blocks) + "\n      ]")
        fleet_blocks.append(
            f'    "{fleet}": [\n' + ",\n".join(repeat_blocks) + "\n    ]"
        )
    lines.append(",\n".join(fleet_blocks))
    lines += ["  }", "}", ""]
    return "\n".join(lines)


def save_folds(
    folds: dict[str, list[list[Fold]]], cfg: FoldsConfig, path: str | Path
) -> None:
    Path(path).write_text(render_folds(folds, cfg), encoding="utf-8", newline="\n")


def load_folds(
    path: str | Path, cfg: FoldsConfig, fleets: list[str]
) -> dict[str, list[list[Fold]]]:
    """Read the committed folds, refusing a file written with other settings or broken folds."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw["settings"] != json.loads(_settings(cfg, fleets)):
        raise ValueError(
            f"{path} was written with other fold settings; rerun make research-data"
        )
    folds = {
        fleet: [
            [Fold(**{name: tuple(fold[name]) for name in SPLITS}) for fold in repeat]
            for repeat in raw["fleets"][fleet]
        ]
        for fleet in fleets
    }
    for fleet, repeats in folds.items():
        check_fleet_folds(repeats, cfg, fleet)
    return folds


def check_fleet_folds(repeats: list[list[Fold]], cfg: FoldsConfig, fleet: str) -> None:
    """Raise unless every repeat partitions the same engines into valid outer folds."""
    if len(repeats) != cfg.n_repeats or any(len(r) != cfg.n_outer for r in repeats):
        raise ValueError(f"{fleet}: expected {cfg.n_repeats} x {cfg.n_outer} folds")
    units = repeats[0][0].units()
    for r, repeat in enumerate(repeats):
        tested = [u for fold in repeat for u in fold.outer_test]
        if sorted(tested) != sorted(units):
            raise ValueError(
                f"{fleet} repeat {r}: outer_test must hold every engine once"
            )
        sizes = [len(fold.outer_test) for fold in repeat]
        if max(sizes) - min(sizes) > 1:
            raise ValueError(
                f"{fleet} repeat {r}: outer fold sizes {sizes} are unbalanced"
            )
        for k, fold in enumerate(repeat):
            parts = [set(getattr(fold, name)) for name in SPLITS]
            if sum(map(len, parts)) != len(units) or set().union(*parts) != units:
                raise ValueError(
                    f"{fleet} repeat {r} fold {k}: splits must be disjoint and cover "
                    "every engine"
                )


def _ids(values: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(int(v) for v in values))


def _settings(cfg: FoldsConfig, fleets: list[str]) -> str:
    return json.dumps(
        {
            "seed": cfg.seed,
            "n_outer": cfg.n_outer,
            "n_repeats": cfg.n_repeats,
            "inner_val_fraction": cfg.inner_val_fraction,
            "fleets": list(fleets),
        }
    )
