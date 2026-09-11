"""Cross-regime transfer from a single-condition source fleet to harder targets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from maintadapt.data import SENSORS
from maintadapt.features import PreparedSubset


def shared_sensors(prepared: Sequence[PreparedSubset]) -> list[str]:
    """Sensors kept by screening in every subset, so one network can read all of them."""
    kept = set.intersection(*(set(p.sensors) for p in prepared))
    if not kept:
        raise ValueError("no sensor survives screening in every subset")
    return [s for s in SENSORS if s in kept]


def budget_units(train_units: Sequence[int], fraction: float, seed: int) -> list[int]:
    """A random share of training units; smaller budgets are nested inside larger ones."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction}")
    order = np.random.default_rng(seed).permutation(np.sort(np.asarray(train_units)))
    return sorted(order[: max(1, round(fraction * len(order)))].tolist())
