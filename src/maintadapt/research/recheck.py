"""An independent recomputation of the point estimates of E1-E3 (make paper-check).

Written apart from the pipeline: it imports nothing from maintadapt, and reads only the
prediction files and sidecars of an archive, the fold file and the pool file. Thresholds,
outcomes, age replacement, savings shares, the selections and Kendall's tau-b are coded
again here from the definitions in PREREGISTRATION.md sections 5 and 6 and DECISIONS.md, so that
an error in analysis.py, stats.py, policies.py, costs.py or derived.py cannot reproduce itself.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

COLUMNS = ["unit_id", "cycle", "rul", "split", "pred"]
MIN_STEPS = 5


@dataclass(frozen=True)
class Setting:
    cost_ratio: float
    lead: int
    levels: int
    # truth cap of the validation RMSE and of jitter: 125 on C-MAPSS, 50 on N-CMAPSS
    cap: int
    # C-MAPSS: last-cycle RMSE on official test engines, capped at 125; N-CMAPSS: all flights of
    # the outer_test units against the cap
    leaderboard: str
    leaderboard_cap: int = 125


@dataclass(frozen=True, eq=False)
class Engines:
    """Trajectories of one split of one fit, in unit order."""

    units: np.ndarray
    pred: list[np.ndarray]
    rul: list[np.ndarray]

    @property
    def lives(self) -> np.ndarray:
        return np.array([len(p) for p in self.pred], dtype=np.int64)


def split_engines(table: dict[str, np.ndarray], split: str) -> Engines:
    keep = table["split"] == split
    unit, cycle = table["unit_id"][keep], table["cycle"][keep]
    order = np.lexsort((cycle, unit))
    unit, cycle = unit[order], cycle[order]
    pred, rul = table["pred"][keep][order], table["rul"][keep][order]
    units, starts = np.unique(unit, return_index=True)
    bounds = [*starts[1:], len(unit)]
    out_pred, out_rul = [], []
    for start, stop in zip(starts, bounds, strict=True):
        if not np.array_equal(cycle[start:stop], np.arange(1, stop - start + 1)):
            raise ValueError(f"{split}: an engine does not hold cycles 1..T")
        out_pred.append(pred[start:stop])
        out_rul.append(rul[start:stop])
    return Engines(units, out_pred, out_rul)


def read_fit(path: Path) -> dict[str, np.ndarray]:
    frame = pq.read_table(path, columns=COLUMNS).to_pandas()
    return {
        "unit_id": frame["unit_id"].to_numpy(np.int64),
        "cycle": frame["cycle"].to_numpy(np.int64),
        "rul": frame["rul"].to_numpy(np.float64),
        "split": frame["split"].astype(str).to_numpy(),
        "pred": frame["pred"].to_numpy(np.float64),
    }


def first_alarms(pred: Sequence[np.ndarray], thetas: np.ndarray) -> np.ndarray:
    """thetas x engines: the first cycle (from 1) with prediction <= theta; 0 for none."""
    out = np.zeros((len(thetas), len(pred)), dtype=np.int64)
    for j, p in enumerate(pred):
        low = np.minimum.accumulate(p)
        # low never rises, so the cycles at or below theta form a suffix; count it from the end
        first = len(low) - np.searchsorted(low[::-1], thetas, side="right")
        out[:, j] = np.where(first < len(low), first + 1, 0)
    return out


def costs_and_cycles(
    alarms: np.ndarray, lives: np.ndarray, lead: int, cost_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per alarm: cost 1 and m = a + L cycles if m <= T, otherwise cost K and T cycles."""
    m = alarms + lead
    preventive = (alarms > 0) & (m <= lives)
    return (
        np.where(preventive, 1.0, float(cost_ratio)),
        np.where(preventive, m, lives).astype(np.float64),
    )


def median_of_minimisers(values: np.ndarray) -> int:
    where = np.flatnonzero(values == np.min(values))
    return int(where[(len(where) - 1) // 2])


def tune_threshold(engines: Engines, s: Setting) -> tuple[float, float]:
    """The validation threshold and the cost rate it reaches on the engines it was tuned on."""
    pooled = np.concatenate(engines.pred)
    levels = np.linspace(0.0, 1.0, s.levels)
    thetas = np.concatenate(
        [[-np.inf], np.quantile(pooled, levels, method="inverted_cdf"), [np.inf]]
    )
    cost, used = costs_and_cycles(
        first_alarms(engines.pred, thetas), engines.lives, s.lead, s.cost_ratio
    )
    rates = cost.sum(axis=1) / used.sum(axis=1)
    best = median_of_minimisers(rates)
    return float(thetas[best]), float(rates[best])


def age_policy(lives: np.ndarray, cost_ratio: float) -> float:
    """A* on outer-train lifetimes: a lifetime itself or never (inf), median minimiser."""
    candidates = np.concatenate([np.unique(lives).astype(float), [np.inf]])
    rates = []
    for a in candidates:
        preventive = a <= lives
        cost = np.where(preventive, 1.0, float(cost_ratio)).sum()
        rates.append(cost / np.where(preventive, a, lives).sum())
    return float(candidates[median_of_minimisers(np.array(rates))])


def tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b from the signs of every pair."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    upper = np.triu_indices(len(x), k=1)
    dx = np.sign(x[:, None] - x[None, :])[upper]
    dy = np.sign(y[:, None] - y[None, :])[upper]
    denominator = np.sqrt(np.count_nonzero(dx) * np.count_nonzero(dy))
    return float(np.sum(dx * dy) / denominator) if denominator else float("nan")


def share(
    cost: float, used: float, age_cost: float, age_used: float, perfect: float
) -> float:
    rate, age = cost / used, age_cost / age_used
    return (age - rate) / (age - perfect)


def pool_configs(path: str | Path) -> list[str]:
    """Configuration ids in pool order: families by index, then as listed."""
    pool = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    families = sorted(pool["families"].values(), key=lambda f: f["index"])
    return [c["id"] for f in families for c in f["configs"]]


def fleet_folds(path: str | Path, fleet: str) -> list[list[dict[str, list[int]]]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["fleets"][fleet]


@dataclass
class FleetPoints:
    configs: list[str]
    leaderboard_rmse: np.ndarray
    jitter: np.ndarray
    savings_share: np.ndarray
    by_cost: np.ndarray
    by_rmse: np.ndarray
    s_by_cost: float
    s_by_rmse: float

    @property
    def endpoints(self) -> dict[str, float]:
        loss = 1.0 - self.savings_share
        e1 = tau_b(self.leaderboard_rmse, loss)
        jit = tau_b(self.jitter, loss)
        return {
            "tau_b": e1,
            "selection_loss": self.s_by_cost - self.s_by_rmse,
            "s_by_cost": self.s_by_cost,
            "s_by_rmse": self.s_by_rmse,
            "delta_tau": jit - e1,
            "tau_jitter": jit,
        }


def _jitter_of(engines: Engines, cap: int) -> float:
    """Mean over engines with at least five steps of their mean |y_t - y_(t-1) + 1|."""
    means = []
    for p, r in zip(engines.pred, engines.rul, strict=True):
        counted = r[:-1] < cap
        if counted.sum() >= MIN_STEPS:
            means.append(np.abs(np.diff(p) + 1.0)[counted].mean())
    return float(np.mean(means))


def _leaderboard_of(table: dict[str, np.ndarray], s: Setting) -> tuple[float, float]:
    if s.leaderboard == "official_last":
        test = split_engines(table, "official_test")
        last = np.array([p[-1] for p in test.pred])
        truth = np.minimum([r[-1] for r in test.rul], s.leaderboard_cap)
        return float(np.sqrt(np.mean((last - truth) ** 2))), _jitter_of(test, s.cap)
    test = split_engines(table, "outer_test")
    pred, rul = np.concatenate(test.pred), np.concatenate(test.rul)
    rmse = float(np.sqrt(np.mean((pred - np.minimum(rul, s.cap)) ** 2)))
    return rmse, _jitter_of(test, s.cap)


def recompute_fleet(
    root: str | Path,
    fleet: str,
    folds: list[list[dict[str, list[int]]]],
    pool: Sequence[str],
    s: Setting,
    log=print,
) -> FleetPoints:
    """Point estimates of every configuration of one fleet, and the two selections."""
    root = Path(root) / fleet
    n_repeats, n_folds = len(folds), len(folds[0])
    units = np.array(sorted({u for fold in folds[0] for u in fold["outer_test"]}))
    position = {int(u): i for i, u in enumerate(units)}
    found = {p.parent.name: p.parent for p in root.glob("*/*/r0_f0.json")}
    configs = []
    for c in pool:
        sidecars = [
            json.loads((found[c] / f"r{r}_f{f}.json").read_text(encoding="utf-8"))
            for r in range(n_repeats)
            for f in range(n_folds)
        ]
        if all(x["status"] == "ok" for x in sidecars):
            configs.append(c)
    shape = (len(configs), n_repeats, len(units))
    cost, used = np.zeros(shape), np.zeros(shape)
    val_cost = np.zeros((len(configs), n_repeats, n_folds))
    val_rmse = np.zeros_like(val_cost)
    lb = np.zeros((len(configs), n_repeats * n_folds))
    jit = np.zeros_like(lb)
    lives = np.zeros(len(units), dtype=np.int64)
    for i, c in enumerate(configs):
        for r in range(n_repeats):
            for f in range(n_folds):
                table = read_fit(found[c] / f"r{r}_f{f}.parquet")
                val = split_engines(table, "inner_val")
                theta, val_cost[i, r, f] = tune_threshold(val, s)
                errors = np.concatenate(
                    [p - np.minimum(t, s.cap) for p, t in zip(val.pred, val.rul)]
                )
                val_rmse[i, r, f] = np.sqrt(np.mean(errors**2))
                test = split_engines(table, "outer_test")
                if sorted(test.units) != sorted(folds[r][f]["outer_test"]):
                    raise ValueError(
                        f"{c} r{r} f{f}: outer_test differs from the folds"
                    )
                at = [position[int(u)] for u in test.units]
                lives[at] = test.lives
                k, u = costs_and_cycles(
                    first_alarms(test.pred, np.array([theta]))[0],
                    test.lives,
                    s.lead,
                    s.cost_ratio,
                )
                cost[i, r, at], used[i, r, at] = k, u
                lb[i, r * n_folds + f], jit[i, r * n_folds + f] = _leaderboard_of(
                    table, s
                )
        log(f"{fleet} {c}")
    fold_of = np.zeros((n_repeats, len(units)), dtype=np.int64)
    age_cost = np.zeros((n_repeats, len(units)))
    age_used = np.zeros_like(age_cost)
    for r in range(n_repeats):
        for f in range(n_folds):
            at = [position[u] for u in folds[r][f]["outer_test"]]
            fold_of[r, at] = f
            outer_train = np.ones(len(units), dtype=bool)
            outer_train[at] = False
            a = age_policy(lives[outer_train], s.cost_ratio)
            preventive = a <= lives[at]
            age_cost[r, at] = np.where(preventive, 1.0, float(s.cost_ratio))
            age_used[r, at] = np.where(preventive, a, lives[at])
    perfect = len(units) / lives.sum()

    def value(k: np.ndarray, u: np.ndarray) -> float:
        """S pooled over the folds of each repeat, averaged over repeats."""
        return float(
            np.mean(
                [
                    share(
                        k[r].sum(),
                        u[r].sum(),
                        age_cost[r].sum(),
                        age_used[r].sum(),
                        perfect,
                    )
                    for r in range(n_repeats)
                ]
            )
        )

    def chosen(choice: np.ndarray) -> float:
        picked = choice[np.arange(n_repeats)[:, None], fold_of]
        rows, cols = np.arange(n_repeats)[:, None], np.arange(len(units))[None, :]
        return value(cost[picked, rows, cols], used[picked, rows, cols])

    by_cost, by_rmse = np.argmin(val_cost, axis=0), np.argmin(val_rmse, axis=0)
    return FleetPoints(
        configs=configs,
        leaderboard_rmse=lb.mean(axis=1),
        jitter=jit.mean(axis=1),
        savings_share=np.array([value(cost[i], used[i]) for i in range(len(configs))]),
        by_cost=by_cost,
        by_rmse=by_rmse,
        s_by_cost=chosen(by_cost),
        s_by_rmse=chosen(by_rmse),
    )
