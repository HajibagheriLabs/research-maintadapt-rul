from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthetic_fleet import CFG

from maintadapt.research.derived import (
    TUNINGS,
    age_table,
    engines_of,
    fit_metrics,
    inventory,
    metric_components,
    repeat_outcomes,
    savings,
    truncation_points,
)
from maintadapt.research.folds import Fold
from maintadapt.research.guards import (
    GuardError,
    require_frozen_archive,
    resolve_root,
)

DECISION = replace(CFG.decision, cost_ratios=[2.0, 10.0], lead_times=[0, 10])
CAP = 125


def _rows(
    units: dict[int, np.ndarray], split: str, lives: dict[int, int]
) -> pd.DataFrame:
    parts = []
    for unit, pred in units.items():
        cycle = np.arange(1, len(pred) + 1)
        parts.append(
            pd.DataFrame(
                {
                    "unit_id": unit,
                    "cycle": cycle,
                    "rul": lives[unit] - cycle,
                    "split": split,
                    "pred": pred,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _oracle(life: int) -> np.ndarray:
    return np.minimum(np.arange(life - 1, -1, -1), CAP).astype(float)


LIVES = {u: 130 + 7 * u for u in range(1, 13)}
# three folds of four engines; inner_val holds two engines of every other fold
FOLDS = [
    Fold(inner_fit=(), inner_val=(5, 9), outer_test=(1, 2, 3, 4)),
    Fold(inner_fit=(), inner_val=(1, 9), outer_test=(5, 6, 7, 8)),
    Fold(inner_fit=(), inner_val=(1, 5), outer_test=(9, 10, 11, 12)),
]


def _fits(predict=_oracle, noisy_fold: int | None = None) -> dict[int, pd.DataFrame]:
    fits = {}
    for f, fold in enumerate(FOLDS):
        val = {u: predict(LIVES[u]) for u in fold.inner_val}
        test = {u: predict(LIVES[u]) for u in fold.outer_test}
        if f == noisy_fold:
            test = {u: np.full(LIVES[u], 60.0) for u in fold.outer_test}
        fits[f] = pd.concat(
            [_rows(val, "inner_val", LIVES), _rows(test, "outer_test", LIVES)],
            ignore_index=True,
        )
    return fits


def test_engines_are_read_back_in_unit_order_with_their_lives() -> None:
    table = _fits()[0]
    engines = engines_of(table, "outer_test")
    assert engines.units.tolist() == [1, 2, 3, 4]
    assert engines.lives.tolist() == [LIVES[u] for u in (1, 2, 3, 4)]
    np.testing.assert_array_equal(engines.predictions[2], _oracle(LIVES[3]))
    with pytest.raises(ValueError, match="cycles 1..T"):
        engines_of(table[table["cycle"] != 2], "outer_test")


def test_every_tuning_rule_lead_and_ratio_covers_each_engine_once() -> None:
    policies, found = repeat_outcomes(_fits(), DECISION)
    n = len(TUNINGS) * 2 * len(DECISION.lead_times) * len(DECISION.cost_ratios)
    assert len(policies) == len(FOLDS) * n
    assert len(found) == len(LIVES) * n
    per_policy = found.groupby(["tuning", "rule", "lead", "cost_ratio"])["unit_id"]
    assert (per_policy.nunique() == len(LIVES)).all()
    assert (per_policy.size() == len(LIVES)).all()
    # the capped truth tuned anywhere maintains every engine at the end of its life
    assert not found["failed"].any() and (found["waste"] == 0).all()
    assert set(policies["theta"]) == set(map(float, DECISION.lead_times))


def test_crossfit_and_validation_ignore_the_fold_they_are_applied_to() -> None:
    base, _ = repeat_outcomes(_fits(), DECISION)
    moved, _ = repeat_outcomes(_fits(noisy_fold=0), DECISION)
    key = ["fold", "tuning", "rule", "lead", "cost_ratio"]
    both = base.merge(moved, on=key, suffixes=("", "_moved"))
    same = both["theta"] == both["theta_moved"]
    # fold 0's outer_test feeds the oracle everywhere and the crossfit of folds 1 and 2
    assert same[(both["tuning"] == "validation")].all()
    assert same[(both["tuning"] == "crossfit") & (both["fold"] == 0)].all()
    assert not same[(both["tuning"] == "oracle")].all()
    assert not same[(both["tuning"] == "crossfit") & (both["fold"] > 0)].all()


def test_savings_pool_the_folds_and_match_the_hand_worked_fleet() -> None:
    # the playbook fleet: A (T = 10) and B (T = 6), theta 4, L = 2, K = 10, A* = 6
    found = pd.DataFrame(
        {
            "config_id": "m",
            "repeat": 0,
            "tuning": "validation",
            "rule": "R1",
            "lead": 2,
            "cost_ratio": 10.0,
            "fold": [0, 1],
            "unit_id": [1, 2],
            "failed": [False, False],
            "used": [9, 6],
            "waste": [1, 0],
        }
    )
    age = pd.DataFrame(
        {
            "repeat": 0,
            "fold": [0, 1],
            "cost_ratio": 10.0,
            "age": 6.0,
            "unit_id": [1, 2],
            "failed": [False, False],
            "used": [6, 6],
            "waste": [4, 0],
        }
    )
    share = savings(found, age, pd.Series({1: 10, 2: 6}))
    row = share.iloc[0]
    assert (row["cost_rate"], row["age_rate"], row["perfect_rate"]) == (
        2 / 15,
        2 / 12,
        2 / 16,
    )
    assert row["savings_share"] == 0.8


def test_age_replacement_is_tuned_on_outer_train_lifetimes_only() -> None:
    lives = pd.Series({1: 10, 2: 6, 3: 8, 4: 100})
    folds = [[Fold(inner_fit=(1,), inner_val=(2,), outer_test=(3, 4))]]
    table = age_table(lives, folds, [0], [10.0])
    # A* on lifetimes 10 and 6 at K = 10 is 6: engines 3 and 4 are maintained at 6
    assert table["age"].tolist() == [6.0, 6.0]
    assert table["used"].tolist() == [6, 6]
    assert table["waste"].tolist() == [2, 94]


def test_metric_components_follow_their_definitions() -> None:
    lives = {1: 4, 2: 60, 3: 3}
    official = _rows({1: np.array([5.0, 2.0, 3.0, 1.0])}, "official_test", {1: 4 + 140})
    outer = _rows({2: _oracle(60) + 1.0}, "outer_test", lives)
    inner = _rows({3: np.array([2.0, 2.0, 2.0])}, "inner_val", lives)
    table = pd.concat([official, outer, inner], ignore_index=True)
    points = pd.DataFrame({"unit_id": [2, 2], "cycle": [31, 60]})
    parts = metric_components(table, points, CAP).set_index("split")
    o = parts.loc["official_test"]
    # official engine: true RUL 143..140, so the last cycle's truth is 125 capped, 130, 140
    assert o["sq_last_125"] == (1 - 125) ** 2
    assert o["sq_last_130"] == (1 - 130) ** 2 and o["sq_last_raw"] == (1 - 140) ** 2
    assert o["nasa_last"] == pytest.approx(np.expm1(124 / 13))
    # previous true RUL never drops below 125, so this engine has no jitter cycles
    assert o["jit_n"] == 0 and o["n_late"] == 0
    t = parts.loc["outer_test"]
    # predictions one above the capped truth: unit error everywhere, and a one-cycle step
    # of -1 per cycle, so |y_t - y_(t-1) + 1| = 0 once the truth is below the cap
    assert t["sse_trunc"] == 2 and t["n_trunc"] == 2
    assert t["jit_sum"] == 0 and t["jit_n"] == 59
    assert t["n_late"] == 51 and t["sse_late"] == 51 and t["err_late"] == 51
    v = parts.loc["inner_val"]
    # truth 2, 1, 0 against predictions of 2
    assert (v["sse_all"], v["n_all"]) == (0 + 1 + 4, 3)
    assert np.isnan(v["jit_n"]) and np.isnan(t["sq_last_125"])
    metrics = fit_metrics(parts.reset_index().assign(fit=0), ["fit"]).iloc[0]
    assert metrics["rmse_last"] == 124.0
    assert metrics["val_rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert metrics["rmse_trunc_outer"] == 1.0 and metrics["jitter_outer"] == 0.0
    # the official engine has fewer than five jitter cycles, so no jitter is defined
    assert np.isnan(metrics["jitter"])


def test_truncation_points_are_seeded_uniform_and_shared() -> None:
    lives = pd.Series({1: 200, 2: 20})
    points = truncation_points(lives, [2026, 1])
    pd.testing.assert_frame_equal(points, truncation_points(lives, [2026, 1]))
    assert points.groupby("unit_id").size().tolist() == [20, 20]
    long = points[points["unit_id"] == 1]["cycle"]
    assert long.between(31, 200).all()
    # a life shorter than 31 cycles truncates at its last cycle only
    assert (points[points["unit_id"] == 2]["cycle"] == 20).all()


def test_inventory_excludes_failed_configurations_and_refuses_pending_fits() -> None:
    rows = [
        (c, r, f, "failed" if (c, r, f) == ("knn-00", 1, 2) else "ok")
        for c in ("knn-00", "ridge-01", "ridge-00", "control_oracle-00")
        for r in (0, 1)
        for f in range(3)
    ]
    found = pd.DataFrame(rows, columns=["config_id", "repeat", "fold", "status"])
    order = ("ridge", "knn", "control_oracle")
    configs, repeats, excluded = inventory(found, 3, order)
    assert configs == ["ridge-00", "ridge-01", "control_oracle-00"]
    assert repeats == [0, 1]
    assert excluded["config_id"].tolist() == ["knn-00"]
    with pytest.raises(ValueError, match="pending"):
        inventory(found.iloc[1:], 3, order)


def test_the_guard_refuses_the_confirmatory_archive_until_it_is_frozen_and_valid(
    tmp_path: Path,
) -> None:
    cfg = replace(CFG, paths=replace(CFG.paths, archive=str(tmp_path / "archive")))
    assert resolve_root("smoke", cfg) == Path(cfg.paths.smoke)
    assert resolve_root(None, cfg) == Path(cfg.paths.archive)
    assert resolve_root("elsewhere/x", cfg) == Path("elsewhere/x")

    def valid(fleet, root, cfg):
        return [], {}

    def broken(fleet, root, cfg):
        return ["r0 f0: 3 non-finite predictions"], {}

    archive = resolve_root("archive", cfg)
    # the smoke archive is never guarded
    require_frozen_archive(
        resolve_root("smoke", cfg), cfg, ["FD001"], broken, lambda: False
    )
    with pytest.raises(GuardError, match="analysis-frozen"):
        require_frozen_archive(archive, cfg, ["FD001"], valid, lambda: False)
    with pytest.raises(GuardError, match="non-finite"):
        require_frozen_archive(archive, cfg, ["FD001"], broken, lambda: True)
    require_frozen_archive(archive, cfg, ["FD001"], valid, lambda: True)
