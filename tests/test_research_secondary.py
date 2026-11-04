import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from synthetic_fleet import CFG, TRAINING, fold_data, repeats, subset
from test_research_analysis import _inputs, _tables

from maintadapt.research.analysis import (
    decision_loss,
    fleet_inputs,
    leaderboard_rmse,
)
from maintadapt.research.archive import read_record
from maintadapt.research.config import CapsConfig, NestedConfig
from maintadapt.research.crossfit import (
    Job,
    control_configs,
    family_representatives,
    fit_seed,
    nested_seed,
    plan,
    run_jobs,
)
from maintadapt.research.folds import Fold, nested_folds
from maintadapt.research.models.registry import FitOutput, fit
from maintadapt.research.secondary import (
    Draws,
    RegretInputs,
    check_oof,
    competitive,
    failure_and_waste,
    family_tables,
    inversions,
    nasa_score,
    nested_outcomes,
    pooled_rmse,
    regret_loss,
    regret_outcomes,
    regret_selected,
    representatives,
    settings_grid,
    tune_age_regret,
    tune_regret,
    with_validation_selection,
)

PRIMARY = CFG.decision.primary
CPU = torch.device("cpu")
ENV = {"python": "3.11", "packages": {}, "hardware": {}, "git_commit": "test"}


def _oracle(life: int) -> np.ndarray:
    return np.minimum(np.arange(life - 1, -1, -1), 125).astype(float)


def _rows(units: dict[int, np.ndarray]) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "unit_id": u,
                    "cycle": np.arange(1, len(p) + 1),
                    "rul": np.arange(len(p) - 1, -1, -1),
                    "pred": p,
                }
            )
            for u, p in units.items()
        ],
        ignore_index=True,
    )


# nested folds and the extra fits


def test_nested_folds_split_the_outer_train_engines_by_lifetime() -> None:
    lives = pd.Series({u: 100 + 3 * u for u in range(1, 101)})
    outer = Fold(
        inner_fit=tuple(range(1, 61)),
        inner_val=tuple(range(61, 81)),
        outer_test=tuple(range(81, 101)),
    )
    inner = nested_folds(lives, outer, CFG.folds, "FD001", 0, 2, 4, 5)
    train = set(range(1, 81))
    assert len(inner) == 4
    # every outer-train engine is predicted exactly once and no outer_test engine is used
    predicted = [u for f in inner for u in f.outer_test]
    assert sorted(predicted) == sorted(train)
    for f in inner:
        assert f.units() == train
        assert (len(f.outer_test), len(f.inner_val), len(f.inner_fit)) == (20, 12, 48)
    # lifetime-sorted blocks of four engines put one engine in each inner fold
    which = {u: k for k, f in enumerate(inner) for u in f.outer_test}
    blocks = [sorted(which[u] for u in range(s, s + 4)) for s in range(1, 81, 4)]
    assert all(b == [0, 1, 2, 3] for b in blocks)
    # one in five of each inner training set stops early, again by lifetime block
    rest = sorted(set(train) - set(inner[0].outer_test))
    held = [u in inner[0].inner_val for u in rest]
    assert all(sum(held[s : s + 5]) == 1 for s in range(0, 60, 5))
    assert nested_folds(lives, outer, CFG.folds, "FD001", 0, 2, 4, 5) == inner
    assert nested_folds(lives, outer, CFG.folds, "FD001", 0, 3, 4, 5) != inner


def test_nested_seeds_are_their_own_and_representatives_follow_validation_rmse() -> (
    None
):
    assert nested_seed("FD001", 0, 2, 1, "lstm-03") == fit_seed(
        "FD001", 0, 2, "lstm-03|inner1"
    )
    assert nested_seed("FD001", 0, 2, 1, "lstm-03") != fit_seed(
        "FD001", 0, 1, "lstm-03"
    )
    found = pd.DataFrame(
        [
            ("FD001", "ridge", "ridge-01", 0, 2, "ok", 15.0),
            ("FD001", "ridge", "ridge-00", 0, 2, "ok", 15.0),
            ("FD001", "ridge", "ridge-02", 0, 2, "ok", 16.0),
            ("FD001", "lstm", "lstm-04", 0, 2, "ok", 12.0),
            ("FD001", "lstm", "lstm-01", 0, 2, "failed", 11.0),
            ("FD001", "lstm", "lstm-02", 0, 1, "ok", 10.0),
        ],
        columns=[
            "fleet",
            "family",
            "config_id",
            "repeat",
            "fold",
            "status",
            "val_rmse",
        ],
    )
    # ties go to the configuration listed first; failed fits and other folds never count
    assert family_representatives(found, "FD001", 0, 2, ["ridge", "lstm"]) == [
        "ridge-00",
        "lstm-04",
    ]
    with pytest.raises(ValueError, match="no finished fit"):
        family_representatives(found, "FD001", 0, 2, ["ridge", "knn"])


def test_extra_fits_take_their_seed_and_sidecar_fields_from_the_caller(
    tmp_path: Path,
) -> None:
    jobs = plan(control_configs()[:1], ["FD001"], [0], [0])

    def fit_job(job: Job, data, seed: int) -> FitOutput:
        return fit(job.config.family, {}, data, seed, CPU, TRAINING)

    run_jobs(
        jobs,
        tmp_path,
        lambda fleet, repeat, fold: fold_data(subset(), repeats()[repeat][fold]),
        fit_job,
        ENV,
        lambda f: "cpu",
        log=lambda _: None,
        seed_of=lambda job: 77,
        extra={"outer_fold": 3},
    )
    record = read_record(jobs[0].key.sidecar(tmp_path))
    assert (record["seed"], record["outer_fold"], record["fold"]) == (77, 3, 0)


def test_secondary_settings_reject_impossible_values() -> None:
    caps = CFG.secondary.caps
    assert [caps.name(v) for v in caps.values] == ["100", "150", "200", "none"]
    assert caps.training_cap(None) == 600 and caps.training_cap(150) == 150
    assert CFG.secondary.nested.early_stopping_block == 5
    with pytest.raises(ValueError, match="uncapped_scale"):
        CapsConfig([100, 700], ["xgboost"], 0, 600)
    with pytest.raises(ValueError, match="pool families"):
        CapsConfig([100], ["arima"], 0, 600)
    with pytest.raises(ValueError, match="1/m"):
        NestedConfig(["FD001"], 0, 4, 0.3)


# S1 and S7: other accuracy metrics


def test_nasa_score_sums_engines_and_pooled_rmse_pools_them() -> None:
    nasa = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    assert nasa_score(nasa, np.ones((1, 2))).tolist() == [[5.0]]
    # engine 2 drawn twice: sums 1 + 2 * 2 = 5 and 3 + 2 * 4 = 11 over the two fits
    assert nasa_score(nasa, np.array([[1.0, 2.0]])).tolist() == [[(5.0 + 11.0) / 2]]
    sse = np.array([[[8.0, 0.0], [2.0, 2.0]]])
    rows = np.array([[[2.0, 2.0], [1.0, 1.0]]])
    # per fit sqrt(8 / 4) and sqrt(4 / 2), then their mean
    assert pooled_rmse(sse, rows, np.ones((1, 2)))[0, 0] == pytest.approx(np.sqrt(2))


# S3: family representatives


def _family_tables() -> dict[str, pd.DataFrame]:
    """Two configurations of one family (ridge) and one of another (knn), by hand."""
    tables = _tables()
    extra = tables["outcomes"][tables["outcomes"]["config_id"] == "knn-00"]
    tables["outcomes"] = pd.concat(
        [tables["outcomes"], extra.assign(config_id="ridge-01", family="ridge")],
        ignore_index=True,
    )
    pol = tables["policies"]
    tables["policies"] = pd.concat(
        [pol, pol[pol["config_id"] == "knn-00"].assign(config_id="ridge-01")],
        ignore_index=True,
    )
    parts = tables["components"]
    knn = parts[parts["config_id"] == "knn-00"].copy()
    # ridge-01 validates better than ridge-00 in fold 1 only
    knn.loc[knn["split"] == "inner_val", "sse_all"] = [8.0, 2.0]
    tables["components"] = pd.concat(
        [parts, knn.assign(config_id="ridge-01")], ignore_index=True
    )
    return tables


def test_a_family_takes_the_outcomes_of_its_representative_in_each_fold() -> None:
    tables = _family_tables()
    configs = ["ridge-00", "ridge-01", "knn-00"]
    base = fleet_inputs(tables, PRIMARY, configs)
    reps = representatives(base)
    # validation RMSE: ridge-00 1 in both folds; ridge-01 sqrt(2) then sqrt(0.5)
    assert reps.set_index(["family", "fold"])["config_id"].to_dict() == {
        ("ridge", 0): "ridge-00",
        ("ridge", 1): "ridge-01",
        ("knn", 0): "knn-00",
        ("knn", 1): "knn-00",
    }
    families = family_tables(tables, reps)
    inputs = fleet_inputs(families, PRIMARY, ["ridge", "knn"])
    once = np.ones((1, 4))
    # fold 0 (engines 1, 2) from ridge-00, fold 1 (engines 3, 4) from ridge-01
    picked = base.cost[[0, 0, 1, 1], 0, [0, 1, 2, 3]]
    assert inputs.cost[0, 0].tolist() == picked.tolist() == [1.0, 1.0, 1.0, 10.0]
    assert decision_loss(inputs, once)[0, 1] == decision_loss(base, once)[0, 2]
    rmse = leaderboard_rmse(inputs, np.ones((1, 2)))[0]
    # ridge's fits: ridge-00 in fold 0 (squared errors 1, 2) and ridge-01 in fold 1 (4, 8)
    expected = np.mean([np.sqrt(1 * 1 * 1.5), np.sqrt(2 * 2 * 1.5)])
    assert rmse[0] == pytest.approx(expected)


def test_failure_share_and_waste_pool_a_repeat_then_average() -> None:
    found = pd.DataFrame(
        {
            "config_id": "m",
            "repeat": [0, 0, 0, 1, 1],
            "failed": [True, False, False, False, False],
            "waste": [0, 4, 8, 10, 20],
        }
    )
    row = failure_and_waste(found).loc["m"]
    assert row["failure_rate"] == pytest.approx((1 / 3 + 0) / 2)
    assert row["waste_per_maintenance"] == pytest.approx((6 + 15) / 2)


def test_inversions_need_both_differences_rejected_in_opposite_directions() -> None:
    pairs = pd.DataFrame(
        {
            "a": ["x", "x", "y"],
            "b": ["y", "z", "z"],
            "difference_rmse": [-1.0, -1.0, 1.0],
            "lo_rmse": [-2.0, -2.0, 0.5],
            "hi_rmse": [-0.5, -0.5, 2.0],
            "p_value_rmse": [0.001, 0.001, 0.001],
            "difference_loss": [0.2, -0.2, 0.2],
            "lo_loss": [0.1, -0.3, -0.1],
            "hi_loss": [0.3, -0.1, 0.4],
            "p_value_loss": [0.001, 0.001, 0.4],
        }
    )
    out = inversions(pairs, 0.05)
    # x is more accurate than y but loses more value: inverted; x-z agree; y-z loss unclear
    assert out["inversion"].tolist() == [True, False, False]
    assert out["inversion_unadjusted"].tolist() == [True, False, False]
    assert out["q_value_loss"].iloc[2] == pytest.approx(0.4)


# S4: threshold variants


def test_variants_keep_the_validation_choice() -> None:
    base = _inputs([1.0, 2.0], [0, 3], val_cost=np.ones((2, 2, 3)))
    variant = _inputs([1.0, 2.0], [1, 1])
    kept = with_validation_selection(variant, base)
    assert (kept.val_cost == base.val_cost).all()
    assert (kept.cost == variant.cost).all()
    with pytest.raises(ValueError, match="same fits"):
        with_validation_selection(_inputs([1.0], [0]), base)


def test_nested_tuning_on_oracle_predictions_maintains_every_engine_at_its_end() -> (
    None
):
    lives = {u: 130 + 10 * u for u in range(1, 9)}
    oof = {0: _rows({u: _oracle(lives[u]) for u in (1, 2)})}
    oof[1] = _rows({u: _oracle(lives[u]) for u in (3, 4)})
    test = {0: _rows({u: _oracle(lives[u]) for u in (5, 6)})}
    test[1] = _rows({u: _oracle(lives[u]) for u in (7, 8)})
    decision = replace(CFG.decision, cost_ratios=[10.0], lead_times=[10])
    policies, found = nested_outcomes(oof, test, decision)
    assert set(policies["tuning"]) == set(found["tuning"]) == {"nested"}
    r1 = found[found["rule"] == "R1"]
    assert not r1["failed"].any() and (r1["waste"] == 0).all()
    assert r1["used"].tolist() == [lives[u] for u in (5, 6, 7, 8)]
    assert policies.loc[policies["rule"] == "R1", "theta"].tolist() == [10.0, 10.0]
    check_oof(oof[0], Fold((1,), (2,), (5, 6)))
    with pytest.raises(ValueError, match="once"):
        check_oof(pd.concat([oof[0], oof[0]]), Fold((1,), (2,), (5, 6)))
    with pytest.raises(ValueError, match="once"):
        check_oof(oof[0], Fold((1, 3), (2,), (5, 6)))


# S6: regret


def test_regret_tuning_on_oracle_predictions_alarms_exactly_the_lead_time_early() -> (
    None
):
    scores = [_oracle(100), _oracle(130)]
    tuned = tune_regret(scores, 10, [100.0], 401)
    # alarm at T - theta and maintenance at T - theta + 10: theta = 10 wastes nothing
    assert tuned == {100.0: (10.0, 0.0)}


def test_regret_tuning_trades_failures_against_waste() -> None:
    flat = [np.full(100, 50.0), np.full(100, 50.0)]
    # an alarm fires at cycle 1 (maintenance at 11, wasting 89) or never (a failure)
    cheap = tune_regret(flat, 10, [5.0, 1000.0], 401)
    assert cheap[5.0] == (-np.inf, 5.0)
    assert cheap[1000.0] == (50.0, 89.0)


def test_age_replacement_regret_is_tuned_by_hand() -> None:
    lifetimes = np.array([100, 120])
    # C_f 50: A = 100 wastes 0 and 20 (10); A = 120 fails one (25); never fails both (50)
    assert tune_age_regret(lifetimes, 50.0) == 100.0
    # C_f 10: A = 100 still 10; A = 120 gives (10 + 0) / 2 = 5; never 10
    assert tune_age_regret(lifetimes, 10.0) == 120.0


def test_regret_outcomes_apply_each_folds_threshold_to_its_test_engines() -> None:
    fits = {
        0: pd.concat(
            [
                _rows({1: _oracle(120)}).assign(split="inner_val"),
                _rows({2: np.full(150, 40.0)}).assign(split="outer_test"),
            ]
        )
    }
    policies, found = regret_outcomes(fits, 10, [100.0], 401)
    assert policies[["theta", "tuned_regret"]].values.tolist() == [[10.0, 0.0]]
    # a flat 40 never reaches 10: the test engine fails
    assert found[["unit_id", "failed", "waste"]].values.tolist() == [[2, True, 0]]


def test_regret_share_and_selection_by_hand() -> None:
    base = _inputs([1.0, 2.0], [0, 0])
    n = base.cost.shape[-1]
    reg = np.zeros((2, 2, n))
    reg[1] = 30.0
    age = np.full((2, n), 60.0)
    once = np.ones((1, n))
    # regret relative to age replacement: 0 / 60 and 30 / 60
    assert regret_loss(reg, age, once)[0].tolist() == [0.0, 0.5]
    r = RegretInputs(reg, age, np.zeros((2, 2, 3)))
    # configuration 1 chosen in fold 0 only: a third of the engines carry 30
    choice = np.array([[1, 0, 0], [1, 0, 0]])
    share = regret_selected(base, r, choice, once)[0]
    assert share == pytest.approx(1 - (30 * 4 / 12) / 60)


# S2 and S9


def test_the_settings_grid_spans_rules_lead_times_and_cost_ratios() -> None:
    grid = settings_grid(CFG.decision, "validation")
    assert len(grid) == 2 * 3 * 7
    assert PRIMARY in grid
    assert {g.tuning for g in grid} == {"validation"}


def test_the_competitive_region_is_measured_from_the_best_point_estimate() -> None:
    points = pd.DataFrame(
        {
            "config_id": ["a", "b", "c", "d"],
            "leaderboard_rmse": [14.0, 12.0, 13.9, 14.1],
        }
    )
    assert competitive(points, 2.0) == ["a", "b", "c"]


def test_draws_repeat_the_confirmatory_bootstrap() -> None:
    from maintadapt.research.stats import bootstrap_counts

    base = _inputs([1.0, 2.0], [0, 3])
    draws = Draws.of(base, 50, [2026, 1])
    assert (draws.train == bootstrap_counts(12, 50, [2026, 1, 0])).all()
    assert (draws.test == bootstrap_counts(8, 50, [2026, 1, 1])).all()
    assert draws.once_train.shape == (1, 12)
    assert json.dumps(draws.once_test.tolist()) == json.dumps([[1.0] * 8])
