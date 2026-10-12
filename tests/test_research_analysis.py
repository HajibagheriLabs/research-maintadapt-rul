from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthetic_fleet import CFG

from maintadapt.research.analysis import (
    HYPOTHESES,
    RULES,
    Estimate,
    FleetInputs,
    analyse_fleet,
    band,
    choose,
    decision_loss,
    fleet_inputs,
    jitter,
    leaderboard_rmse,
    selected_share,
    verdict_h1,
    verdict_h2,
    verdict_h3,
)
from maintadapt.research.derived import savings

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = CFG.decision.primary
K = PRIMARY.cost_ratio
N_ENGINES, N_TEST, LIFE = 12, 8, 100
REPEATS, FOLDS = 2, 3


def _inputs(
    rmse: list[float],
    failures: list[int],
    val_cost: np.ndarray | None = None,
    val_rmse: np.ndarray | None = None,
) -> FleetInputs:
    """Configuration c fails on engines 0..failures[c]-1 and maintains the rest at cycle 90.

    Age replacement maintains every engine at 80; lives are all 100, so CR* = 1/100 and
    CR_age = 1/80. Every official test engine has squared error rmse[c]^2.
    """
    c = len(rmse)
    failed = np.arange(N_ENGINES)[None, :] < np.array(failures)[:, None]
    cost = np.where(failed, K, 1.0)[:, None, :].repeat(REPEATS, axis=1)
    used = np.where(failed, LIFE, 90.0)[:, None, :].repeat(REPEATS, axis=1)
    fold_of = np.tile(np.arange(N_ENGINES) % FOLDS, (REPEATS, 1))
    sq = (np.array(rmse, dtype=float) ** 2)[:, None, None]
    fits = REPEATS * FOLDS
    return FleetInputs(
        configs=[f"m-{i:02d}" for i in range(c)],
        repeats=list(range(REPEATS)),
        folds=list(range(FOLDS)),
        lives=np.full(N_ENGINES, float(LIFE)),
        cost=cost,
        used=used,
        fold_of=fold_of,
        age_cost=np.ones((REPEATS, N_ENGINES)),
        age_used=np.full((REPEATS, N_ENGINES), 80.0),
        val_cost=np.zeros((c, REPEATS, FOLDS)) if val_cost is None else val_cost,
        val_rmse=np.zeros((c, REPEATS, FOLDS)) if val_rmse is None else val_rmse,
        sq_last=np.broadcast_to(sq, (c, fits, N_TEST)).copy(),
        jitter=np.broadcast_to(
            np.array(rmse, dtype=float)[:, None, None], (c, fits, 5)
        ).copy(),
        jitter_engines=np.arange(5),
    )


def _share(failures: int) -> float:
    """Hand-worked S for f failures among the twelve engines."""
    rate = (K * failures + (N_ENGINES - failures)) / (
        LIFE * failures + 90 * (N_ENGINES - failures)
    )
    return (1 / 80 - rate) / (1 / 80 - 1 / 100)


def test_rules_and_hypotheses_are_quoted_verbatim_from_the_preregistration() -> None:
    text = " ".join((ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8").split())
    for rule in RULES.values():
        assert rule in text
    for key, hypothesis in HYPOTHESES.items():
        assert f"{key}: {hypothesis}" in text


def test_decision_loss_increasing_with_rmse_gives_tau_b_one() -> None:
    inputs = _inputs(rmse=[1, 2, 3, 4, 5, 6], failures=[0, 1, 2, 3, 4, 5])
    result = analyse_fleet(inputs, 200, 0.95, [1, 1])
    assert result.e1.point == pytest.approx(1.0)
    # one shared fail-free point: S for zero failures is 5/9 exactly
    assert result.points["savings_share"].iloc[0] == pytest.approx(5 / 9)
    assert result.points["leaderboard_rmse"].tolist() == pytest.approx(
        [1, 2, 3, 4, 5, 6]
    )
    decreasing = analyse_fleet(
        _inputs([6, 5, 4, 3, 2, 1], [0, 1, 2, 3, 4, 5]), 200, 0.95, [1, 1]
    )
    assert decreasing.e1.point == pytest.approx(-1.0)
    # jitter tracks RMSE here, so it is exactly as strongly associated: delta-tau = 0
    assert result.e3.point == pytest.approx(0.0)


def test_selection_by_rmse_that_is_known_to_be_worse_gives_a_positive_loss() -> None:
    # A (index 0) has the lower validation RMSE but fails on three engines; B never fails
    val_rmse = np.stack([np.zeros((REPEATS, FOLDS)), np.ones((REPEATS, FOLDS))])
    val_cost = np.stack([np.ones((REPEATS, FOLDS)), np.zeros((REPEATS, FOLDS))])
    inputs = _inputs([1, 2], [3, 0], val_cost=val_cost, val_rmse=val_rmse)
    result = analyse_fleet(inputs, 200, 0.95, [2, 1])
    assert result.s_by_cost.point == pytest.approx(_share(0))
    assert result.s_by_rmse.point == pytest.approx(_share(3))
    assert result.e2.point == pytest.approx(_share(0) - _share(3))
    assert result.e2.point > 0 and result.e2.lo > 0


def test_each_fold_contributes_the_outcomes_of_its_own_choice() -> None:
    val_rmse = np.stack([np.zeros((REPEATS, FOLDS)), np.ones((REPEATS, FOLDS))])
    # by validation cost: B in folds 0 and 1, A in fold 2
    val_cost = np.stack([np.ones((REPEATS, FOLDS)), np.zeros((REPEATS, FOLDS))])
    val_cost[0, :, 2], val_cost[1, :, 2] = 0.0, 1.0
    inputs = _inputs([1, 2], [3, 0], val_cost=val_cost, val_rmse=val_rmse)
    once = np.ones((1, N_ENGINES))
    # A fails on engines 0, 1, 2; only engine 2 lies in fold 2: one failure in twelve
    assert selected_share(inputs, choose(inputs.val_cost), once)[0] == pytest.approx(
        _share(1)
    )
    assert selected_share(inputs, choose(inputs.val_rmse), once)[0] == pytest.approx(
        _share(3)
    )


def test_choice_ties_go_to_the_configuration_listed_first() -> None:
    assert choose(np.zeros((3, 1, 2))).tolist() == [[0, 0]]
    with pytest.raises(ValueError, match="missing"):
        choose(np.array([[[np.nan]]]))


def test_bootstrap_intervals_are_deterministic_for_a_seed() -> None:
    # validation cost picks configuration 3, which fails on four engines, so its savings
    # share moves with the resampled engines; tau-b over six configurations takes few values
    val_cost = np.ones((6, REPEATS, FOLDS))
    val_cost[3] = 0.0
    inputs = _inputs([1, 3, 2, 5, 4, 6], [0, 2, 1, 4, 3, 5], val_cost=val_cost)
    # engines that differ in cycles used make every resample's share distinct
    spread = np.random.default_rng(0).uniform(60, 90, N_ENGINES)
    inputs = replace(inputs, used=np.minimum(inputs.used, spread))
    first = analyse_fleet(inputs, 300, 0.95, [7, 1])
    again = analyse_fleet(inputs, 300, 0.95, [7, 1])
    assert first.e1 == again.e1 and first.e2 == again.e2 and first.e3 == again.e3
    other = analyse_fleet(inputs, 300, 0.95, [8, 1])
    assert (other.s_by_cost.lo, other.s_by_cost.hi) != (
        first.s_by_cost.lo,
        first.s_by_cost.hi,
    )
    assert first.e1.lo <= first.e1.point <= first.e1.hi


@pytest.mark.parametrize(
    ("tau", "expected"),
    [
        (0.7, "strong"),
        (0.6999, "moderate"),
        (0.4, "moderate"),
        (0.3999, "weak"),
        (-0.5, "weak"),
    ],
)
def test_bands_follow_the_registered_cut_points(tau: float, expected: str) -> None:
    assert band(tau) == expected


def _fleets(*estimates: tuple[float, float, float]) -> dict[str, Estimate]:
    return {
        f"FD00{i + 1}": Estimate(p, lo, hi, 2000)
        for i, (p, lo, hi) in enumerate(estimates)
    }


def test_h1_verdicts_at_their_boundaries() -> None:
    weak = (0.2, 0.0, 0.5999)
    assert verdict_h1(_fleets(weak, weak, weak, (0.9, 0.8, 1.0))) == "supported"
    # an upper bound of exactly 0.6 is not below 0.6
    assert verdict_h1(_fleets(weak, weak, (0.2, 0.0, 0.6), weak)) == "supported"
    assert verdict_h1(_fleets(weak, weak, (0.2, 0.0, 0.6), (0.2, 0.0, 0.6))) == "mixed"
    strong = (0.8, 0.5001, 0.9)
    assert verdict_h1(_fleets(strong, strong, strong, weak)) == "rejected"
    assert verdict_h1(_fleets(strong, strong, (0.8, 0.5, 0.9), weak)) == "mixed"
    # a point estimate of 0.4 is moderate, not weak
    assert verdict_h1(_fleets(weak, weak, (0.4, 0.1, 0.5), (0.4, 0.1, 0.5))) == "mixed"


def test_h2_verdicts_at_their_boundaries() -> None:
    material = (0.05, 0.001, 0.1)
    flat = (0.0, -0.02, 0.02)
    assert verdict_h2(_fleets(material, material, flat, flat)) == "supported"
    assert (
        verdict_h2(_fleets(material, (0.0499, 0.001, 0.1), flat, flat))
        == "inconclusive"
    )
    # an interval touching 0 does not exclude it
    assert verdict_h2(_fleets(material, (0.06, 0.0, 0.1), flat, flat)) == "inconclusive"
    assert verdict_h2(_fleets(flat, flat, flat, material)) == "rejected"
    assert (
        verdict_h2(_fleets(flat, flat, (0.0, -0.0201, 0.02), material))
        == "inconclusive"
    )


def test_h3_verdicts_at_their_boundaries() -> None:
    better = (0.1, 0.01, 0.2)
    assert verdict_h3(_fleets(better, better, better, (-0.1, -0.2, 0.0))) == "supported"
    assert verdict_h3(_fleets(better, better, (0.1, -0.01, 0.2), (0.0, -0.1, 0.1))) == (
        "inconclusive"
    )
    # delta-tau of exactly 0 counts as not stronger
    zero = (0.0, -0.1, 0.1)
    assert verdict_h3(_fleets(zero, zero, (-0.2, -0.3, -0.1), better)) == "rejected"


def _tables() -> dict[str, pd.DataFrame]:
    """Derived tables of two configurations, one repeat and two folds, by hand."""
    lives = {1: 100, 2: 120, 3: 90, 4: 110}
    fold = {1: 0, 2: 0, 3: 1, 4: 1}
    rows = []
    for config, failing in (("ridge-00", {3}), ("knn-00", {1, 4})):
        for unit, life in lives.items():
            failed = unit in failing
            for lead, k in ((PRIMARY.lead_time, K), (0, K), (PRIMARY.lead_time, 2.0)):
                rows.append(
                    {
                        "config_id": config,
                        "family": config[:-3],
                        "repeat": 0,
                        "fold": fold[unit],
                        "tuning": "validation",
                        "rule": "R1",
                        "lead": lead,
                        "cost_ratio": k,
                        "unit_id": unit,
                        "failed": failed,
                        "used": life if failed else life - 7,
                        "waste": 0 if failed else 7,
                    }
                )
    outcomes = pd.DataFrame(rows)
    age = pd.DataFrame(
        [
            {
                "repeat": 0,
                "fold": fold[u],
                "cost_ratio": k,
                "age": 85.0,
                "unit_id": u,
                "failed": False,
                "used": 85,
                "waste": life - 85,
            }
            for u, life in lives.items()
            for k in (K, 2.0)
        ]
    )
    policies = pd.DataFrame(
        [
            {
                "config_id": c,
                "family": c[:-3],
                "repeat": 0,
                "fold": f,
                "tuning": "validation",
                "rule": "R1",
                "lead": PRIMARY.lead_time,
                "cost_ratio": K,
                "theta": 20.0,
                "alpha": 1.0,
                "tuned_cost_rate": 0.01 * (i + 1) + f,
            }
            for i, c in enumerate(("ridge-00", "knn-00"))
            for f in (0, 1)
        ]
    )
    parts = []
    for i, c in enumerate(("ridge-00", "knn-00")):
        for f in (0, 1):
            for u in (1, 2):
                parts.append(
                    {
                        "config_id": c,
                        "repeat": 0,
                        "fold": f,
                        "split": "official_test",
                        "unit_id": u,
                        "sq_last_125": (i + 1) * (f + 1) * u * 1.0,
                        "jit_sum": 12.0 * u,
                        "jit_n": 6.0 if u == 1 else 2.0,
                        "sse_all": 0.0,
                        "n_all": 1.0,
                    }
                )
            parts.append(
                {
                    "config_id": c,
                    "repeat": 0,
                    "fold": f,
                    "split": "inner_val",
                    "unit_id": 9,
                    "sq_last_125": np.nan,
                    "jit_sum": np.nan,
                    "jit_n": np.nan,
                    "sse_all": 4.0 * (i + 1),
                    "n_all": 4.0,
                }
            )
    engines = pd.DataFrame({"unit_id": list(lives), "life": list(lives.values())})
    return {
        "outcomes": outcomes,
        "policies": policies,
        "age": age,
        "engines": engines,
        "components": pd.DataFrame(parts),
    }


def test_inputs_from_derived_tables_reproduce_their_savings_and_metrics() -> None:
    tables = _tables()
    configs = ["ridge-00", "knn-00"]
    inputs = fleet_inputs(tables, PRIMARY, configs)
    primary = tables["outcomes"].query(
        "lead == @PRIMARY.lead_time and cost_ratio == @K"
    )
    lives = tables["engines"].set_index("unit_id")["life"]
    expected = savings(primary, tables["age"].query("cost_ratio == @K"), lives)
    once = np.ones((1, 4))
    loss = decision_loss(inputs, once)[0]
    for config, d in zip(configs, loss, strict=True):
        s = expected.loc[expected["config_id"] == config, "savings_share"].iloc[0]
        assert d == pytest.approx(1 - s, rel=1e-12)
    # official RMSE: mean over the two folds of sqrt(mean over engines 1 and 2)
    rmse = leaderboard_rmse(inputs, np.ones((1, 2)))[0]
    for i, value in enumerate(rmse):
        by_fold = [np.sqrt((i + 1) * (f + 1) * 1.5) for f in (0, 1)]
        assert value == pytest.approx(np.mean(by_fold))
    # only engine 1 has the five jitter cycles needed; its mean step is 12 / 6 = 2
    assert inputs.jitter_engines.tolist() == [0]
    assert jitter(inputs, np.ones((1, 1)))[0].tolist() == [2.0, 2.0]
    assert inputs.val_rmse[:, 0, 0].tolist() == [1.0, np.sqrt(2.0)]
    assert inputs.val_cost[1, 0, 1] == pytest.approx(1.02)
    with pytest.raises(ValueError, match="missing"):
        broken = dict(tables)
        broken["outcomes"] = tables["outcomes"].iloc[1:]
        fleet_inputs(broken, PRIMARY, configs)
