import numpy as np
import pytest

from maintadapt.research.costs import (
    age_outcomes,
    cost_rate,
    perfect_rate,
    regret,
    savings_share,
    tune_age,
)
from maintadapt.research.policies import (
    NO_ALARM,
    TunedPolicy,
    alarm_times,
    apply_policy,
    ema,
    lives_of,
    median_minimiser,
    outcomes,
    scores,
    threshold_candidates,
    tune,
)

# the playbook's hand-worked fleet, K = 10
ENGINE_A = np.array([9, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
ENGINE_B = np.array([9, 8, 6, 4, 2, 0], dtype=float)
FLEET = [ENGINE_A, ENGINE_B]
LIVES = np.array([10, 6])
K = 10.0
LEVELS = 401
COST_RATIOS = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]


def _hand(theta: float, lead: int):
    alarms = alarm_times(scores(FLEET, "R1"), np.array([theta]))
    return alarms, outcomes(alarms, LIVES, lead)


def test_hand_worked_fleet_at_lead_two_has_savings_share_exactly_0_8() -> None:
    alarms, found = _hand(4.0, 2)
    # A first predicts 4 at cycle 7, B at cycle 4; maintenance two cycles later
    assert alarms.tolist() == [[7, 4]]
    assert found.failed.tolist() == [[False, False]]
    assert found.used.tolist() == [[9, 6]]
    assert found.waste.tolist() == [[1, 0]]
    rate = cost_rate(found, K)[0]
    assert rate == 2 / 15
    assert perfect_rate(LIVES) == 2 / 16
    # age replacement at A = 6 maintains both engines at cycle 6
    assert tune_age(LIVES, K) == 6.0
    age = age_outcomes(6.0, LIVES)
    assert age.used.tolist() == [[6, 6]] and not age.failed.any()
    age_rate = cost_rate(age, K)[0]
    assert age_rate == 2 / 12
    assert savings_share(rate, age_rate, perfect_rate(LIVES)) == 0.8
    assert regret(found, 100.0).mean() == 0.5


def test_hand_worked_fleet_at_lead_three_lets_engine_b_fail() -> None:
    _, found = _hand(4.0, 3)
    # A is maintained at cycle 10 with nothing wasted; B would be maintained at 7 > 6
    assert found.failed.tolist() == [[False, True]]
    assert found.used.tolist() == [[10, 6]]
    assert found.waste.tolist() == [[0, 0]]
    assert cost_rate(found, K)[0] == 11 / 16
    assert regret(found, 100.0).tolist() == [[0, 100]]


def test_ema_is_causal_and_alpha_one_is_the_prediction() -> None:
    # s = 4, then 0.5 * 2 + 0.5 * 4 = 3, then 0.5 * 0 + 0.5 * 3 = 1.5
    assert ema(np.array([4.0, 2.0, 0.0]), 0.5).tolist() == [4.0, 3.0, 1.5]
    assert ema(ENGINE_B, 1.0).tolist() == ENGINE_B.tolist()
    long = np.linspace(100, 0, 50)
    np.testing.assert_array_equal(ema(long, 0.3)[:20], ema(long[:20], 0.3))
    with pytest.raises(ValueError, match="alpha"):
        ema(ENGINE_B, 0.0)


def test_alarms_follow_the_running_minimum_and_cover_both_infinite_thresholds() -> None:
    bumpy = [np.array([5.0, 3.0, 6.0, 2.0])]
    # the score rises again after cycle 2, but the first crossing of 3 stays at cycle 2
    assert alarm_times(bumpy, np.array([3.0, 2.5, 2.0, 1.0])).tolist() == [
        [2],
        [4],
        [4],
        [NO_ALARM],
    ]
    infinite = alarm_times(scores(FLEET, "R1"), np.array([-np.inf, np.inf]))
    assert infinite[0].tolist() == [NO_ALARM, NO_ALARM]
    assert infinite[1].tolist() == [1, 1]
    never = outcomes(infinite[:1], LIVES, 0)
    assert never.failed.all() and never.used.tolist() == [[10, 6]]
    assert never.waste.sum() == 0


def test_a_lead_time_beyond_the_life_means_failure() -> None:
    alarms = np.array([[1, 1]])
    just_in_time = outcomes(alarms, LIVES, 5)
    # B: alarm at 1, maintained at 6 = T; A: maintained at 6, wasting 4
    assert just_in_time.failed.tolist() == [[False, False]]
    assert just_in_time.waste.tolist() == [[4, 0]]
    too_late = outcomes(alarms, LIVES, 6)
    assert too_late.failed.tolist() == [[False, True]]
    assert outcomes(alarms, LIVES, 10).failed.all()


def test_candidates_are_quantiles_of_the_pooled_scores_framed_by_infinities() -> None:
    candidates = threshold_candidates(scores(FLEET, "R1"), LEVELS)
    assert len(candidates) == LEVELS + 2
    assert candidates[0] == -np.inf and candidates[-1] == np.inf
    assert (np.diff(candidates) >= 0).all()
    assert set(candidates[1:-1]) <= set(np.concatenate(FLEET))
    assert candidates[1] == 0.0 and candidates[-2] == 9.0


def test_ties_go_to_the_median_minimiser() -> None:
    assert median_minimiser(np.array([3.0, 1.0, 2.0, 1.0, 1.0])) == 3
    # with an even number of minimisers, the lower of the two middle ones
    assert median_minimiser(np.array([1.0, 0.0, 0.0])) == 1


def test_tuning_finds_the_cheapest_candidate_and_reports_its_cost() -> None:
    for lead in (0, 2, 3):
        tuned = tune(FLEET, "R1", [1.0], lead, COST_RATIOS, LEVELS)
        candidates = threshold_candidates(scores(FLEET, "R1"), LEVELS)
        every = outcomes(alarm_times(scores(FLEET, "R1"), candidates), LIVES, lead)
        for k, policy in tuned.items():
            assert policy.cost_rate == cost_rate(every, k).min()
            again = cost_rate(apply_policy(FLEET, "R1", policy, lead), k)[0]
            assert again == policy.cost_rate


def test_r2_ties_between_alphas_go_to_the_lightest_smoothing() -> None:
    # constant predictions make every alpha's EMA identical
    flat = [np.full(30, 40.0), np.full(50, 20.0), np.full(70, 60.0)]
    tuned = tune(flat, "R2", [0.1, 0.5, 1.0, 0.3], 0, [10.0], LEVELS)
    assert tuned[10.0].alpha == 1.0


def _fleet(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """Noisy predictions of the capped truth, clipped to [0, 125] like the models' outputs."""
    engines = []
    for life in rng.integers(60, 250, size=n):
        truth = np.minimum(np.arange(life - 1, -1, -1), 125)
        engines.append(np.clip(truth + rng.normal(0, 15, life), 0, 125))
    return engines


def _same(a, b) -> None:
    for name in ("failed", "used", "waste"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


TRANSFORMS_R1 = {
    "x + 15": lambda x: x + 15,
    "1.25x": lambda x: 1.25 * x,
    "x^2 / 125": lambda x: x**2 / 125,
    "exp(x / 50)": lambda x: np.exp(x / 50),
}
TRANSFORMS_R2 = {
    "x + 15": lambda x: x + 15,
    "1.25x": lambda x: 1.25 * x,
    "1.25x + 15": lambda x: 1.25 * x + 15,
}


@pytest.mark.parametrize(
    ("rule", "alphas", "transform"),
    [("R1", [1.0], f) for f in TRANSFORMS_R1.values()]
    + [("R2", [1.0, 0.5, 0.3, 0.2, 0.1], f) for f in TRANSFORMS_R2.values()],
    ids=[f"R1 {n}" for n in TRANSFORMS_R1] + [f"R2 {n}" for n in TRANSFORMS_R2],
)
def test_tuned_outcomes_are_invariant_to_increasing_transforms(
    rule: str, alphas: list[float], transform
) -> None:
    rng = np.random.default_rng(11)
    tuning, applied = _fleet(rng, 25), _fleet(rng, 15)
    moved_tuning = [transform(e) for e in tuning]
    moved_applied = [transform(e) for e in applied]
    for lead in (0, 10, 20):
        base = tune(tuning, rule, alphas, lead, COST_RATIOS, LEVELS)
        moved = tune(moved_tuning, rule, alphas, lead, COST_RATIOS, LEVELS)
        for k in COST_RATIOS:
            assert moved[k].alpha == base[k].alpha
            assert moved[k].cost_rate == base[k].cost_rate
            _same(
                apply_policy(moved_applied, rule, moved[k], lead),
                apply_policy(applied, rule, base[k], lead),
            )


def test_the_capped_truth_reaches_savings_share_one_below_the_cap() -> None:
    rng = np.random.default_rng(5)
    lives = rng.integers(130, 300, size=20)
    oracle = [np.minimum(np.arange(t - 1, -1, -1), 125).astype(float) for t in lives]
    assert lives_of(oracle).tolist() == lives.tolist()
    perfect = perfect_rate(lives)
    for k in (1.5, 10.0, 50.0):
        age_rate = cost_rate(age_outcomes(tune_age(lives, k), lives), k)[0]
        for lead in (0, 10, 20, 60, 124):
            policy = tune(oracle, "R1", [1.0], lead, [k], LEVELS)[k]
            # alarm when T - t = L, maintenance at T exactly
            assert policy.theta == lead
            rate = cost_rate(apply_policy(oracle, "R1", policy, lead), k)[0]
            assert savings_share(rate, age_rate, perfect) == 1.0
        # at L = cap the capped truth is 125 from cycle 1 to T - 125, so no threshold can
        # place the alarm at T - 125 and some life is always wasted or lost
        policy = tune(oracle, "R1", [1.0], 125, [k], LEVELS)[k]
        rate = cost_rate(apply_policy(oracle, "R1", policy, 125), k)[0]
        assert savings_share(rate, age_rate, perfect) < 1.0


def test_age_replacement_lets_short_lives_fail_when_failures_are_cheap() -> None:
    lives = np.array([10, 100])
    # K = 1.5: A = 10 costs 2 per 20 cycles (0.1); A = 100 lets the short engine fail and
    # maintains the long one at its end, 2.5 per 110 cycles (0.0227); never, 3 per 110
    assert tune_age(lives, 1.5) == 100.0
    assert tune_age(lives, 50.0) == 10.0
    never = age_outcomes(np.inf, lives)
    assert never.failed.all() and cost_rate(never, 1.5)[0] == 3 / 110
    with pytest.raises(ValueError, match="more than maintenance"):
        cost_rate(age_outcomes(10.0, lives), 1.0)


def test_policies_reject_unknown_rules_and_negative_leads() -> None:
    with pytest.raises(ValueError, match="unknown rule"):
        scores(FLEET, "R3")
    with pytest.raises(ValueError, match="lead"):
        outcomes(np.array([[1, 1]]), LIVES, -1)
    assert isinstance(tune(FLEET, "R1", [1.0], 0, [10.0], LEVELS)[10.0], TunedPolicy)
    with pytest.raises(ValueError, match="some alphas"):
        tune(FLEET, "R2", [], 0, [10.0], LEVELS)
