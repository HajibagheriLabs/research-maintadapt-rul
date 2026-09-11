import numpy as np
import pytest

from maintadapt.decision import CostModel, fixed_schedule, replay_alarms
from maintadapt.uncertainty import conformal_margin, coverage


def test_policy_costs_match_a_hand_worked_fleet() -> None:
    costs = CostModel(failure=100.0, per_cycle=1.0)
    # engine A crosses threshold 3 when its true RUL is 1; engine B never crosses and fails
    engine_a = (
        np.array([9.0, 7.0, 5.0, 3.0, 1.0]),
        np.array([4.0, 3.0, 2.0, 1.0, 0.0]),
    )
    engine_b = (
        np.array([9.0, 8.0, 7.0, 6.0, 5.0]),
        np.array([4.0, 3.0, 2.0, 1.0, 0.0]),
    )
    outcomes = replay_alarms([engine_a, engine_b], np.array([3.0]), run_to_failure=True)
    assert outcomes.failed.tolist() == [[False, True]]
    assert outcomes.cost(costs).tolist() == [pytest.approx((1.0 + 100.0) / 2)]

    # cut 10 cycles before failure: booking 18 - 3 = 15 ahead fails, 18 - 10 = 8 ahead wastes 2
    truncated = (np.array([20.0, 18.0]), np.array([12.0, 10.0]))
    outcomes = replay_alarms([truncated], np.array([3.0, 10.0]), run_to_failure=False)
    assert outcomes.cost(costs).tolist() == [pytest.approx(100.0), pytest.approx(2.0)]

    # lives of 5 and 10 cycles serviced at age 6: the first fails, the second loses 4 cycles
    schedule = fixed_schedule(np.array([5.0, 10.0]), np.array([6.0]))
    assert schedule.cost(costs).tolist() == [pytest.approx((100.0 + 4.0) / 2)]


def test_interval_coverage_and_conformal_margin_by_hand() -> None:
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    lower, upper = np.array([0.0, 2.5, 2.0, 5.0]), np.array([2.0, 3.0, 4.0, 6.0])
    # only the first and third truths fall inside their intervals
    assert coverage(truth, lower, upper) == 0.5
    # scores max(lower - y, y - upper) are -1, 0.5, -1, 1; a 50% target needs the 3rd smallest
    assert conformal_margin(truth, lower, upper, level=0.5) == 0.5
