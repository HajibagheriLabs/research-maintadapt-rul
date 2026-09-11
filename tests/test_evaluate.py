import numpy as np
import pytest

from maintadapt.evaluate import nasa_score, rmse


def test_metrics_match_hand_computed_values() -> None:
    truth = np.array([50.0, 50.0])
    predicted = np.array([40.0, 60.0])
    assert rmse(truth, predicted) == pytest.approx(10.0)

    # 10 cycles early: exp(10/13) - 1 = 1.1581055; 10 cycles late: exp(10/10) - 1 = 1.7182818
    early = nasa_score(truth[:1], predicted[:1])
    late = nasa_score(truth[1:], predicted[1:])
    assert early == pytest.approx(1.1581055, abs=1e-6)
    assert late == pytest.approx(1.7182818, abs=1e-6)
    assert late > early
    assert nasa_score(truth, predicted) == pytest.approx(2.8763874, abs=1e-6)
    assert nasa_score(truth, truth) == 0.0
