from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest
from scipy.stats import norm
from synthetic_fleet import CFG, TRAINING, fold_data, subset

from maintadapt.research.archive import SCHEMA
from maintadapt.research.fold_data import PREDICTED
from maintadapt.research.models.spaces import load_pool
from maintadapt.research.policies import tune
from maintadapt.research.prob_rules import (
    apply_r3,
    apply_r4,
    plugin_probability,
    quantile_scores,
    scarce_tuning_sets,
    tune_r3,
)
from maintadapt.research.probabilistic import (
    LEVELS,
    QUANTILE_COLUMNS,
    cdf_at,
    column,
    fit_probabilistic,
    gaussian_quantiles,
    member_seed,
    monotone,
    probabilistic_table,
    quantiles_from_draws,
)
from maintadapt.sequence import resolve_device


def test_level_columns_are_ten_thousandths() -> None:
    assert [column(a) for a in (0.001, 0.0025, 0.5, 0.99)] == [
        "q0010",
        "q0025",
        "q5000",
        "q9900",
    ]
    assert len(set(QUANTILE_COLUMNS)) == len(LEVELS)
    assert list(LEVELS) == sorted(LEVELS)


def test_cdf_interpolates_between_stored_quantiles_by_hand() -> None:
    q, levels = np.array([[10.0, 20.0, 30.0]] * 6), [0.1, 0.5, 0.9]
    x = np.array([5.0, 10.0, 15.0, 25.0, 30.0, 40.0])
    assert cdf_at(q, x, levels) == pytest.approx([0.0, 0.1, 0.3, 0.7, 1.0, 1.0])
    # tied quantiles at 0: P(RUL <= 0) is the largest tied level
    assert cdf_at(np.array([[0.0, 0.0, 10.0]]), 0.0, levels)[0] == pytest.approx(0.5)


def test_quantile_helpers() -> None:
    q = gaussian_quantiles(np.array([50.0]), np.array([10.0]), [0.5, 0.975])
    assert q[0] == pytest.approx([50.0, 50.0 + 10.0 * norm.ppf(0.975)])
    draws = np.random.default_rng(0).normal(size=(200, 3))
    assert quantiles_from_draws(draws, [0.1, 0.9]) == pytest.approx(
        np.quantile(draws, [0.1, 0.9], axis=0).T
    )
    assert monotone(np.array([[5.0, -1.0, 130.0]]), 125).tolist() == [[0.0, 5.0, 125.0]]
    assert len({member_seed(7, k) for k in range(5)}) == 5


def test_plugin_probability_by_hand() -> None:
    # (L + 1) / ((K - 1) * mean life) = 11 / (9 * 200)
    assert plugin_probability(10, 10, 200.0) == pytest.approx(11 / 1800)
    assert plugin_probability(1.5, 20, 10.0) == 1.0
    with pytest.raises(ValueError):
        plugin_probability(1.0, 10, 200.0)


def _degenerate(path: list[float]) -> np.ndarray:
    """Every quantile equal to the given RUL at each cycle: a distribution with no spread."""
    return np.repeat(np.asarray(path, dtype=float)[:, None], len(LEVELS), axis=1)


def test_r4_on_exact_predictions_maintains_at_the_end_of_life() -> None:
    # true RUL 4, 3, 2, 1, 0; with L = 0, P(RUL <= 0) reaches 1 at cycle 5
    found = apply_r4([_degenerate([4, 3, 2, 1, 0])], 0, 0.01)
    assert (found.failed[0, 0], found.used[0, 0], found.waste[0, 0]) == (False, 5, 0)
    # with L = 1 the alarm comes at cycle 4 and maintenance at 5, still in time
    found = apply_r4([_degenerate([4, 3, 2, 1, 0])], 1, 0.01)
    assert (found.failed[0, 0], found.used[0, 0]) == (False, 5)
    # a prediction 2 cycles late alarms at cycle 5 with L = 1: maintenance at 6 > 5 fails
    found = apply_r4([_degenerate([6, 5, 4, 3, 2])], 1, 0.01)
    assert found.failed[0, 0]


def test_r3_is_rule_r1_on_a_lower_quantile() -> None:
    rng = np.random.default_rng(4)
    engines = [
        monotone(np.arange(n)[::-1, None] + rng.normal(0, 5, (n, len(LEVELS))), 125)
        for n in rng.integers(30, 60, 6)
    ]
    tuned = tune_r3(engines, 0.1, 10, [10.0], 41)[10.0]
    expected = tune(quantile_scores(engines, 0.1), "R1", [1.0], 10, [10.0], 41)[10.0]
    assert tuned == expected
    found = apply_r3(engines, 0.1, tuned, 10)
    assert found.failed.shape == (1, 6)


def test_scarce_tuning_sets() -> None:
    sets = scarce_tuning_sets(range(1, 21), 5, 3, np.random.default_rng(0))
    assert [len(s) for s in sets] == [5, 5, 5]
    assert all(len(set(s)) == 5 for s in sets)
    with pytest.raises(ValueError):
        scarce_tuning_sets(range(4), 5, 1, np.random.default_rng(0))


@pytest.mark.parametrize("family", ["xgboost_quantile", "mc_dropout", "ensemble"])
def test_probabilistic_families_store_sorted_quantiles_for_every_cycle(
    family: str,
) -> None:
    pool = {
        c.id: c for c in load_pool(CFG.paths.pool, CFG.pool.seed, CFG.pool.n_configs)
    }
    base = "xgboost-00" if family == "xgboost_quantile" else "cnn-00"
    params = dict(pool[base].params)
    if family == "xgboost_quantile":
        params["n_estimators"] = 30
    data = fold_data(subset())
    training = replace(TRAINING, max_epochs=1, patience=1)
    out = fit_probabilistic(family, params, data, 0, resolve_device("cpu"), training)
    table = probabilistic_table(data, out)
    assert table.schema.names[: len(SCHEMA)] == SCHEMA.names
    assert all(table.schema.field(c).type == pa.float32() for c in QUANTILE_COLUMNS)
    frame = table.to_pandas()
    assert len(frame) == sum(len(data.frame(s)) for s in PREDICTED)
    q = frame[list(QUANTILE_COLUMNS)].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()
    assert ((q >= 0) & (q <= data.cap)).all()
    if family == "ensemble":
        assert {f"member{k}" for k in range(5)} <= set(frame.columns)
