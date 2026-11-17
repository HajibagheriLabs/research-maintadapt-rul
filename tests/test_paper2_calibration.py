import math

import numpy as np
import pandas as pd
import pytest

from maintadapt.uncertainty import (
    conformal_level,
    conformal_margin,
    coverage_by_true_rul,
    cut_points,
    engine_weights,
    interval_scores,
    matched_margin,
    mondrian_margins,
    quantile_shifts,
    rul_bins,
    weighted_quantile,
)


def test_cut_points_draw_the_same_number_of_cycles_within_every_life() -> None:
    lives = pd.Series({1: 3, 2: 50})
    points = cut_points(lives, 7, np.random.default_rng(0))
    assert points.groupby("unit_id").size().to_dict() == {1: 7, 2: 7}
    for unit, life in lives.items():
        cycles = points.loc[points["unit_id"] == unit, "cycle"]
        assert cycles.between(1, life).all()


def test_engine_weights_give_every_engine_the_same_total() -> None:
    assert engine_weights(np.array([1, 1, 2])).tolist() == [0.25, 0.25, 0.5]


def test_weighted_quantile_by_hand() -> None:
    # sorted: 1 (0.5), 2 (0.3), 3 (0.2); cumulative 0.5, 0.8, 1.0
    values, weights = np.array([3.0, 1.0, 2.0]), np.array([0.2, 0.5, 0.3])
    assert weighted_quantile(values, weights, 0.5) == 1.0
    assert weighted_quantile(values, weights, 0.6) == 2.0
    assert weighted_quantile(values, weights, 1.0) == 3.0


def test_conformal_level_counts_engines() -> None:
    assert conformal_level(0.9, 9) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        conformal_level(0.9, 8)


def test_interval_scores_are_negative_inside() -> None:
    scores = interval_scores(
        np.array([5.0, 10.0]), np.array([4.0, 11.0]), np.array([6.0, 12.0])
    )
    assert scores.tolist() == [-1.0, 1.0]


def test_one_point_per_engine_reproduces_split_conformal() -> None:
    rng = np.random.default_rng(1)
    y, lower = rng.normal(0, 5, 40), rng.normal(-3, 1, 40)
    upper = lower + 6
    units = np.arange(40)
    assert matched_margin(y, lower, upper, units, 0.9) == conformal_margin(
        y, lower, upper, 0.9
    )


def test_matched_margin_weights_engines_not_cycles() -> None:
    # engine 1: three points with score 0; engine 2: one point with score 10. Per engine the
    # weights are 1/6 each and 1/2; level 0.5 becomes 0.75 with two engines, reached at 10.
    y = np.array([0.0, 0.0, 0.0, 10.0])
    zeros = np.zeros(4)
    units = np.array([1, 1, 1, 2])
    assert matched_margin(y, zeros, zeros, units, 0.5) == 10.0
    # weighting the four points equally would have stopped at the third point, a score of 0
    assert np.sort(y)[math.ceil(0.75 * 4) - 1] == 0.0


def test_rul_bins_are_closed_on_the_left() -> None:
    edges = (0, 25, 50, math.inf)
    assert rul_bins(np.array([0.0, 24.9, 25.0, 130.0]), edges).tolist() == [0, 0, 1, 2]


def test_mondrian_bins_with_few_engines_take_the_pooled_margin() -> None:
    rng = np.random.default_rng(2)
    units = np.repeat(np.arange(20), 5)
    point = np.where(units < 18, 10.0, 80.0)
    y = point + rng.normal(0, np.where(units < 18, 1.0, 10.0))
    margins = mondrian_margins(y, point, point, point, units, (0, 50, math.inf), 0.8, 5)
    pooled = matched_margin(y, point, point, units, 0.8)
    late_only = matched_margin(
        y[point < 50], point[point < 50], point[point < 50], units[point < 50], 0.8
    )
    assert margins[0] == late_only
    assert margins[1] == pooled


def test_quantile_shifts_recover_a_constant_offset() -> None:
    q = np.array([[1.0, 2.0], [3.0, 5.0], [0.0, 4.0]])
    levels = np.array([0.1, 0.9])
    shifts = quantile_shifts(q[:, 0] + 2.0, q, levels, np.array([1, 2, 3]))
    assert shifts[0] == 2.0
    # column 2: residuals 1, 0, -2; the 0.9 weighted quantile is the largest, 1
    assert shifts[1] == 1.0


def test_coverage_by_true_rul_by_hand() -> None:
    table = coverage_by_true_rul(
        np.array([10.0, 60.0, 200.0]),
        np.array([5.0, 70.0, 100.0]),
        np.array([15.0, 80.0, 300.0]),
    ).set_index("bin")
    assert table.loc["[0, 25)", "coverage"] == 1.0
    assert table.loc["[50, 75)", "coverage"] == 0.0
    assert table.loc["[125, inf)", "coverage"] == 1.0
    assert table.loc["true RUL <= 50", ["n", "coverage", "width"]].tolist() == [
        1,
        1.0,
        10.0,
    ]
    assert np.isnan(table.loc["[25, 50)", "coverage"])
