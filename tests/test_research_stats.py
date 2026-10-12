import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau

from maintadapt.research.stats import (
    benjamini_hochberg,
    bootstrap_counts,
    bootstrap_p_value,
    choose,
    kendall_tau_b,
    pairwise_differences,
    percentile_interval,
    selection_loss,
)


def test_bootstrap_counts_are_seeded_multinomial_draws() -> None:
    counts = bootstrap_counts(5, 2000, [2026, 1])
    assert counts.shape == (2000, 5)
    assert (counts.sum(axis=1) == 5).all()
    np.testing.assert_array_equal(counts, bootstrap_counts(5, 2000, [2026, 1]))
    assert not np.array_equal(counts, bootstrap_counts(5, 2000, [2026, 2]))
    # every engine enters a replicate once on average
    np.testing.assert_allclose(counts.mean(axis=0), 1.0, atol=0.1)


def test_pooled_sums_over_resamples_are_matrix_products() -> None:
    values = np.array([1.0, 10.0, 100.0])
    # engine 0 drawn twice and engine 2 once: 1 + 1 + 100
    assert (np.array([[2, 0, 1]]) @ values).tolist() == [102.0]
    counts = bootstrap_counts(3, 50, 7)
    explicit = [values[np.repeat(np.arange(3), row)].sum() for row in counts]
    np.testing.assert_allclose(counts @ values, explicit)


def test_percentile_interval_takes_the_central_share() -> None:
    # the 5th and 95th percentiles of 0..100; 0.05 * 100 carries one rounding error
    lo, hi = percentile_interval(np.arange(101.0), 0.9)
    assert (lo, hi) == (pytest.approx(5.0), pytest.approx(95.0))
    both = percentile_interval(np.arange(101.0)[:, None] * [1.0, 2.0], 0.9)
    np.testing.assert_allclose(both, [[5.0, 10.0], [95.0, 190.0]])


def test_kendall_tau_b_corrects_for_ties() -> None:
    x, y = [1, 2, 2, 3], [1, 2, 3, 3]
    # 4 concordant, 0 discordant pairs; one pair tied in x only, one in y only:
    # (4 - 0) / sqrt((6 - 1) * (6 - 1)) = 0.8
    assert kendall_tau_b(x, y) == pytest.approx(0.8)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 4, 30), rng.integers(0, 4, 30)
    assert kendall_tau_b(a, b) == kendalltau(a, b, variant="b").statistic


def test_bootstrap_p_value_doubles_the_smaller_tail() -> None:
    d = np.array([-1.0, 1, 2, 3, 4, 5, 6, 7, 8])
    # one replicate at or below zero: 2 * (1 + 1) / (9 + 1) = 0.4
    assert bootstrap_p_value(d) == pytest.approx(0.4)
    assert bootstrap_p_value(np.full(99, 3.0)) == pytest.approx(0.02)
    assert bootstrap_p_value(np.array([-1.0, 1.0])) == 1.0


def test_pairwise_differences_report_every_pair_once() -> None:
    samples = pd.DataFrame(
        {"a": np.arange(1.0, 10.0), "b": np.zeros(9), "c": np.full(9, 5.0)}
    )
    estimates = pd.Series({"a": 5.0, "b": 0.0, "c": 5.0})
    table = pairwise_differences(estimates, samples, ci=0.8).set_index(["a", "b"])
    assert list(table.index) == [("a", "b"), ("a", "c"), ("b", "c")]
    assert table.loc[("a", "b"), "difference"] == 5.0
    # a - b runs 1..9, never at or below zero: 2 * (0 + 1) / 10
    assert table.loc[("a", "b"), "p_value"] == pytest.approx(0.2)
    assert table.loc[("b", "c"), "p_value"] == pytest.approx(0.2)
    assert table.loc[("a", "c"), "lo"] < 0 < table.loc[("a", "c"), "hi"]


def test_benjamini_hochberg_matches_a_hand_computed_example() -> None:
    p = np.array([0.01, 0.039, 0.03, 0.005, 0.20])
    adjusted, rejected = benjamini_hochberg(p, 0.05)
    # sorted p: 0.005, 0.01, 0.03, 0.039, 0.2; times m / rank: 0.025, 0.025, 0.05, 0.04875,
    # 0.2; the step-up minimum from the top gives 0.025, 0.025, 0.04875, 0.04875, 0.2
    np.testing.assert_allclose(adjusted, [0.025, 0.04875, 0.04875, 0.025, 0.2])
    assert rejected.tolist() == [True, True, True, True, False]
    capped, none = benjamini_hochberg(np.array([0.9, 0.95]), 0.05)
    assert capped.max() <= 1.0 and not none.any()


def test_selection_picks_the_lowest_validation_metric_per_fold() -> None:
    validation = pd.DataFrame(
        {
            "repeat": [0, 0, 0, 0, 1, 1],
            "fold": [0, 0, 1, 1, 0, 0],
            "config_id": ["a", "b", "a", "b", "a", "b"],
            "val_rmse": [10.0, 12.0, 11.0, 9.0, 8.0, 8.0],
            "val_cost": [0.5, 0.4, 0.3, 0.6, 0.2, 0.1],
        }
    )
    by_rmse = choose(validation, "val_rmse")
    # repeat 1 fold 0 ties on RMSE, and the configuration listed first wins
    assert by_rmse["config_id"].tolist() == ["a", "b", "a"]
    assert choose(validation, "val_cost")["config_id"].tolist() == ["b", "a", "b"]
    assert selection_loss(0.6, 0.45) == pytest.approx(0.15)
    np.testing.assert_allclose(
        selection_loss(np.array([0.6, 0.5]), np.array([0.6, 0.7])), [0.0, -0.2]
    )
