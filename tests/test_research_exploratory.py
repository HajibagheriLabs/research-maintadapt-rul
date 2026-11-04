from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from synthetic_fleet import CFG

from maintadapt.decision import CostModel
from maintadapt.research.derived import repeat_outcomes
from maintadapt.research.exploratory import (
    ema_all,
    fit_quantities,
    last_cycle_squared_errors,
    late_squared_errors,
    legacy_booking,
    recalibrate,
    splits_of,
    tune_r1,
)
from maintadapt.research.policies import ema, tune

PRIMARY = CFG.decision.primary
COSTS = CostModel(failure=100.0, per_cycle=1.0)


def test_the_filtered_ema_is_the_policy_ema_for_every_engine() -> None:
    rng = np.random.default_rng(0)
    engines = [rng.normal(50, 10, size=n) for n in (1, 7, 40, 200)]
    for alpha in (1.0, 0.5, 0.1):
        for got, p in zip(ema_all(engines, alpha), engines, strict=True):
            assert np.allclose(got, ema(p, alpha), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="alpha"):
        ema_all(engines, 0.0)


def test_isotonic_recalibration_pools_violators_and_clips_outside_the_range() -> None:
    # truth 30 then 20 violates the order, so both become their mean, 25
    mapped = recalibrate(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([10.0, 30.0, 20.0, 40.0]),
        np.array([0.0, 2.5, 4.0, 9.0]),
    )
    assert mapped.tolist() == [10.0, 25.0, 40.0, 40.0]


def _fit(rows: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]) -> pd.DataFrame:
    parts = []
    for split, engines in rows.items():
        for unit, (pred, rul) in engines.items():
            parts.append(
                pd.DataFrame(
                    {
                        "unit_id": unit,
                        "cycle": np.arange(1, len(pred) + 1),
                        "rul": rul,
                        "split": split,
                        "pred": pred,
                    }
                )
            )
    table = pd.concat(parts, ignore_index=True)
    table["split"] = pd.Categorical(table["split"])
    return table


def test_splits_are_read_as_engines_in_unit_order() -> None:
    fit = _fit(
        {
            "inner_val": {
                2: (np.array([5.0, 4.0]), np.array([1, 0])),
                3: (np.array([9.0]), np.array([0])),
            },
            "outer_test": {7: (np.array([1.0, 2.0, 3.0]), np.array([2, 1, 0]))},
        }
    )
    parts = splits_of(fit)
    assert parts["inner_val"].units.tolist() == [2, 3]
    assert parts["inner_val"].lives.tolist() == [2, 1]
    assert parts["outer_test"].booking()[0][1].tolist() == [2.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="in order"):
        splits_of(fit.iloc[[1, 0, 2, 3, 4, 5]])


def test_late_and_last_cycle_errors_by_hand() -> None:
    rul = [np.array([60.0, 50.0, 49.0]), np.array([70.0, 69.0])]
    pred = [np.array([0.0, 52.0, 46.0]), np.array([0.0, 0.0])]
    sse, n = late_squared_errors(rul, pred)
    # cycles with true RUL <= 50: (52 - 50)^2 + (46 - 49)^2; none on the second engine
    assert sse.tolist() == [13.0, 0.0] and n.tolist() == [2, 0]
    # last cycles: 46 against 49, and 0 against the truth 69 capped at 60
    assert last_cycle_squared_errors(pred, rul, 60).tolist() == [9.0, 3600.0]


def test_r1_tuning_matches_the_policy_tuning() -> None:
    rng = np.random.default_rng(3)
    engines = [
        np.clip(np.arange(life - 1, -1, -1) + rng.normal(0, 8, life), 0, 125)
        for life in (120, 150, 180, 210)
    ]
    for lead, k in ((0, 10.0), (10, 2.0)):
        got = tune_r1(engines, lead, k, 401)
        expected = tune(engines, "R1", [], lead, [k], 401)[k]
        assert (got.theta, got.cost_rate) == (expected.theta, expected.cost_rate)


def test_the_legacy_booking_by_hand() -> None:
    life = np.arange(9, -1, -1, dtype=float)
    # a perfect validation history: every cut wastes exactly theta, so theta 0 is cheapest
    val = [(life, life)]
    truth = np.array([30.0, 29.0, 28.0])
    test = [(truth, truth), (truth + 5.0, truth)]
    theta, cost = legacy_booking(val, test, COSTS, np.arange(0.0, 6.0))
    # the exact engine is booked for its last cycle; the one 5 cycles optimistic is booked
    # 33 cycles ahead with 28 left and fails
    assert theta == 0.0
    assert cost.tolist() == [0.0, 100.0]


def test_every_fit_quantity_and_alpha_one_reproduces_the_derived_outcomes() -> None:
    rng = np.random.default_rng(7)

    def noisy(life: int) -> tuple[np.ndarray, np.ndarray]:
        rul = np.arange(life - 1, -1, -1)
        return np.clip(rul + rng.normal(0, 10, life), 0, 125), rul

    def truncated(life: int, cut: int) -> tuple[np.ndarray, np.ndarray]:
        pred, rul = noisy(life)
        return pred[:cut], rul[:cut]

    fits = {
        f: _fit(
            {
                "inner_val": {u: noisy(130 + 9 * u) for u in (1, 2, 3)},
                "outer_test": {u: noisy(140 + 7 * u) for u in (4 + 2 * f, 5 + 2 * f)},
                "official_test": {1: truncated(150, 120), 2: truncated(160, 60)},
            }
        )
        for f in (0, 1)
    }
    lead, k = PRIMARY.lead_time, PRIMARY.cost_ratio
    decision = replace(CFG.decision, cost_ratios=[k], lead_times=[lead])
    _, derived = repeat_outcomes(fits, decision)
    for f, fit in fits.items():
        got = fit_quantities(
            fit, [1.0, 0.5], lead, k, 401, 125, COSTS, np.arange(0.0, 126.0)
        )
        assert got["official"].columns.tolist() == [
            "unit_id",
            "sse_late_iso",
            "n_late",
            "legacy_cost",
            "sq_last_ema_1",
            "sq_last_ema_0.5",
        ]
        # the second official engine stops 100 cycles before failure: no late cycles
        assert got["official"]["n_late"].tolist()[1] == 0
        mine = got["outer"][got["outer"]["alpha"] == 1.0]
        theirs = derived[
            (derived["fold"] == f)
            & (derived["tuning"] == "validation")
            & (derived["rule"] == "R1")
        ]
        assert mine["failed"].tolist() == theirs["failed"].tolist()
        assert mine["used"].tolist() == theirs["used"].tolist()
        assert got["policies"]["alpha"].tolist() == [1.0, 0.5]
