from dataclasses import replace
from functools import cache

import numpy as np
import pandas as pd
import pytest
import torch
from synthetic_fleet import CFG, CONSTANT, FOLD, ROOT, TRAINING, fold_data, subset

from maintadapt.data import SENSORS, Subset
from maintadapt.research.fold_data import PREDICTED, FoldData
from maintadapt.research.models.registry import (
    CONTROL_FAMILIES,
    NEURAL_FAMILIES,
    TRAINERS,
    Fitted,
    build_network,
    check_params,
    fit,
    train,
)
from maintadapt.research.models.spaces import FAMILY_NAMES, load_pool

POOL = load_pool(ROOT / CFG.paths.pool, CFG.pool.seed, CFG.pool.n_configs)
REFERENCE = {c.family: c.params for c in POOL if c.id.endswith("-00")}
CPU = torch.device("cpu")
SEED = 7


def _subset() -> Subset:
    return subset()


def _perturbed() -> Subset:
    """Outer_test and official test engines with wildly different readings."""
    fleet = subset()
    train, test = fleet.train.copy(), fleet.test.copy()
    held = train["unit_id"].isin(FOLD.outer_test)
    train.loc[held, SENSORS] = train.loc[held, SENSORS] * 1000.0 + 7.0
    test[SENSORS] = test[SENSORS] * -500.0 + 3.0
    return Subset("FD001", train, test)


def _fold_data(fleet: Subset) -> FoldData:
    return fold_data(fleet)


@cache
def _data() -> FoldData:
    return _fold_data(_subset())


@cache
def _trained(family: str) -> Fitted:
    return train(family, REFERENCE[family], _data(), SEED, CPU, TRAINING)


def _restrict(data: FoldData, keep: dict[str, pd.Series]) -> FoldData:
    """The same fitted fold with fewer rows in some splits and an empty cache."""
    frames = dict(data.frames)
    for split, mask in keep.items():
        frames[split] = frames[split][mask].reset_index(drop=True)
    return replace(data, frames=frames)


def _assert_same(family: str, got: np.ndarray, expected: np.ndarray) -> None:
    """Equal up to summation order, far below anything a look-ahead would change."""
    if family in NEURAL_FAMILIES:
        # float32 networks output RUL/cap; another batch composition moves its last bits
        cap = _data().cap
        np.testing.assert_allclose(got / cap, expected / cap, rtol=0.0, atol=1e-5)
    else:
        # float64; BLAS sums rows in blocks whose boundaries move when rows are removed
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)


def test_registry_covers_every_family_of_the_pool_and_two_controls() -> None:
    assert set(TRAINERS) == set(FAMILY_NAMES) | set(CONTROL_FAMILIES)
    assert set(REFERENCE) == set(FAMILY_NAMES)


def test_controls_predict_the_capped_truth_and_the_cap() -> None:
    data = _data()
    oracle = fit("control_oracle", {}, data, SEED, CPU, TRAINING)
    constant = fit("control_constant", {}, data, SEED, CPU, TRAINING)
    for split in PREDICTED:
        truth = data.target(split).to_numpy()
        np.testing.assert_array_equal(oracle.predictions[split], truth)
        assert (constant.predictions[split] == data.cap).all()
    assert oracle.val_rmse == 0.0
    assert (oracle.n_inputs, constant.n_inputs) == (1, 0)
    with pytest.raises(ValueError, match="unknown keys"):
        check_params("control_oracle", {"alpha": 1.0})


def test_fold_data_fits_every_statistic_on_inner_fit_engines() -> None:
    data, subset = _data(), _subset()
    units = {split: set(frame["unit_id"]) for split, frame in data.frames.items()}
    assert units["inner_fit"] == set(FOLD.inner_fit)
    assert units["inner_val"] == set(FOLD.inner_val)
    assert units["outer_test"] == set(FOLD.outer_test)
    assert units["official_test"] == set(subset.test["unit_id"])
    for frame in data.frames.values():
        order = frame[["unit_id", "cycle"]]
        pd.testing.assert_frame_equal(
            order, order.sort_values(["unit_id", "cycle"], ignore_index=True)
        )
    assert not set(CONSTANT) & set(data.sensors)
    assert len(data.sensors) == len(SENSORS) - len(CONSTANT)
    # z-scores fitted on inner_fit alone are exactly centred and scaled there, not elsewhere
    fitted = data.frame("inner_fit")[data.sensors]
    np.testing.assert_allclose(fitted.mean(), 0.0, atol=1e-9)
    np.testing.assert_allclose(fitted.std(ddof=0), 1.0, atol=1e-9)
    assert data.frame("outer_test")[data.sensors].mean().abs().max() > 1e-3
    assert data.rolling("inner_val") is data.rolling("inner_val")
    assert data.windows("inner_val", 20) is data.windows("inner_val", 20)
    assert data.windows("inner_val", 20).shape == (
        len(data.frame("inner_val")),
        20,
        len(data.sensors),
    )


def test_perturbing_held_out_engines_leaves_the_fitted_statistics_unchanged() -> None:
    base, moved = _data(), _fold_data(_perturbed())
    assert moved.sensors == base.sensors
    pd.testing.assert_frame_equal(moved.screening, base.screening)
    pd.testing.assert_frame_equal(moved.normaliser.mean, base.normaliser.mean)
    pd.testing.assert_frame_equal(moved.normaliser.std, base.normaliser.std)
    # KMeans sums in a thread- and alignment-dependent order, so even two runs on the same
    # rows differ in the last bits; the regime of every row (in the frames below) is exact
    np.testing.assert_allclose(
        moved.conditions.centers, base.conditions.centers, rtol=0.0, atol=1e-12
    )
    for split in ("inner_fit", "inner_val"):
        pd.testing.assert_frame_equal(moved.frame(split), base.frame(split))
    # the perturbation reached the held-out frames, so the checks above are not vacuous
    for split in ("outer_test", "official_test"):
        assert not moved.frame(split)[base.sensors].equals(
            base.frame(split)[base.sensors]
        )


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_held_out_engines_never_change_inner_val_predictions(family: str) -> None:
    moved = _fold_data(_perturbed())
    refit = train(family, REFERENCE[family], moved, SEED, CPU, TRAINING)
    np.testing.assert_array_equal(
        refit.predict(moved, "inner_val"),
        _trained(family).predict(_data(), "inner_val"),
    )


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_predictions_are_causal(family: str) -> None:
    data, fitted = _data(), _trained(family)
    # 12 cycles is shorter than every window and rolling span, 45 longer than most
    for t0 in (12, 45):
        cut = _restrict(data, {s: data.frame(s)["cycle"] <= t0 for s in PREDICTED})
        for split in PREDICTED:
            before = (data.frame(split)["cycle"] <= t0).to_numpy()
            _assert_same(
                family,
                fitted.predict(cut, split),
                fitted.predict(data, split)[before],
            )


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_same_seed_gives_the_same_fit_on_cpu(family: str) -> None:
    data, fitted = _data(), _trained(family)
    out = fit(family, REFERENCE[family], data, SEED, CPU, TRAINING)
    for split in PREDICTED:
        np.testing.assert_array_equal(
            out.predictions[split], fitted.predict(data, split)
        )
    truth = data.target("inner_val").to_numpy()
    rmse = np.sqrt(np.mean((out.predictions["inner_val"] - truth) ** 2))
    assert out.val_rmse == pytest.approx(rmse, rel=1e-12)
    assert out.fit_seconds > 0
    assert out.n_inputs == fitted.n_inputs
    assert out.best_iteration == fitted.best_iteration
    has_early_stopping = family in NEURAL_FAMILIES or family.startswith("xgboost")
    assert (out.best_iteration is not None) == has_early_stopping


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_one_prediction_per_cycle_in_frame_order(family: str) -> None:
    data, fitted = _data(), _trained(family)
    for split in PREDICTED:
        full = fitted.predict(data, split)
        frame = data.frame(split)
        assert full.shape == (len(frame),)
        # one engine predicted alone must reproduce its own rows of the full split
        unit = frame["unit_id"].value_counts().idxmax()
        alone = _restrict(data, {split: frame["unit_id"] == unit})
        rows = (frame["unit_id"] == unit).to_numpy()
        _assert_same(family, fitted.predict(alone, split), full[rows])
    n_inputs = {"xgboost_raw": 1 + len(data.sensors)}
    sequence = ("lstm", "tcn", "cnn", "transformer")
    n_inputs |= {f: len(data.sensors) for f in sequence}
    expected = n_inputs.get(family, data.rolling("inner_fit").shape[1])
    assert fitted.n_inputs == expected


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_predictions_are_clipped_to_the_cap(family: str) -> None:
    data, fitted = _data(), _trained(family)
    for split in PREDICTED:
        predicted = fitted.predict(data, split)
        assert np.isfinite(predicted).all()
        assert predicted.min() >= 0.0 and predicted.max() <= data.cap
        raw = np.asarray(fitted.model(fitted.inputs(data, split)), dtype=float)
        np.testing.assert_array_equal(predicted, np.clip(raw, 0.0, data.cap))


def test_clipping_is_exercised_by_a_linear_model() -> None:
    data, fitted = _data(), _trained("ridge")
    raw = np.concatenate(
        [np.asarray(fitted.model(fitted.inputs(data, s))) for s in PREDICTED]
    )
    # a linear trend overshoots the capped plateau or runs below zero at failure
    assert raw.min() < 0.0 or raw.max() > data.cap


def test_every_pool_configuration_is_accepted_and_every_network_runs() -> None:
    torch.manual_seed(0)
    for config in POOL:
        check_params(config.family, config.params)
        if config.family not in NEURAL_FAMILIES:
            continue
        if config.family == "mlp":
            x = torch.randn(3, 40)
        else:
            x = torch.randn(3, config.params["window"], 6)
        net = build_network(config.family, x.shape[-1], config.params).eval()
        assert net(x).shape == (3,)


def test_unknown_or_missing_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        check_params("ridge", {"alpha": 1.0, "alhpa": 1.0})
    with pytest.raises(ValueError, match="missing keys"):
        check_params("knn", {"n_neighbors": 5})
    with pytest.raises(ValueError, match="unknown family"):
        check_params("svm", {})
