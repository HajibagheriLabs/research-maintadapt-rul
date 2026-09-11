import numpy as np
import pandas as pd

from maintadapt.features import rolling_features


def test_rolling_features_never_look_ahead_and_recover_a_linear_trend() -> None:
    rng = np.random.default_rng(0)
    units = [pd.DataFrame({"unit_id": u, "cycle": np.arange(1, 61)}) for u in (3, 8)]
    frame = pd.concat(units, ignore_index=True)
    frame["noisy"] = rng.normal(size=len(frame))
    frame["ramp"] = 2.0 * frame["cycle"] + 5.0

    full = rolling_features(frame, ["noisy", "ramp"], windows=[5, 20])
    past = frame["cycle"] <= 30
    # recomputing on a history cut at cycle 30 must reproduce every earlier value exactly
    truncated = rolling_features(
        frame[past].reset_index(drop=True), ["noisy", "ramp"], [5, 20]
    )
    pd.testing.assert_frame_equal(full[past].reset_index(drop=True), truncated)

    assert np.allclose(full.loc[frame["cycle"] >= 2, "ramp_slope_20"], 2.0)
    assert (full.loc[frame["cycle"] == 1, "ramp_slope_20"] == 0.0).all()
    # the 5-cycle mean at cycle 10 averages cycles 6..10 of the ramp: 2 * 8 + 5
    at_ten = full.loc[(frame["cycle"] == 10) & (frame["unit_id"] == 8), "ramp_mean_5"]
    assert np.allclose(at_ten, 21.0)
