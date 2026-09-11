from pathlib import Path

import numpy as np
import pandas as pd
import torch

from maintadapt.config import load_config
from maintadapt.sequence import build_model, make_windows, predict, train_model

ROOT = Path(__file__).resolve().parents[1]


def test_windows_end_at_each_row_and_pad_with_the_first_reading() -> None:
    frame = pd.DataFrame(
        {"unit_id": [1] * 3 + [2] * 5, "cycle": [1, 2, 3, 1, 2, 3, 4, 5]}
    )
    frame["reading"] = frame["cycle"] * 10 + frame["unit_id"]

    windows = make_windows(frame, ["reading"], length=4)
    assert windows.shape == (8, 4, 1)
    assert windows.dtype == np.float32
    # unit 1 at cycle 2 has one real predecessor, so the window repeats its first reading
    assert windows[1, :, 0].tolist() == [11, 11, 11, 21]

    last = make_windows(frame, ["reading"], length=4, last_only=True)
    assert last.shape == (2, 4, 1)
    assert last[1, :, 0].tolist() == [22, 32, 42, 52]


def test_two_training_steps_on_synthetic_data_write_a_usable_checkpoint(
    tmp_path: Path,
) -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml").sequence
    rng = np.random.default_rng(0)
    x = rng.normal(size=(48, cfg.window, 5)).astype(np.float32)
    y = rng.uniform(0, 125, size=48)
    device = torch.device("cpu")
    for kind in cfg.models:
        torch.manual_seed(0)
        path = tmp_path / f"{kind}.pt"
        result = train_model(
            build_model(kind, 5, cfg),
            (x, y),
            (x[:8], y[:8]),
            cfg.training,
            125,
            device,
            path,
            max_steps=2,
        )
        assert result.steps == 2
        restored = build_model(kind, 5, cfg)
        restored.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
        assert predict(restored, x[:4], 125, device, batch_size=4).shape == (4,)
