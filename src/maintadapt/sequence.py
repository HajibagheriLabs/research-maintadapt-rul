"""Deep sequence models over sliding windows of sensor readings."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from maintadapt.config import SequenceConfig, TrainingConfig
from maintadapt.evaluate import rmse


@dataclass(frozen=True, eq=False)
class TrainResult:
    model: nn.Module
    best_epoch: int
    best_val_rmse: float
    steps: int


class LSTMRegressor(nn.Module):
    def __init__(
        self, n_features: int, hidden: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        between = dropout if layers > 1 else 0.0
        self.lstm = nn.LSTM(
            n_features, hidden, layers, batch_first=True, dropout=between
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class CausalBlock(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # left padding only, so the output at cycle t never sees cycles after t
        y = self.dropout(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))
        y = self.dropout(F.relu(self.conv2(F.pad(y, (self.pad, 0)))))
        return F.relu(y + self.skip(x))


class TCNRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: int,
        kernel: int,
        dilations: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        widths = [n_features] + [channels] * len(dilations)
        self.blocks = nn.Sequential(
            *(
                CausalBlock(widths[i], widths[i + 1], kernel, d, dropout)
                for i, d in enumerate(dilations)
            )
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(x.transpose(1, 2))
        return self.head(features[:, :, -1]).squeeze(-1)


def build_model(kind: str, n_features: int, cfg: SequenceConfig) -> nn.Module:
    if kind == "tcn":
        tcn = cfg.tcn
        return TCNRegressor(
            n_features, tcn.channels, tcn.kernel_size, tcn.dilations, cfg.dropout
        )
    if kind == "lstm":
        return LSTMRegressor(n_features, cfg.lstm.hidden, cfg.lstm.layers, cfg.dropout)
    raise ValueError(f"unknown sequence model {kind!r}; expected 'tcn' or 'lstm'")


def make_windows(
    frame: pd.DataFrame, columns: list[str], length: int, *, last_only: bool = False
) -> np.ndarray:
    """Windows of `length` cycles ending at each row, left-padded with the unit's first reading."""
    windows = []
    for _, unit in frame.groupby("unit_id", sort=True):
        values = unit.sort_values("cycle")[columns].to_numpy(dtype=np.float32)
        padded = np.concatenate([np.repeat(values[:1], length - 1, axis=0), values])
        view = np.lib.stride_tricks.sliding_window_view(padded, length, axis=0)
        view = view.transpose(0, 2, 1)
        windows.append(view[-1:] if last_only else view)
    return np.ascontiguousarray(np.concatenate(windows))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def predict(
    model: nn.Module, x: np.ndarray, cap: int, device: torch.device, batch_size: int
) -> np.ndarray:
    model.eval()
    return np.clip(_forward(model, x, device, batch_size) * cap, 0, cap)


@torch.no_grad()
def predict_mc_dropout(
    model: nn.Module,
    x: np.ndarray,
    cap: int,
    device: torch.device,
    batch_size: int,
    samples: int,
) -> np.ndarray:
    """Stochastic forward passes with dropout left on; returns (samples, n) RUL draws."""
    # the models contain no batch-norm, so train mode changes nothing but dropout
    model.train()
    draws = [_forward(model, x, device, batch_size) * cap for _ in range(samples)]
    model.eval()
    return np.clip(np.stack(draws), 0, cap)


def train_model(
    model: nn.Module,
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    cfg: TrainingConfig,
    cap: int,
    device: torch.device,
    checkpoint: Path,
    *,
    lr: float | None = None,
    max_steps: int | None = None,
) -> TrainResult:
    """Adam on the MSE of RUL/cap, keeping the epoch with the lowest validation RMSE."""
    model.to(device)
    x_train = torch.as_tensor(train[0], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train[1] / cap, dtype=torch.float32, device=device)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr if lr is None else lr,
        weight_decay=cfg.weight_decay,
    )
    best_rmse, best_state, best_epoch, stale, steps = math.inf, None, 0, 0, 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), device=device)
        for batch in order.split(cfg.batch_size):
            loss = F.mse_loss(model(x_train[batch]), y_train[batch])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        val_rmse = rmse(val[1], predict(model, val[0], cap, device, cfg.batch_size))
        if val_rmse < best_rmse:
            best_rmse, best_epoch, stale = val_rmse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= cfg.patience or (max_steps is not None and steps >= max_steps):
            break
    model.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_state, "epoch": best_epoch, "val_rmse": best_rmse},
        checkpoint,
    )
    return TrainResult(model, best_epoch, best_rmse, steps)


def _forward(
    model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    outputs = [
        model(
            torch.as_tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        )
        for i in range(0, len(x), batch_size)
    ]
    return torch.cat(outputs).float().cpu().numpy()
