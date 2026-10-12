"""Networks of the research pool that the legacy sequence module does not define."""

from __future__ import annotations

from itertools import pairwise

import torch
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for units in hidden:
            layers += [nn.Linear(width, units), nn.ReLU(), nn.Dropout(dropout)]
            width = units
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SamePadTime(nn.Module):
    """Zero padding along time that keeps the length, with the extra cycle after (as Keras)."""

    def __init__(self, kernel: int) -> None:
        super().__init__()
        before = (kernel - 1) // 2
        self.pad = nn.ZeroPad2d((0, 0, before, kernel - 1 - before))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pad(x)


class CNNRegressor(nn.Module):
    """Li, Ding and Sun (2018): convolutions along time only, shared across sensors.

    The window is a one-channel image of cycles by sensors; kernels span k cycles and one
    sensor, so sensors meet only in the fully-connected layer. Windows end at the predicted
    cycle, so the zero padding after it stands for nothing in the future.
    """

    def __init__(
        self,
        n_features: int,
        window: int,
        filters: int,
        kernel_size: int,
        conv_layers: int,
        fc: int,
        dropout: float,
    ) -> None:
        super().__init__()
        widths = [1] + [filters] * conv_layers
        layers: list[nn.Module] = []
        for c_in, c_out in pairwise(widths):
            layers += [
                SamePadTime(kernel_size),
                nn.Conv2d(c_in, c_out, (kernel_size, 1)),
                nn.Tanh(),
            ]
        # the paper's last convolution: one filter of three cycles
        layers += [SamePadTime(3), nn.Conv2d(filters, 1, (3, 1)), nn.Tanh()]
        self.convs = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(window * n_features, fc),
            nn.Tanh(),
            nn.Linear(fc, 1),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.convs(x.unsqueeze(1))).squeeze(-1)


class TransformerRegressor(nn.Module):
    """Encoder over a window; the last cycle's representation gives the prediction."""

    def __init__(
        self,
        n_features: int,
        window: int,
        d_model: int,
        heads: int,
        layers: int,
        ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.project = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.empty(1, window, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, ff, dropout, batch_first=True
        )
        # no padding masks, so nested tensors would only add a warning
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.project(x) + self.position)
        return self.head(encoded[:, -1]).squeeze(-1)
