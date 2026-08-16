"""CRNN architecture: CNN feature extractor -> BiLSTM sequence model -> CTC head.

Input images are (1, 32, W) grayscale. The CNN downsamples height to 1 and
width by 4, so a 32x160 crop yields 40 time steps.
"""
from __future__ import annotations

import torch
from torch import nn

from .charset import NUM_CLASSES

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 160


def conv_block(in_ch: int, out_ch: int, pool: tuple[int, int] | None) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(pool, pool))
    return nn.Sequential(*layers)


class CRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 192) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            conv_block(1, 32, (2, 2)),    # 16 x W/2
            conv_block(32, 64, (2, 2)),   # 8  x W/4
            conv_block(64, 128, (2, 1)),  # 4  x W/4
            conv_block(128, 128, (2, 1)), # 2  x W/4
            conv_block(128, 256, (2, 1)), # 1  x W/4
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, batch_first=False, dropout=0.1)
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 1, 32, W) -> log-probabilities (T, N, C) ready for CTC loss."""
        features = self.cnn(x)              # (N, C, 1, T)
        features = features.squeeze(2)      # (N, C, T)
        sequence = features.permute(2, 0, 1)  # (T, N, C)
        output, _ = self.rnn(sequence)
        return self.head(output).log_softmax(dim=2)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(x)
