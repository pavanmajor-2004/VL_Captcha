"""Classical fixed-length DeepCAPTCHA baseline (Noury & Rezaei, 2020).

This is a faithful re-implementation of the Deep-CAPTCHA solver used as the
primary fixed-length comparison point. The network treats recognition as ``L``
independent ``V``-way classification problems via ``L`` parallel softmax heads
hanging off a shared CNN trunk, and therefore requires the CAPTCHA length ``L``
to be known a priori.

Architecture (per the original paper):

* three ``Conv(5x5, 'same') -> ReLU -> MaxPool(2x2)`` blocks with channel
  widths ``32 -> 48 -> 64``;
* a ``512``-unit fully-connected layer with ReLU and 30% dropout;
* ``L`` parallel linear "softmax" heads, each producing ``V`` class logits.

The module emits raw logits of shape ``(B, L, V)``; apply cross-entropy per
position (or softmax for inference) downstream.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["DeepCaptcha"]


class DeepCaptcha(nn.Module):
    """Fixed-length CNN CAPTCHA classifier with ``L`` parallel heads.

    Parameters
    ----------
    num_classes:
        Alphabet size ``V`` (e.g. ``10`` numeric, ``62`` alphanumeric). This is
        the per-position class count and does **not** include a CTC blank.
    length:
        Fixed number of characters ``L`` the model predicts.
    in_channels:
        Number of input image channels (``1`` for the paper's grayscale path).
    input_size:
        ``(height, width)`` of the input images, used for shape validation and
        to size the flattened feature vector.
    hidden_dim:
        Width of the dense bottleneck layer.
    dropout:
        Dropout probability applied after the dense layer.
    """

    def __init__(
        self,
        num_classes: int,
        length: int,
        in_channels: int = 1,
        input_size: Tuple[int, int] = (25, 67),
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}.")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")

        self.num_classes = num_classes
        self.length = length
        self.in_channels = in_channels
        self.input_size = input_size

        # Three Conv(5x5, same) -> ReLU -> MaxPool(2x2) blocks.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Each MaxPool halves the spatial dims (floor division).
        h, w = input_size
        feat_h = h // 8
        feat_w = w // 8
        if feat_h < 1 or feat_w < 1:
            raise ValueError(
                f"input_size {input_size} is too small; spatial dims collapse "
                "below 1 after three 2x2 pools."
            )
        self._flattened_dim = 64 * feat_h * feat_w

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flattened_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # L parallel heads packed as a single (hidden -> L * V) projection.
        self.heads = nn.Linear(hidden_dim, length * num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-init convolutions and linear layers."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _check_input(self, x: Tensor) -> None:
        """Validate the entry-point tensor against the contract."""
        if x.dim() != 4:
            raise ValueError(
                f"DeepCaptcha expects (B, C, H, W), got shape {tuple(x.shape)}."
            )
        _, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {c}."
            )
        if (h, w) != self.input_size:
            raise ValueError(
                f"Expected input size {self.input_size}, got {(h, w)}."
            )

    def forward(self, x: Tensor) -> Tensor:
        """Classify a fixed-length CAPTCHA batch.

        Parameters
        ----------
        x:
            Image batch of shape ``(B, in_channels, H, W)``.

        Returns
        -------
        Tensor
            Per-position logits of shape ``(B, L, V)``.
        """
        self._check_input(x)
        batch = x.shape[0]

        feats = self.features(x)
        embedded = self.classifier(feats)
        logits = self.heads(embedded).view(batch, self.length, self.num_classes)

        if logits.shape != (batch, self.length, self.num_classes):
            raise ValueError(
                f"DeepCaptcha output contract violated: expected "
                f"{(batch, self.length, self.num_classes)}, got {tuple(logits.shape)}."
            )
        return logits
