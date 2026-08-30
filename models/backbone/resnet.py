"""Lightweight ResNet-18 backbone specialized for sequential OCR.

A vanilla ResNet-18 downsamples height and width symmetrically by a factor of 32,
which destroys the horizontal (time-step) resolution that CTC-based recognizers
depend on. This module reconfigures the per-stage strides so that:

* the **width** (interpreted as the temporal axis) is reduced by a factor of 8,
  yielding ``T = W // 8`` time steps (``320 -> 40`` for the project contract), and
* the **height** is progressively collapsed and finally pooled to a singleton,
  producing a per-time-step feature vector.

The output therefore satisfies the sequence contract ``(B, T, C)`` with
``T = 40`` and ``C = 512`` for the default ``48 x 320`` input. Intermediate
feature maps from each stage can optionally be returned for downstream
multi-scale fusion (e.g. the KAN feature pyramid).
"""

from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["BasicBlock", "ResNet18Seq"]

_StrideT = Union[int, Tuple[int, int]]


def _as_pair(value: _StrideT) -> Tuple[int, int]:
    """Normalize an int or 2-tuple stride into a ``(height, width)`` tuple."""
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"Stride tuple must have length 2, got {value!r}.")
    return (int(value[0]), int(value[1]))


def conv3x3(in_planes: int, out_planes: int,
            stride: Tuple[int, int] = (1, 1)) -> nn.Conv2d:
    """3x3 convolution with ``same``-style padding and configurable stride."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False,
    )


def conv1x1(in_planes: int, out_planes: int,
            stride: Tuple[int, int] = (1, 1)) -> nn.Conv2d:
    """1x1 convolution used by the residual down-projection."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
    )


class BasicBlock(nn.Module):
    """Standard ResNet basic residual block with asymmetric stride support.

    Parameters
    ----------
    in_planes:
        Number of input channels.
    out_planes:
        Number of output channels.
    stride:
        ``(height_stride, width_stride)`` applied by the first convolution and
        the identity down-projection.
    """

    expansion: int = 1

    def __init__(self, in_planes: int, out_planes: int,
                 stride: Tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes, (1, 1))
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.downsample: nn.Module
        if stride != (1, 1) or in_planes != out_planes:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, out_planes, stride),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block to ``(B, C_in, H, W)``."""
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNet18Seq(nn.Module):
    """ResNet-18 variant that preserves width resolution for OCR.

    The stride schedule (defaults) maps a ``(B, 3, 48, 320)`` input to a
    ``(B, 40, 512)`` sequence:

    ======== ================ ============ ============ =========
    stage    width stride     height       width        channels
    ======== ================ ============ ============ =========
    stem     2                24           160          64
    layer1   1                12           160          64
    layer2   2                6            80           128
    layer3   2                3            40           256
    layer4   1                2 -> pool 1  40           512
    ======== ================ ============ ============ =========

    Parameters
    ----------
    in_channels:
        Number of input image channels (``3`` for RGB).
    layers:
        Block counts per stage (``(2, 2, 2, 2)`` reproduces ResNet-18).
    channels:
        Output channels per stage.
    expected_input_size:
        ``(height, width)`` used for entry-point shape validation.
    width_downsample:
        Total width reduction factor; defines ``T = W // width_downsample``.
    num_stages:
        Number of residual stages to build (``1..4``). Use ``3`` to stop at the
        stride-16 / 256-channel tap (e.g. for the MS-KAN pyramid), which avoids
        constructing the otherwise-unused ``layer4`` parameters. The default
        ``4`` reproduces the full ResNet-18 trunk used by the CRNN-style baseline.
    """

    def __init__(
        self,
        in_channels: int = 3,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
        expected_input_size: Tuple[int, int] = (48, 320),
        width_downsample: int = 8,
        num_stages: int = 4,
    ) -> None:
        super().__init__()
        if not 1 <= num_stages <= 4:
            raise ValueError(f"num_stages must be in 1..4, got {num_stages}.")
        self.in_channels = in_channels
        self.expected_input_size = expected_input_size
        self.width_downsample = width_downsample
        self.num_stages = num_stages
        self.out_channels = channels[num_stages - 1]

        # Stem: stride 2 in both axes -> (H/2, W/2).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Per-stage strides. Width strides 1,2,2,1 give an extra x4 (x2 from the
        # stem = x8 total); height strides 2,2,2,2 aggressively collapse height.
        stage_strides = [(2, 1), (2, 2), (2, 2), (2, 1)]
        self._in_planes = channels[0]
        self.stages = nn.ModuleList(
            self._make_layer(channels[i], layers[i], stride=stage_strides[i])
            for i in range(num_stages)
        )

        # Collapse whatever height remains to a singleton, keep width as T.
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))

        self._init_weights()

    def _make_layer(self, planes: int, blocks: int,
                    stride: Tuple[int, int]) -> nn.Sequential:
        """Build one residual stage with ``blocks`` :class:`BasicBlock` modules."""
        strides = [stride] + [(1, 1)] * (blocks - 1)
        modules: List[nn.Module] = []
        for blk_stride in strides:
            modules.append(BasicBlock(self._in_planes, planes, blk_stride))
            self._in_planes = planes
        return nn.Sequential(*modules)

    def _init_weights(self) -> None:
        """Kaiming-init convolutions; constant-init norm layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    # --- Shape validation ----------------------------------------------------

    def _check_input(self, x: Tensor) -> None:
        """Validate the entry-point tensor against the architecture contract."""
        if x.dim() != 4:
            raise ValueError(
                f"ResNet18Seq expects a 4-D (B, C, H, W) tensor, "
                f"got {x.dim()}-D shape {tuple(x.shape)}."
            )
        _, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {c}."
            )
        exp_h, exp_w = self.expected_input_size
        if (h, w) != (exp_h, exp_w):
            raise ValueError(
                f"Expected input spatial size {(exp_h, exp_w)}, got {(h, w)}."
            )

    def _check_output(self, seq: Tensor, batch: int, width: int) -> None:
        """Validate the exit-point sequence shape ``(B, T, C)``."""
        expected_t = width // self.width_downsample
        if seq.shape != (batch, expected_t, self.out_channels):
            raise ValueError(
                f"ResNet18Seq output contract violated: expected "
                f"{(batch, expected_t, self.out_channels)}, got {tuple(seq.shape)}."
            )

    # --- Forward -------------------------------------------------------------

    def forward(
        self, x: Tensor, return_intermediates: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """Extract a per-time-step feature sequence from an image batch.

        Parameters
        ----------
        x:
            Image batch of shape ``(B, in_channels, 48, 320)``.
        return_intermediates:
            If ``True``, also return the list of per-stage feature maps (one per
            built stage, each ``(B, C_s, H_s, W_s)``) for multi-scale downstream
            modules.

        Returns
        -------
        Tensor or tuple
            The sequence tensor ``(B, T, C)``; or, if ``return_intermediates``,
            the tuple ``(sequence, [stage maps])``.
        """
        self._check_input(x)
        batch, _, _, width = x.shape

        feat = self.stem(x)
        taps: List[Tensor] = []
        for stage in self.stages:
            feat = stage(feat)
            taps.append(feat)

        pooled = self.height_pool(feat)          # (B, C, 1, T)
        seq = pooled.squeeze(2).transpose(1, 2)  # (B, T, C)

        self._check_output(seq, batch, width)

        if return_intermediates:
            return seq, taps
        return seq
