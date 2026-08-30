r"""End-to-end VL-KAN CAPTCHA recognizer (top-level assembly).

This wires the custom components into the full pipeline described in the paper::

    Image -> ResNet18Seq trunk -> MS-KAN Feature Pyramid -> A-KAN blocks
          -> { CTC head, Length head, Boundary head }

Data flow (default ``B x 3 x 48 x 320`` input, ``T = 40``, ``d = 256``):

================== ======================= =========================
component          input                   output
================== ======================= =========================
ResNet18Seq        (B, 3, 48, 320)         taps @ strides {4,8,16}
MS-KAN-FP          3 trunk maps            h: (B, 40, 256)
A-KAN x N          (B, 40, 256)            (B, 40, 256)
CTC head           (B, 40, 256)            (B, 40, V+1)
Length head        (B, 40, 256) mean-pool  (B, 8)
Boundary head      (B, 40, 256)            (B, 40)
================== ======================= =========================

The length head is fed the **pre-attention** fused feature (a global-layout
summary), while the CTC and boundary heads consume the **post-attention**
sequence, matching the approved contract. All three heads are returned as raw
logits so the multi-task loss can apply CTC, cross-entropy and BCE respectively.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .backbone.resnet import ResNet18Seq
from .kan.a_kan_block import AKANBlock
from .kan.kan_layer import KANLayer
from .kan.ms_kan_fp import MultiScaleKANPyramid
from .kan.spline import DEFAULT_NUM_GRIDS, DEFAULT_SPLINE_ORDER

__all__ = ["LengthHead", "BoundaryHead", "CTCHead", "VLKAN"]


class CTCHead(nn.Module):
    """Per-time-step projection to vocabulary logits (blank included).

    Parameters
    ----------
    dim:
        Input feature width.
    num_classes:
        Alphabet size ``V``; the head emits ``V + 1`` logits (blank at index 0).
    """

    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ctc_classes = num_classes + 1
        self.proj = nn.Linear(dim, self.ctc_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, T, dim)`` to ``(B, T, V + 1)`` logits."""
        return self.proj(x)


class LengthHead(nn.Module):
    """KAN-based length classifier over the admissible length set.

    Parameters
    ----------
    dim:
        Input (pooled) feature width.
    num_lengths:
        Number of admissible lengths (``8`` for ``3..10``).
    num_grids, spline_order:
        Spline hyperparameters for the internal KAN.
    """

    def __init__(self, dim: int, num_lengths: int,
                 num_grids: int = DEFAULT_NUM_GRIDS,
                 spline_order: int = DEFAULT_SPLINE_ORDER) -> None:
        super().__init__()
        self.num_lengths = num_lengths
        self.kan = KANLayer(dim, dim // 2, num_grids=num_grids,
                            spline_order=spline_order)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim // 2, num_lengths)

    def forward(self, pooled: Tensor) -> Tensor:
        """Map a pooled feature ``(B, dim)`` to length logits ``(B, num_lengths)``."""
        return self.proj(self.act(self.kan(pooled)))


class BoundaryHead(nn.Module):
    """1-D convolutional head predicting per-time-step boundary logits.

    Parameters
    ----------
    dim:
        Input feature width.
    kernel_size:
        Temporal receptive field of the convolution (odd, ``same`` padded).
    """

    def __init__(self, dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.conv = nn.Conv1d(dim, dim // 2, kernel_size,
                              padding=kernel_size // 2)
        self.act = nn.GELU()
        self.proj = nn.Conv1d(dim // 2, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, T, dim)`` to per-step boundary logits ``(B, T)``."""
        feat = x.transpose(1, 2)                 # (B, dim, T)
        feat = self.act(self.conv(feat))
        logits = self.proj(feat).squeeze(1)      # (B, T)
        return logits


class VLKAN(nn.Module):
    r"""Variable-length CAPTCHA recognizer with KAN components.

    Parameters
    ----------
    num_classes:
        Alphabet size ``V`` (excludes CTC blank).
    min_length, max_length:
        Inclusive admissible length bounds (length head emits
        ``max_length - min_length + 1`` logits).
    in_channels:
        Number of input image channels.
    input_size:
        Expected ``(height, width)`` of inputs.
    dim:
        Common model width ``d`` / ``C'``.
    num_akan_blocks:
        Number of stacked :class:`AKANBlock` layers.
    num_heads:
        Attention heads inside each A-KAN block.
    pyramid_strides_channels:
        Channel counts of the trunk taps used by the pyramid (strides
        ``{4, 8, 16}`` -> ``(64, 128, 256)``).
    num_grids, spline_order:
        Spline hyperparameters shared across KAN components.
    grid_update_freq:
        Adaptive grid-tracking cadence forwarded to the A-KAN KAN FFNs.
    dynamic_fusion:
        Whether the pyramid uses content-based (dynamic) fusion weights.
    """

    def __init__(
        self,
        num_classes: int,
        min_length: int = 3,
        max_length: int = 10,
        in_channels: int = 3,
        input_size: Tuple[int, int] = (48, 320),
        dim: int = 256,
        num_akan_blocks: int = 4,
        num_heads: int = 4,
        pyramid_strides_channels: Sequence[int] = (64, 128, 256),
        num_grids: int = DEFAULT_NUM_GRIDS,
        spline_order: int = DEFAULT_SPLINE_ORDER,
        grid_update_freq: int = 0,
        dynamic_fusion: bool = True,
    ) -> None:
        super().__init__()
        if max_length < min_length:
            raise ValueError("max_length must be >= min_length.")

        self.num_classes = num_classes
        self.blank_index = 0
        self.ctc_classes = num_classes + 1
        self.min_length = min_length
        self.max_length = max_length
        self.num_lengths = max_length - min_length + 1
        self.in_channels = in_channels
        self.input_size = input_size
        self.dim = dim

        self.backbone = ResNet18Seq(
            in_channels=in_channels, expected_input_size=input_size,
            num_stages=len(pyramid_strides_channels),
        )
        self.seq_len = input_size[1] // self.backbone.width_downsample  # T = 40
        self.num_pyramid_scales = len(pyramid_strides_channels)

        self.pyramid = MultiScaleKANPyramid(
            in_channels=pyramid_strides_channels,
            out_dim=dim,
            seq_len=self.seq_len,
            num_grids=num_grids,
            spline_order=spline_order,
            dynamic_fusion=dynamic_fusion,
        )

        self.akan_blocks = nn.ModuleList(
            AKANBlock(
                dim=dim, num_heads=num_heads,
                num_grids=num_grids, spline_order=spline_order,
                grid_update_freq=grid_update_freq,
            )
            for _ in range(num_akan_blocks)
        )

        self.ctc_head = CTCHead(dim, num_classes)
        self.length_head = LengthHead(dim, self.num_lengths,
                                      num_grids=num_grids, spline_order=spline_order)
        self.boundary_head = BoundaryHead(dim)

    # --- Validation ----------------------------------------------------------

    def _check_input(self, x: Tensor) -> None:
        """Validate the entry-point image tensor against the contract."""
        if x.dim() != 4:
            raise ValueError(f"VLKAN expects (B, C, H, W), got {tuple(x.shape)}.")
        _, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}.")
        if (h, w) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {(h, w)}.")

    def _check_outputs(self, out: Dict[str, Tensor], batch: int) -> None:
        """Validate all head outputs against the contract."""
        expected = {
            "ctc_logits": (batch, self.seq_len, self.ctc_classes),
            "length_logits": (batch, self.num_lengths),
            "boundary_logits": (batch, self.seq_len),
        }
        for key, shape in expected.items():
            if out[key].shape != shape:
                raise ValueError(
                    f"VLKAN '{key}' contract violated: expected {shape}, "
                    f"got {tuple(out[key].shape)}."
                )

    # --- Forward -------------------------------------------------------------

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        r"""Run the full pipeline and return all three head outputs.

        Parameters
        ----------
        x:
            Image batch of shape ``(B, in_channels, 48, 320)``.

        Returns
        -------
        dict
            ``{"ctc_logits": (B, T, V+1), "length_logits": (B, num_lengths),
            "boundary_logits": (B, T)}`` -- all raw logits.
        """
        self._check_input(x)
        batch = x.shape[0]

        # Trunk: take the first `num_pyramid_scales` taps (strides 4, 8, 16).
        _, taps = self.backbone(x, return_intermediates=True)
        pyramid_feats: List[Tensor] = taps[: self.num_pyramid_scales]

        # Multi-scale fusion -> h (pre-attention fused feature).
        fused = self.pyramid(pyramid_feats)              # (B, T, d)

        # Length head reads the pre-attention global layout summary.
        length_logits = self.length_head(fused.mean(dim=1))  # (B, num_lengths)

        # Sequence refinement through stacked A-KAN blocks.
        h = fused
        for block in self.akan_blocks:
            h = block(h)                                 # (B, T, d)

        ctc_logits = self.ctc_head(h)                    # (B, T, V+1)
        boundary_logits = self.boundary_head(h)          # (B, T)

        out = {
            "ctc_logits": ctc_logits,
            "length_logits": length_logits,
            "boundary_logits": boundary_logits,
        }
        self._check_outputs(out, batch)
        return out

    # --- Helpers -------------------------------------------------------------

    def regularization_loss(self, l1_weight: float = 1.0,
                            entropy_weight: float = 1.0) -> Tensor:
        """Aggregate KAN sparsity regularizers across pyramid and A-KAN FFNs."""
        terms: List[Tensor] = [
            block.regularization_loss(l1_weight, entropy_weight)
            for block in self.akan_blocks
        ]
        return torch.stack(terms).sum()

    @torch.no_grad()
    def update_grids(self) -> None:
        """Trigger adaptive grid refits in every A-KAN KAN FFN (cadence-gated)."""
        for block in self.akan_blocks:
            block.kan_ffn.maybe_update_grids()
