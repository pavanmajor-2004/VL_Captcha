r"""Multi-Scale KAN Feature Pyramid (MS-KAN-FP) -- Contribution 1.

This module fuses feature maps drawn from three strides of the custom ResNet
trunk into a single per-time-step sequence. The paper defines the fusion as::

    h_t = sum_{s=1}^{3} alpha_s * Psi^{(s)}( F^{(s)}_t ),   alpha_s = softmax(beta_s)

where ``F^{(s)}`` is the stage-``s`` feature map (strides ``{4, 8, 16}``),
``Psi^{(s)}`` is a per-scale KAN projection to a common width ``C'``, and the
``alpha_s`` are learnable, softmax-normalized fusion weights.

Tensor flow (default ``B x 3 x 48 x 320`` input -> ``T = 40``, ``C' = 256``):

================ ===================== ==================== =================
stage / stride   trunk map (B,C,H,W)   height-collapsed     KAN-projected + T
================ ===================== ==================== =================
s=1 / 4          (B, 64, 12, 160)      (B, 160, 64)         (B, 40, 256)
s=2 / 8          (B, 128, 6, 80)       (B, 80, 128)         (B, 40, 256)
s=3 / 16         (B, 256, 3, 40)       (B, 40, 256)         (B, 40, 256)
================ ===================== ==================== =================

The three projected sequences are stacked and combined with the softmax fusion
weights to yield ``h in (B, 40, 256)``. An optional content-based attention mode
makes the fusion weights *dynamic* (per-sample, per-time-step) rather than global.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .kan_layer import KANLayer
from .spline import DEFAULT_NUM_GRIDS, DEFAULT_SPLINE_ORDER

__all__ = ["MultiScaleKANPyramid"]


class MultiScaleKANPyramid(nn.Module):
    r"""Fuse multi-stride trunk features into one KAN-projected sequence.

    Parameters
    ----------
    in_channels:
        Channel counts of the input feature maps, ordered fine-to-coarse, e.g.
        ``(64, 128, 256)`` for strides ``{4, 8, 16}``.
    out_dim:
        Common projected width ``C'`` (defaults to ``256``).
    seq_len:
        Target temporal length ``T`` all scales are resampled to (defaults ``40``).
    num_grids, spline_order:
        Spline hyperparameters forwarded to each scale's KAN projection.
    dynamic_fusion:
        If ``True`` the fusion weights are predicted per time-step from content
        (attention-weighted dynamic layer); if ``False`` they are global learnable
        scalars ``beta_s`` passed through a softmax.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (64, 128, 256),
        out_dim: int = 256,
        seq_len: int = 40,
        num_grids: int = DEFAULT_NUM_GRIDS,
        spline_order: int = DEFAULT_SPLINE_ORDER,
        dynamic_fusion: bool = True,
    ) -> None:
        super().__init__()
        if len(in_channels) < 1:
            raise ValueError("in_channels must list at least one scale.")

        self.in_channels = tuple(in_channels)
        self.num_scales = len(in_channels)
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.dynamic_fusion = dynamic_fusion

        # Per-scale KAN projection C_s -> C'. The "spatial KAN kernel" acts on the
        # per-(time, height) channel vector; height is reduced first (below).
        self.projections = nn.ModuleList(
            KANLayer(c, out_dim, num_grids=num_grids, spline_order=spline_order)
            for c in in_channels
        )
        # Per-scale height pooling to a singleton, turning (B,C,H,W) -> (B,W,C).
        self.height_pools = nn.ModuleList(
            nn.AdaptiveAvgPool2d((1, None)) for _ in in_channels
        )
        # LayerNorm stabilizes each projected stream before fusion.
        self.scale_norms = nn.ModuleList(
            nn.LayerNorm(out_dim) for _ in in_channels
        )

        if dynamic_fusion:
            # Content-based attention: predicts a weight per scale per time-step.
            self.attn = nn.Sequential(
                nn.Linear(out_dim * self.num_scales, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, self.num_scales),
            )
        else:
            # Global learnable logits beta_s -> alpha_s = softmax(beta).
            self.beta = nn.Parameter(torch.zeros(self.num_scales))

        self.out_norm = nn.LayerNorm(out_dim)

    # --- Validation ----------------------------------------------------------

    def _check_inputs(self, feats: Sequence[Tensor]) -> None:
        """Validate the number and channel dims of the incoming feature maps."""
        if len(feats) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} feature maps, got {len(feats)}."
            )
        for s, (f, c) in enumerate(zip(feats, self.in_channels)):
            if f.dim() != 4:
                raise ValueError(
                    f"Scale {s} map must be 4-D (B,C,H,W), got {tuple(f.shape)}."
                )
            if f.shape[1] != c:
                raise ValueError(
                    f"Scale {s} expected {c} channels, got {f.shape[1]}."
                )

    # --- Per-scale processing ------------------------------------------------

    def _project_scale(self, idx: int, feat: Tensor) -> Tensor:
        """Collapse height, KAN-project channels, and resample to ``seq_len``.

        Parameters
        ----------
        idx:
            Scale index.
        feat:
            Feature map ``(B, C_s, H_s, W_s)``.

        Returns
        -------
        Tensor
            Projected, length-``T`` sequence ``(B, T, C')``.
        """
        b = feat.shape[0]
        # (B, C, H, W) -> (B, C, 1, W) -> (B, W, C): per-column channel vectors.
        pooled = self.height_pools[idx](feat).squeeze(2).transpose(1, 2)

        # KAN projection over the channel axis (broadcasts over the W/time axis).
        projected = self.projections[idx](pooled)        # (B, W, C')
        projected = self.scale_norms[idx](projected)

        # Temporal resample W -> T via linear interpolation (length alignment).
        if projected.shape[1] != self.seq_len:
            projected = F.interpolate(
                projected.transpose(1, 2),               # (B, C', W)
                size=self.seq_len, mode="linear", align_corners=False,
            ).transpose(1, 2)                            # (B, T, C')
        if projected.shape != (b, self.seq_len, self.out_dim):
            raise ValueError(
                f"Scale {idx} projection contract violated: expected "
                f"{(b, self.seq_len, self.out_dim)}, got {tuple(projected.shape)}."
            )
        return projected

    # --- Forward -------------------------------------------------------------

    def forward(self, feats: Sequence[Tensor]) -> Tensor:
        r"""Fuse multi-scale trunk features into ``h in (B, T, C')``.

        Parameters
        ----------
        feats:
            Sequence of feature maps ordered fine-to-coarse, matching
            :attr:`in_channels`.

        Returns
        -------
        Tensor
            Fused sequence of shape ``(B, seq_len, out_dim)``.
        """
        self._check_inputs(feats)

        # Project every scale to (B, T, C').
        projected: List[Tensor] = [
            self._project_scale(s, f) for s, f in enumerate(feats)
        ]
        stack = torch.stack(projected, dim=2)            # (B, T, S, C')
        b, t, s, c = stack.shape

        if self.dynamic_fusion:
            # Content attention: concat scales -> per-(B,T) weight over S scales.
            concat = stack.reshape(b, t, s * c)          # (B, T, S*C')
            logits = self.attn(concat)                   # (B, T, S)
            alpha = F.softmax(logits, dim=-1).unsqueeze(-1)  # (B, T, S, 1)
        else:
            # Global softmax weights broadcast over (B, T).
            alpha = F.softmax(self.beta, dim=0).view(1, 1, s, 1)

        fused = (stack * alpha).sum(dim=2)               # (B, T, C')
        fused = self.out_norm(fused)

        if fused.shape != (b, self.seq_len, self.out_dim):
            raise ValueError(
                f"MS-KAN-FP output contract violated: expected "
                f"{(b, self.seq_len, self.out_dim)}, got {tuple(fused.shape)}."
            )
        return fused
