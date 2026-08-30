r"""Character Boundary Prediction Head (CBPH) -- Contribution 5.

The CBPH is a small 1-D convolutional "tracker" that, at every time step ``t`` of
the recognizer's feature sequence, predicts the probability that ``t`` is a
character boundary (a transition between two glyphs). The auxiliary boundary
signal disambiguates overlapping characters and biases the CTC beam search toward
alignments that switch labels at predicted boundaries.

Supervision uses **soft** targets rather than hard 0/1 indicators: each true glyph
centre is rendered as a Gaussian bump and the per-step target is the maximum over
all bumps. This tolerates the inherent localization uncertainty of where, exactly,
a boundary lands in the down-sampled timeline.

    b_t            = sigma( w_b^T h_t + c_b )                 (predicted logit -> prob)
    tilde_b_t      = max_g exp( -(t - mu_g)^2 / (2 s^2) )      (soft Gaussian target)
    L_boundary     = BCE( b, tilde_b )

All tensors keep the sequence shape ``(B, T)`` for the boundary axis so the head
composes with the rest of the ``(B, T, d)`` pipeline.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["BoundaryHead", "build_soft_boundary_targets"]


def build_soft_boundary_targets(
    boundary_centres: Sequence[Sequence[float]],
    seq_len: int,
    sigma: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    r"""Render per-sample glyph centres into soft Gaussian boundary targets.

    Each normalized centre ``c in [0, 1]`` maps to a fractional time index
    ``mu = c * (seq_len - 1)``; the target at step ``t`` is the maximum Gaussian
    response over all of a sample's centres::

        tilde_b_t = max_g exp( -(t - mu_g)^2 / (2 * sigma^2) )

    Parameters
    ----------
    boundary_centres:
        Ragged batch of normalized glyph-centre positions in ``[0, 1]``; one list
        per sample (lengths may differ across samples).
    seq_len:
        Length ``T`` of the feature timeline.
    sigma:
        Standard deviation of the Gaussian bump, in time-step units.
    device, dtype:
        Placement and precision of the returned tensor.

    Returns
    -------
    Tensor
        Soft target tensor of shape ``(B, T)`` with values in ``[0, 1]``.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")
    batch = len(boundary_centres)
    steps = torch.arange(seq_len, device=device, dtype=dtype)  # (T,)
    targets = torch.zeros(batch, seq_len, device=device, dtype=dtype)

    two_sigma_sq = 2.0 * sigma * sigma
    for i, centres in enumerate(boundary_centres):
        if len(centres) == 0:
            continue
        mu = torch.as_tensor(centres, device=device, dtype=dtype) * (seq_len - 1)
        # (G, T) Gaussian responses, reduced by max over the G centres.
        resp = torch.exp(-((steps.unsqueeze(0) - mu.unsqueeze(1)) ** 2) / two_sigma_sq)
        targets[i] = resp.max(dim=0).values
    return targets


class BoundaryHead(nn.Module):
    """1-D convolutional boundary tracker over sequence features.

    Parameters
    ----------
    dim:
        Input feature width ``d``.
    hidden_dim:
        Channels of the intermediate convolution (defaults to ``dim // 2``).
    kernel_size:
        Temporal receptive field (forced odd, ``same`` padded).
    num_layers:
        Number of stacked conv-GELU blocks before the 1-channel projection.
    dropout:
        Dropout applied between conv blocks.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        kernel_size: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.dim = dim
        hidden_dim = hidden_dim or max(1, dim // 2)
        pad = kernel_size // 2

        blocks: List[nn.Module] = []
        in_ch = dim
        for _ in range(num_layers):
            blocks.append(nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=pad))
            blocks.append(nn.GELU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            in_ch = hidden_dim
        self.encoder = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def _check_input(self, x: Tensor) -> None:
        """Validate the entry-point sequence tensor ``(B, T, dim)``."""
        if x.dim() != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"BoundaryHead expected (B, T, {self.dim}), got {tuple(x.shape)}."
            )

    def forward(self, x: Tensor) -> Tensor:
        """Compute per-time-step boundary logits.

        Parameters
        ----------
        x:
            Sequence features of shape ``(B, T, dim)``.

        Returns
        -------
        Tensor
            Raw boundary logits of shape ``(B, T)``. Apply :func:`torch.sigmoid`
            for probabilities, or feed directly to BCE-with-logits.
        """
        self._check_input(x)
        batch, seq_len, _ = x.shape
        feat = x.transpose(1, 2)                  # (B, dim, T)
        feat = self.encoder(feat)
        logits = self.proj(feat).squeeze(1)       # (B, T)

        if logits.shape != (batch, seq_len):
            raise ValueError(
                f"BoundaryHead output contract violated: expected "
                f"{(batch, seq_len)}, got {tuple(logits.shape)}."
            )
        return logits

    @staticmethod
    def loss(
        logits: Tensor,
        soft_targets: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Soft-target binary cross-entropy boundary loss.

        Parameters
        ----------
        logits:
            Predicted boundary logits ``(B, T)``.
        soft_targets:
            Gaussian-smoothed targets ``(B, T)`` in ``[0, 1]`` (e.g. from
            :func:`build_soft_boundary_targets`).
        mask:
            Optional ``(B, T)`` float/bool mask of valid (non-padded) steps; if
            given, the loss is averaged only over valid positions.

        Returns
        -------
        Tensor
            Scalar boundary loss. Computed in fp32 for mixed-precision stability.
        """
        if logits.shape != soft_targets.shape:
            raise ValueError(
                f"logits {tuple(logits.shape)} and targets "
                f"{tuple(soft_targets.shape)} must match."
            )
        # BCE in fp32 regardless of autocast: sigmoid/log are precision-sensitive.
        per_step = F.binary_cross_entropy_with_logits(
            logits.float(), soft_targets.float(), reduction="none"
        )
        if mask is not None:
            m = mask.to(per_step.dtype)
            denom = m.sum().clamp_min(1.0)
            return (per_step * m).sum() / denom
        return per_step.mean()
