r"""Attention-Augmented KAN Block (A-KAN) -- Contribution 2.

A standard Transformer encoder block interleaves multi-head self-attention with a
position-wise MLP feed-forward network (FFN). A-KAN keeps the attention sublayer
but **replaces the FFN with a KAN**, so the position-wise nonlinearity is a sum of
learnable, locally-controlled spline activations instead of two affine layers with
a fixed pointwise nonlinearity. The forward pass (pre-LayerNorm residual form)::

    u_t  = LN(h_t + MHSA(h)_t)
    h'_t = LN(u_t + KAN(u_t))

The spline edges give smooth, locally adaptive control that the paper argues is
well suited to disambiguating overlapping character boundaries.

All tensors keep the sequence shape ``(B, T, d)`` throughout, so blocks stack
transparently and gradients flow through both the attention and KAN paths.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .kan_block import KANBlock
from .spline import DEFAULT_NUM_GRIDS, DEFAULT_SPLINE_ORDER

__all__ = ["AKANBlock"]


class AKANBlock(nn.Module):
    r"""Transformer-style block whose FFN is a KAN.

    Parameters
    ----------
    dim:
        Model width ``d`` (input and output feature size).
    num_heads:
        Number of self-attention heads. Must divide ``dim``.
    kan_hidden:
        Hidden width of the KAN feed-forward sublayer (analogous to the MLP
        expansion). Defaults to ``dim`` (KANs need far less width than MLPs).
    num_grids, spline_order:
        Spline hyperparameters forwarded to the KAN FFN.
    attn_dropout, ffn_dropout, residual_dropout:
        Dropout probabilities for attention weights, inside the KAN FFN, and on
        the residual branches respectively.
    grid_update_freq:
        Forwarded to the internal :class:`KANBlock` for adaptive grid tracking.

    Attributes
    ----------
    attn:
        The multi-head self-attention module.
    kan_ffn:
        The KAN replacing the position-wise FFN (``dim -> kan_hidden -> dim``).
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        kan_hidden: Optional[int] = None,
        num_grids: int = DEFAULT_NUM_GRIDS,
        spline_order: int = DEFAULT_SPLINE_ORDER,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        grid_update_freq: int = 0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        kan_hidden = kan_hidden or dim

        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=attn_dropout, batch_first=True,
        )
        self.drop_attn = nn.Dropout(residual_dropout)

        self.norm_ffn = nn.LayerNorm(dim)
        # KAN feed-forward: dim -> kan_hidden -> dim, with adaptive grids.
        self.kan_ffn = KANBlock(
            dims=[dim, kan_hidden, dim],
            num_grids=num_grids,
            spline_order=spline_order,
            grid_update_freq=grid_update_freq,
            dropout=ffn_dropout,
        )
        self.drop_ffn = nn.Dropout(residual_dropout)

    # --- Validation ----------------------------------------------------------

    def _check_io(self, x: Tensor, name: str) -> None:
        """Validate that ``x`` is ``(B, T, dim)``."""
        if x.dim() != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"AKANBlock {name} expected (B, T, {self.dim}), "
                f"got {tuple(x.shape)}."
            )

    # --- Forward -------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Apply attention then the KAN FFN, both with pre-LN residuals.

        Parameters
        ----------
        x:
            Input sequence of shape ``(B, T, dim)``.
        key_padding_mask:
            Optional ``(B, T)`` boolean mask where ``True`` marks padded steps.
        attn_mask:
            Optional additive/boolean attention mask of shape ``(T, T)``.

        Returns
        -------
        Tensor
            Output sequence of shape ``(B, T, dim)``.
        """
        self._check_io(x, "input")

        # --- Self-attention sublayer (pre-LN residual). ---------------------
        normed = self.norm_attn(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.drop_attn(attn_out)

        # --- KAN feed-forward sublayer (pre-LN residual). -------------------
        normed = self.norm_ffn(x)
        ffn_out = self.kan_ffn(normed)
        x = x + self.drop_ffn(ffn_out)

        self._check_io(x, "output")
        return x

    def regularization_loss(self, l1_weight: float = 1.0,
                            entropy_weight: float = 1.0) -> Tensor:
        """Expose the KAN FFN's sparsity regularizer for the training loss."""
        return self.kan_ffn.regularization_loss(l1_weight, entropy_weight)
