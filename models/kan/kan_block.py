r"""Stacked multi-layer KAN with adaptive grid tracking.

This composes several :class:`~models.kan.kan_layer.KANLayer` modules into a deep
Kolmogorov-Arnold network::

    KAN(x) = (Phi_L o Phi_{L-1} o ... o Phi_1)(x)

Each ``Phi_l`` is a KAN layer realizing the inner univariate decomposition
``y_j = sum_i phi_{j,i}(x_i)``; stacking them yields the depth that the original
shallow ``2n + 1`` Kolmogorov-Arnold representation lacks.

Because spline expressivity is concentrated within each layer's knot domain, the
block also performs **adaptive grid tracking**: during training it maintains an
exponential moving average of the per-feature activation range entering every
layer ("historical activation domains"). On a configurable cadence it refits each
layer's knot boundaries to that tracked domain, so the splines keep their
resolution where the data actually lives even as upstream representations drift
during backpropagation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .kan_layer import KANLayer
from .spline import DEFAULT_NUM_GRIDS, DEFAULT_SPLINE_ORDER

__all__ = ["KANBlock"]


class KANBlock(nn.Module):
    r"""A stack of KAN layers with optional adaptive grid maintenance.

    Parameters
    ----------
    dims:
        Layer widths, e.g. ``[256, 128, 64]`` builds two KAN layers
        ``256->128`` and ``128->64``. Must contain at least two entries.
    num_grids, spline_order:
        Spline hyperparameters ``G`` and ``K`` shared by all layers.
    grid_range:
        Initial spline domain before extension.
    grid_update_freq:
        If ``> 0``, :meth:`maybe_update_grids` refits grids every this many
        recorded training steps. If ``0``, automatic updates are disabled.
    grid_momentum:
        EMA momentum in ``[0, 1)`` for tracking per-feature activation ranges.
    enable_standalone_scale_spline:
        Forwarded to each :class:`KANLayer`.
    dropout:
        Optional dropout applied between layers (``0`` disables it).

    Attributes
    ----------
    layers:
        The :class:`~torch.nn.ModuleList` of KAN layers.
    """

    def __init__(
        self,
        dims: Sequence[int],
        num_grids: int = DEFAULT_NUM_GRIDS,
        spline_order: int = DEFAULT_SPLINE_ORDER,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        grid_update_freq: int = 0,
        grid_momentum: float = 0.9,
        enable_standalone_scale_spline: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError(f"dims must list >= 2 widths, got {list(dims)}.")
        if not 0.0 <= grid_momentum < 1.0:
            raise ValueError("grid_momentum must be in [0, 1).")

        self.dims = list(dims)
        self.in_dim = dims[0]
        self.out_dim = dims[-1]
        self.grid_update_freq = grid_update_freq
        self.grid_momentum = grid_momentum

        self.layers = nn.ModuleList(
            KANLayer(
                in_dim=dims[i],
                out_dim=dims[i + 1],
                num_grids=num_grids,
                spline_order=spline_order,
                grid_range=grid_range,
                enable_standalone_scale_spline=enable_standalone_scale_spline,
            )
            for i in range(len(dims) - 1)
        )
        self.dropouts = nn.ModuleList(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            for _ in range(len(self.layers) - 1)
        )

        # Tracked historical activation domain (per layer, per input feature).
        # These are non-learnable EMAs used only to drive grid adaptation.
        for i, dim in enumerate(self.dims[:-1]):
            self.register_buffer(f"domain_min_{i}", torch.full((dim,), float("inf")))
            self.register_buffer(f"domain_max_{i}", torch.full((dim,), float("-inf")))
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    # --- Domain tracking helpers --------------------------------------------

    def _domain_buffers(self, i: int) -> Tuple[Tensor, Tensor]:
        """Return the ``(min, max)`` EMA buffers for layer ``i``."""
        return getattr(self, f"domain_min_{i}"), getattr(self, f"domain_max_{i}")

    @torch.no_grad()
    def _update_domain(self, i: int, x: Tensor) -> None:
        """Blend the current batch's per-feature range into the EMA buffers."""
        flat = x.reshape(-1, x.shape[-1])
        batch_min = flat.min(dim=0).values
        batch_max = flat.max(dim=0).values
        dmin, dmax = self._domain_buffers(i)

        # First observation seeds the buffer; thereafter use an EMA blend.
        if torch.isinf(dmin).any():
            dmin.copy_(batch_min)
            dmax.copy_(batch_max)
        else:
            m = self.grid_momentum
            dmin.mul_(m).add_(batch_min, alpha=1 - m)
            dmax.mul_(m).add_(batch_max, alpha=1 - m)

    # --- Forward -------------------------------------------------------------

    def _check_input(self, x: Tensor) -> None:
        """Validate the entry-point trailing feature dimension."""
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"KANBlock expected last dim {self.in_dim}, got {x.shape[-1]} "
                f"(full shape {tuple(x.shape)})."
            )

    def forward(self, x: Tensor) -> Tensor:
        """Run the stacked KAN, tracking activation domains during training.

        Parameters
        ----------
        x:
            Input of shape ``(..., dims[0])``.

        Returns
        -------
        Tensor
            Output of shape ``(..., dims[-1])``.
        """
        self._check_input(x)
        h = x
        for i, layer in enumerate(self.layers):
            if self.training:
                # Record the domain *entering* this layer (no autograd impact).
                self._update_domain(i, h.detach())
            h = layer(h)
            if i < len(self.dropouts):
                h = self.dropouts[i](h)

        if self.training:
            self._step += 1

        if h.shape[-1] != self.out_dim:
            raise ValueError(
                f"KANBlock output contract violated: expected last dim "
                f"{self.out_dim}, got {h.shape[-1]}."
            )
        return h

    # --- Grid adaptation -----------------------------------------------------

    @torch.no_grad()
    def update_grids(self, x: Tensor) -> None:
        """Refit every layer's knot grid by streaming ``x`` through the stack.

        Each layer's grid is adapted to the *actual* activations it receives, so
        the refit is done sequentially: update layer ``i`` from ``h`` then advance
        ``h = layer_i(h)``.

        Parameters
        ----------
        x:
            A representative batch of shape ``(..., dims[0])``.
        """
        self._check_input(x)
        h = x
        for layer in self.layers:
            layer.update_grid(h)
            h = layer(h)

    @torch.no_grad()
    def maybe_update_grids(self, x: Optional[Tensor] = None) -> bool:
        """Conditionally refit grids based on the configured cadence.

        Parameters
        ----------
        x:
            Batch to drive the refit. If ``None``, a synthetic batch is sampled
            uniformly from the tracked historical activation domain of the first
            layer (the downstream layers then receive their own true activations).

        Returns
        -------
        bool
            ``True`` if a grid update was performed this call.
        """
        if self.grid_update_freq <= 0:
            return False
        if int(self._step) == 0 or int(self._step) % self.grid_update_freq != 0:
            return False

        if x is None:
            dmin, dmax = self._domain_buffers(0)
            if torch.isinf(dmin).any():
                return False
            # Sample 512 points uniformly across the tracked domain.
            u = torch.rand(512, self.in_dim, device=dmin.device, dtype=dmin.dtype)
            x = dmin.unsqueeze(0) + u * (dmax - dmin).unsqueeze(0)

        self.update_grids(x)
        return True

    # --- Regularization ------------------------------------------------------

    def regularization_loss(self, l1_weight: float = 1.0,
                            entropy_weight: float = 1.0) -> Tensor:
        """Sum the per-layer KAN sparsity regularizers.

        Parameters
        ----------
        l1_weight, entropy_weight:
            Forwarded to each layer's :meth:`KANLayer.regularization_loss`.

        Returns
        -------
        Tensor
            Scalar total regularization loss.
        """
        terms: List[Tensor] = [
            layer.regularization_loss(l1_weight, entropy_weight)
            for layer in self.layers
        ]
        return torch.stack(terms).sum()
