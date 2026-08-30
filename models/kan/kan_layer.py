r"""A single Kolmogorov-Arnold Network layer (from scratch, GPU-optimized).

A KAN layer of width ``(in_dim, out_dim)`` replaces the affine-then-activation
pattern of an MLP with a sum of learnable univariate edge functions::

    y_j = sum_{i=1}^{in_dim} phi_{j,i}(x_i)

where each edge activation is parameterized as a residual combination of a
scalable base path (SiLU) and a trainable B-spline::

    phi(x) = w_b * SiLU(x) + w_s * sum_k c_k * B_k(x)

This is the internal-degree-of-freedom side of the Kolmogorov-Arnold
decomposition ``f(x) = sum_q Phi_q( sum_p phi_{q,p}(x_p) )``: the inner sum over
``p`` is the per-output reduction over input edges, and stacking layers composes
the outer ``Phi_q``.

The forward pass is expressed entirely with batched matrix multiplies / einsums
so it runs as a few fused GPU kernels with no Python-side per-sample work.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .spline import (
    DEFAULT_NUM_GRIDS,
    DEFAULT_SPLINE_ORDER,
    b_spline_basis,
    curve_to_coeff,
    generate_grid,
    num_spline_bases,
)

__all__ = ["KANLayer"]


class KANLayer(nn.Module):
    r"""Standalone KAN layer with residual SiLU + B-spline edge activations.

    Parameters
    ----------
    in_dim:
        Number of input features.
    out_dim:
        Number of output features.
    num_grids:
        Number of uniform knot intervals ``G``.
    spline_order:
        B-spline order ``K``.
    grid_range:
        Base spline domain ``(a, b)`` before the order-``K`` extension.
    scale_base:
        Standard deviation used to initialize the base (SiLU) weights ``w_b``.
    scale_spline:
        Initial value / scale of the spline gate ``w_s``.
    scale_noise:
        Magnitude of the random curve used to initialize the spline coefficients.
    enable_standalone_scale_spline:
        If ``True`` the spline gate ``w_s`` is a free per-edge parameter; if
        ``False`` it is folded into the coefficients (set to 1).
    grid_eps:
        Blend factor in ``[0, 1]`` between a purely uniform grid (``1``) and a
        purely data-adaptive (quantile) grid (``0``) during grid updates.

    Attributes
    ----------
    base_weight:
        ``w_b`` of shape ``(out_dim, in_dim)``.
    spline_coef:
        Spline coefficients ``c`` of shape ``(out_dim, in_dim, G + K)``.
    spline_scaler:
        ``w_s`` of shape ``(out_dim, in_dim)`` (present iff standalone scaling).
    grid:
        Knot buffer of shape ``(in_dim, G + 2K + 1)``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_grids: int = DEFAULT_NUM_GRIDS,
        spline_order: int = DEFAULT_SPLINE_ORDER,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        scale_noise: float = 0.1,
        enable_standalone_scale_spline: bool = True,
        grid_eps: float = 0.02,
    ) -> None:
        super().__init__()
        if in_dim < 1 or out_dim < 1:
            raise ValueError("in_dim and out_dim must be >= 1.")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_grids = num_grids
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.grid_eps = grid_eps
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.n_bases = num_spline_bases(num_grids, spline_order)

        # Knot grid is a (non-learnable) buffer so it moves with .to()/.cuda()
        # and is saved in the state dict, but receives no gradient.
        grid = generate_grid(in_dim, num_grids, spline_order, grid_range)
        self.register_buffer("grid", grid)

        # Learnable parameters.
        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.spline_coef = nn.Parameter(torch.empty(out_dim, in_dim, self.n_bases))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_dim, in_dim))
        else:
            self.register_parameter("spline_scaler", None)

        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.scale_noise = scale_noise
        self.reset_parameters()

    # --- Initialization ------------------------------------------------------

    def reset_parameters(self) -> None:
        """Initialize base weights (Kaiming-style) and spline coefficients.

        Spline coefficients are fit, via least squares, to a small random curve
        sampled at the interior knot locations. This starts each edge as a tiny,
        smooth perturbation rather than an arbitrary high-frequency function.
        """
        # Base path: scaled Kaiming-uniform on w_b.
        bound = self.scale_base / (self.in_dim ** 0.5)
        nn.init.uniform_(self.base_weight, -bound, bound)

        with torch.no_grad():
            # Sample points at the G+1 interior knots for each input dim.
            interior = self.grid[:, self.spline_order : -self.spline_order]  # (in, G+1)
            x = interior.transpose(0, 1).contiguous()                        # (G+1, in)
            # Random target curve: (G+1, in, out), small magnitude.
            noise = (
                torch.rand(x.shape[0], self.in_dim, self.out_dim,
                           device=x.device, dtype=x.dtype)
                - 0.5
            ) * self.scale_noise / self.num_grids
            coef = curve_to_coeff(x, noise, self.grid, self.spline_order)
            # If the gate is folded in, pre-multiply by scale_spline.
            gate = 1.0 if self.enable_standalone_scale_spline else self.scale_spline
            self.spline_coef.data.copy_(coef * gate)

            if self.enable_standalone_scale_spline:
                nn.init.constant_(self.spline_scaler, self.scale_spline)

    # --- Spline helpers ------------------------------------------------------

    @property
    def scaled_coef(self) -> Tensor:
        """Return coefficients with the spline gate ``w_s`` folded in.

        Shape ``(out_dim, in_dim, G + K)``. When standalone scaling is enabled the
        per-edge gate broadcasts over the basis axis.
        """
        if self.enable_standalone_scale_spline:
            return self.spline_coef * self.spline_scaler.unsqueeze(-1)
        return self.spline_coef

    # --- Validation ----------------------------------------------------------

    def _check_input(self, x: Tensor) -> None:
        """Validate the entry-point tensor's trailing feature dimension."""
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"KANLayer expected last dim {self.in_dim}, got {x.shape[-1]} "
                f"(full shape {tuple(x.shape)})."
            )

    # --- Forward -------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the edge activations and reduce over input features.

        Implements ``y_j = sum_i [ w_b_{j,i} SiLU(x_i) + w_s_{j,i} sum_k c_{j,i,k} B_k(x_i) ]``.

        Parameters
        ----------
        x:
            Input of shape ``(..., in_dim)`` (any number of leading dims).

        Returns
        -------
        Tensor
            Output of shape ``(..., out_dim)``.
        """
        self._check_input(x)
        lead_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_dim)            # (N, in_dim)

        # Base path: SiLU(x) projected by w_b. einsum == (N,in) x (out,in) -> (N,out).
        base = F.silu(x_flat)
        base_out = torch.einsum("ni,oi->no", base, self.base_weight)

        # Spline path: basis (N, in, n_bases) contracted with coef (out, in, n_bases).
        bases = b_spline_basis(x_flat, self.grid, self.spline_order)
        spline_out = torch.einsum("nik,oik->no", bases, self.scaled_coef)

        out = base_out + spline_out                    # (N, out_dim)
        return out.reshape(*lead_shape, self.out_dim)

    # --- Adaptive grid update ------------------------------------------------

    @torch.no_grad()
    def update_grid(self, x: Tensor, margin: float = 0.01) -> None:
        r"""Re-fit the knot grid to the empirical activation domain of ``x``.

        Adapting the knots keeps spline resolution concentrated where inputs
        actually land. The procedure:

        1. evaluate the *current* per-edge spline at ``x`` (the function to keep);
        2. sort ``x`` per input dimension and form an adaptive (quantile) grid;
        3. blend it with a uniform grid via ``grid_eps``;
        4. extend by ``K`` knots on each side;
        5. solve least squares so the new coefficients reproduce step 1's curve.

        Parameters
        ----------
        x:
            A representative batch of shape ``(..., in_dim)``.
        margin:
            Fractional padding added beyond the observed min/max before extension.
        """
        self._check_input(x)
        x_flat = x.reshape(-1, self.in_dim)
        n = x_flat.shape[0]
        if n < 2:
            return

        # Step 1: current per-edge spline values y = (N, in, out).
        bases = b_spline_basis(x_flat, self.grid, self.spline_order)
        y = torch.einsum("nik,oik->nio", bases, self.scaled_coef)

        # Step 2: per-dimension sorted samples and adaptive (quantile) knots.
        x_sorted = torch.sort(x_flat, dim=0).values                 # (N, in)
        idx = torch.linspace(0, n - 1, self.num_grids + 1,
                             device=x_flat.device).long()
        adaptive = x_sorted[idx]                                    # (G+1, in)

        # Step 3: uniform grid spanning the (padded) observed range.
        lo = x_sorted[0] - margin * (x_sorted[-1] - x_sorted[0]).clamp_min(1e-6)
        hi = x_sorted[-1] + margin * (x_sorted[-1] - x_sorted[0]).clamp_min(1e-6)
        steps = torch.linspace(0, 1, self.num_grids + 1,
                               device=x_flat.device).unsqueeze(1)   # (G+1, 1)
        uniform = lo.unsqueeze(0) + steps * (hi - lo).unsqueeze(0)   # (G+1, in)

        interior = self.grid_eps * uniform + (1 - self.grid_eps) * adaptive

        # Step 4: extend by K knots on each side using the local interior spacing.
        h = (interior[-1] - interior[0]) / self.num_grids           # (in,)
        left = interior[0].unsqueeze(0) - h.unsqueeze(0) * torch.arange(
            self.spline_order, 0, -1, device=x_flat.device).unsqueeze(1)
        right = interior[-1].unsqueeze(0) + h.unsqueeze(0) * torch.arange(
            1, self.spline_order + 1, device=x_flat.device).unsqueeze(1)
        new_knots = torch.cat([left, interior, right], dim=0)       # (G+2K+1, in)
        new_grid = new_knots.transpose(0, 1).contiguous()           # (in, G+2K+1)

        # Step 5: re-project the preserved curve onto the new basis.
        new_coef = curve_to_coeff(x_flat, y, new_grid, self.spline_order)

        self.grid.copy_(new_grid)
        if self.enable_standalone_scale_spline:
            # Fold the gate back out so scaled_coef stays consistent.
            self.spline_coef.data.copy_(
                new_coef / self.spline_scaler.unsqueeze(-1).clamp_min(1e-6)
            )
        else:
            self.spline_coef.data.copy_(new_coef)

    # --- Regularization ------------------------------------------------------

    def regularization_loss(self, l1_weight: float = 1.0,
                            entropy_weight: float = 1.0) -> Tensor:
        r"""Sparsity-promoting regularizer over the spline coefficients.

        Combines an L1 term on the per-edge mean magnitude with an entropy term on
        the induced edge-importance distribution, mirroring the penalty proposed by
        Liu et al. (2024) to encourage interpretable, sparse KANs.

        Parameters
        ----------
        l1_weight, entropy_weight:
            Relative weights of the two terms.

        Returns
        -------
        Tensor
            A scalar regularization loss.
        """
        # Per-edge L1: mean absolute coefficient -> (out, in).
        edge_l1 = self.spline_coef.abs().mean(dim=-1)
        total_l1 = edge_l1.sum()

        prob = edge_l1 / (total_l1 + 1e-8)
        entropy = -(prob * (prob + 1e-8).log()).sum()
        return l1_weight * total_l1 + entropy_weight * entropy
