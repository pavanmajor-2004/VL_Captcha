r"""B-spline primitives for Kolmogorov-Arnold Networks (from scratch).

The Kolmogorov-Arnold representation theorem states that any continuous
multivariate function decomposes into sums of univariate continuous functions::

    f(x) = sum_{q=1}^{2n+1} Phi_q( sum_{p=1}^{n} phi_{q,p}(x_p) )

KAN operationalizes this by making every univariate edge function ``phi`` a
learnable B-spline. This module provides the *differentiable* machinery to
evaluate those splines.

A B-spline of order ``K`` (degree ``K``) over ``G`` uniform knot intervals is a
linear combination of ``G + K`` basis functions ``B_k``. The basis is built with
the **Cox-de Boor recursion**::

    B_{i,0}(x) = 1 if knot_i <= x < knot_{i+1} else 0
    B_{i,k}(x) =   (x - knot_i)        / (knot_{i+k}   - knot_i)     * B_{i,k-1}(x)
                 + (knot_{i+k+1} - x)  / (knot_{i+k+1} - knot_{i+1}) * B_{i+1,k-1}(x)

To support order-``K`` splines that retain full support at the domain edges, the
``G + 1`` interior knots are padded with ``K`` extra knots on each side, giving a
knot vector of length ``G + 2K + 1``. Every operation here is fully vectorized and
broadcasts over arbitrary leading (batch/time) dimensions so autograd flows
through ``x``, the knot grid, and (downstream) the spline coefficients without any
Python-level loops over samples.

Defaults follow the paper: ``G = 5`` knot intervals, order ``K = 3``.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

__all__ = [
    "DEFAULT_NUM_GRIDS",
    "DEFAULT_SPLINE_ORDER",
    "num_spline_bases",
    "generate_grid",
    "b_spline_basis",
    "curve_to_coeff",
]

#: Number of uniform knot intervals ``G`` in the spline domain.
DEFAULT_NUM_GRIDS: int = 5
#: B-spline order / degree ``K``.
DEFAULT_SPLINE_ORDER: int = 3


def num_spline_bases(num_grids: int = DEFAULT_NUM_GRIDS,
                     spline_order: int = DEFAULT_SPLINE_ORDER) -> int:
    """Return the number of B-spline basis functions ``G + K`` per edge.

    Parameters
    ----------
    num_grids:
        Number of uniform knot intervals ``G``.
    spline_order:
        Spline order ``K``.

    Returns
    -------
    int
        The basis count ``G + K`` (``8`` for the defaults ``G=5, K=3``).
    """
    return num_grids + spline_order


def generate_grid(
    in_dim: int,
    num_grids: int = DEFAULT_NUM_GRIDS,
    spline_order: int = DEFAULT_SPLINE_ORDER,
    grid_range: Tuple[float, float] = (-1.0, 1.0),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    r"""Build a per-input-dimension extended uniform knot vector.

    For a domain ``[a, b]`` with ``G`` intervals the spacing is ``h = (b - a) / G``.
    The knot vector is extended by ``K`` knots on each side so that order-``K``
    basis functions have full support across ``[a, b]``::

        knots = a + (i * h)  for i in {-K, ..., G + K}      (length G + 2K + 1)

    The same grid is replicated for every input dimension; downstream layers may
    later adapt each row independently.

    Parameters
    ----------
    in_dim:
        Number of input features (rows of the grid).
    num_grids, spline_order:
        ``G`` and ``K`` respectively.
    grid_range:
        The base domain ``(a, b)`` before extension.
    device, dtype:
        Tensor placement and precision.

    Returns
    -------
    Tensor
        Contiguous knot tensor of shape ``(in_dim, G + 2K + 1)``. Stored
        contiguously so the per-row knot lookups inside the recursion are
        cache-friendly.
    """
    a, b = grid_range
    h = (b - a) / num_grids
    # Index offsets from -K to G+K inclusive -> G + 2K + 1 knots.
    steps = torch.arange(
        -spline_order, num_grids + spline_order + 1, device=device, dtype=dtype
    )
    knots = a + steps * h                      # (G + 2K + 1,)
    grid = knots.unsqueeze(0).expand(in_dim, -1).contiguous()
    return grid


def b_spline_basis(
    x: Tensor,
    grid: Tensor,
    spline_order: int = DEFAULT_SPLINE_ORDER,
    eps: float = 1e-8,
) -> Tensor:
    r"""Evaluate B-spline basis functions via the Cox-de Boor recursion.

    The computation is vectorized: ``x`` may carry any number of leading
    dimensions (e.g. ``(batch, time)``) and the final basis axis is produced by
    broadcasting the knot grid against the inputs. The recursion is unrolled over
    the (small, fixed) order ``K`` only -- never over samples -- so the whole
    pass is a handful of elementwise tensor ops and remains fully differentiable.

    Parameters
    ----------
    x:
        Input activations of shape ``(..., in_dim)``.
    grid:
        Knot grid of shape ``(in_dim, n_knots)`` with ``n_knots = G + 2K + 1``.
    spline_order:
        Spline order ``K``.
    eps:
        Small constant guarding the recursion's denominators against repeated
        knots (zero-width intervals).

    Returns
    -------
    Tensor
        Basis tensor of shape ``(..., in_dim, G + K)``. For any ``x`` inside the
        base domain the ``G + K`` values along the last axis form a partition of
        unity (they sum to 1).
    """
    if grid.dim() != 2:
        raise ValueError(f"grid must be 2-D (in_dim, n_knots), got {tuple(grid.shape)}.")
    if x.shape[-1] != grid.shape[0]:
        raise ValueError(
            f"x last dim {x.shape[-1]} must match grid in_dim {grid.shape[0]}."
        )

    # Promote x to (..., in_dim, 1) so it broadcasts against (in_dim, n_knots).
    xe = x.unsqueeze(-1)

    # --- Order 0: indicator over each knot interval [knot_i, knot_{i+1}). ----
    # Shape: (..., in_dim, n_knots - 1).
    bases = ((xe >= grid[:, :-1]) & (xe < grid[:, 1:])).to(x.dtype)

    # --- Recursively lift order 0 -> K via the Cox-de Boor formula. ----------
    for k in range(1, spline_order + 1):
        # Left coefficient: (x - knot_i) / (knot_{i+k} - knot_i).
        left_num = xe - grid[:, : -(k + 1)]
        left_den = grid[:, k:-1] - grid[:, : -(k + 1)]
        left = left_num / (left_den + eps)

        # Right coefficient: (knot_{i+k+1} - x) / (knot_{i+k+1} - knot_{i+1}).
        right_num = grid[:, k + 1:] - xe
        right_den = grid[:, k + 1:] - grid[:, 1:-k]
        right = right_num / (right_den + eps)

        # Combine adjacent lower-order bases; last axis shrinks by 1 each step.
        bases = left * bases[..., :-1] + right * bases[..., 1:]

    return bases.contiguous()


def curve_to_coeff(
    x: Tensor,
    y: Tensor,
    grid: Tensor,
    spline_order: int = DEFAULT_SPLINE_ORDER,
    eps: float = 1e-8,
) -> Tensor:
    r"""Least-squares fit spline coefficients reproducing samples ``y`` at ``x``.

    Solves, per input dimension, the linear system ``B(x) @ c = y`` where ``B(x)``
    is the basis matrix. This is the inverse of evaluating the spline and is used
    to (a) initialize coefficients from noise and (b) re-project an existing
    spline onto a freshly adapted knot grid without changing the function it
    represents.

    Parameters
    ----------
    x:
        Sample inputs of shape ``(N, in_dim)``.
    y:
        Target per-edge spline values of shape ``(N, in_dim, out_dim)``.
    grid:
        Knot grid of shape ``(in_dim, n_knots)``.
    spline_order:
        Spline order ``K``.
    eps:
        Denominator guard forwarded to :func:`b_spline_basis`.

    Returns
    -------
    Tensor
        Coefficient tensor of shape ``(out_dim, in_dim, G + K)``.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2-D (N, in_dim), got {tuple(x.shape)}.")
    if y.dim() != 3:
        raise ValueError(f"y must be 3-D (N, in_dim, out_dim), got {tuple(y.shape)}.")

    # Basis matrix A: (N, in_dim, n_bases) -> (in_dim, N, n_bases) for batched lstsq.
    bases = b_spline_basis(x, grid, spline_order, eps)
    a_mat = bases.transpose(0, 1)            # (in_dim, N, n_bases)
    b_mat = y.transpose(0, 1)                # (in_dim, N, out_dim)

    # Batched least squares: one independent system per input dimension.
    solution = torch.linalg.lstsq(a_mat, b_mat).solution  # (in_dim, n_bases, out)

    # Reorder to the canonical coefficient layout (out, in, n_bases).
    return solution.permute(2, 0, 1).contiguous()
