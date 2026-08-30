"""Batch-compatible, GPU-friendly CAPTCHA augmentations (PyTorch 2.x).

Every transform here operates on a batched image tensor of shape
``(B, C, H, W)`` with float values in ``[0, 1]`` and returns a tensor of the same
shape and range. All geometry is implemented with
:func:`torch.nn.functional.grid_sample`, and all stochastic fields are generated
with explicit :class:`torch.Generator` support for reproducibility.

Implemented effects
-------------------
* :class:`RandomRotationJitter` -- affine rotation + horizontal/vertical shift.
* :class:`ElasticDistortion` -- localized smoothed displacement fields.
* :class:`PerspectiveWarp` -- per-sample homography (corner perturbation).
* :class:`PerlinClutter` -- additive multi-octave Perlin background texture.
* :class:`SplineOcclusion` -- random quadratic Bezier strokes / lines.
* :class:`CaptchaAugmentor` -- probabilistic composition of the above.

The displacement-field, Perlin and occlusion helpers are fully vectorized across
the batch; only the homography solve loops over the (tiny, fixed) 4-corner system.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "gaussian_blur",
    "RandomRotationJitter",
    "ElasticDistortion",
    "PerspectiveWarp",
    "PerlinClutter",
    "SplineOcclusion",
    "CaptchaAugmentor",
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _identity_grid(batch: int, height: int, width: int, device: torch.device,
                   dtype: torch.dtype) -> Tensor:
    """Return the identity sampling grid in normalized ``[-1, 1]`` coords.

    Parameters
    ----------
    batch, height, width:
        Target grid dimensions.
    device, dtype:
        Tensor placement and precision.

    Returns
    -------
    Tensor
        Grid of shape ``(B, H, W, 2)`` where the last axis is ``(x, y)``, ready
        for :func:`torch.nn.functional.grid_sample` (``align_corners=False``).
    """
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
    return grid.unsqueeze(0).expand(batch, height, width, 2)


def _gaussian_kernel1d(sigma: float, device: torch.device,
                       dtype: torch.dtype) -> Tensor:
    """Build a normalized 1-D Gaussian kernel.

    The radius is ``ceil(3 * sigma)`` so the kernel captures >99% of the mass.
    """
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur(image: Tensor, sigma: float) -> Tensor:
    """Apply a separable Gaussian blur to a batched image tensor.

    Parameters
    ----------
    image:
        Tensor of shape ``(B, C, H, W)``.
    sigma:
        Standard deviation of the Gaussian, in pixels. ``sigma <= 0`` is a no-op.

    Returns
    -------
    Tensor
        The blurred tensor, same shape as the input.
    """
    if sigma <= 0:
        return image
    channels = image.shape[1]
    kernel = _gaussian_kernel1d(sigma, image.device, image.dtype)
    radius = kernel.numel() // 2

    kx = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)

    blurred = F.conv2d(
        F.pad(image, (radius, radius, 0, 0), mode="reflect"),
        kx, groups=channels,
    )
    blurred = F.conv2d(
        F.pad(blurred, (0, 0, radius, radius), mode="reflect"),
        ky, groups=channels,
    )
    return blurred


def _rand(shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype,
          generator: Optional[torch.Generator]) -> Tensor:
    """Uniform ``[0, 1)`` sample honoring an optional generator."""
    return torch.rand(shape, device=device, dtype=dtype, generator=generator)


# ---------------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------------

class RandomRotationJitter(torch.nn.Module):
    """Random global rotation plus horizontal / vertical translation jitter.

    Parameters
    ----------
    max_angle:
        Maximum absolute rotation in degrees.
    max_translate:
        Maximum absolute translation as a fraction of width/height.
    padding_mode:
        ``grid_sample`` padding mode for out-of-bounds samples.
    """

    def __init__(self, max_angle: float = 12.0, max_translate: float = 0.06,
                 padding_mode: str = "border") -> None:
        super().__init__()
        self.max_angle = max_angle
        self.max_translate = max_translate
        self.padding_mode = padding_mode

    def forward(self, image: Tensor,
                generator: Optional[torch.Generator] = None) -> Tensor:
        """Apply the transform to ``(B, C, H, W)`` and return the same shape."""
        b, _, h, w = image.shape
        device, dtype = image.device, image.dtype

        angles = (_rand((b,), device, dtype, generator) * 2 - 1) * math.radians(
            self.max_angle
        )
        cos, sin = torch.cos(angles), torch.sin(angles)
        tx = (_rand((b,), device, dtype, generator) * 2 - 1) * self.max_translate * 2
        ty = (_rand((b,), device, dtype, generator) * 2 - 1) * self.max_translate * 2

        # Inverse (output->input) affine matrix for grid generation.
        theta = torch.zeros(b, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, (b, image.shape[1], h, w), align_corners=False)
        return F.grid_sample(
            image, grid, mode="bilinear",
            padding_mode=self.padding_mode, align_corners=False,
        )


class ElasticDistortion(torch.nn.Module):
    """Elastic deformation via a smoothed random displacement field.

    A white-noise displacement field is low-pass filtered with a Gaussian
    (controlling correlation length ``sigma``) and scaled by ``alpha`` to produce
    locally coherent warps, following Simard et al. (2003).

    Parameters
    ----------
    alpha:
        Displacement magnitude in pixels.
    sigma:
        Gaussian smoothing of the displacement field; larger is smoother.
    padding_mode:
        ``grid_sample`` padding mode.
    """

    def __init__(self, alpha: float = 6.0, sigma: float = 6.0,
                 padding_mode: str = "border") -> None:
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.padding_mode = padding_mode

    def forward(self, image: Tensor,
                generator: Optional[torch.Generator] = None) -> Tensor:
        """Warp ``(B, C, H, W)`` with a per-sample elastic field."""
        b, _, h, w = image.shape
        device, dtype = image.device, image.dtype

        # Random displacement in pixels for x and y, then smooth.
        disp = _rand((b, 2, h, w), device, dtype, generator) * 2 - 1
        disp = gaussian_blur(disp, self.sigma) * self.alpha

        # Convert pixel displacements to normalized grid units ([-1, 1] span).
        dx = disp[:, 0] * (2.0 / max(1, w - 1))
        dy = disp[:, 1] * (2.0 / max(1, h - 1))

        base = _identity_grid(b, h, w, device, dtype).clone()
        base[..., 0] = base[..., 0] + dx
        base[..., 1] = base[..., 1] + dy

        return F.grid_sample(
            image, base, mode="bilinear",
            padding_mode=self.padding_mode, align_corners=False,
        )


class PerspectiveWarp(torch.nn.Module):
    """Per-sample perspective warp by perturbing the four image corners.

    A destination quadrilateral is formed by jittering the corners; the homography
    mapping output pixels back to input pixels is solved per sample and used to
    build a ``grid_sample`` grid.

    Parameters
    ----------
    distortion_scale:
        Maximum corner displacement as a fraction of width/height.
    padding_mode:
        ``grid_sample`` padding mode.
    """

    def __init__(self, distortion_scale: float = 0.10,
                 padding_mode: str = "border") -> None:
        super().__init__()
        self.distortion_scale = distortion_scale
        self.padding_mode = padding_mode

    @staticmethod
    def _find_homography(src: Tensor, dst: Tensor) -> Tensor:
        """Solve the homography mapping ``src`` to ``dst`` for each batch item.

        Parameters
        ----------
        src, dst:
            Corner tensors of shape ``(B, 4, 2)`` in pixel coordinates.

        Returns
        -------
        Tensor
            Homography matrices of shape ``(B, 3, 3)``.
        """
        b = src.shape[0]
        device, dtype = src.dtype, src.device  # noqa: F841 (clarity)
        a = torch.zeros(b, 8, 8, device=src.device, dtype=src.dtype)
        rhs = torch.zeros(b, 8, 1, device=src.device, dtype=src.dtype)

        sx, sy = src[..., 0], src[..., 1]
        dx, dy = dst[..., 0], dst[..., 1]
        for i in range(4):
            r = 2 * i
            a[:, r, 0] = sx[:, i]
            a[:, r, 1] = sy[:, i]
            a[:, r, 2] = 1.0
            a[:, r, 6] = -sx[:, i] * dx[:, i]
            a[:, r, 7] = -sy[:, i] * dx[:, i]
            rhs[:, r, 0] = dx[:, i]

            a[:, r + 1, 3] = sx[:, i]
            a[:, r + 1, 4] = sy[:, i]
            a[:, r + 1, 5] = 1.0
            a[:, r + 1, 6] = -sx[:, i] * dy[:, i]
            a[:, r + 1, 7] = -sy[:, i] * dy[:, i]
            rhs[:, r + 1, 0] = dy[:, i]

        solution = torch.linalg.solve(a, rhs).squeeze(-1)  # (B, 8)
        ones = torch.ones(b, 1, device=src.device, dtype=src.dtype)
        homography = torch.cat((solution, ones), dim=1).view(b, 3, 3)
        return homography

    def forward(self, image: Tensor,
                generator: Optional[torch.Generator] = None) -> Tensor:
        """Apply a perspective warp to ``(B, C, H, W)``."""
        b, _, h, w = image.shape
        device, dtype = image.device, image.dtype

        corners = torch.tensor(
            [[0.0, 0.0], [w - 1, 0.0], [w - 1, h - 1], [0.0, h - 1]],
            device=device, dtype=dtype,
        ).unsqueeze(0).expand(b, 4, 2)

        offset = (_rand((b, 4, 2), device, dtype, generator) * 2 - 1)
        offset = offset * self.distortion_scale * torch.tensor(
            [w, h], device=device, dtype=dtype
        )
        dst = corners + offset

        # Homography mapping output (dst) coordinates back to input (src) coords.
        homography = self._find_homography(dst, corners)

        ys = torch.arange(h, device=device, dtype=dtype)
        xs = torch.arange(w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        ones = torch.ones_like(grid_x)
        out_coords = torch.stack((grid_x, grid_y, ones), dim=-1)  # (H, W, 3)
        out_coords = out_coords.view(1, h * w, 3).expand(b, h * w, 3)

        mapped = torch.bmm(out_coords, homography.transpose(1, 2))  # (B, H*W, 3)
        eps = torch.finfo(dtype).eps
        src_x = mapped[..., 0] / (mapped[..., 2] + eps)
        src_y = mapped[..., 1] / (mapped[..., 2] + eps)

        norm_x = src_x / max(1, w - 1) * 2 - 1
        norm_y = src_y / max(1, h - 1) * 2 - 1
        grid = torch.stack((norm_x, norm_y), dim=-1).view(b, h, w, 2)

        return F.grid_sample(
            image, grid, mode="bilinear",
            padding_mode=self.padding_mode, align_corners=False,
        )


# ---------------------------------------------------------------------------
# Photometric / clutter transforms
# ---------------------------------------------------------------------------

def _fade(t: Tensor) -> Tensor:
    """Quintic smoothing curve ``6t^5 - 15t^4 + 10t^3`` (Perlin's fade)."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin_noise_2d(batch: int, height: int, width: int, cells: Tuple[int, int],
                    device: torch.device, dtype: torch.dtype,
                    generator: Optional[torch.Generator] = None) -> Tensor:
    """Generate batched 2-D Perlin gradient noise in ``[-1, 1]``.

    Parameters
    ----------
    batch, height, width:
        Output dimensions ``(B, H, W)``.
    cells:
        Number of grid cells ``(cells_y, cells_x)`` -- controls feature scale.
    device, dtype:
        Tensor placement and precision.
    generator:
        Optional RNG for reproducibility.

    Returns
    -------
    Tensor
        Noise of shape ``(B, H, W)`` approximately in ``[-1, 1]``.
    """
    cy, cx = cells
    # Random unit gradient vectors at each lattice node.
    angles = _rand((batch, cy + 1, cx + 1), device, dtype, generator) * 2 * math.pi
    grad = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    # Pixel coordinates expressed in lattice space (slightly inside the border).
    lin_y = torch.linspace(0, cy, height + 1, device=device, dtype=dtype)[:-1]
    lin_x = torch.linspace(0, cx, width + 1, device=device, dtype=dtype)[:-1]
    gy, gx = torch.meshgrid(lin_y, lin_x, indexing="ij")  # (H, W)

    y0 = torch.floor(gy).long().clamp(0, cy - 1)
    x0 = torch.floor(gx).long().clamp(0, cx - 1)
    y1, x1 = y0 + 1, x0 + 1
    fy = (gy - y0.to(dtype)).unsqueeze(0)  # (1, H, W)
    fx = (gx - x0.to(dtype)).unsqueeze(0)

    def corner_dot(iy: Tensor, ix: Tensor, dy: Tensor, dx: Tensor) -> Tensor:
        g = grad[:, iy, ix]              # (B, H, W, 2)
        return g[..., 0] * dx + g[..., 1] * dy

    n00 = corner_dot(y0, x0, fy, fx)
    n10 = corner_dot(y0, x1, fy, fx - 1)
    n01 = corner_dot(y1, x0, fy - 1, fx)
    n11 = corner_dot(y1, x1, fy - 1, fx - 1)

    u, v = _fade(fx), _fade(fy)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return nx0 + v * (nx1 - nx0)


class PerlinClutter(torch.nn.Module):
    """Blend multi-octave Perlin noise into the image as background texture.

    Parameters
    ----------
    octaves:
        Number of noise octaves summed with decreasing amplitude.
    base_cells:
        Lattice resolution of the first octave; each octave doubles it.
    strength:
        Maximum blend weight of the clutter onto the image.
    """

    def __init__(self, octaves: int = 3, base_cells: Tuple[int, int] = (3, 8),
                 strength: float = 0.35) -> None:
        super().__init__()
        self.octaves = octaves
        self.base_cells = base_cells
        self.strength = strength

    def forward(self, image: Tensor,
                generator: Optional[torch.Generator] = None) -> Tensor:
        """Composite Perlin clutter onto ``(B, C, H, W)``."""
        b, c, h, w = image.shape
        device, dtype = image.device, image.dtype

        noise = torch.zeros(b, h, w, device=device, dtype=dtype)
        amplitude, total = 1.0, 0.0
        cy, cx = self.base_cells
        for octave in range(self.octaves):
            cells = (max(1, cy * (2 ** octave)), max(1, cx * (2 ** octave)))
            noise = noise + amplitude * perlin_noise_2d(
                b, h, w, cells, device, dtype, generator
            )
            total += amplitude
            amplitude *= 0.5
        noise = noise / max(total, 1e-6)
        noise = (noise * 0.5 + 0.5).clamp(0.0, 1.0)  # -> [0, 1]
        noise = noise.unsqueeze(1).expand(b, c, h, w)

        weight = _rand((b, 1, 1, 1), device, dtype, generator) * self.strength
        return (image * (1.0 - weight) + noise * weight).clamp(0.0, 1.0)


class SplineOcclusion(torch.nn.Module):
    """Overlay random quadratic Bezier strokes (anti-segmentation lines).

    Each stroke is sampled as a quadratic Bezier curve; its points are splatted
    onto a mask which is then thickened via max-pooling and softened, then
    alpha-composited over the image.

    Parameters
    ----------
    max_curves:
        Upper bound on the number of strokes per image (the actual count is
        random per sample, with unused strokes given zero opacity).
    thickness:
        Stroke thickness in pixels (odd values recommended).
    samples:
        Number of points sampled along each Bezier curve.
    max_opacity:
        Maximum stroke opacity.
    """

    def __init__(self, max_curves: int = 6, thickness: int = 3,
                 samples: int = 200, max_opacity: float = 0.8) -> None:
        super().__init__()
        self.max_curves = max_curves
        self.thickness = thickness if thickness % 2 == 1 else thickness + 1
        self.samples = samples
        self.max_opacity = max_opacity

    def forward(self, image: Tensor,
                generator: Optional[torch.Generator] = None) -> Tensor:
        """Draw and composite random strokes over ``(B, C, H, W)``."""
        b, c, h, w = image.shape
        device, dtype = image.device, image.dtype
        mask = torch.zeros(b, 1, h, w, device=device, dtype=dtype)

        t = torch.linspace(0, 1, self.samples, device=device, dtype=dtype)
        t = t.view(1, 1, self.samples)  # (1, 1, S)

        for _ in range(self.max_curves):
            # Three control points per curve in pixel coordinates, per sample.
            p = _rand((b, 3, 2), device, dtype, generator)
            p[..., 0] *= (w - 1)
            p[..., 1] *= (h - 1)
            p0, p1, p2 = p[:, 0], p[:, 1], p[:, 2]  # each (B, 2)

            one_minus = 1 - t
            # Quadratic Bezier: (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2.
            bx = (one_minus ** 2) * p0[:, 0:1, None] \
                + 2 * one_minus * t * p1[:, 0:1, None] \
                + (t ** 2) * p2[:, 0:1, None]
            by = (one_minus ** 2) * p0[:, 1:2, None] \
                + 2 * one_minus * t * p1[:, 1:2, None] \
                + (t ** 2) * p2[:, 1:2, None]
            xs = bx.view(b, self.samples).round().long().clamp(0, w - 1)
            ys = by.view(b, self.samples).round().long().clamp(0, h - 1)

            flat_idx = ys * w + xs                       # (B, S)
            # Randomly disable some strokes by zeroing their contribution.
            active = (_rand((b, 1), device, dtype, generator) < 0.6).to(dtype)
            stroke = mask.view(b, h * w)
            stroke.scatter_(1, flat_idx, active.expand(b, self.samples))
            mask = stroke.view(b, 1, h, w)

        # Thicken strokes, then soften the edges.
        if self.thickness > 1:
            pad = self.thickness // 2
            mask = F.max_pool2d(mask, kernel_size=self.thickness, stride=1, padding=pad)
        mask = gaussian_blur(mask, sigma=0.8).clamp(0.0, 1.0)

        opacity = _rand((b, 1, 1, 1), device, dtype, generator) * self.max_opacity
        # Stroke colour: dark, matching CAPTCHA ink.
        colour = _rand((b, c, 1, 1), device, dtype, generator) * 0.2
        alpha = mask * opacity
        return (image * (1.0 - alpha) + colour * alpha).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

class CaptchaAugmentor(torch.nn.Module):
    """Probabilistic composition of the full distortion pipeline.

    Each component transform is applied independently with its own probability.
    The order mirrors the rendering physics: geometry first (elastic, rotation,
    perspective), then background clutter, then foreground occlusions, then
    sensor noise.

    Parameters
    ----------
    p_elastic, p_rotation, p_perspective, p_clutter, p_occlusion, p_noise:
        Per-transform application probabilities.
    noise_std:
        Maximum standard deviation of additive Gaussian sensor noise.
    """

    def __init__(
        self,
        p_elastic: float = 0.7,
        p_rotation: float = 0.8,
        p_perspective: float = 0.5,
        p_clutter: float = 0.7,
        p_occlusion: float = 0.6,
        p_noise: float = 0.5,
        noise_std: float = 0.05,
        elastic: Optional[ElasticDistortion] = None,
        rotation: Optional[RandomRotationJitter] = None,
        perspective: Optional[PerspectiveWarp] = None,
        clutter: Optional[PerlinClutter] = None,
        occlusion: Optional[SplineOcclusion] = None,
    ) -> None:
        super().__init__()
        self.p_elastic = p_elastic
        self.p_rotation = p_rotation
        self.p_perspective = p_perspective
        self.p_clutter = p_clutter
        self.p_occlusion = p_occlusion
        self.p_noise = p_noise
        self.noise_std = noise_std

        self.elastic = elastic or ElasticDistortion()
        self.rotation = rotation or RandomRotationJitter()
        self.perspective = perspective or PerspectiveWarp()
        self.clutter = clutter or PerlinClutter()
        self.occlusion = occlusion or SplineOcclusion()

    @staticmethod
    def _flip(prob: float, generator: Optional[torch.Generator],
              device: torch.device) -> bool:
        """Return ``True`` with probability ``prob`` (generator-aware)."""
        if prob >= 1.0:
            return True
        if prob <= 0.0:
            return False
        return bool(torch.rand((), device=device, generator=generator).item() < prob)

    def forward(self, image: Tensor,
                generator: Optional[torch.Generator] = None) -> Tensor:
        """Apply the randomized pipeline to ``(B, C, H, W)`` in ``[0, 1]``.

        Parameters
        ----------
        image:
            Batched image tensor in ``[0, 1]``.
        generator:
            Optional RNG for fully reproducible augmentation.

        Returns
        -------
        Tensor
            Augmented tensor, same shape and range as the input.
        """
        if image.dim() != 4:
            raise ValueError(f"Expected (B, C, H, W) tensor, got shape {tuple(image.shape)}.")
        device = image.device
        out = image

        if self._flip(self.p_elastic, generator, device):
            out = self.elastic(out, generator)
        if self._flip(self.p_rotation, generator, device):
            out = self.rotation(out, generator)
        if self._flip(self.p_perspective, generator, device):
            out = self.perspective(out, generator)
        if self._flip(self.p_clutter, generator, device):
            out = self.clutter(out, generator)
        if self._flip(self.p_occlusion, generator, device):
            out = self.occlusion(out, generator)
        if self._flip(self.p_noise, generator, device):
            std = torch.rand((), device=device, generator=generator) * self.noise_std
            noise = torch.randn(out.shape, device=device, dtype=out.dtype,
                                generator=generator) * std
            out = (out + noise).clamp(0.0, 1.0)

        return out
