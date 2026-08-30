"""Procedural CAPTCHA image renderer (Pillow).

This module dynamically synthesizes variable-length text CAPTCHA images that
match the canonical input contract of VL-KAN: RGB, height ``H = 48``,
width ``W = 320``. It renders each glyph independently so that per-character
rotation, size, vertical baseline jitter and horizontal **tracking** (inter-glyph
spacing, including overlap) can be controlled.

The renderer also returns ground-truth **character boundary centres** (normalized
horizontal positions in ``[0, 1]``). These feed the Character Boundary Prediction
Head's soft-target construction further down the pipeline.

Heavy geometric/photometric distortions (elastic deformation, perspective warp,
Perlin clutter, occlusions, noise) are intentionally **not** applied here; they
live in :mod:`datasets.augmentations` so they can run batched on the GPU.
"""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .vocabulary import Vocabulary

__all__ = ["CaptchaSample", "CaptchaGenerator"]


# Directories searched for TrueType fonts when none are supplied explicitly.
_SYSTEM_FONT_DIRS: Tuple[str, ...] = (
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "/Library/Fonts",
    "/System/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
    "C:/Windows/Fonts",
)


@dataclass
class CaptchaSample:
    """A single rendered CAPTCHA and its annotations.

    Attributes
    ----------
    image:
        The rendered ``RGB`` :class:`PIL.Image.Image` of size ``(W, H)``.
    text:
        The ground-truth string.
    length:
        Number of characters (``len(text)``), in ``[3, 10]``.
    boundaries:
        Normalized horizontal centres (in ``[0, 1]``) of each glyph, length
        equal to :attr:`length`. Used to build soft boundary targets.
    """

    image: Image.Image
    text: str
    length: int
    boundaries: List[float]


class CaptchaGenerator:
    """Render variable-length CAPTCHA images with randomized typography.

    Parameters
    ----------
    vocabulary:
        The :class:`~datasets.vocabulary.Vocabulary` whose symbols are sampled.
    image_size:
        Output ``(width, height)``. Defaults to the contract ``(320, 48)``.
    length_range:
        Inclusive ``(min, max)`` character count. Defaults to ``(3, 10)``.
    font_paths:
        Explicit list of ``.ttf``/``.otf`` paths. If ``None`` the renderer
        discovers system fonts; if none are found it falls back to Pillow's
        built-in bitmap font (rendering still succeeds).
    font_size_range:
        Inclusive ``(min, max)`` per-glyph font size in pixels.
    rotation_range:
        Inclusive ``(min, max)`` per-glyph rotation in degrees.
    tracking_range:
        Inter-glyph spacing as a fraction of glyph width. Negative values
        produce overlapping characters. Defaults to ``(-0.30, 0.30)``.
    baseline_jitter:
        Maximum vertical offset as a fraction of image height.
    seed:
        Optional seed for the renderer's private :class:`random.Random`.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        image_size: Tuple[int, int] = (320, 48),
        length_range: Tuple[int, int] = (3, 10),
        font_paths: Optional[Sequence[str]] = None,
        font_size_range: Tuple[int, int] = (28, 40),
        rotation_range: Tuple[float, float] = (-25.0, 25.0),
        tracking_range: Tuple[float, float] = (-0.30, 0.30),
        baseline_jitter: float = 0.10,
        seed: Optional[int] = None,
    ) -> None:
        if length_range[0] < 1 or length_range[1] < length_range[0]:
            raise ValueError(f"Invalid length_range: {length_range}.")
        if font_size_range[0] < 1 or font_size_range[1] < font_size_range[0]:
            raise ValueError(f"Invalid font_size_range: {font_size_range}.")

        self.vocabulary = vocabulary
        self.width, self.height = image_size
        self.length_range = length_range
        self.font_size_range = font_size_range
        self.rotation_range = rotation_range
        self.tracking_range = tracking_range
        self.baseline_jitter = baseline_jitter
        self._rng = random.Random(seed)

        self._font_paths: List[str] = self._resolve_font_paths(font_paths)
        # Cache of (path, size) -> FreeTypeFont to avoid repeated disk loads.
        self._font_cache: dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    # --- Font handling -------------------------------------------------------

    @staticmethod
    def _resolve_font_paths(font_paths: Optional[Sequence[str]]) -> List[str]:
        """Return a usable list of font paths or an empty list (default font)."""
        if font_paths:
            resolved = [p for p in font_paths if os.path.isfile(p)]
            if resolved:
                return resolved

        discovered: List[str] = []
        for directory in _SYSTEM_FONT_DIRS:
            if not os.path.isdir(directory):
                continue
            for pattern in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
                discovered.extend(
                    glob.glob(os.path.join(directory, "**", pattern), recursive=True)
                )
        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique = [p for p in discovered if not (p in seen or seen.add(p))]
        return unique

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load a random TrueType font at ``size`` (cached), or the default font."""
        if not self._font_paths:
            # Pillow's default bitmap font ignores size but keeps rendering valid.
            return ImageFont.load_default()
        path = self._rng.choice(self._font_paths)
        key = (path, size)
        font = self._font_cache.get(key)
        if font is None:
            try:
                font = ImageFont.truetype(path, size=size)
            except OSError:
                # Corrupt/unsupported font file: drop it and retry with default.
                self._font_paths.remove(path)
                return self._load_font(size)
            self._font_cache[key] = font
        return font

    # --- Glyph rendering -----------------------------------------------------

    def _render_glyph(self, char: str) -> Image.Image:
        """Render a single character to a tight RGBA tile and rotate it.

        Returns
        -------
        PIL.Image.Image
            An ``RGBA`` tile containing the (possibly rotated) glyph with a fully
            transparent background, cropped to its bounding box.
        """
        size = self._rng.randint(*self.font_size_range)
        font = self._load_font(size)

        # Measure the glyph so the tile is just large enough to hold it.
        try:
            left, top, right, bottom = font.getbbox(char)
        except (AttributeError, TypeError):  # very old Pillow fallback
            right, bottom = font.getsize(char)  # type: ignore[attr-defined]
            left, top = 0, 0
        glyph_w = max(1, right - left)
        glyph_h = max(1, bottom - top)

        # Pad to give rotation room and avoid clipping antialiased edges.
        pad = max(4, size // 4)
        tile = Image.new("RGBA", (glyph_w + 2 * pad, glyph_h + 2 * pad), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tile)

        # Dark, slightly randomized ink for contrast against light backgrounds.
        shade = self._rng.randint(0, 90)
        color = (shade, shade, shade, 255)
        draw.text((pad - left, pad - top), char, font=font, fill=color)

        angle = self._rng.uniform(*self.rotation_range)
        tile = tile.rotate(angle, resample=Image.BICUBIC, expand=True)
        return tile.crop(tile.getbbox() or (0, 0, tile.width, tile.height))

    # --- Public API ----------------------------------------------------------

    def sample_text(self, length: Optional[int] = None) -> str:
        """Sample a random string from the alphabet.

        Parameters
        ----------
        length:
            Desired length; if ``None`` a length is drawn uniformly from
            :attr:`length_range`.

        Returns
        -------
        str
            The sampled string.
        """
        if length is None:
            length = self._rng.randint(*self.length_range)
        alphabet = self.vocabulary.alphabet
        return "".join(self._rng.choice(alphabet) for _ in range(length))

    def render(self, text: str) -> CaptchaSample:
        """Render a specific string into a :class:`CaptchaSample`.

        The glyphs are laid out left to right using randomized tracking; if the
        composed line would exceed the canvas width it is uniformly compressed so
        the full string always fits.

        Parameters
        ----------
        text:
            The string to render (its characters must be in the alphabet).

        Returns
        -------
        CaptchaSample
            The rendered image plus annotations.
        """
        # Light, faintly tinted background.
        base = self._rng.randint(210, 255)
        bg = (
            min(255, base + self._rng.randint(-10, 10)),
            min(255, base + self._rng.randint(-10, 10)),
            min(255, base + self._rng.randint(-10, 10)),
        )
        canvas = Image.new("RGB", (self.width, self.height), bg)

        glyphs = [self._render_glyph(ch) for ch in text]

        # First pass: advance widths with tracking applied.
        advances: List[float] = []
        for tile in glyphs:
            track = self._rng.uniform(*self.tracking_range)
            advances.append(tile.width * (1.0 + track))
        total = sum(advances)

        # Side margins (a fraction of the canvas) then compress if overflowing.
        margin = self.width * 0.04
        usable = self.width - 2.0 * margin
        scale = min(1.0, usable / total) if total > 0 else 1.0
        advances = [a * scale for a in advances]

        # Centre the (possibly compressed) line horizontally.
        line_width = sum(advances)
        cursor = margin + (usable - line_width) / 2.0

        boundaries: List[float] = []
        for tile, advance in zip(glyphs, advances):
            scaled_w = max(1, int(round(tile.width * scale)))
            scaled_h = max(1, int(round(tile.height * scale)))
            if (scaled_w, scaled_h) != (tile.width, tile.height):
                tile = tile.resize((scaled_w, scaled_h), Image.BICUBIC)

            # Vertical placement with baseline jitter, kept inside the canvas.
            jitter = self._rng.uniform(-self.baseline_jitter, self.baseline_jitter)
            y = int(round((self.height - scaled_h) / 2.0 + jitter * self.height))
            y = max(0, min(self.height - scaled_h, y))
            x = int(round(cursor))

            canvas.paste(tile, (x, y), tile)

            centre_x = (x + scaled_w / 2.0) / self.width
            boundaries.append(float(min(1.0, max(0.0, centre_x))))
            cursor += advance

        return CaptchaSample(
            image=canvas,
            text=text,
            length=len(text),
            boundaries=boundaries,
        )

    def generate(self, length: Optional[int] = None) -> CaptchaSample:
        """Sample a random string and render it.

        Parameters
        ----------
        length:
            Optional fixed length; otherwise drawn from :attr:`length_range`.

        Returns
        -------
        CaptchaSample
            The rendered sample.
        """
        return self.render(self.sample_text(length))
