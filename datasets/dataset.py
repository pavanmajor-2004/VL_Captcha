r"""Procedural CAPTCHA dataset and collation for VL-KAN training.

This module wraps the procedural renderer in a map-style :class:`torch.utils.data.Dataset`
so it plugs into a standard :class:`~torch.utils.data.DataLoader`. Generation is
*deterministic per index*: each ``__getitem__`` reseeds the underlying renderer
RNG from ``(base_seed, epoch, index)`` so that runs are reproducible and
resume-safe regardless of worker count.

Key design choices
-------------------
* **Charset vs. vocabulary decoupling.** The renderer samples glyphs from a
  phase-specific ``charset`` (e.g. digits only for the numeric curriculum phases),
  while targets are encoded with a single fixed ``Vocabulary`` (the 62-class
  alphanumeric one). Digits map to the same indices in both, so the model's CTC
  head dimension stays constant across the whole curriculum.
* **Clean images only.** The dataset returns undistorted ``[0, 1]`` tensors;
  batched geometric/photometric augmentation runs on-device in the trainer.
* **No NumPy dependency.** PIL images are converted via ``torch.frombuffer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .captcha_generator import CaptchaGenerator
from .vocabulary import Vocabulary

__all__ = ["CaptchaBatch", "CaptchaDataset", "make_collate_fn"]


def _pil_to_tensor(img: Image.Image) -> Tensor:
    """Convert an ``RGB`` PIL image to a ``(3, H, W)`` float tensor in ``[0, 1]``.

    Uses ``torch.frombuffer`` to avoid a NumPy dependency. The intermediate
    ``.float()`` makes an owned copy, so the read-only buffer is never mutated.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    buf = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return buf.view(h, w, 3).permute(2, 0, 1).contiguous().float().div_(255.0)


@dataclass
class CaptchaBatch:
    """A collated training batch.

    Attributes
    ----------
    images:
        Clean image tensor ``(B, 3, H, W)`` in ``[0, 1]``.
    targets:
        Flat concatenated CTC target indices ``(sum(target_lengths),)`` in
        ``1 .. V`` (blank excluded).
    target_lengths:
        Per-sample target lengths ``(B,)``.
    boundaries:
        Ragged list of normalized glyph-centre positions, one list per sample,
        used to build soft boundary targets at the model's time resolution.
    texts:
        Ground-truth strings (for logging / metric computation).
    """

    images: Tensor
    targets: Tensor
    target_lengths: Tensor
    boundaries: List[List[float]]
    texts: List[str]

    def to(self, device: torch.device, non_blocking: bool = False) -> "CaptchaBatch":
        """Move tensor fields to ``device`` (ragged/python fields untouched)."""
        return CaptchaBatch(
            images=self.images.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
            target_lengths=self.target_lengths.to(device, non_blocking=non_blocking),
            boundaries=self.boundaries,
            texts=self.texts,
        )

    def __len__(self) -> int:
        return self.images.shape[0]


class CaptchaDataset(Dataset):
    """Map-style procedural CAPTCHA dataset for one curriculum phase.

    Parameters
    ----------
    vocabulary:
        The fixed encoding vocabulary (typically 62-class alphanumeric).
    charset:
        String of glyphs the renderer is allowed to sample for this phase. Must
        be a subset of ``vocabulary.alphabet``.
    length_range:
        Inclusive ``(min, max)`` character count for this phase.
    samples_per_epoch:
        Number of samples exposed per epoch (defines ``__len__``).
    image_size:
        Renderer output ``(width, height)``.
    font_paths:
        Optional explicit font list forwarded to the renderer.
    base_seed:
        Seed root for per-index determinism.
    epoch:
        Current epoch index, mixed into the per-sample seed (call
        :meth:`set_epoch` between epochs to decorrelate samples).
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        charset: str,
        length_range: Tuple[int, int] = (3, 10),
        samples_per_epoch: int = 100_000,
        image_size: Tuple[int, int] = (320, 48),
        font_paths: Optional[Sequence[str]] = None,
        base_seed: int = 1234,
        epoch: int = 0,
    ) -> None:
        super().__init__()
        missing = set(charset) - set(vocabulary.alphabet)
        if missing:
            raise ValueError(
                f"charset contains symbols absent from the vocabulary: {sorted(missing)}."
            )
        self.vocabulary = vocabulary
        self.charset = charset
        self.length_range = length_range
        self.samples_per_epoch = samples_per_epoch
        self.image_size = image_size
        self.base_seed = base_seed
        self.epoch = epoch

        # Renderer over a charset-restricted vocabulary (encoding uses the full one).
        render_vocab = Vocabulary(alphabet=charset, blank_index=0, name="render")
        self.generator = CaptchaGenerator(
            vocabulary=render_vocab,
            image_size=image_size,
            length_range=length_range,
            font_paths=font_paths,
        )

    def set_epoch(self, epoch: int) -> None:
        """Update the epoch used in per-sample seeding (decorrelates epochs)."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _seed_for(self, index: int) -> int:
        """Derive a stable 63-bit seed for a given ``(epoch, index)`` pair."""
        h = (self.base_seed * 0x9E3779B1) ^ (self.epoch * 0x85EBCA77) ^ (index + 1)
        return h & 0x7FFF_FFFF_FFFF_FFFF

    def __getitem__(self, index: int) -> Dict[str, object]:
        """Render one deterministic sample.

        Returns
        -------
        dict
            ``{"image": (3,H,W) tensor, "text": str, "length": int,
            "boundaries": list[float]}``.
        """
        # Deterministic reseed so the sample is reproducible across workers/resumes.
        self.generator._rng.seed(self._seed_for(index))
        sample = self.generator.generate()
        return {
            "image": _pil_to_tensor(sample.image),
            "text": sample.text,
            "length": sample.length,
            "boundaries": list(sample.boundaries),
        }


def make_collate_fn(vocabulary: Vocabulary) -> Callable[[List[Dict[str, object]]], CaptchaBatch]:
    """Build a collate function that encodes targets with ``vocabulary``.

    Parameters
    ----------
    vocabulary:
        The encoding vocabulary (must match the model's CTC head).

    Returns
    -------
    callable
        A function mapping a list of dataset items to a :class:`CaptchaBatch`.
    """

    def collate(items: List[Dict[str, object]]) -> CaptchaBatch:
        images = torch.stack([it["image"] for it in items], dim=0)  # type: ignore[arg-type]
        texts = [str(it["text"]) for it in items]
        boundaries = [list(it["boundaries"]) for it in items]  # type: ignore[arg-type]

        flat, lengths = vocabulary.encode_batch(texts)
        targets = torch.tensor(flat, dtype=torch.long)
        target_lengths = torch.tensor(lengths, dtype=torch.long)

        return CaptchaBatch(
            images=images,
            targets=targets,
            target_lengths=target_lengths,
            boundaries=boundaries,
            texts=texts,
        )

    return collate
