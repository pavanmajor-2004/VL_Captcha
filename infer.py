r"""VL-KAN inference loop.

Loads a trained VL-KAN checkpoint and transcribes CAPTCHA images, either from a
directory/glob of image files or from a batch of freshly generated synthetic
samples (useful for a quick smoke check). Decoding uses the length- and
boundary-aware beam search by default, with a greedy fallback.

Examples
--------
Transcribe a folder of PNGs::

    python infer.py --checkpoint artifacts/run1/checkpoints/last.pt --images data/*.png

Generate and transcribe synthetic samples (no input files needed)::

    python infer.py --checkpoint artifacts/run1/checkpoints/last.pt --synthetic 8
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from datasets.dataset import _pil_to_tensor
from datasets.vocabulary import build_vocabulary
from decode.beam_search import BeamSearchConfig
from evaluation.decoding import beam_ctc_decode, greedy_ctc_decode
from evaluation.metrics import LatencyTracker
from models.kan_captcha import VLKAN

IMAGE_SIZE: Tuple[int, int] = (320, 48)  # (width, height)


def _load_images(paths: List[str]) -> Tuple[Tensor, List[str]]:
    """Load and resize image files into a model-ready batch.

    Parameters
    ----------
    paths:
        Image file paths.

    Returns
    -------
    tuple
        ``(images (N,3,48,320), names)``.
    """
    from PIL import Image

    tensors: List[Tensor] = []
    names: List[str] = []
    for path in paths:
        img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
        tensors.append(_pil_to_tensor(img))
        names.append(os.path.basename(path))
    if not tensors:
        raise ValueError("No images were loaded.")
    return torch.stack(tensors, dim=0), names


def _synthetic_batch(n: int, seed: int) -> Tuple[Tensor, List[str]]:
    """Generate ``n`` synthetic samples for a quick inference check."""
    from datasets.captcha_generator import CaptchaGenerator
    from datasets.vocabulary import build_vocabulary as _bv

    gen = CaptchaGenerator(
        vocabulary=_bv("alphanumeric"), image_size=IMAGE_SIZE,
        length_range=(3, 10), seed=seed,
    )
    tensors: List[Tensor] = []
    truths: List[str] = []
    for _ in range(n):
        sample = gen.generate()
        tensors.append(_pil_to_tensor(sample.image))
        truths.append(sample.text)
    return torch.stack(tensors, dim=0), truths


def load_model(checkpoint: str, num_classes: int, device: torch.device) -> VLKAN:
    """Instantiate ``VLKAN`` and load weights from a checkpoint.

    Parameters
    ----------
    checkpoint:
        Path to a checkpoint saved by the trainer (or a raw ``state_dict``).
    num_classes:
        Vocabulary size ``V`` (model CTC head is ``V + 1``).
    device:
        Target device.

    Returns
    -------
    VLKAN
        The model in eval mode with weights loaded.
    """
    model = VLKAN(num_classes=num_classes).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state))
    model.eval()
    return model


def run_inference(
    model: VLKAN,
    images: Tensor,
    device: torch.device,
    use_beam: bool,
    batch_size: int,
) -> Tuple[List[str], LatencyTracker]:
    """Transcribe a batch of images, returning predictions and latency stats.

    Parameters
    ----------
    model:
        A loaded VL-KAN model.
    images:
        Image tensor ``(N, 3, 48, 320)`` in ``[0, 1]``.
    device:
        Compute device.
    use_beam:
        Use length/boundary-aware beam search if ``True``, else greedy CTC.
    batch_size:
        Inference mini-batch size.

    Returns
    -------
    tuple
        ``(predictions, latency_tracker)``.
    """
    vocabulary = build_vocabulary("alphanumeric")
    beam_cfg = BeamSearchConfig()
    predictions: List[str] = []
    latency = LatencyTracker()

    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            chunk = images[start:start + batch_size].to(device)
            with latency.measure(num_items=chunk.shape[0]):
                outputs = model(chunk)
                if use_beam:
                    preds = beam_ctc_decode(outputs, vocabulary, beam_cfg)
                else:
                    preds = greedy_ctc_decode(outputs["ctc_logits"], vocabulary)
            predictions.extend(preds)
    return predictions, latency


def parse_args() -> argparse.Namespace:
    """Parse inference command-line arguments."""
    p = argparse.ArgumentParser(description="Run VL-KAN inference.")
    p.add_argument("--checkpoint", required=True, help="Trained model checkpoint.")
    p.add_argument("--images", nargs="*", default=None,
                   help="Image paths or globs to transcribe.")
    p.add_argument("--synthetic", type=int, default=0,
                   help="Generate and transcribe N synthetic samples instead.")
    p.add_argument("--device", default=None, help="cuda | cpu (auto if unset).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--greedy", action="store_true",
                   help="Use greedy CTC decoding instead of beam search.")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main() -> None:
    """Program entry: load, transcribe, and print results + latency."""
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    vocabulary = build_vocabulary("alphanumeric")
    model = load_model(args.checkpoint, vocabulary.size, device)

    truths: Optional[List[str]] = None
    if args.synthetic > 0:
        images, truths = _synthetic_batch(args.synthetic, args.seed)
        names = [f"synthetic_{i}" for i in range(args.synthetic)]
    else:
        paths: List[str] = []
        for pattern in (args.images or []):
            paths.extend(sorted(glob.glob(pattern)))
        images, names = _load_images(paths)

    predictions, latency = run_inference(
        model, images, device, use_beam=not args.greedy, batch_size=args.batch_size,
    )

    print(f"{'name':<18} {'prediction':<14} truth")
    for i, (name, pred) in enumerate(zip(names, predictions)):
        truth = truths[i] if truths is not None else ""
        print(f"{name:<18} {pred:<14} {truth}")

    stats = latency.compute()
    print(
        f"\nlatency: mean={stats.mean_ms:.2f}ms p95={stats.p95_ms:.2f}ms "
        f"throughput={stats.throughput:.1f} img/s over {stats.num_items} images."
    )


if __name__ == "__main__":
    main()
