r"""Automated structural ablation harness for VL-KAN.

Steps through four model variants spanning the two architectural axes the paper
isolates -- **variable-length CTC decoding** and **Kolmogorov-Arnold (KAN)
feature/​sequence modelling** -- then reports a structured Markdown comparison.

Variants
--------
============================  ===========  ======  ====================================
Variant                       VarLen CTC   KAN     Concrete model
============================  ===========  ======  ====================================
Baseline                      no           no      ``DeepCaptcha`` (fixed-length CNN)
Baseline + VarLen             yes          no      ``VariableLengthDeepCaptcha`` (CRNN)
Baseline + KAN                yes          part    ``VLKAN`` w/o A-KAN attention stack
Full VL-KAN                   yes          yes     ``VLKAN`` (pyramid + A-KAN blocks)
============================  ===========  ======  ====================================

For each variant the harness (optionally) runs a short, native-regime warm-up
training loop on procedurally generated data, evaluates on a fixed held-out set,
and records recognition metrics (CAcc / SAcc / LAcc / CER), inference latency,
and parameter count. Results are emitted as Markdown tables.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from datasets.dataset import CaptchaBatch, CaptchaDataset, make_collate_fn
from datasets.vocabulary import Vocabulary, build_vocabulary
from evaluation.decoding import (
    fixed_length_decode,
    greedy_ctc_decode,
)
from evaluation.metrics import (
    LatencyTracker,
    MetricAccumulator,
)
from models.deepcaptcha import DeepCaptcha
from models.kan_captcha import VLKAN
from models.variable_length_deepcaptcha import VariableLengthDeepCaptcha

__all__ = ["VariantSpec", "VariantResult", "build_variants", "run_ablation"]

NUMERIC_CHARSET = "0123456789"


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

@dataclass
class VariantSpec:
    """Declarative description of one ablation variant.

    Attributes
    ----------
    name:
        Human-readable variant label (table row).
    kind:
        ``"ctc"`` for CTC sequence models or ``"fixed"`` for the fixed-length
        baseline -- selects the training regime and decode adapter.
    builder:
        Factory ``(num_classes, max_length) -> nn.Module``.
    uses_varlen, uses_kan:
        Capability flags reported in the table.
    """

    name: str
    kind: str
    builder: Callable[[int, int], nn.Module]
    uses_varlen: bool
    uses_kan: bool


@dataclass
class VariantResult:
    """Measured outcome for one variant."""

    name: str
    uses_varlen: bool
    uses_kan: bool
    params_m: float
    char_accuracy: float
    sequence_accuracy: float
    length_accuracy: float
    cer: float
    latency_ms: float
    throughput: float


def build_variants(dim: int, num_akan_blocks: int) -> List[VariantSpec]:
    """Construct the four-variant registry.

    Parameters
    ----------
    dim:
        Embedding width for the KAN variants.
    num_akan_blocks:
        Number of A-KAN blocks in the *full* model.

    Returns
    -------
    list of VariantSpec
        The ordered ablation grid.
    """
    return [
        VariantSpec(
            name="Baseline",
            kind="fixed",
            builder=lambda v, L: DeepCaptcha(
                num_classes=v, length=L, in_channels=3, input_size=(48, 320),
            ),
            uses_varlen=False,
            uses_kan=False,
        ),
        VariantSpec(
            name="Baseline + VarLen",
            kind="ctc",
            builder=lambda v, L: VariableLengthDeepCaptcha(num_classes=v),
            uses_varlen=True,
            uses_kan=False,
        ),
        VariantSpec(
            name="Baseline + KAN",
            kind="ctc",
            builder=lambda v, L: VLKAN(
                num_classes=v, dim=dim, num_akan_blocks=0, dynamic_fusion=False,
            ),
            uses_varlen=True,
            uses_kan=True,
        ),
        VariantSpec(
            name="Full VL-KAN",
            kind="ctc",
            builder=lambda v, L: VLKAN(
                num_classes=v, dim=dim, num_akan_blocks=num_akan_blocks,
                dynamic_fusion=True,
            ),
            uses_varlen=True,
            uses_kan=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Native-regime training / evaluation
# ---------------------------------------------------------------------------

def _count_params_m(model: nn.Module) -> float:
    """Trainable parameter count in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def _seq_len_of(model: nn.Module, default: int = 40) -> int:
    """Best-effort retrieval of a CTC model's output time length ``T``."""
    return int(getattr(model, "seq_len", default))


def _ctc_train_step(
    model: nn.Module, batch: CaptchaBatch, device: torch.device,
) -> Tensor:
    """One CTC training step for a sequence model; returns the scalar loss."""
    out = model(batch.images)
    log_probs = out["ctc_logits"].float().log_softmax(dim=-1)  # (B, T, V+1)
    b, t, _ = log_probs.shape
    log_probs = log_probs.permute(1, 0, 2).contiguous()        # (T, B, V+1)
    input_lengths = torch.full((b,), t, dtype=torch.long, device=device)
    return F.ctc_loss(
        log_probs, batch.targets, input_lengths, batch.target_lengths,
        blank=0, zero_infinity=True,
    )


def _fixed_train_step(
    model: nn.Module, batch: CaptchaBatch, device: torch.device, max_length: int,
) -> Tensor:
    """One cross-entropy step for the fixed-length baseline (ignore_index pad)."""
    logits = model(batch.images)                # (B, L, V)
    b, L, V = logits.shape
    targets = torch.full((b, L), -100, dtype=torch.long, device=device)
    # CTC targets are flat; rebuild per-sample 0-based class ids for CE.
    cursor = 0
    for i in range(b):
        n = int(batch.target_lengths[i])
        seq = batch.targets[cursor:cursor + n] - 1   # 1..V -> 0..V-1
        cursor += n
        fill = min(n, L)
        targets[i, :fill] = seq[:fill]
    return F.cross_entropy(
        logits.reshape(b * L, V), targets.reshape(b * L), ignore_index=-100,
    )


def _decode(model: nn.Module, spec: VariantSpec, batch: CaptchaBatch,
            vocabulary: Vocabulary) -> List[str]:
    """Decode a batch's predictions according to the variant kind."""
    out = model(batch.images)
    if spec.kind == "fixed":
        return fixed_length_decode(out, vocabulary)
    return greedy_ctc_decode(out["ctc_logits"], vocabulary)


def _build_loader(
    vocabulary: Vocabulary, charset: str, samples: int, batch_size: int,
    seed: int, epoch: int,
) -> DataLoader:
    """Build a deterministic procedural loader for ablation."""
    dataset = CaptchaDataset(
        vocabulary=vocabulary, charset=charset, length_range=(3, 10),
        samples_per_epoch=samples, base_seed=seed, epoch=epoch,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=make_collate_fn(vocabulary), drop_last=True,
    )


def _evaluate_variant(
    spec: VariantSpec,
    vocabulary: Vocabulary,
    device: torch.device,
    max_length: int,
    train_steps: int,
    eval_samples: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> VariantResult:
    """Warm-train (optionally), then evaluate one variant end-to-end."""
    model = spec.builder(vocabulary.size, max_length).to(device)

    # --- Native-regime warm-up training. --------------------------------
    if train_steps > 0:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loader = _build_loader(
            vocabulary, NUMERIC_CHARSET, train_steps * batch_size,
            batch_size, seed, epoch=0,
        )
        step = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            if spec.kind == "fixed":
                loss = _fixed_train_step(model, batch, device, max_length)
            else:
                loss = _ctc_train_step(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            if step >= train_steps:
                break

    # --- Held-out evaluation (fixed seed, disjoint epoch index). --------
    model.eval()
    metrics = MetricAccumulator()
    latency = LatencyTracker()
    eval_loader = _build_loader(
        vocabulary, NUMERIC_CHARSET, eval_samples, batch_size,
        seed + 9999, epoch=1,
    )
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            with latency.measure(num_items=batch.images.shape[0]):
                preds = _decode(model, spec, batch, vocabulary)
            metrics.update(preds, batch.texts)

    rec = metrics.compute()
    lat = latency.compute()
    return VariantResult(
        name=spec.name,
        uses_varlen=spec.uses_varlen,
        uses_kan=spec.uses_kan,
        params_m=_count_params_m(model),
        char_accuracy=rec.char_accuracy,
        sequence_accuracy=rec.sequence_accuracy,
        length_accuracy=rec.length_accuracy,
        cer=rec.cer,
        latency_ms=lat.mean_ms,
        throughput=lat.throughput,
    )


# ---------------------------------------------------------------------------
# Markdown reporting
# ---------------------------------------------------------------------------

def _bool_mark(flag: bool) -> str:
    """Render a capability flag as a check / dash."""
    return "yes" if flag else "-"


def results_to_markdown(results: List[VariantResult]) -> str:
    """Render variant results as Markdown comparison tables.

    Parameters
    ----------
    results:
        The measured variant outcomes, in registry order.

    Returns
    -------
    str
        A Markdown document containing a capability matrix and a metrics table.
    """
    lines: List[str] = []
    lines.append("# VL-KAN Structural Ablation\n")

    lines.append("## Architectural capability matrix\n")
    lines.append("| Variant | Variable length | KAN modelling | Params (M) |")
    lines.append("| --- | :---: | :---: | ---: |")
    for r in results:
        lines.append(
            f"| {r.name} | {_bool_mark(r.uses_varlen)} | "
            f"{_bool_mark(r.uses_kan)} | {r.params_m:.2f} |"
        )
    lines.append("")

    lines.append("## Recognition & performance metrics\n")
    lines.append(
        "| Variant | CAcc | SAcc | LAcc | CER | Latency (ms/img) | "
        "Throughput (img/s) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            f"| {r.name} | {r.char_accuracy * 100:.2f}% | "
            f"{r.sequence_accuracy * 100:.2f}% | {r.length_accuracy * 100:.2f}% | "
            f"{r.cer:.4f} | {r.latency_ms:.2f} | {r.throughput:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_ablation(args: argparse.Namespace) -> List[VariantResult]:
    """Run the full ablation grid and return the per-variant results."""
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    vocabulary = build_vocabulary("numeric") if args.numeric_only \
        else build_vocabulary("alphanumeric")

    variants = build_variants(dim=args.dim, num_akan_blocks=args.num_akan_blocks)
    results: List[VariantResult] = []
    for spec in variants:
        print(f"[ablation] evaluating: {spec.name}")
        result = _evaluate_variant(
            spec=spec,
            vocabulary=vocabulary,
            device=device,
            max_length=args.max_length,
            train_steps=args.train_steps,
            eval_samples=args.eval_samples,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
        results.append(result)
        print(
            f"  CAcc={result.char_accuracy * 100:.2f}% "
            f"SAcc={result.sequence_accuracy * 100:.2f}% "
            f"LAcc={result.length_accuracy * 100:.2f}% "
            f"lat={result.latency_ms:.2f}ms params={result.params_m:.2f}M"
        )

    markdown = results_to_markdown(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    print(f"\n[ablation] wrote Markdown report -> {args.output}\n")
    print(markdown)
    return results


def parse_args() -> argparse.Namespace:
    """Parse ablation command-line arguments."""
    p = argparse.ArgumentParser(description="Run VL-KAN structural ablations.")
    p.add_argument("--device", default=None, help="cuda | cpu (auto if unset).")
    p.add_argument("--numeric-only", action="store_true",
                   help="Use the 10-class numeric vocabulary.")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--num-akan-blocks", type=int, default=4)
    p.add_argument("--max-length", type=int, default=10)
    p.add_argument("--train-steps", type=int, default=200,
                   help="Warm-up steps per variant (0 to evaluate untrained).")
    p.add_argument("--eval-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", default="artifacts/ablation.md")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU run for pipeline validation.")
    return p.parse_args()


def main() -> None:
    """Program entry."""
    args = parse_args()
    if args.smoke:
        args.device = "cpu"
        args.numeric_only = True
        args.dim = 64
        args.num_akan_blocks = 1
        args.train_steps = 2
        args.eval_samples = 16
        args.batch_size = 4
        args.output = "artifacts/ablation_smoke.md"
    run_ablation(args)


if __name__ == "__main__":
    main()
