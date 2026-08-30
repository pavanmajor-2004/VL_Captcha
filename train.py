r"""VL-KAN training entry point.

Builds the model, optimizer, warmup+cosine schedule, LAD-CTC loss, the optional
ACA surrogate, and the three-phase curriculum, then launches :class:`Trainer`.

Examples
--------
Fresh run::

    python train.py --output-dir artifacts/run1

Resume from the last checkpoint::

    python train.py --output-dir artifacts/run1 --resume artifacts/run1/checkpoints/last.pt

A tiny CPU dry-run for validation::

    python train.py --smoke
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List

import torch
import torch.nn as nn

from datasets.vocabulary import build_vocabulary
from losses.ctc_lad import LADCTCConfig, LengthAwareDynamicCTCLoss
from models.kan_captcha import VLKAN
from models.variable_length_deepcaptcha import VariableLengthDeepCaptcha
from training.trainer import (
    PhaseConfig,
    Trainer,
    TrainerConfig,
    create_writer,
)

NUMERIC_CHARSET = "0123456789"


class CTCLogitsSurrogate(nn.Module):
    """Adapter exposing a dict-output recognizer as a bare ``ctc_logits`` tensor.

    The PGD adversary expects ``model(images) -> (B, T, V+1)``; the CRNN-style
    :class:`VariableLengthDeepCaptcha` returns a dict, so this thin wrapper selects
    the CTC logits.

    Parameters
    ----------
    model:
        A recognizer whose ``forward`` returns ``{"ctc_logits": ...}``.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the CTC logits of the wrapped model."""
        return self.model(x)["ctc_logits"]


def build_phases(args: argparse.Namespace, charset_alnum: str) -> List[PhaseConfig]:
    """Construct the three-phase curriculum from CLI arguments."""
    return [
        PhaseConfig(
            name="P1-numeric-short",
            charset=NUMERIC_CHARSET,
            length_range=(3, 5),
            epochs=args.phase1_epochs,
            aca_enabled=False,
        ),
        PhaseConfig(
            name="P2-numeric-full",
            charset=NUMERIC_CHARSET,
            length_range=(3, 10),
            epochs=args.phase2_epochs,
            aca_enabled=False,
        ),
        PhaseConfig(
            name="P3-alphanumeric-aca",
            charset=charset_alnum,
            length_range=(3, 10),
            epochs=args.phase3_epochs,
            aca_enabled=True,
            aca_fraction=args.aca_fraction,
        ),
    ]


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to ``min_lr_ratio * base_lr``.

    Parameters
    ----------
    optimizer:
        The optimizer to schedule.
    warmup_steps:
        Number of linear-warmup optimizer steps.
    total_steps:
        Total optimizer steps across the whole curriculum.
    min_lr_ratio:
        Floor as a fraction of the base LR.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        The configured scheduler.
    """
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train VL-KAN.")
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--resume", default=None, help="Path to a checkpoint to resume.")
    p.add_argument("--device", default=None, help="cuda | cpu (auto if unset).")
    p.add_argument("--alphabet", default="alphanumeric",
                   choices=["alphanumeric"], help="Encoding vocabulary.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--samples-per-epoch", type=int, default=100_000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--amp-dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--phase1-epochs", type=int, default=20)
    p.add_argument("--phase2-epochs", type=int, default=40)
    p.add_argument("--phase3-epochs", type=int, default=60)
    p.add_argument("--aca-fraction", type=float, default=0.1)
    p.add_argument("--num-akan-blocks", type=int, default=4)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--surrogate-ckpt", default=None,
                   help="Optional weights for the frozen ACA surrogate.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU dry-run with minimal steps for validation.")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of samples_per_epoch used as validation set (default 0.2).")
    p.add_argument("--val-every", type=int, default=1,
                   help="Run validation every N epochs within each phase (default 1).")
    return p.parse_args()


def main() -> None:
    """Program entry: assemble components and run the curriculum."""
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.smoke:
        # Shrink everything to a few steps so the full pipeline runs on CPU.
        args.device = "cpu"
        args.batch_size = 4
        args.samples_per_epoch = 16
        args.num_workers = 0
        args.amp_dtype = "float32"
        args.phase1_epochs = args.phase2_epochs = args.phase3_epochs = 1
        args.num_akan_blocks = 1

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    vocab = build_vocabulary(args.alphabet)          # 62-class alphanumeric
    num_classes = vocab.size

    model = VLKAN(
        num_classes=num_classes,
        min_length=3,
        max_length=10,
        dim=args.dim,
        num_akan_blocks=args.num_akan_blocks,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Total optimizer steps across the curriculum (for the cosine horizon).
    steps_per_epoch = max(1, args.samples_per_epoch // args.batch_size)
    total_epochs = args.phase1_epochs + args.phase2_epochs + args.phase3_epochs
    total_steps = max(1, total_epochs * steps_per_epoch // max(1, args.grad_accum))
    warmup_steps = 1 if args.smoke else 500
    scheduler = build_lr_scheduler(
        optimizer, warmup_steps, total_steps, min_lr_ratio=1e-6 / args.lr,
    )

    loss_fn = LengthAwareDynamicCTCLoss(
        LADCTCConfig(blank=0, min_length=3, lambda_max=0.5,
                     warmup_steps=max(1, total_steps // 2)),
    )

    # ACA surrogate: a frozen CRNN-style recognizer (optionally pre-trained).
    surrogate = CTCLogitsSurrogate(VariableLengthDeepCaptcha(num_classes=num_classes))
    if args.surrogate_ckpt and os.path.isfile(args.surrogate_ckpt):
        sd = torch.load(args.surrogate_ckpt, map_location="cpu", weights_only=False)
        surrogate.model.load_state_dict(sd.get("model", sd))
        print(f"[train] loaded surrogate weights from {args.surrogate_ckpt}")

    config = TrainerConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        grad_accum_steps=args.grad_accum,
        amp_dtype=args.amp_dtype,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        num_workers=args.num_workers,
        base_seed=args.seed,
        output_dir=args.output_dir,
        log_every=1 if args.smoke else 20,
        save_every=4 if args.smoke else 1000,
        val_split=args.val_split,
        val_every=args.val_every,
    )

    phases = build_phases(args, vocab.alphabet)
    writer = create_writer(os.path.join(args.output_dir, "logs"))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        vocabulary=vocab,
        phases=phases,
        config=config,
        device=device,
        surrogate=surrogate,
        writer=writer,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume, resume_training=True)

    trainer.train()
    print("[train] curriculum complete.")


if __name__ == "__main__":
    main()
