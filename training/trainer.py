r"""End-to-end training engine for VL-KAN (PyTorch 2.x).

Implements a scalable trainer with:

* **Mixed precision** via :mod:`torch.amp` (defaulting to ``bfloat16``; ``float16``
  transparently enables a :class:`~torch.amp.GradScaler`).
* **Gradient accumulation** with automatic global-norm clipping (``<= 5.0``).
* A clean **multi-task objective** ``alpha * LAD-CTC + beta * length_CE +
  gamma * boundary_BCE`` (defaults ``1.0 / 0.3 / 0.2``).
* A **three-phase curriculum** (short numeric -> extended numeric ->
  alphanumeric + online ACA) driven by per-phase :class:`PhaseConfig` objects.
* **Native checkpointing** capturing model / optimizer / scheduler / loss / RNG /
  curriculum position for seamless pause-and-resume, plus structured TensorBoard
  telemetry.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from data.adversarial import AdversarialBatch, PGDAdversary
from datasets.augmentations import CaptchaAugmentor
from datasets.dataset import CaptchaBatch, CaptchaDataset, make_collate_fn
from datasets.vocabulary import Vocabulary
from evaluation.decoding import greedy_ctc_decode
from evaluation.metrics import MetricAccumulator, RecognitionMetrics
from losses.ctc_lad import LengthAwareDynamicCTCLoss
from models.heads.boundary_head import build_soft_boundary_targets

__all__ = ["PhaseConfig", "TrainerConfig", "Trainer", "NullWriter", "create_writer"]


# ---------------------------------------------------------------------------
# TensorBoard writer (with graceful fallback)
# ---------------------------------------------------------------------------

class NullWriter:
    """No-op stand-in used when TensorBoard is unavailable."""

    def add_scalar(self, *args, **kwargs) -> None:  # noqa: D401, ANN002, ANN003
        """Ignore scalar logging."""

    def add_text(self, *args, **kwargs) -> None:
        """Ignore text logging."""

    def flush(self) -> None:
        """Ignore flush."""

    def close(self) -> None:
        """Ignore close."""


def create_writer(log_dir: str):
    """Return a TensorBoard ``SummaryWriter`` or a :class:`NullWriter` fallback.

    Parameters
    ----------
    log_dir:
        Directory for the event files.

    Returns
    -------
    object
        A ``SummaryWriter`` if ``torch.utils.tensorboard`` imports, else a
        :class:`NullWriter`.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[trainer] TensorBoard unavailable ({exc}); using NullWriter.")
        return NullWriter()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PhaseConfig:
    """One curriculum phase.

    Attributes
    ----------
    name:
        Human-readable phase label.
    charset:
        Glyphs the renderer may sample in this phase.
    length_range:
        Inclusive ``(min, max)`` sequence length.
    epochs:
        Number of epochs to train in this phase.
    aca_enabled:
        Whether online Adversarial CAPTCHA Augmentation is active.
    aca_fraction:
        Fraction of each batch to perturb when ``aca_enabled``.
    """

    name: str
    charset: str
    length_range: tuple[int, int]
    epochs: int
    aca_enabled: bool = False
    aca_fraction: float = 0.1


@dataclass
class TrainerConfig:
    """Top-level training configuration.

    Attributes are grouped into optimization, precision, multi-task weighting,
    data, ACA, and bookkeeping. All have sensible paper-aligned defaults.
    """

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.05
    min_lr: float = 1e-6
    warmup_steps: int = 500
    grad_accum_steps: int = 1
    max_grad_norm: float = 5.0

    # Precision
    amp_dtype: str = "bfloat16"          # "bfloat16" | "float16" | "float32"

    # Multi-task loss weights
    alpha: float = 1.0                   # LAD-CTC
    beta: float = 0.3                    # length cross-entropy
    gamma: float = 0.2                   # boundary BCE
    reg_weight: float = 1e-4             # KAN sparsity regularizer
    boundary_sigma: float = 1.5          # soft boundary target width

    # Data
    batch_size: int = 256
    samples_per_epoch: int = 100_000
    num_workers: int = 4
    image_size: tuple[int, int] = (320, 48)
    base_seed: int = 1234

    # ACA (PGD surrogate attack)
    aca_epsilon: float = 8.0 / 255.0
    aca_alpha: float = 2.0 / 255.0
    aca_steps: int = 7

    # Bookkeeping
    output_dir: str = "artifacts"
    log_every: int = 20
    save_every: int = 1000
    min_length: int = 3
    max_length: int = 10

    # Validation
    val_split: float = 0.2           # fraction of samples_per_epoch used for val
    val_every: int = 1               # run validation every N epochs (per phase)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Drives multi-phase, mixed-precision VL-KAN training with checkpointing.

    Parameters
    ----------
    model:
        The VL-KAN model (returns ``ctc_logits`` / ``length_logits`` /
        ``boundary_logits``).
    optimizer:
        Pre-built optimizer (e.g. ``AdamW``).
    scheduler:
        Per-optimizer-step LR scheduler.
    loss_fn:
        The LAD-CTC loss module.
    vocabulary:
        Encoding vocabulary matching the model's CTC head.
    phases:
        Ordered curriculum phases.
    config:
        The :class:`TrainerConfig`.
    device:
        Compute device.
    surrogate:
        Optional frozen surrogate emitting ``(B, T, V+1)`` logits for ACA. Required
        only if any phase has ``aca_enabled``.
    writer:
        TensorBoard writer (or :class:`NullWriter`). Created automatically if
        ``None``.
    font_paths:
        Optional explicit font list for the renderer.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: LengthAwareDynamicCTCLoss,
        vocabulary: Vocabulary,
        phases: List[PhaseConfig],
        config: TrainerConfig,
        device: torch.device,
        surrogate: Optional[nn.Module] = None,
        writer: Optional[object] = None,
        font_paths: Optional[List[str]] = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn.to(device)
        self.vocabulary = vocabulary
        self.phases = phases
        self.cfg = config
        self.device = device
        self.font_paths = font_paths

        # Precision setup. bf16 needs no GradScaler (fp32 exponent range); fp16 does.
        self._amp_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[config.amp_dtype]
        self._amp_enabled = self._amp_dtype in (torch.bfloat16, torch.float16)
        use_scaler = self._amp_dtype is torch.float16
        self.scaler = torch.amp.GradScaler(
            device.type, enabled=use_scaler
        )

        # On-device batched augmentation + its own reproducible RNG.
        self.augmentor = CaptchaAugmentor().to(device)
        self.aug_generator = torch.Generator(device=device)
        self.aug_generator.manual_seed(config.base_seed + 777)

        # ACA attacker (lazily required only when a phase enables it).
        self.surrogate = surrogate
        self.adversary: Optional[PGDAdversary] = None
        if surrogate is not None:
            self.adversary = PGDAdversary(
                surrogate.to(device),
                epsilon=config.aca_epsilon,
                alpha=config.aca_alpha,
                steps=config.aca_steps,
                blank=0,
            ).to(device)

        self.writer = writer if writer is not None else create_writer(
            os.path.join(config.output_dir, "logs")
        )
        self.ckpt_dir = os.path.join(config.output_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Training position (mutated as we go; persisted in checkpoints).
        self.global_step = 0
        self.start_phase = 0
        self.start_epoch = 0
        self.best_val_sacc: float = 0.0  # tracks best sequence accuracy for checkpointing

    def _build_val_loader(self, phase: PhaseConfig, epoch: int) -> DataLoader:
        """Construct a deterministic validation DataLoader for ``phase``.

        The validation set is drawn from a *different* seed space than training
        (offset by ``0xDEAD_BEEF``) so the two splits never share identical
        samples even across resumes. Its size is ``val_split`` × ``samples_per_epoch``
        and it uses ``shuffle=False`` to keep results reproducible.
        """
        val_samples = max(1, int(self.cfg.samples_per_epoch * self.cfg.val_split))
        # Use a seed offset that is orthogonal to the training seed space.
        val_seed = (self.cfg.base_seed ^ 0xDEAD_BEEF) & 0x7FFF_FFFF_FFFF_FFFF
        dataset = CaptchaDataset(
            vocabulary=self.vocabulary,
            charset=phase.charset,
            length_range=phase.length_range,
            samples_per_epoch=val_samples,
            image_size=self.cfg.image_size,
            font_paths=self.font_paths,
            base_seed=val_seed,
            epoch=epoch,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=make_collate_fn(self.vocabulary),
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=self.cfg.num_workers > 0,
        )

    def _validate_epoch(
        self, phase: PhaseConfig, phase_idx: int, epoch: int
    ) -> RecognitionMetrics:
        """Run a full validation pass and return aggregate recognition metrics.

        The model is set to eval mode and no gradients are computed. Augmentation
        and ACA are *not* applied — validation uses clean images to measure the
        model's true generalisation. Greedy CTC decoding is used for speed;
        it is a consistent proxy for the beam-search numbers.

        The metrics (CAcc, SAcc, LAcc, CER) and the aggregate validation loss are
        written to TensorBoard under the ``val/`` prefix.
        """
        self.model.eval()
        loader = self._build_val_loader(phase, epoch)
        seq_len = self.model.seq_len
        accumulator = MetricAccumulator()
        total_val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                b = batch.images.shape[0]
                input_lengths = torch.full(
                    (b,), seq_len, dtype=torch.long, device=self.device
                )

                # Forward (AMP for speed, but no backward).
                with torch.autocast(
                    self.device.type, dtype=self._amp_dtype, enabled=self._amp_enabled
                ):
                    out = self.model(batch.images)

                # Greedy CTC decode → recognition metrics.
                preds = greedy_ctc_decode(out["ctc_logits"], self.vocabulary)
                accumulator.update(preds, batch.texts)

                # Validation loss (LAD-CTC component only; no regularizer noise).
                lad = self.loss_fn(
                    out["ctc_logits"],
                    out["length_logits"],
                    batch.targets,
                    input_lengths,
                    batch.target_lengths,
                )
                total_val_loss += float(lad["loss"])
                num_batches += 1

        metrics = accumulator.compute()
        avg_val_loss = total_val_loss / max(1, num_batches)

        # --- TensorBoard logging -------------------------------------------
        self.writer.add_scalar("val/loss_lad_ctc", avg_val_loss, self.global_step)
        self.writer.add_scalar("val/CAcc", metrics.char_accuracy, self.global_step)
        self.writer.add_scalar("val/SAcc", metrics.sequence_accuracy, self.global_step)
        self.writer.add_scalar("val/LAcc", metrics.length_accuracy, self.global_step)
        self.writer.add_scalar("val/CER", metrics.cer, self.global_step)

        # --- Console summary -----------------------------------------------
        print(
            f"[{phase.name}] VAL ep{epoch} step{self.global_step} | "
            f"loss={avg_val_loss:.4f} | "
            f"CAcc={metrics.char_accuracy:.4f} "
            f"SAcc={metrics.sequence_accuracy:.4f} "
            f"LAcc={metrics.length_accuracy:.4f} "
            f"CER={metrics.cer:.4f} "
            f"(N={metrics.num_samples})"
        )

        # --- Save best checkpoint if SAcc improved -------------------------
        if metrics.sequence_accuracy > self.best_val_sacc:
            self.best_val_sacc = metrics.sequence_accuracy
            best_path = self.save_checkpoint(phase_idx, epoch, tag="best")
            print(
                f"[{phase.name}] New best SAcc={self.best_val_sacc:.4f} "
                f"→ saved {best_path}"
            )

        self.writer.flush()
        return metrics

    # --- Data ----------------------------------------------------------------

    def _build_loader(self, phase: PhaseConfig, epoch: int) -> DataLoader:
        """Construct a DataLoader for ``phase`` at a given ``epoch``."""
        dataset = CaptchaDataset(
            vocabulary=self.vocabulary,
            charset=phase.charset,
            length_range=phase.length_range,
            samples_per_epoch=self.cfg.samples_per_epoch,
            image_size=self.cfg.image_size,
            font_paths=self.font_paths,
            base_seed=self.cfg.base_seed,
            epoch=epoch,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=make_collate_fn(self.vocabulary),
            drop_last=True,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=self.cfg.num_workers > 0,
        )

    # --- Loss ----------------------------------------------------------------

    def _compute_losses(
        self, batch: CaptchaBatch, seq_len: int,
    ) -> tuple[Tensor, Dict[str, float]]:
        """Forward the model and assemble the weighted multi-task loss.

        Parameters
        ----------
        batch:
            A device-resident :class:`CaptchaBatch` (images already augmented).
        seq_len:
            The model's output time length ``T`` (for boundary targets / CTC).

        Returns
        -------
        tuple
            ``(total_loss, scalar_logs)``.
        """
        cfg = self.cfg
        b = batch.images.shape[0]
        input_lengths = torch.full((b,), seq_len, dtype=torch.long, device=self.device)

        with torch.autocast(self.device.type, dtype=self._amp_dtype,
                            enabled=self._amp_enabled):
            out = self.model(batch.images)

        ctc_logits = out["ctc_logits"]
        length_logits = out["length_logits"]
        boundary_logits = out["boundary_logits"]

        # --- (1) LAD-CTC term (computed in fp32 internally). ----------------
        lad = self.loss_fn(
            ctc_logits, length_logits, batch.targets,
            input_lengths, batch.target_lengths,
        )

        # --- (2) Explicit length cross-entropy. -----------------------------
        length_idx = (batch.target_lengths - cfg.min_length).clamp(
            0, length_logits.shape[1] - 1
        )
        length_ce = F.cross_entropy(length_logits.float(), length_idx)

        # --- (3) Soft Gaussian boundary BCE. --------------------------------
        soft_targets = build_soft_boundary_targets(
            batch.boundaries, seq_len=seq_len, sigma=cfg.boundary_sigma,
            device=self.device, dtype=torch.float32,
        )
        from models.heads.boundary_head import BoundaryHead
        boundary_bce = BoundaryHead.loss(boundary_logits, soft_targets)

        # --- (4) Optional KAN sparsity regularizer. -------------------------
        reg = torch.zeros((), device=self.device)
        if cfg.reg_weight > 0 and hasattr(self.model, "regularization_loss"):
            reg = self.model.regularization_loss().float()

        total = (
            cfg.alpha * lad["loss"]
            + cfg.beta * length_ce
            + cfg.gamma * boundary_bce
            + cfg.reg_weight * reg
        )

        logs = {
            "loss/total": float(total.detach()),
            "loss/lad_ctc": float(lad["loss"].detach()),
            "loss/ctc": float(lad["ctc"]),
            "loss/length_nll": float(lad["length"]),
            "loss/length_ce": float(length_ce.detach()),
            "loss/boundary_bce": float(boundary_bce.detach()),
            "loss/kan_reg": float(reg.detach()),
            "sched/lambda": float(lad["lambda"]),
        }
        return total, logs

    # --- ACA -----------------------------------------------------------------

    def _apply_aca(self, batch: CaptchaBatch, phase: PhaseConfig) -> Tensor:
        """Return possibly-perturbed images for an ACA-enabled phase."""
        if not phase.aca_enabled or self.adversary is None:
            return batch.images
        b = batch.images.shape[0]
        adv_batch = AdversarialBatch(
            images=batch.images,
            targets=batch.targets,
            input_lengths=torch.full((b,), self.model.seq_len,
                                     dtype=torch.long, device=self.device),
            target_lengths=batch.target_lengths,
        )
        return self.adversary.mix(adv_batch, fraction=phase.aca_fraction)

    # --- Train loop ----------------------------------------------------------

    def _train_epoch(self, phase: PhaseConfig, phase_idx: int, epoch: int) -> None:
        """Run a single epoch of a curriculum phase."""
        self.model.train()
        loader = self._build_loader(phase, epoch)
        seq_len = self.model.seq_len
        accum = max(1, self.cfg.grad_accum_steps)

        self.optimizer.zero_grad(set_to_none=True)
        running = 0.0
        t0 = time.time()

        for it, batch in enumerate(loader):
            batch = batch.to(self.device, non_blocking=True)

            # On-device batched augmentation, then optional ACA perturbation.
            with torch.no_grad():
                aug_images = self.augmentor(batch.images, generator=self.aug_generator)
            batch.images = aug_images
            batch.images = self._apply_aca(batch, phase)

            total, logs = self._compute_losses(batch, seq_len)
            loss = total / accum

            # Backward (scaled only under fp16).
            self.scaler.scale(loss).backward()
            running += logs["loss/total"]

            is_step = ((it + 1) % accum == 0)
            if is_step:
                # Unscale before clipping so the norm is computed in real units.
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.loss_fn.step()  # advance LAD-CTC lambda anneal
                self.global_step += 1

                if self.global_step % self.cfg.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    imgs_per_s = (
                        self.cfg.log_every * self.cfg.batch_size * accum
                        / max(1e-6, time.time() - t0)
                    )
                    for key, val in logs.items():
                        self.writer.add_scalar(key, val, self.global_step)
                    self.writer.add_scalar("sched/lr", lr, self.global_step)
                    self.writer.add_scalar("opt/grad_norm", float(grad_norm),
                                           self.global_step)
                    self.writer.add_scalar("perf/imgs_per_s", imgs_per_s,
                                           self.global_step)
                    print(
                        f"[{phase.name}] ep{epoch} step{self.global_step} "
                        f"loss={logs['loss/total']:.3f} "
                        f"ctc={logs['loss/ctc']:.3f} "
                        f"len_ce={logs['loss/length_ce']:.3f} "
                        f"bdy={logs['loss/boundary_bce']:.3f} "
                        f"lr={lr:.2e} |g|={float(grad_norm):.2f} "
                        f"{imgs_per_s:.0f} img/s"
                    )
                    t0 = time.time()

                if self.global_step % self.cfg.save_every == 0:
                    self.save_checkpoint(phase_idx, epoch, tag="last")

        self.writer.flush()

    def train(self) -> None:
        """Execute the full curriculum from the current resume position."""
        for phase_idx in range(self.start_phase, len(self.phases)):
            phase = self.phases[phase_idx]
            if phase.aca_enabled and self.adversary is None:
                raise RuntimeError(
                    f"Phase '{phase.name}' enables ACA but no surrogate was provided."
                )
            self.writer.add_text(
                "curriculum/phase",
                f"{phase_idx}: {phase.name} charset={len(phase.charset)} "
                f"len={phase.length_range} aca={phase.aca_enabled}",
                self.global_step,
            )
            start_ep = self.start_epoch if phase_idx == self.start_phase else 0
            for epoch in range(start_ep, phase.epochs):
                self._train_epoch(phase, phase_idx, epoch)
                self.save_checkpoint(phase_idx, epoch + 1, tag="last")
                # Run validation every val_every epochs.
                if (epoch + 1) % max(1, self.cfg.val_every) == 0:
                    self._validate_epoch(phase, phase_idx, epoch + 1)
            # Reset per-phase epoch cursor once a phase completes.
            self.start_epoch = 0
        self.writer.close()

    # --- Checkpointing -------------------------------------------------------

    def save_checkpoint(self, phase_idx: int, epoch: int, tag: str = "last") -> str:
        """Persist full training state for seamless resume.

        Parameters
        ----------
        phase_idx, epoch:
            Curriculum position to resume *from* on reload.
        tag:
            Filename tag (e.g. ``"last"`` or ``"best"``).

        Returns
        -------
        str
            The checkpoint path written.
        """
        path = os.path.join(self.ckpt_dir, f"{tag}.pt")
        state = {
            "global_step": self.global_step,
            "phase_idx": phase_idx,
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss_fn": self.loss_fn.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val_sacc": self.best_val_sacc,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
                "python": random.getstate(),
                "aug_generator": self.aug_generator.get_state(),
            },
            "config": self.cfg.__dict__,
        }
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)  # atomic swap guards against mid-write crashes
        return path

    def load_checkpoint(self, path: str, resume_training: bool = True) -> None:
        """Restore training state saved by :meth:`save_checkpoint`.

        Parameters
        ----------
        path:
            Checkpoint file path.
        resume_training:
            If ``True``, also restore optimizer/scheduler/RNG and the curriculum
            cursor so :meth:`train` continues exactly where it left off. If
            ``False``, only model weights are loaded (e.g. for evaluation).
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        if not resume_training:
            return

        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.loss_fn.load_state_dict(state["loss_fn"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step = int(state["global_step"])
        self.start_phase = int(state["phase_idx"])
        self.start_epoch = int(state["epoch"])
        self.best_val_sacc = float(state.get("best_val_sacc", 0.0))

        rng = state.get("rng", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("aug_generator") is not None:
            self.aug_generator.set_state(rng["aug_generator"])
        print(
            f"[trainer] resumed from {path}: phase={self.start_phase} "
            f"epoch={self.start_epoch} step={self.global_step}"
        )
