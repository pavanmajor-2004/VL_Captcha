r"""Length-Aware Dynamic CTC Loss (LAD-CTC) -- Contribution 3.

Standard CTC marginalizes over every monotonic alignment ``pi`` that collapses
(via the blank-removing map ``B``) to the target ``y``::

    L_ctc = -log sum_{pi in B^{-1}(y)} prod_t p(pi_t | z_t)

LAD-CTC couples the recognizer's **length-prediction branch** into this objective
rather than only at decode time. Each target is re-weighted by the log-probability
the length head assigns to that target's true length, with an annealed strength
``lambda``::

    L_lad = E[ ctc(z, y) - lambda * log p(ell = |y| | I) ]

Equivalently this is the CTC term plus ``lambda`` times the length cross-entropy,
so minimizing it simultaneously sharpens character alignment and the length
posterior. ``lambda`` is annealed ``0 -> lambda_max`` over training so the length
prior is introduced gradually once the CTC head is no longer random.

Mixed-precision notes
---------------------
* CTC and all log-domain reductions are evaluated in **fp32** even under autocast;
  ``log_softmax`` and the CTC forward-backward are numerically fragile in fp16.
* ``zero_infinity=True`` discards the ``+inf`` losses that occur when an input is
  too short to contain its target, preventing NaN gradients.
* The length log-prob is gathered with ``log_softmax`` (not ``log(softmax)``) for a
  stable, single-kernel reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["LADCTCConfig", "LengthAwareDynamicCTCLoss"]


def _linear_anneal(lambda_max: float, warmup_steps: int) -> Callable[[int], float]:
    """Return a schedule ramping ``0 -> lambda_max`` linearly over warmup steps."""
    warmup_steps = max(1, warmup_steps)

    def schedule(step: int) -> float:
        return lambda_max * min(1.0, step / warmup_steps)

    return schedule


@dataclass
class LADCTCConfig:
    """Configuration for :class:`LengthAwareDynamicCTCLoss`.

    Attributes
    ----------
    blank:
        CTC blank index (``0`` by project convention).
    min_length:
        Smallest admissible target length (``3``); used to map a length to its
        class index in the length head's output.
    lambda_max:
        Maximum length-prior strength after annealing.
    warmup_steps:
        Number of steps over which ``lambda`` ramps from 0 to ``lambda_max``.
    """

    blank: int = 0
    min_length: int = 3
    lambda_max: float = 0.5
    warmup_steps: int = 50_000


class LengthAwareDynamicCTCLoss(nn.Module):
    """CTC loss re-weighted online by the length branch's log-probability.

    Parameters
    ----------
    config:
        A :class:`LADCTCConfig`. If ``None`` the defaults are used.
    lambda_schedule:
        Optional custom callable ``step -> lambda``. Overrides the linear anneal
        derived from ``config`` when provided.
    """

    def __init__(
        self,
        config: Optional[LADCTCConfig] = None,
        lambda_schedule: Optional[Callable[[int], float]] = None,
    ) -> None:
        super().__init__()
        self.config = config or LADCTCConfig()
        self._schedule = lambda_schedule or _linear_anneal(
            self.config.lambda_max, self.config.warmup_steps
        )
        # Training-step counter persisted with the module state.
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    # --- Schedule ------------------------------------------------------------

    @property
    def current_lambda(self) -> float:
        """Current length-prior weight ``lambda`` for the recorded step."""
        return float(self._schedule(int(self._step)))

    def step(self) -> None:
        """Advance the annealing counter by one (call once per optimizer step)."""
        self._step += 1

    # --- Forward -------------------------------------------------------------

    def forward(
        self,
        ctc_logits: Tensor,
        length_logits: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Dict[str, Tensor]:
        r"""Compute the LAD-CTC loss and its components.

        Parameters
        ----------
        ctc_logits:
            Raw per-step logits of shape ``(B, T, V + 1)``.
        length_logits:
            Raw length-head logits of shape ``(B, num_lengths)``.
        targets:
            Flat (concatenated) target indices of shape ``(sum(target_lengths),)``
            with classes in ``1 .. V`` (blank excluded).
        input_lengths:
            Per-sample CTC input lengths ``(B,)``.
        target_lengths:
            Per-sample target lengths ``(B,)``.

        Returns
        -------
        dict
            ``{"loss", "ctc", "length", "lambda"}`` where ``loss`` is the combined
            objective and the others are detached diagnostics (except gradients
            still flow through ``loss``).
        """
        cfg = self.config

        # --- CTC term (fp32, log-domain). -----------------------------------
        # CTCLoss wants (T, B, V+1) log-probs; compute in fp32 under autocast.
        log_probs = ctc_logits.float().log_softmax(dim=-1).permute(1, 0, 2)
        ctc_per_sample = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=cfg.blank,
            reduction="none",
            zero_infinity=True,
        )  # (B,)

        # --- Length log-prior term (fp32 log_softmax + gather). -------------
        length_logp = length_logits.float().log_softmax(dim=-1)  # (B, num_lengths)
        num_lengths = length_logp.shape[1]
        # Map true length -> class index, clamped into the valid range.
        idx = (target_lengths - cfg.min_length).clamp(0, num_lengths - 1)
        lp_true = length_logp.gather(1, idx.unsqueeze(1)).squeeze(1)  # (B,)

        lam = self.current_lambda
        # Combined per-sample objective: ctc - lambda * log p(true length).
        per_sample = ctc_per_sample - lam * lp_true
        loss = per_sample.mean()

        return {
            "loss": loss,
            "ctc": ctc_per_sample.mean().detach(),
            "length": (-lp_true.mean()).detach(),  # length NLL diagnostic
            "lambda": torch.as_tensor(lam, device=loss.device),
        }
