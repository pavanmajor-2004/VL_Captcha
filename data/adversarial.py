"""Online Adversarial CAPTCHA Augmentation (ACA) via Projected Gradient Descent.

This module implements the surrogate-based PGD engine used to harden VL-KAN
against adversarial perturbations (paper Sections 6.3 and 7.4). During training a
fraction of the batch is perturbed *online* with an :math:`\\ell_\\infty`-bounded
7-step PGD attack crafted against a **frozen surrogate** sequence model
(typically a CRNN). The surrogate's parameters are never updated; only the input
image receives gradients.

The default loss is the CTC objective, matching the sequence-recognition heads of
both the surrogate and VL-KAN, but any callable ``loss_fn(logits, batch) -> scalar``
may be supplied.

Mathematically, for a clean image :math:`\\mathbf{I}` with target :math:`\\mathbf{y}`::

    delta_0 ~ U(-eps, eps)
    delta_{k+1} = clip_eps( delta_k + alpha * sign( grad_delta L(g(I + delta_k), y) ) )
    I_adv = clip_[0,1]( I + delta_K )

where ``g`` is the frozen surrogate and ``clip_eps`` projects back onto the
:math:`\\ell_\\infty` ball of radius ``eps``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["AdversarialBatch", "ctc_surrogate_loss", "PGDAdversary"]


@dataclass
class AdversarialBatch:
    """Container for the tensors an attack step needs from the training batch.

    Attributes
    ----------
    images:
        Clean images, shape ``(B, C, H, W)`` in ``[0, 1]``.
    targets:
        Flat (concatenated) CTC target indices, shape ``(sum(target_lengths),)``.
    input_lengths:
        Per-sample CTC input (time) lengths, shape ``(B,)``.
    target_lengths:
        Per-sample target lengths, shape ``(B,)``.
    """

    images: Tensor
    targets: Tensor
    input_lengths: Tensor
    target_lengths: Tensor


def ctc_surrogate_loss(logits: Tensor, batch: AdversarialBatch,
                       blank: int = 0) -> Tensor:
    """Compute the (summed) CTC loss for a surrogate's logits.

    Parameters
    ----------
    logits:
        Surrogate output of shape ``(B, T, V + 1)`` (time-major conversion is
        handled here).
    batch:
        The :class:`AdversarialBatch` providing targets and lengths.
    blank:
        Blank class index (``0`` by project convention).

    Returns
    -------
    Tensor
        Scalar CTC loss summed over the batch. Maximizing this loss w.r.t. the
        input is the adversarial objective.
    """
    # CTCLoss expects (T, B, V+1) log-probabilities.
    log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
    return F.ctc_loss(
        log_probs,
        batch.targets,
        batch.input_lengths,
        batch.target_lengths,
        blank=blank,
        reduction="sum",
        zero_infinity=True,
    )


class PGDAdversary(nn.Module):
    """:math:`\\ell_\\infty` PGD attack against a frozen surrogate model.

    Parameters
    ----------
    surrogate:
        A sequence model mapping ``(B, C, H, W) -> (B, T, V + 1)`` logits. It is
        frozen (``requires_grad_(False)`` and ``eval()``) on construction.
    epsilon:
        Maximum :math:`\\ell_\\infty` perturbation magnitude (default ``8/255``).
    alpha:
        Per-step size (default ``2/255``).
    steps:
        Number of PGD iterations (default ``7``).
    loss_fn:
        Callable ``(logits, batch) -> scalar`` to maximize. Defaults to
        :func:`ctc_surrogate_loss`.
    random_start:
        If ``True``, initialize the perturbation uniformly in the epsilon ball.
    clip_min, clip_max:
        Valid pixel range for the perturbed image (default ``[0, 1]``).
    blank:
        Blank index forwarded to the default CTC loss.
    """

    def __init__(
        self,
        surrogate: nn.Module,
        epsilon: float = 8.0 / 255.0,
        alpha: float = 2.0 / 255.0,
        steps: int = 7,
        loss_fn: Optional[Callable[[Tensor, AdversarialBatch], Tensor]] = None,
        random_start: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        blank: int = 0,
    ) -> None:
        super().__init__()
        if epsilon < 0 or alpha < 0 or steps < 1:
            raise ValueError("epsilon/alpha must be >= 0 and steps >= 1.")

        self.surrogate = surrogate
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.random_start = random_start
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.blank = int(blank)
        self.loss_fn = loss_fn or (lambda logits, batch: ctc_surrogate_loss(
            logits, batch, blank=self.blank
        ))

        # Freeze the surrogate: it is a fixed attacker, never trained here.
        self.surrogate.eval()
        for param in self.surrogate.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True) -> "PGDAdversary":
        """Keep the surrogate in eval mode regardless of parent ``train`` calls."""
        super().train(mode)
        self.surrogate.eval()
        return self

    @torch.enable_grad()
    def attack(self, batch: AdversarialBatch) -> Tensor:
        """Run PGD and return the adversarial images (detached).

        Parameters
        ----------
        batch:
            The clean batch to perturb.

        Returns
        -------
        Tensor
            Adversarial images of shape ``(B, C, H, W)`` in ``[clip_min, clip_max]``,
            detached from the autograd graph and safe to feed to the main model.
        """
        images = batch.images.detach()

        if self.random_start:
            delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            delta = (images + delta).clamp(self.clip_min, self.clip_max) - images
        else:
            delta = torch.zeros_like(images)
        delta = delta.detach()

        for _ in range(self.steps):
            delta.requires_grad_(True)
            adv = images + delta
            logits = self.surrogate(adv)
            loss = self.loss_fn(logits, batch)

            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]

            # Ascend the loss surface, then project back onto the eps-ball and
            # the valid pixel range.
            delta = delta.detach() + self.alpha * grad.sign()
            delta = delta.clamp(-self.epsilon, self.epsilon)
            delta = (images + delta).clamp(self.clip_min, self.clip_max) - images
            delta = delta.detach()

        return (images + delta).clamp(self.clip_min, self.clip_max).detach()

    def forward(self, batch: AdversarialBatch) -> Tensor:
        """Alias for :meth:`attack` so the module is callable."""
        return self.attack(batch)

    @torch.no_grad()
    def mix(self, batch: AdversarialBatch, fraction: float,
            generator: Optional[torch.Generator] = None) -> Tensor:
        """Perturb a random subset of the batch (online ACA helper).

        Only ``fraction`` of the samples are attacked (paper: 10%); the remainder
        pass through unchanged. The attack itself still requires gradients, so the
        per-subset call re-enables grad internally.

        Parameters
        ----------
        batch:
            The clean batch.
        fraction:
            Fraction of samples in ``[0, 1]`` to replace with adversarial versions.
        generator:
            Optional RNG controlling which samples are selected.

        Returns
        -------
        Tensor
            Images of shape ``(B, C, H, W)`` with a random subset adversarially
            perturbed.
        """
        if fraction <= 0.0:
            return batch.images.detach()

        b = batch.images.shape[0]
        device = batch.images.device
        num = max(1, int(round(b * min(1.0, fraction))))
        perm = torch.randperm(b, device=device, generator=generator)[:num]

        # Attacking the full batch and scattering back is simpler and keeps the
        # CTC length bookkeeping intact (targets are concatenated per sample).
        adv_full = self.attack(batch)
        out = batch.images.detach().clone()
        out[perm] = adv_full[perm]
        return out
