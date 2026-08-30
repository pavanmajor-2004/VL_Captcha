r"""Length- and Boundary-Aware CTC Beam Search decoder.

This is a prefix-beam-search CTC decoder (Graves et al.) extended with two
inference-time priors specific to VL-KAN:

* **Boundary bias.** When a beam *extends* its label sequence (emits a new,
  non-repeated character) the move is rewarded by ``mu * log b_t``, where ``b_t``
  is the boundary head's probability that step ``t`` is a character transition.
  Emissions are thus encouraged to occur where the tracker predicts boundaries.

* **Length prior.** At the end of decoding each surviving prefix is re-scored by
  ``lambda * log p(ell = |prefix| | I)`` using the length head's posterior, so the
  final ranking favours transcriptions whose length matches the length branch.

The core recursion maintains, per prefix, two log-probabilities:

    p_b  -- prob of the prefix with the last alignment step being **blank**
    p_nb -- prob of the prefix with the last alignment step being **non-blank**

updated across time with the standard CTC prefix-beam transitions, all in the log
domain via ``logsumexp`` for numerical stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

__all__ = ["BeamHypothesis", "BeamSearchConfig", "LengthBoundaryBeamSearch"]

_NEG_INF = -1e30


def _logsumexp(a: float, b: float) -> float:
    """Stable scalar ``log(exp(a) + exp(b))`` for the beam accumulators."""
    if a <= _NEG_INF:
        return b
    if b <= _NEG_INF:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


@dataclass
class BeamSearchConfig:
    """Hyperparameters for :class:`LengthBoundaryBeamSearch`.

    Attributes
    ----------
    beam_width:
        Number of prefixes retained per time step.
    blank:
        CTC blank index.
    min_length, max_length:
        Inclusive admissible transcription length bounds (``3``..``10``). Used to
        map prefix length to a length-head class and to prune over-long beams.
    boundary_weight:
        ``mu`` -- strength of the boundary log-prob bias on emissions.
    length_weight:
        ``lambda`` -- strength of the final length-prior re-scoring.
    prune_threshold:
        Per-step probability floor; symbols below this are skipped to bound the
        branching factor.
    """

    beam_width: int = 10
    blank: int = 0
    min_length: int = 3
    max_length: int = 10
    boundary_weight: float = 0.3
    length_weight: float = 0.5
    prune_threshold: float = 1e-3


@dataclass
class BeamHypothesis:
    """A decoded prefix and its diagnostic scores.

    Attributes
    ----------
    tokens:
        Decoded class indices (in ``1 .. V``; blanks already collapsed).
    score:
        Final combined log-score (CTC + boundary + length prior).
    ctc_score:
        Log-probability contribution from CTC alone.
    """

    tokens: Tuple[int, ...]
    score: float
    ctc_score: float = field(default=0.0)


@dataclass
class _PrefixState:
    """Internal mutable log-prob accumulators for one prefix."""

    p_b: float = _NEG_INF      # ends in blank
    p_nb: float = _NEG_INF     # ends in non-blank

    @property
    def total(self) -> float:
        """Total prefix log-prob ``logsumexp(p_b, p_nb)``."""
        return _logsumexp(self.p_b, self.p_nb)


class LengthBoundaryBeamSearch:
    """CTC prefix beam search with boundary and length priors.

    Parameters
    ----------
    config:
        A :class:`BeamSearchConfig`. If ``None`` the defaults are used.
    """

    def __init__(self, config: Optional[BeamSearchConfig] = None) -> None:
        self.config = config or BeamSearchConfig()

    # --- Single-sequence decode ---------------------------------------------

    def decode_sequence(
        self,
        log_probs: Tensor,
        boundary_logp: Optional[Tensor] = None,
        length_logp: Optional[Tensor] = None,
    ) -> List[BeamHypothesis]:
        r"""Decode one sequence's CTC log-probabilities.

        Parameters
        ----------
        log_probs:
            Per-step log-probabilities of shape ``(T, V + 1)`` (already
            ``log_softmax``-normalized).
        boundary_logp:
            Optional per-step boundary log-probabilities ``(T,)`` (i.e.
            ``log sigmoid(boundary_logits)``). When provided, emissions at step
            ``t`` receive a ``mu * boundary_logp[t]`` bonus.
        length_logp:
            Optional length-head log-probabilities ``(num_lengths,)`` over lengths
            ``min_length .. max_length``; applied as a final re-scoring prior.

        Returns
        -------
        list of BeamHypothesis
            Surviving hypotheses sorted by descending final score.
        """
        cfg = self.config
        T, vocab = log_probs.shape
        lp = log_probs.detach().cpu()
        b_lp = boundary_logp.detach().cpu() if boundary_logp is not None else None

        # Beam: prefix tokens -> accumulator state. Seed with the empty prefix.
        beam: Dict[Tuple[int, ...], _PrefixState] = {(): _PrefixState(p_b=0.0)}

        for t in range(T):
            next_beam: Dict[Tuple[int, ...], _PrefixState] = {}
            # Boundary bonus for an emission at this step (0 if unavailable).
            bound_bonus = (
                cfg.boundary_weight * float(b_lp[t]) if b_lp is not None else 0.0
            )

            # Candidate symbols above the pruning floor (always include blank).
            row = lp[t]
            cand = [cfg.blank]
            cand.extend(
                s for s in range(vocab)
                if s != cfg.blank and float(row[s].exp()) >= cfg.prune_threshold
            )

            for prefix, state in beam.items():
                tot = state.total

                # --- Case 1: emit blank. Prefix unchanged, mass goes to p_b. ---
                blank_lp = float(row[cfg.blank])
                nb = next_beam.setdefault(prefix, _PrefixState())
                nb.p_b = _logsumexp(nb.p_b, tot + blank_lp)

                # --- Case 2: emit a non-blank symbol s. ----------------------
                for s in cand:
                    if s == cfg.blank:
                        continue
                    sym_lp = float(row[s])
                    last = prefix[-1] if prefix else None

                    if s == last:
                        # Repeat: extend only from p_b (a blank must separate
                        # identical labels); staying yields p_nb on same prefix.
                        same = next_beam.setdefault(prefix, _PrefixState())
                        same.p_nb = _logsumexp(same.p_nb, state.p_nb + sym_lp)

                        new_prefix = prefix + (s,)
                        if len(new_prefix) <= cfg.max_length:
                            ext = next_beam.setdefault(new_prefix, _PrefixState())
                            ext.p_nb = _logsumexp(
                                ext.p_nb, state.p_b + sym_lp + bound_bonus
                            )
                    else:
                        # New character: extend from the full prefix mass.
                        new_prefix = prefix + (s,)
                        if len(new_prefix) <= cfg.max_length:
                            ext = next_beam.setdefault(new_prefix, _PrefixState())
                            ext.p_nb = _logsumexp(
                                ext.p_nb, tot + sym_lp + bound_bonus
                            )

            # Prune to the top-`beam_width` prefixes by total log-prob.
            ranked = sorted(
                next_beam.items(), key=lambda kv: kv[1].total, reverse=True
            )
            beam = dict(ranked[: cfg.beam_width])

        # --- Final length-prior re-scoring. ---------------------------------
        results: List[BeamHypothesis] = []
        for prefix, state in beam.items():
            ctc_score = state.total
            score = ctc_score
            n = len(prefix)
            if (
                length_logp is not None
                and cfg.min_length <= n <= cfg.max_length
            ):
                cls = n - cfg.min_length
                if 0 <= cls < length_logp.shape[0]:
                    score = score + cfg.length_weight * float(length_logp[cls])
            results.append(
                BeamHypothesis(tokens=prefix, score=score, ctc_score=ctc_score)
            )

        results.sort(key=lambda h: h.score, reverse=True)
        return results

    # --- Batched decode ------------------------------------------------------

    def decode_batch(
        self,
        ctc_logits: Tensor,
        boundary_logits: Optional[Tensor] = None,
        length_logits: Optional[Tensor] = None,
        input_lengths: Optional[Sequence[int]] = None,
    ) -> List[BeamHypothesis]:
        r"""Decode a batch of recognizer outputs to best hypotheses.

        Parameters
        ----------
        ctc_logits:
            Raw CTC logits ``(B, T, V + 1)`` (un-normalized; ``log_softmax`` is
            applied here).
        boundary_logits:
            Optional raw boundary logits ``(B, T)``; converted to log-probs via
            ``logsigmoid``.
        length_logits:
            Optional raw length logits ``(B, num_lengths)``; converted via
            ``log_softmax``.
        input_lengths:
            Optional per-sample valid time lengths; if given, decoding for sample
            ``i`` only consumes the first ``input_lengths[i]`` steps.

        Returns
        -------
        list of BeamHypothesis
            The single best hypothesis per batch element (length ``B``).
        """
        if ctc_logits.dim() != 3:
            raise ValueError(
                f"ctc_logits must be (B, T, V+1), got {tuple(ctc_logits.shape)}."
            )
        batch, T, _ = ctc_logits.shape
        log_probs = ctc_logits.float().log_softmax(dim=-1)
        bound_lp = (
            torch.nn.functional.logsigmoid(boundary_logits.float())
            if boundary_logits is not None else None
        )
        len_lp = (
            length_logits.float().log_softmax(dim=-1)
            if length_logits is not None else None
        )

        best: List[BeamHypothesis] = []
        for i in range(batch):
            t_valid = T if input_lengths is None else int(input_lengths[i])
            t_valid = max(1, min(T, t_valid))
            hyps = self.decode_sequence(
                log_probs[i, :t_valid],
                bound_lp[i, :t_valid] if bound_lp is not None else None,
                len_lp[i] if len_lp is not None else None,
            )
            best.append(
                hyps[0] if hyps else BeamHypothesis(tokens=(), score=_NEG_INF)
            )
        return best
