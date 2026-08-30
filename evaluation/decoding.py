r"""Reusable output-decoding helpers shared by ablation and inference.

These adapters turn raw model outputs into transcription strings under the
project CTC conventions (blank at index 0, symbols at ``1 .. V``):

* :func:`greedy_ctc_decode` -- argmax CTC path collapse for sequence models.
* :func:`beam_ctc_decode` -- length/boundary-aware beam search for the full
  VL-KAN model.
* :func:`fixed_length_decode` -- per-position argmax for the fixed-length
  Deep-CAPTCHA baseline (no blank class).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor

from datasets.vocabulary import Vocabulary
from decode.beam_search import BeamSearchConfig, LengthBoundaryBeamSearch

__all__ = ["greedy_ctc_decode", "beam_ctc_decode", "fixed_length_decode"]


def greedy_ctc_decode(ctc_logits: Tensor, vocabulary: Vocabulary) -> List[str]:
    """Greedy CTC decode: per-step argmax then collapse repeats and blanks.

    Parameters
    ----------
    ctc_logits:
        Raw CTC logits ``(B, T, V + 1)``.
    vocabulary:
        Vocabulary whose ``ctc_greedy_decode`` implements the collapse.

    Returns
    -------
    list of str
        One decoded string per batch element.
    """
    if ctc_logits.dim() != 3:
        raise ValueError(f"ctc_logits must be (B, T, V+1), got {tuple(ctc_logits.shape)}.")
    paths = ctc_logits.argmax(dim=-1).detach().cpu().tolist()
    return [vocabulary.ctc_greedy_decode(path) for path in paths]


def beam_ctc_decode(
    outputs: Dict[str, Tensor],
    vocabulary: Vocabulary,
    config: Optional[BeamSearchConfig] = None,
) -> List[str]:
    """Length/boundary-aware beam-search decode of a full VL-KAN output dict.

    Parameters
    ----------
    outputs:
        Model output dict with ``ctc_logits`` and optionally ``boundary_logits`` /
        ``length_logits``.
    vocabulary:
        Vocabulary used to map token indices to characters.
    config:
        Optional beam-search configuration.

    Returns
    -------
    list of str
        Best transcription per batch element.
    """
    decoder = LengthBoundaryBeamSearch(config)
    hyps = decoder.decode_batch(
        outputs["ctc_logits"],
        outputs.get("boundary_logits"),
        outputs.get("length_logits"),
    )
    return [vocabulary.decode(h.tokens) for h in hyps]


def fixed_length_decode(logits: Tensor, vocabulary: Vocabulary) -> List[str]:
    """Decode fixed-length baseline logits ``(B, L, V)`` (no blank class).

    Each position's argmax indexes the *real* alphabet directly (class ``c`` maps
    to vocabulary index ``c + 1``). The model always emits ``L`` characters.

    Parameters
    ----------
    logits:
        Per-position logits ``(B, L, V)`` over the ``V`` real symbols.
    vocabulary:
        Vocabulary used for index-to-char mapping.

    Returns
    -------
    list of str
        One fixed-length string per batch element.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be (B, L, V), got {tuple(logits.shape)}.")
    preds = logits.argmax(dim=-1).detach().cpu().tolist()  # 0-based real classes
    out: List[str] = []
    for row in preds:
        out.append("".join(vocabulary.index_to_char[c + 1] for c in row))
    return out
