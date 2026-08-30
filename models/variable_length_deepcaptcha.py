"""Variable-length DeepCAPTCHA baseline: CNN + BiLSTM + CTC + length head.

This extends the fixed-length DeepCAPTCHA into a sequence model capable of
handling variable-length challenges (lengths 3-10). It is the strong CRNN-style
baseline used in the comparative study, augmented with an explicit length
branch:

* a :class:`~models.backbone.resnet.ResNet18Seq` trunk produces a length-``T``
  feature sequence ``(B, T, C)``;
* a 2-layer **bidirectional LSTM** contextualizes the sequence;
* a **CTC head** projects each time step to ``V + 1`` vocabulary logits (blank
  included) for alignment-free decoding;
* an independent, parallel **length-prediction head** pools the contextualized
  sequence and emits logits over the 8 admissible lengths ``{3, ..., 10}`` for
  training with cross-entropy.

The two heads are intentionally decoupled: the CTC head models *what* characters
appear while the length head models *how many*, providing a soft prior the
decoder can exploit.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .backbone.resnet import ResNet18Seq

__all__ = ["VariableLengthDeepCaptcha"]


class VariableLengthDeepCaptcha(nn.Module):
    """CNN-BiLSTM recognizer with parallel CTC and length-prediction heads.

    Parameters
    ----------
    num_classes:
        Alphabet size ``V`` (excludes the CTC blank). The CTC head emits
        ``V + 1`` logits, with the blank reserved at index ``0``.
    min_length, max_length:
        Inclusive bounds of the admissible length set; the length head produces
        ``max_length - min_length + 1`` logits (``8`` for ``3..10``).
    in_channels:
        Number of input image channels.
    input_size:
        Expected ``(height, width)`` of input images.
    lstm_hidden:
        Hidden size of each LSTM direction.
    lstm_layers:
        Number of stacked BiLSTM layers.
    dropout:
        Dropout applied between LSTM layers and in the length head.
    """

    def __init__(
        self,
        num_classes: int,
        min_length: int = 3,
        max_length: int = 10,
        in_channels: int = 3,
        input_size: Tuple[int, int] = (48, 320),
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if max_length < min_length:
            raise ValueError("max_length must be >= min_length.")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")

        self.num_classes = num_classes
        self.blank_index = 0
        self.ctc_classes = num_classes + 1
        self.min_length = min_length
        self.max_length = max_length
        self.num_lengths = max_length - min_length + 1
        self.in_channels = in_channels
        self.input_size = input_size

        self.backbone = ResNet18Seq(
            in_channels=in_channels,
            expected_input_size=input_size,
        )
        feat_dim = self.backbone.out_channels

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        seq_dim = lstm_hidden * 2  # bidirectional concat

        # CTC head: per-time-step vocabulary logits (blank included).
        self.ctc_head = nn.Linear(seq_dim, self.ctc_classes)

        # Length head: global temporal pooling -> length logits.
        self.length_head = nn.Sequential(
            nn.Linear(seq_dim, seq_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(seq_dim // 2, self.num_lengths),
        )

        self._reset_head_parameters()

    def _reset_head_parameters(self) -> None:
        """Xavier-init the linear heads for stable early training."""
        for module in [self.ctc_head, *self.length_head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _check_input(self, x: Tensor) -> None:
        """Validate the entry-point tensor against the contract."""
        if x.dim() != 4:
            raise ValueError(
                f"VariableLengthDeepCaptcha expects (B, C, H, W), "
                f"got shape {tuple(x.shape)}."
            )
        _, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {c}."
            )
        if (h, w) != self.input_size:
            raise ValueError(
                f"Expected input size {self.input_size}, got {(h, w)}."
            )

    def _check_output(self, ctc_logits: Tensor, length_logits: Tensor,
                      batch: int, expected_t: int) -> None:
        """Validate both head outputs against the contract."""
        if ctc_logits.shape != (batch, expected_t, self.ctc_classes):
            raise ValueError(
                f"CTC head contract violated: expected "
                f"{(batch, expected_t, self.ctc_classes)}, "
                f"got {tuple(ctc_logits.shape)}."
            )
        if length_logits.shape != (batch, self.num_lengths):
            raise ValueError(
                f"Length head contract violated: expected "
                f"{(batch, self.num_lengths)}, got {tuple(length_logits.shape)}."
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Run the recognizer and return both head outputs.

        Parameters
        ----------
        x:
            Image batch of shape ``(B, in_channels, 48, 320)``.

        Returns
        -------
        dict
            ``{"ctc_logits": (B, T, V + 1), "length_logits": (B, 8)}``. The CTC
            logits are raw (apply ``log_softmax`` before :class:`~torch.nn.CTCLoss`);
            the length logits are raw (apply cross-entropy with targets
            ``length - min_length``).
        """
        self._check_input(x)
        batch, _, _, width = x.shape
        expected_t = width // self.backbone.width_downsample

        seq = self.backbone(x)              # (B, T, C)
        context, _ = self.lstm(seq)         # (B, T, 2*hidden)

        ctc_logits = self.ctc_head(context)            # (B, T, V + 1)
        pooled = context.mean(dim=1)                   # (B, 2*hidden)
        length_logits = self.length_head(pooled)       # (B, num_lengths)

        self._check_output(ctc_logits, length_logits, batch, expected_t)
        return {"ctc_logits": ctc_logits, "length_logits": length_logits}
