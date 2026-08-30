"""VL-KAN loss functions."""

from .ctc_lad import LADCTCConfig, LengthAwareDynamicCTCLoss

__all__ = ["LADCTCConfig", "LengthAwareDynamicCTCLoss"]
