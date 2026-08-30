"""Prediction heads for VL-KAN."""

from .boundary_head import BoundaryHead, build_soft_boundary_targets

__all__ = ["BoundaryHead", "build_soft_boundary_targets"]
