"""Experiment drivers (ablations, profiling) for VL-KAN."""

from .ablation import (
    VariantResult,
    VariantSpec,
    build_variants,
    results_to_markdown,
    run_ablation,
)

__all__ = [
    "VariantResult",
    "VariantSpec",
    "build_variants",
    "results_to_markdown",
    "run_ablation",
]
