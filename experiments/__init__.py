"""
CGM Experiments Package

This package contains experimental modules for testing and validating
the Common Governance Model - Recursive Gyrovector Formalism.
"""

from .physical_constants import ElectricCalibrationValidator
from .singularity_infinity import SingularityInfinityValidator

__all__ = [
    "ElectricCalibrationValidator",
    "SingularityInfinityValidator",
]
