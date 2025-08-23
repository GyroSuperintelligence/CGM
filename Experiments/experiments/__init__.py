"""
CGM-RGF Experiments Package

This package contains experimental modules for testing and validating
the Common Governance Model - Recursive Gyrovector Formalism.
"""

from .core_experiments import CoreTheoremTester
from .physical_constants import PhysicalConstantsValidator
from .singularity_infinity import SingularityInfinityValidator

__all__ = [
    "CoreTheoremTester",
    "PhysicalConstantsValidator",
    "SingularityInfinityValidator",
]
