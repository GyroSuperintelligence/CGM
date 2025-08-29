"""
Core mathematical operations for CGM (Common Governance Model)

This package provides the fundamental mathematical operations for the CGM framework,
including dimensional calibration, gyrovector operations, and recursive memory structures.

Modules:
    dimensions: Dimensional calibration engine with {ħ, c, m⋆} basis
    gyrovector_ops: Einstein-Ungar gyrovector space operations
    gyrotriangle: Gyrotriangle defect and area calculations
    recursive_memory: Recursive memory structure for κ prediction
"""

__version__ = "1.0.0"
__author__ = "CGM Framework"
__description__ = "Core mathematical operations for Common Governance Model"

from .dimensions import DimensionalCalibrator, DimVec
from .gyrovector_ops import GyroVectorSpace, RecursivePath
from .gyrotriangle import GyroTriangle
from .recursive_memory import RecursiveMemory

__all__ = ["dimensions", "gyrovector_ops", "gyrotriangle", "recursive_memory"]
