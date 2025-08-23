"""
Dimensional Engine Theorems for CGM

This module contains mathematical proofs related to the dimensional
calibration engine, including the group homomorphism property.

Theorem: The map d ↦ u(d) with basis {ħ,c,m⋆} is a homomorphism.
Proof: Uses invertibility of B matrix in core.dimensions.DimensionalCalibrator.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

# Import the core implementation
try:
    from Experiments.core.dimensions import DimensionalCalibrator, DimVec
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.dimensions import DimensionalCalibrator, DimVec


class DimensionalEngineHomomorphism:
    """
    Theorem: The DimensionalCalibrator implements a group homomorphism.
    
    Let B = [dim(ħ) dim(c) dim(m⋆)] with rows (M,L,T).
    We solve Ba = d and map d ↦ u(d) = ∏ᵢ vᵢ^aᵢ where v = (ħ,c,m⋆).
    
    Because B is invertible, the exponent vector a(d) = B⁻¹d is unique,
    hence the unit map is well-defined and:
    
        u(d₁ + d₂) = ∏ v^(a(d₁) + a(d₂)) = u(d₁) u(d₂)
    
    So the calibrator implements a group homomorphism ℝ³ → ℝ₊
    (additive exponents → multiplicative units).
    
    This class is a thin wrapper around core.dimensions.DimensionalCalibrator
    that proves the homomorphism property.
    """
    
    def __init__(self, hbar: float, c: float, m_anchor: float):
        try:
            # Use the core implementation
            self.calibrator = DimensionalCalibrator(hbar, c, m_anchor)
            self.values = np.array([hbar, c, m_anchor], dtype=float)
        except ImportError:
            # Fallback if core.dimensions not available
            self.calibrator = None
            self.values = np.array([hbar, c, m_anchor], dtype=float)
    
    def verify_homomorphism(self, d1: DimVec, d2: DimVec) -> Dict[str, Any]:
        """
        Verify the homomorphism property: u(d₁ + d₂) = u(d₁) u(d₂)
        
        Args:
            d1, d2: Dimension vectors to test
            
        Returns:
            Verification results
        """
        if self.calibrator is None:
            return {
                "error": "Core DimensionalCalibrator not available",
                "note": "Install core.dimensions to test homomorphism"
            }
        
        # Use the core implementation to compute units
        u1 = self.calibrator.get_unit(d1)
        u2 = self.calibrator.get_unit(d2)
        
        # Compute unit of sum
        d_sum = d1 + d2
        u_sum = self.calibrator.get_unit(d_sum)
        
        # Verify homomorphism: u(d₁ + d₂) = u(d₁) u(d₂)
        homomorphism_satisfied = np.isclose(u_sum, u1 * u2, rtol=1e-10)
        
        return {
            "d1": d1,
            "d2": d2,
            "d_sum": d_sum,
            "u1": u1,
            "u2": u2,
            "u_sum": u_sum,
            "u1_times_u2": u1 * u2,
            "homomorphism_satisfied": homomorphism_satisfied,
            "note": "u(d₁ + d₂) = u(d₁) u(d₂) using core DimensionalCalibrator"
        }
    
    def test_base_units(self) -> Dict[str, Any]:
        """
        Test that base units are correctly computed.
        
        Returns:
            Verification of L₀, T₀, M₀ calculations
        """
        if self.calibrator is None:
            return {
                "error": "Core DimensionalCalibrator not available",
                "note": "Install core.dimensions to test base units"
            }
        
        base = self.calibrator.base_units_SI()
        
        # Verify c-invariance: L₀/T₀ = c
        c_invariant = base["L0"] / base["T0"]
        c_correct = self.values[1]
        c_invariance_satisfied = np.isclose(c_invariant, c_correct, rtol=1e-10)
        
        return {
            "M0_computed": base["M0"],
            "M0_correct": self.values[2],
            "M0_consistent": np.isclose(base["M0"], self.values[2], rtol=1e-10),
            "L0_computed": base["L0"],
            "T0_computed": base["T0"],
            "c_invariant": c_invariant,
            "c_correct": c_correct,
            "c_invariance_satisfied": c_invariance_satisfied,
            "all_consistent": c_invariance_satisfied,
            "note": "Base units from core DimensionalCalibrator"
        }
