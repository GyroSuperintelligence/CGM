"""
Tests for CGM-RGF Gyro Math Theorems

This module tests the mathematical foundations and theorems
that underpin the Common Governance Model.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.gyrovector_ops import GyroVectorSpace
from ..core.gyrotriangle import GyroTriangle
from ..core.dimensions import DimensionalCalibrator, DimVec


def test_thomas_wigner_small_angle():
    """Test that small-velocity gyration produces proper rotation matrices."""
    gs = GyroVectorSpace(c=1.0)
    u, v = np.array([1e-3, 0, 0]), np.array([0, 1e-3, 0])
    R = gs.gyration(u, v)
    
    # Check rotation properties: orthogonal, det=1
    np.testing.assert_allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, rtol=0, atol=1e-10)
    
    print("✅ Thomas-Wigner small angle test passed")


def test_gauss_bonnet_zero_defect():
    """Test that closure case gives zero defect (Euclidean limit)."""
    gs = GyroVectorSpace()
    gt = GyroTriangle(gs)
    a, b, g = gt.cgm_standard_angles()
    
    assert gt.is_closed(a, b, g)
    assert np.isclose(gt.compute_defect(a, b, g), 0.0, atol=1e-12)
    
    print("✅ Gauss-Bonnet zero defect test passed")


def test_dimensional_calibrator_base_units():
    """Test that base units are correctly computed from {ħ, c, m⋆}."""
    calib = DimensionalCalibrator(
        hbar=1.054571817e-34, 
        c=2.99792458e8, 
        m_anchor=9.1093837015e-31
    )
    base = calib.base_units_SI()
    
    # L₀ = ħ/(m c); T₀ = ħ/(m c²)
    L0 = 1.054571817e-34 / (9.1093837015e-31 * 2.99792458e8)
    T0 = 1.054571817e-34 / (9.1093837015e-31 * (2.99792458e8**2))
    
    np.testing.assert_allclose(base["L0"], L0, rtol=1e-12)
    np.testing.assert_allclose(base["T0"], T0, rtol=1e-12)
    
    print("✅ Dimensional calibrator base units test passed")


def test_gyration_proof_guard():
    """Test that gyration proof-guard catches and fixes non-orthogonal matrices."""
    gs = GyroVectorSpace(c=1.0)
    
    # Test with vectors that might trigger the proof-guard
    u = np.array([0.9, 0.1, 0.0])  # Near light speed
    v = np.array([0.1, 0.9, 0.0])  # Near light speed
    
    R = gs.gyration(u, v)
    
    # Should always be a proper rotation matrix
    orthogonality_error = np.linalg.norm(R.T @ R - np.eye(3))
    det_error = abs(np.linalg.det(R) - 1.0)
    
    assert orthogonality_error < 1e-10, f"Orthogonality error: {orthogonality_error}"
    assert det_error < 1e-10, f"Determinant error: {det_error}"
    
    print("✅ Gyration proof-guard test passed")


def test_bu_amplitude_identity():
    """Test that BU amplitude satisfies the theoretical identity."""
    from theorems.gyrogeometry import BUAmplitudeIdentity
    
    bu_test = BUAmplitudeIdentity()
    amplitude = 1.0 / (2.0 * np.sqrt(2.0 * np.pi))
    result = bu_test.verify_identity(amplitude)
    
    assert result["identity_satisfied"], f"BU identity failed: {result}"
    
    print("✅ BU amplitude identity test passed")


def test_gyrotriangle_defect_theorem():
    """Test Gauss-Bonnet theorem for gyrotriangles."""
    from theorems.gyrogeometry import GyroTriangleDefectTheorem
    
    theorem = GyroTriangleDefectTheorem(c=1.0)
    closure_test = theorem.test_closure_case()
    
    assert closure_test["is_zero_defect"], f"Closure case failed: {closure_test}"
    
    print("✅ Gyrotriangle defect theorem test passed")


def run_all_tests():
    """Run all gyro math tests."""
    print("Running CGM-RGF Gyro Math Tests")
    print("=" * 40)
    
    try:
        test_thomas_wigner_small_angle()
        test_gauss_bonnet_zero_defect()
        test_dimensional_calibrator_base_units()
        test_gyration_proof_guard()
        test_bu_amplitude_identity()
        test_gyrotriangle_defect_theorem()
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
