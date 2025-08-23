#!/usr/bin/env python3
"""
CGM-RGF Experimental Framework

Main entry point for running CGM experiments and validations.
CLEAN VERSION - No duplication, minimal output
"""

import numpy as np
import os
import sys
from typing import Dict, Any

# Import core components
from core.gyrovector_ops import GyroVectorSpace

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import experiments with error handling
try:
    from experiments.physical_constants import PhysicalConstantsValidator

    PHYSICAL_CONSTANTS_AVAILABLE = True
except ImportError:
    print("Warning: Physical constants experiments not available")
    PHYSICAL_CONSTANTS_AVAILABLE = False

try:
    from experiments.core_experiments import CoreTheoremTester
    CORE_EXPERIMENTS_AVAILABLE = True
except ImportError:
    print("Warning: Core experiments not available")
    CORE_EXPERIMENTS_AVAILABLE = False

try:
    from experiments.singularity_infinity import SingularityInfinityValidator
    SINGULARITY_INFINITY_AVAILABLE = True
except ImportError:
    print("Warning: Singularity/infinity experiments not available")
    SINGULARITY_INFINITY_AVAILABLE = False


def run_comprehensive_physical_dimensions() -> Dict[str, Any]:
    """Run the comprehensive physical dimensions experiment"""
    if not PHYSICAL_CONSTANTS_AVAILABLE:
        return {
            "status": "UNAVAILABLE",
            "reason": "Physical constants experiments not available",
        }

    gyrospace = GyroVectorSpace(c=1.0)
    validator = PhysicalConstantsValidator(gyrospace)
    return validator.run_comprehensive_physical_dimensions_experiment()


def main():
    """Main entry point - CLEAN SINGLE EXECUTION"""
    print("CGM-RGF Experimental Framework")
    print("=" * 30)
    print(
        "Testing the Common Governance Model - Recursive Gyrovector Formalism"
    )
    print()

    # Check what experiments are available
    print("Available experiments:")
    core_status = "✅" if CORE_EXPERIMENTS_AVAILABLE else "❌"
    print(f"• Core experiments: {core_status}")
    phys_status = "✅" if PHYSICAL_CONSTANTS_AVAILABLE else "❌"
    print(f"• Physical constants: {phys_status}")
    sing_status = "✅" if SINGULARITY_INFINITY_AVAILABLE else "❌"
    print(f"• Singularity/infinity: {sing_status}")
    print()

    # Run ONLY the comprehensive experiment (includes everything)
    if PHYSICAL_CONSTANTS_AVAILABLE:
        print("Running Comprehensive CGM Validation...")
        print("=" * 50)
        results = run_comprehensive_physical_dimensions()
        
        # Derive an overall flag if the comprehensive experiment didn't set one
        if 'overall_success' not in results:
                    try:
                        dim_ok = results.get('dimensions', {}).get('validation_passed', False)
                        integ = results.get('integration', {})
                        # Treat strings like "PASS"/"FAIL" as booleans, but prefer actual bools
                        def _as_bool(x): 
                            return (x is True) or (isinstance(x, str) and x.upper() == "PASS")
                        integ_ok = all([
                            bool(integ.get('core_theorems_passed', False)),
                            _as_bool(integ.get('gyrotriangle_closure', False)),
                            _as_bool(integ.get('bu_global_closure', False)),
                            bool(integ.get('una_orthogonality', False)),
                        ])
                        
                        # Constants must be DIAGNOSTIC_ONLY or actually pass
                        consts = results.get('constants', {})
                        const_mode = consts.get('status', '')
                        const_ok = consts.get('validation_passed', False) or const_mode in {"DIAGNOSTIC_ONLY", "QUARANTINED"}
                        
                        # Overall success requires dimensions + integration, and constants must be diagnostic or pass
                        results['overall_success'] = bool(dim_ok and integ_ok and const_ok)
                    except Exception:
                        results['overall_success'] = False

        print("✅ Comprehensive validation completed!")
        print()

        # Print ONLY the final summary
        print("=" * 60)
        print("FINAL CGM VALIDATION SUMMARY")
        print("=" * 60)

        # Extract key results from comprehensive experiment
        if "integration" in results:
            integration: Dict[str, Any] = results["integration"]
            core_passed = integration.get("core_theorem_count", 0)
            core_total = integration.get("total_core_theorems", 4)
            print(f"Core Theorems: {core_passed}/{core_total} PASSED")

        if "dimensions" in results and isinstance(results["dimensions"], dict):
            dimensions: Dict[str, Any] = results["dimensions"]
            gyro_bool = bool(dimensions.get("gyrotriangle_closure", False))
            if gyro_bool:
                print("Gyrotriangle Closure: ✅ PASS")

        if "constants" in results and isinstance(results["constants"], dict):
            constants: Dict[str, Any] = results["constants"]
            if constants.get("status") == "DIAGNOSTIC_ONLY":
                print("Physical constants: (diagnostic only; not scored)")
                if "c_error" in constants:
                    c_error_val = constants["c_error"]
                    if isinstance(c_error_val, (int, float)):
                        print(f"Speed of Light: anchored (error: {c_error_val:.1e})")
                if "G_error" in constants:
                    g_error_val = constants["G_error"]
                    if isinstance(g_error_val, (int, float)):
                        print(f"Gravitational Constant: requires κ theory (error: {g_error_val:.1e})")
            else:
                # Show regular scoring if constants are actually validated
                if "c_error" in constants:
                    c_error_val = constants["c_error"]
                    if isinstance(c_error_val, (int, float)):
                        c_status = "✅" if c_error_val < 1.0 else "❌"
                        print(f"Speed of Light: {c_status} (error: {c_error_val:.1e})")
                if "G_error" in constants:
                    g_error_val = constants["G_error"]
                    if isinstance(g_error_val, (int, float)):
                        g_status = "✅" if g_error_val < 1.0 else "❌"
                        print(f"Gravitational Constant: {g_status} (error: {g_error_val:.1e})")

        overall_success = results.get("overall_success", False)
        overall_status = "✅ PASS" if overall_success else "❌ NEEDS WORK"
        print(f"\nOverall Status: {overall_status}")

        # Save results
        try:
            np.save("cgm_results.npy", results)
            print("\n💾 Results saved to: cgm_results.npy")
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")
    else:
        print("❌ Physical constants experiments not available")
        print("Run individual experiments instead")


if __name__ == "__main__":
    main()


# Alias for backward compatibility
run_all_tests = run_comprehensive_physical_dimensions
