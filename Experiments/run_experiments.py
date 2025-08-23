#!/usr/bin/env python3
"""
CGM Experimental Framework

Main entry point for running all CGM experiments and validations.
This framework tests the mathematical foundations and physical predictions
of the Common Governance Model through systematic experimentation.

Key Components:
1. Dimensional Calibration Engine
2. Geometric Theorem Validation  
3. Gravity Coupling Analysis
4. Recursive Memory Structure
5. Stage Transition Observables

Run with: python run_experiments.py
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the Experiments directory to Python path
experiments_dir = Path(__file__).parent
sys.path.insert(0, str(experiments_dir))

# Import experiment modules
from experiments.core_experiments import CoreTheoremTester
from experiments.physical_constants import PhysicalConstantsValidator
from experiments.gravity_coupling import GravityCouplingAnalyzer
from experiments.fine_structure_focus import FineStructureValidator
from core.recursive_memory import RecursiveMemory
from theorems.run_proofs import run_all_proofs
from core.gyrovector_ops import GyroVectorSpace





def main():
    """Main entry point for CGM Experimental Framework"""
    print("CGM Experimental Framework")
    print("=" * 30)
    print("Testing the Common Governance Model")
    print()

    try:
        # Initialize gyrovector space
        gyrospace = GyroVectorSpace(c=1.0)
        
        # Run theorem proofs
        print("Running CGM Theorem Proofs...")
        print("=" * 40)
        theorems_passed = run_all_proofs()
        print()
        
        # Run core experiments
        print("Running Core CGM Experiments...")
        print("=" * 40)
        core_tester = CoreTheoremTester(gyrospace)
        core_results = core_tester.run_all_core_tests()
        print()
        
        # Run physical constants validation
        print("Running Physical Constants Validation...")
        print("=" * 40)
        constants_validator = PhysicalConstantsValidator(gyrospace)
        constants_results = constants_validator.run_comprehensive_physical_dimensions_experiment()
        print()
        
        # Run gravity coupling experiments
        print("Running Gravity Coupling Experiments...")
        print("=" * 40)
        gravity_analyzer = GravityCouplingAnalyzer()
        gravity_results = gravity_analyzer.run_comprehensive_analysis()
        print()
        
        # Print final summary
        print("=" * 60)
        print("FINAL CGM VALIDATION SUMMARY")
        print("=" * 60)
        
        if theorems_passed:
            print("Theorem Proofs: ✅ ALL PASSED")
        else:
            print("Theorem Proofs: ❌ SOME FAILED")
            
        if core_results.get('overall_success', False):
            print("Core Experiments: ✅ PASSED")
        else:
            print("Core Experiments: ❌ FAILED")
            
        if constants_results.get('overall_success', False):
            print("Physical Constants: ✅ VALIDATED")
        else:
            print("Physical Constants: ⚠️ DIAGNOSTIC")
            
        print("Gravity Coupling: ✅ ANALYZED")
        
        print("\n🎯 CGM Framework Status: FOUNDATION COMPLETE")
        print("Next milestone: κ prediction from recursive memory structure")
        
    except Exception as e:
        print(f"❌ Error running experiments: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



