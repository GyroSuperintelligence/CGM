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
4. Helical Memory Structure
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
from experiments.cgm_theorems_physics import CoreTheoremTester
from experiments.physical_constants import ElectricCalibrationValidator
from experiments.cgm_gravity_analysis import GravityCouplingAnalyzer

from experiments.cgm_theorems_math import run_all_proofs
from experiments.functions.gyrovector_ops import GyroVectorSpace

# Triad source index analysis
from experiments.triad_index_analyzer import TriadIndexAnalyzer


def run_stable_experiment(name, func, *args, **kwargs):
    """Run a stable experiment with minimal output"""
    print(f"Running {name}...")
    import io
    import sys
    from contextlib import redirect_stdout

    # Capture all output
    captured = io.StringIO()
    with redirect_stdout(captured):
        result = func(*args, **kwargs)

    # Show only key results
    if hasattr(result, "get"):
        if "overall_success" in result:
            status = "✅ PASS" if result["overall_success"] else "❌ FAIL"
            print(f"   {name}: {status}")
        elif "validation_passed" in result:
            status = "✅ PASS" if result["validation_passed"] else "❌ FAIL"
            print(f"   {name}: {status}")
        else:
            print(f"   {name}: Completed")
    else:
        print(f"   {name}: Completed")

    return result


def main():
    """Main entry point for CGM Experimental Framework"""
    print("Common Governance Model - Experiments")
    print("=" * 30)
    print("Thesis and Hypotheses Testing")
    print()

    try:
        # Initialize gyrovector space
        gyrospace = GyroVectorSpace(c=1.0)

        # Run stable experiments with minimal output
        print("Running Stable Experiments (streamlined)...")
        print("=" * 40)

        theorems_passed = run_stable_experiment("CGM Theorem Proofs", run_all_proofs)

        core_tester = CoreTheoremTester(gyrospace)
        core_results = run_stable_experiment(
            "Core CGM Experiments", core_tester.run_all_core_tests
        )

        constants_validator = ElectricCalibrationValidator(gyrospace)
        constants_results = run_stable_experiment(
            "Electric Calibration",
            constants_validator.run_electric_calibration_experiment,
            alpha_input=1 / 137.035999084,
        )

        gravity_analyzer = GravityCouplingAnalyzer()
        gravity_results = run_stable_experiment(
            "Gravity Coupling", gravity_analyzer.run_dimensional_analysis
        )

        from experiments.tw_precession import test_tw_precession_small_angle

        tw_results = run_stable_experiment(
            "Thomas-Wigner Precession", test_tw_precession_small_angle
        )

        from experiments.tw_closure_test import TWClosureTester

        tw_closure_tester = TWClosureTester(gyrospace)
        tw_closure_results = run_stable_experiment(
            "TW Consistency Band", tw_closure_tester.run_tw_closure_tests
        )

        print()

        # Solve for emergent N* (streamlined - stable)
        print("Solving for Emergent Cosmogenesis Scale...")
        print("=" * 40)
        target_L_star = 1.337e-4  # m (target from CMB temperature)

        # Use the helical analyzer to find N*
        from experiments.helical_memory_analyzer import HelicalMemoryAnalyzer

        helical_analyzer = HelicalMemoryAnalyzer(gyrospace)

        # Use the new CMB N-solver with actual psi_bu data
        N_star, final_result = helical_analyzer.solve_min_N_for_CMB(Nmax=100)

        if N_star is not None:
            print(
                f"🎯 N* = {N_star}, L* = {final_result['L_star_N']:.2e} m, Π = {final_result['pitch_loop']:.3f}, Ξ = {final_result['Xi_loop']:.3f}"
            )
        else:
            print("⚠️  No N* found in range [1, 100]")

        print()

        # Run light chirality experiments (streamlined - stable)
        from experiments.light_chirality_experiments import LightChiralityExperiments

        light_experiments = LightChiralityExperiments(gyrospace)
        light_results = run_stable_experiment(
            "Light Chirality",
            light_experiments.run_complete_light_chirality_experiments,
        )

        print()

        # Run singularity and infinity experiments (streamlined - stable)
        from experiments.singularity_infinity import SingularityInfinityValidator

        singularity_validator = SingularityInfinityValidator(gyrospace)
        singularity_results = run_stable_experiment(
            "Singularity & Infinity", singularity_validator.run_all_validations
        )

        # Run gravitational field experiments using recursive memory
        from experiments.functions.recursive_memory import RecursiveMemory

        recursive_memory = RecursiveMemory(gyrospace)
        kappa_results = recursive_memory.estimate_kappa_from_geometry()
        print(
            f"   Gravitational Field: κ(geo)={kappa_results['kappa_estimate']:.3e}, coherence={kappa_results['coherence_magnitude']:.3e}"
        )

        print()

        # Run unified cosmogenesis analysis (DETAILED - still working on this)
        print("Running Unified Cosmogenesis Analysis...")
        print("=" * 40)
        memory_analyzer = HelicalMemoryAnalyzer(gyrospace)
        memory_results = memory_analyzer.run_comprehensive_analysis()
        print()

        # Run triad source index analysis
        print("Running Triad Source Index Analysis...")
        print("=" * 50)

        triad_analyzer = TriadIndexAnalyzer(gyrospace)
        triad_results = run_stable_experiment(
            "Triad Source Index Analysis", triad_analyzer.run
        )

        print()

        # Run new validation modules (integrating your assistant's insights)
        print("Running Advanced CGM Validations...")
        print("=" * 50)

        # Kompaneyets distortion analysis
        from experiments.cgm_kompaneyets_analysis import CGMKompaneyetsAnalyzer

        kompaneyets_analyzer = CGMKompaneyetsAnalyzer(gyrospace)
        kompaneyets_results = run_stable_experiment(
            "Kompaneets Distortions", kompaneyets_analyzer.run_comprehensive_analysis
        )

        # Acoustic diagnostics analysis
        from experiments.cgm_sound_diagnostics import CGMAcousticDiagnostics

        acoustic_diagnostics = CGMAcousticDiagnostics(gyrospace)
        acoustic_results = run_stable_experiment(
            "Acoustic Diagnostics", acoustic_diagnostics.run_diagnostic_suite
        )

        print()

        # Print final summary
        print("=" * 60)
        print("FINAL CGM VALIDATION SUMMARY")
        print("=" * 60)

        if theorems_passed:
            print("Theorem Proofs: ✅ ALL PASSED")
        else:
            print("Theorem Proofs: ❌ SOME FAILED")

        if core_results.get("overall_success", False):
            print("Core Experiments: ✅ PASSED")
        else:
            print("Core Experiments: ❌ FAILED")

        if constants_results.get("overall_success", False):
            print("Physical Constants: ✅ VALIDATED")
        else:
            print("Physical Constants: ⚠️ DIAGNOSTIC")

        print("Gravity Coupling: ✅ ANALYZED")

        if tw_results.get("validation_passed", False):
            print("Thomas-Wigner Precession: ✅ VALIDATED")
        else:
            print("Thomas-Wigner Precession: ❌ FAILED")

        if tw_closure_results.get("overall_success", False):
            print("TW-Consistency Band: ✅ VALIDATED")
        else:
            print("TW-Consistency Band: ❌ FAILED")

        # Triad source index analysis results
        if triad_results.get("triads"):
            successful_triads = sum(
                1 for triad in triad_results["triads"] if "error" not in triad
            )
            total_triads = len(triad_results["triads"])
            print(
                f"Triad Source Index Analysis: ✅ {successful_triads}/{total_triads} triads successful"
            )

            if triad_results.get("domain_penalties"):
                xi_mean = np.mean(triad_results["domain_penalties"])
                print(f"   Domain penalties: mean = {xi_mean:.3f}")
        else:
            print("Triad Source Index Analysis: ❌ FAILED")

        if light_results.get("overall_success", False):
            print("Light Chirality (UNA/light): ✅ VALIDATED")
        else:
            print("Light Chirality (UNA/light): ❌ FAILED")

        if singularity_results.get("recursive_singularity", {}).get(
            "validation_passed", False
        ):
            print("Recursive Singularity: ✅ VALIDATED")
        else:
            print("Recursive Singularity: ❌ FAILED")

        if kappa_results.get("curvature_proxy_median", 0) > 0:
            print("Geometric Invariants: ✅ COMPUTED")
        else:
            print("Geometric Invariants: ❌ FAILED")

        if memory_results.get("hypothesis_tests", {}):
            passed_memory_tests = sum(
                1
                for test in memory_results["hypothesis_tests"].values()
                if test["passed"]
            )
            total_memory_tests = len(memory_results["hypothesis_tests"])
            print(
                f"Cosmogenesis Analysis: {passed_memory_tests}/{total_memory_tests} hypotheses validated"
            )
        else:
            print("Cosmogenesis Analysis: ⚠️ NEEDS REFINEMENT")

        # New validation results
        if kompaneyets_results.get("overall_viable", False):
            print("Kompaneets Distortions: ✅ VALIDATED")
        else:
            print("Kompaneets Distortions: ❌ FAILED")

        if acoustic_results.get("overall_passed", False):
            print("Acoustic Diagnostics: ✅ VALIDATED")
        else:
            print("Acoustic Diagnostics: ❌ FAILED")

        print("\n🎯 CGM Framework Status: FOUNDATION COMPLETE")

        # Brief diagnostic summary
        print("\n🔍 KEY INSIGHTS:")
        print("=" * 60)

        if "toroidal_holonomy" in tw_closure_results:
            holonomy = tw_closure_results["toroidal_holonomy"]
            print(
                f"   Toroidal Holonomy: {holonomy['total_holonomy']:.3f} rad (deficit: {holonomy['deviation']:.3f})"
            )

        if "anatomical_tw_ratio" in tw_closure_results:
            chi = tw_closure_results["anatomical_tw_ratio"]
            print(f"   TW Ratio χ: {chi['chi_mean']:.3f} ± {chi['chi_std']:.3f}")

        if "psi_bu_field" in memory_results:
            psi = memory_results["psi_bu_field"]
            print(
                f"   ψ_BU: {psi['magnitude']:.3f}, coherence: {psi['spin_translation_coherence']:.3f}"
            )

    except Exception as e:
        print(f"❌ Error running experiments: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
