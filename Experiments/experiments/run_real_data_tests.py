#!/usr/bin/env python3
"""
CGM Empirical Validation Suite
Run all empirical tests against real observational data.
"""

import os
import sys
from cgm_data_manager import CGMDataManager
from planck_cmb_test import PlanckComptonYTest
from etherington_cmb_coherence_test import EtheringtonComptonYCoherenceTest
from supernova_hubble_residuals_test import SupernovaHubbleResidualsTest


def run_test_a(data_manager):
    """Run Test A: Planck Compton-y Map Test."""
    print("\nTest A: Planck Compton-y Map")
    print("-" * 40)
    
    test = PlanckComptonYTest()
    test.set_data_manager(data_manager)
    return test.run_test()


def run_test_b(data_manager):
    """Run Test B: Etherington Compton-y Coherence Test."""
    print("\nTest B: Etherington Compton-y Coherence")
    print("-" * 40)
    
    test = EtheringtonComptonYCoherenceTest()
    test.set_data_manager(data_manager)
    return test.run_test()


def run_test_c(data_manager):
    """Run Test C: Supernova Hubble Residuals Test."""
    print("\nTest C: Supernova Hubble Residuals")
    print("-" * 40)
    
    test = SupernovaHubbleResidualsTest()
    test.set_data_manager(data_manager)
    return test.run_test()


def main():
    """Run the complete CGM empirical validation suite."""
    print("CGM EMPIRICAL VALIDATION SUITE")
    print("=" * 50)
    print("Testing Common Governance Model against observations")
    print()
    
    print("Starting validation...")
    
    # Initialize data manager and load data
    data_manager = CGMDataManager()
    print("Loading Planck data...")
    data_manager.load_data()
    print("Data loaded successfully!")
    
    # Run all tests
    test_a_results = run_test_a(data_manager)
    test_b_results = run_test_b(data_manager)
    test_c_results = run_test_c(data_manager)
    
    # Compile results
    print("\n" + "=" * 50)
    print("OVERALL RESULTS")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test A results
    if test_a_results.get("overall_passes", False):
        print("Test A (Planck): PASS")
        tests_passed += 1
    else:
        print("Test A (Planck): FAIL")
    
    # Test B results
    if test_b_results.get("overall_passes", False):
        print("Test B (Etherington): PASS")
        tests_passed += 1
    else:
        print("Test B (Etherington): FAIL")
    
    # Test C results
    if test_c_results.get("overall_passes", False):
        print("Test C (Supernova): PASS")
        tests_passed += 1
    else:
        print("Test C (Supernova): FAIL")
    
    # Summary
    print(f"\nSummary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests PASSED - CGM validation successful")
    else:
        print("✗ Some tests FAILED - CGM validation incomplete")
    
    # Show memory usage
    try:
        mem_info = data_manager.get_memory_usage()
        print(f"\nMemory usage: {mem_info}")
    except:
        pass
    
    print("\n" + "=" * 80)
    print("EMPIRICAL VALIDATION COMPLETE")
    print("=" * 80)
    
    return {
        "test_a": test_a_results,
        "test_b": test_b_results,
        "test_c": test_c_results,
        "overall_passed": tests_passed,
        "total_tests": total_tests,
        "data_manager": data_manager
    }


if __name__ == "__main__":
    main()
