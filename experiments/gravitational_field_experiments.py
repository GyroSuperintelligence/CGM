#!/usr/bin/env python3
"""
Gravitational Field Experiments for CGM

This module probes for the dimensionless coupling κ that relates CGM geometry
to the gravitational constant G via: G = ħ c / (κ² m_anchor²)

The goal is to derive κ from CGM's geometric structure rather than fitting
it to match experimental G.
"""

import numpy as np
from typing import Dict, Any, List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.functions.gyrovector_ops import GyroVectorSpace
from experiments.stages.bu_stage import BUStage
from experiments.stages.ona_stage import ONAStage


def estimate_kappa_from_geometry(gs: GyroVectorSpace, n: int = 200) -> Dict[str, Any]:
    """
    Estimate the dimensionless coupling κ from CGM geometric structure.
    
    This probes BU/ONA geometry to find a scale-free observable that can
    be mapped to κ via a theoretical relationship.
    
    Args:
        gs: Gyrovector space instance
        n: Number of random configurations to sample
        
    Returns:
        Dictionary with geometric curvature proxy and κ estimate
    """
    bu = BUStage(gs)
    ona = ONAStage(gs)
    rng = np.random.default_rng(0)  # Fixed seed for reproducibility
    
    # Build scale-free observables from existing defects
    curvature_proxies = []
    commutativity_defects = []
    associativity_defects = []
    monodromy_measures = []
    
    for _ in range(n):
        # Generate random test configuration
        U = rng.normal(0, 1, (3, 3))
        vecs = [U[i] / np.linalg.norm(U[i]) for i in range(3)]
        
        # Measure BU stage properties
        comm_def, assoc_def = bu.coaddition_check(vecs[0], vecs[1])
        commutativity_defects.append(comm_def)
        associativity_defects.append(assoc_def)
        
        # Measure ONA stage monodromy
        mono = ona.monodromy_measure(vecs)
        monodromy_measures.append(mono)
        
        # Candidate scale-free "curvature" proxy
        # This combines BU defects with ONA monodromy in a dimensionless way
        curvature_proxy = (comm_def + assoc_def) / max(mono, 1e-12)
        curvature_proxies.append(curvature_proxy)
    
    # Compute statistics
    curvature_median = float(np.median(curvature_proxies))
    curvature_mean = float(np.mean(curvature_proxies))
    curvature_std = float(np.std(curvature_proxies))
    
    # Report only dimensionless invariants; mapping to κ is a theoretical task
    # But provide κ_estimate for diagnostic purposes
    kappa_estimate = 1.0 / np.sqrt(max(curvature_median, 1e-18))
    
    # Enhanced κ prediction using holonomy-closure relationship
    # κ_geom = C / √(curvature) where C is fixed by theoretical normalization
    # We can estimate C from the canonical gyrotriangle closure condition
    canonical_curvature = 0.5  # Expected curvature for canonical α=π/2, β=γ=π/4
    C_theoretical = kappa_estimate * np.sqrt(canonical_curvature)
    
    # Now predict κ for the actual measured curvature
    kappa_holonomy = C_theoretical / np.sqrt(max(curvature_median, 1e-18))
    
    return {
        "curvature_proxy_median": curvature_median,
        "curvature_proxy_mean": curvature_mean,
        "curvature_proxy_std": curvature_std,
        "commutativity_defects": commutativity_defects,
        "associativity_defects": associativity_defects,
        "monodromy_measures": monodromy_measures,
        "n_samples": n,
        "kappa_estimate": kappa_estimate,
        "kappa_holonomy": kappa_holonomy,
        "C_theoretical": C_theoretical,
        "canonical_curvature": canonical_curvature,
        "note": "Report only dimensionless invariants; mapping to κ is a theoretical task."
    }


def analyze_kappa_scaling(gs: GyroVectorSpace, anchor_masses: List[float]) -> Dict[str, Any]:
    """
    Analyze how κ scales with different anchor masses.
    
    This helps understand the relationship between CGM's geometric κ
    and the physical requirement κ_required = sqrt(ħc/(G m²)).
    
    Args:
        gs: Gyrovector space instance
        anchor_masses: List of anchor masses to test
        
    Returns:
        Dictionary with κ scaling analysis
    """
    hbar = 1.054571817e-34  # J⋅s
    c = 2.99792458e8        # m/s
    G = 6.67430e-11         # m³/kg⋅s²
    
    kappa_required_values = []
    for m_anchor in anchor_masses:
        kappa_req = np.sqrt(hbar * c / (G * m_anchor**2))
        kappa_required_values.append(kappa_req)
    
    # Get geometric κ estimate
    geometry_result = estimate_kappa_from_geometry(gs)
    kappa_geom = geometry_result["kappa_estimate"]
    
    # Analyze scaling relationships
    scaling_analysis = {
        "anchor_masses": anchor_masses,
        "kappa_required": kappa_required_values,
        "kappa_geometric": kappa_geom,
        "scaling_ratios": [k_req / kappa_geom for k_req in kappa_required_values],
        "mass_scaling_exponent": None,  # To be determined from data
        "note": "Compare kappa_required vs kappa_geometric to understand mass scaling"
    }
    
    return scaling_analysis


def run_gravitational_field_experiment() -> Dict[str, Any]:
    """
    Main experiment to probe gravitational coupling from CGM geometry.
    
    Returns:
        Complete experimental results for κ estimation
    """
    print("Running Gravitational Field Experiment")
    print("=" * 40)
    print("Probing CGM geometry for dimensionless coupling κ")
    print("Goal: G = ħ c / (κ² m_anchor²)")
    print()
    
    # Initialize gyrovector space
    gs = GyroVectorSpace(c=1.0)
    
    # Estimate κ from geometry
    geometry_result = estimate_kappa_from_geometry(gs, n=500)
    
    # Test scaling with different anchor masses
    test_masses = [9.1093837015e-31, 1.67262192369e-27, 1.66053906660e-27]  # e⁻, p, n
    scaling_result = analyze_kappa_scaling(gs, test_masses)
    
    # Combine results
    results = {
        "experiment_name": "Gravitational Field κ-Probe",
        "geometry_probe": geometry_result,
        "scaling_analysis": scaling_result,
        "summary": {
            "kappa_geometric": geometry_result["kappa_estimate"],
            "curvature_proxy": geometry_result["curvature_proxy_median"],
            "next_steps": [
                "Theorize the proportionality constant C in κ = C/√(curvature)",
                "Compare κ_geometric with κ_required for various anchor masses",
                "Develop geometric interpretation of the curvature proxy"
            ]
        }
    }
    
    # Print results
    print(f"Geometric curvature proxy: {geometry_result['curvature_proxy_median']:.6f}")
    print(f"κ estimate (geometric): {geometry_result['kappa_estimate']:.3e}")
    print()
    print("κ required for different anchor masses:")
    for i, mass in enumerate(test_masses):
        mass_name = ["electron", "proton", "neutron"][i]
        kappa_req = scaling_result["kappa_required"][i]
        print(f"  {mass_name}: {kappa_req:.3e}")
    print()
    print("Next step: fit κ ∝ curvature^{-1/2} and compare to κ_required")
    
    return results


if __name__ == "__main__":
    results = run_gravitational_field_experiment()
    print("\n💾 Experiment completed. Results available in 'results' variable.")
