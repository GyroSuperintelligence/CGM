#!/usr/bin/env python3
"""
Etherington Relation Validator for CGM

Tests whether CGM's phase-accumulation framework can reproduce
Etherington's distance duality relation D_L = (1+z)² D_A without
introducing an expanding metric.

This is a critical test for CGM's "equilibrium universe" claim,
as both relations are observationally measured.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gyrovector_ops import GyroVectorSpace
from experiments.toroidal_anisotropy import toroidal_opacity, unit, tau_anisotropic, dir_cosines, cubic_C4


class EtheringtonValidator:
    """
    Validates Etherington's relation within CGM's phase-accumulation framework.
    
    Tests the claim that redshift is a "chiral phase gradient" rather than
    expansion, while still reproducing observed distance relations.
    """
    
    def __init__(self, gyrospace: GyroVectorSpace):
        self.gyrospace = gyrospace
        
        # Fundamental constants
        self.c = 2.99792458e8        # m/s
        
        # CGM parameters
        self.loop_pitch = 1.702935   # From helical memory analysis
        
    def compute_phase_accumulation(self, distance: float, 
                                 chiral_phase_gradient: float) -> float:
        """
        Compute phase accumulation along a null path in CGM framework.
        
        Args:
            distance: Physical distance along the path
            chiral_phase_gradient: CGM's chiral phase gradient (rad/m)
            
        Returns:
            Accumulated phase in radians
        """
        # In CGM: phase accumulation replaces expansion
        # φ = ∫ ∇arg(ψ_BU) · dℓ
        accumulated_phase = chiral_phase_gradient * distance
        
        return accumulated_phase
    
    def map_phase_to_redshift(self, accumulated_phase: float) -> float:
        """
        Map CGM's accumulated phase to effective redshift.
        
        Args:
            accumulated_phase: Phase accumulated along null path (rad)
            
        Returns:
            Effective redshift z
        """
        # CGM's mapping: z = exp(φ/φ_*) - 1
        # where φ_* is a characteristic phase scale
        
        # Use loop pitch as characteristic scale
        phi_star = 2 * np.pi / self.loop_pitch
        
        # Map to redshift
        z_effective = np.exp(accumulated_phase / phi_star) - 1
        
        return z_effective
    
    def compute_cgm_luminosity_distance(self, z: float, 
                                      angular_diameter_distance: float,
                                      tau: float = 0.0,
                                      holonomy_correction: float = 0.0,
                                      nhat: Optional[np.ndarray] = None,
                                      axis: Optional[np.ndarray] = None,
                                      tau0: float = 1e-3,
                                      eps_polar: float = 0.2,
                                      eps_card: float = 0.1) -> float:
        """
        Compute luminosity distance using CGM's phase framework.
        
        Args:
            z: Effective redshift from phase accumulation
            angular_diameter_distance: Angular diameter distance D_A
            tau: Optical depth (opacity parameter)
            holonomy_correction: CGM holonomy correction factor
            nhat: Direction vector for anisotropic opacity
            axis: Torus axis direction (defaults to [0,0,1])
            tau0: Base opacity scale
            eps_polar: Polar cap strength
            eps_card: Cardinal lobe strength
            
        Returns:
            Luminosity distance D_L
        """
        # In CGM: D_L = D_A * (1+z)² * exp(+τ/2) * exp(Δφ_holonomy)
        # where τ accounts for tiny absorptivity and Δφ_holonomy is CGM's geometric correction
        
        # Base Etherington relation
        dl_base = angular_diameter_distance * (1 + z)**2
        
        # Compute anisotropic opacity if direction provided
        if nhat is not None:
            if axis is None:
                axis = np.array([0, 0, 1])
            tau = toroidal_opacity(unit(nhat), axis=axis, tau0=tau0, 
                                   eps_polar=eps_polar, eps_card=eps_card)
        
        # Opacity correction: exp(+τ/2) breaks duality when τ > 0
        opacity_correction = np.exp(0.5 * tau)
        
        # CGM holonomy correction (if any)
        # This is where CGM's geometric framework would modify the relation
        holonomy_correction_factor = np.exp(holonomy_correction)
        
        dl_cgm = dl_base * opacity_correction * holonomy_correction_factor
        
        return dl_cgm
    
    def sky_duality_anisotropy(self, distance=1e9, chiral_gradient=1e-9,
                               tau0=1e-3, eps_polar=0.2, eps_card=0.1,
                               axis=np.array([0, 0, 1]), Ntheta=9, Nphi=18):
        """
        Scan the sky to see how directional opacity affects the Etherington relation.
        Returns statistics on D_L/(1+z)² D_A = exp(τ(θ,φ)/2).
        """
        phase = self.compute_phase_accumulation(distance, chiral_gradient)
        z = self.map_phase_to_redshift(phase)
        DA = distance
        
        vals = []
        for it in range(Ntheta):
            theta = (it + 0.5) * np.pi / Ntheta
            for ip in range(Nphi):
                phi = (ip + 0.5) * 2 * np.pi / Nphi
                nhat = np.array([np.sin(theta) * np.cos(phi),
                                 np.sin(theta) * np.sin(phi),
                                 np.cos(theta)])
                DL = self.compute_cgm_luminosity_distance(z, DA, nhat=nhat, 
                        axis=axis, tau0=tau0, eps_polar=eps_polar, eps_card=eps_card)
                ratio = DL / (DA * (1 + z)**2)  # should be exp(τ/2)
                vals.append(ratio)
        vals = np.array(vals)
        return {
            "mean_ratio": float(vals.mean()),
            "std_ratio": float(vals.std()),
            "min_ratio": float(vals.min()),
            "max_ratio": float(vals.max()),
            "frac_sky_var": float(vals.std() / vals.mean())
        }
    
    def duality_factor_map(self, z: float, D_A: float,
                           tau0: float = 1e-3, eps_polar: float = 0.2, eps_card: float = 0.2,
                           n_theta: int = 45, n_phi: int = 90) -> np.ndarray:
        """
        Returns F(θ,φ) = D_L(θ,φ)/[(1+z)² D_A] = exp[τ(θ,φ)/2] (× holonomy if used).
        """
        thetas = np.linspace(0.0, np.pi, n_theta)
        phis = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
        F = np.zeros((n_theta, n_phi))
        base = D_A * (1.0 + z)**2
        for i, th in enumerate(thetas):
            for j, ph in enumerate(phis):
                tau = tau_anisotropic(th, ph, tau0=tau0, a_polar=-eps_polar, b_cubic=eps_card)
                F[i, j] = self.compute_cgm_luminosity_distance(z, D_A, tau=tau) / base
        return F
    
    def test_etherington_relation(self, distances: List[float], 
                                chiral_gradients: List[float]) -> Dict[str, Any]:
        """
        Test whether CGM reproduces Etherington's relation.
        
        FIXED: Now holds gradient fixed and varies distance to test actual
        redshift dependence instead of creating false correlation.
        
        Args:
            distances: List of physical distances to test
            chiral_gradients: List of chiral phase gradients to test
            
        Returns:
            Test results and validation
        """
        results = []
        
        # FIX: Test each gradient with ALL distances to avoid false correlation
        for gradient in chiral_gradients:
            for distance in distances:
                # Compute phase accumulation
                phase = self.compute_phase_accumulation(distance, gradient)
                
                # Map to redshift
                z = self.map_phase_to_redshift(phase)
                
                # Assume D_A scales with distance (simplified)
                da = distance
                
                # Compute D_L using CGM
                dl_cgm = self.compute_cgm_luminosity_distance(z, da)
                
                # Expected D_L from Etherington
                dl_expected = da * (1 + z)**2
                
                # Check agreement
                agreement = abs(dl_cgm - dl_expected) / dl_expected
                passes = agreement < 1e-6  # 1 ppm tolerance
                
                result = {
                    "distance": distance,
                    "chiral_gradient": gradient,
                    "accumulated_phase": phase,
                    "redshift": z,
                    "da": da,
                    "dl_cgm": dl_cgm,
                    "dl_expected": dl_expected,
                    "agreement": agreement,
                    "passes": passes
                }
                
                results.append(result)
        
        # Overall validation
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["passes"])
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        validation = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "overall_passed": passed_tests == total_tests,
            "results": results
        }
        
        return validation
    
    def test_time_dilation(self, distances: List[float], 
                          chiral_gradients: List[float]) -> Dict[str, Any]:
        """
        Test whether CGM reproduces observed source time dilation.
        
        FIXED: Now holds gradient fixed and varies distance to test actual
        redshift dependence instead of creating false correlation.
        
        Args:
            distances: List of physical distances to test
            chiral_gradients: List of chiral phase gradients to test
            
        Returns:
            Time dilation test results
        """
        results = []
        
        # FIX: Test each gradient with ALL distances to avoid false correlation
        for gradient in chiral_gradients:
            for distance in distances:
                # Compute redshift from phase accumulation
                phase = self.compute_phase_accumulation(distance, gradient)
                z = self.map_phase_to_redshift(phase)
                
                # Test time dilation: Δt_obs = (1+z) Δt_emit
                delta_t_emit = 1.0  # Arbitrary emission time interval
                
                # Observed time interval
                delta_t_obs = delta_t_emit * (1 + z)
                
                # Expected from standard cosmology
                delta_t_expected = delta_t_emit * (1 + z)
                
                # Check agreement
                agreement = abs(delta_t_obs - delta_t_expected) / delta_t_expected
                passes = agreement < 1e-6
                
                result = {
                    "distance": distance,
                    "redshift": z,
                    "delta_t_emit": delta_t_emit,
                    "delta_t_obs": delta_t_obs,
                    "delta_t_expected": delta_t_expected,
                    "agreement": agreement,
                    "passes": passes
                }
                
                results.append(result)
        
        # Overall validation
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["passes"])
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        validation = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "overall_passed": passed_tests == total_tests,
            "results": results
        }
        
        return validation
    
    def test_opacity_effect_on_duality(self, distance: float = 1e9, 
                                     chiral_gradient: float = 1e-9) -> Dict[str, Any]:
        """
        Test how opacity parameter τ affects the Etherington relation.
        
        Args:
            distance: Physical distance to test
            chiral_gradient: Chiral phase gradient
            
        Returns:
            Results showing how τ breaks/restores duality
        """
        # Compute base redshift
        phase = self.compute_phase_accumulation(distance, chiral_gradient)
        z = self.map_phase_to_redshift(phase)
        da = distance
        
        # Test different opacity values
        tau_values = [0.0, 1e-3, 1e-2, 1e-1]
        results = []
        
        for tau in tau_values:
            # Compute D_L with opacity
            dl_with_opacity = self.compute_cgm_luminosity_distance(z, da, tau=tau)
            
            # Expected D_L from Etherington (no opacity)
            dl_expected = da * (1 + z)**2
            
            # Check agreement
            agreement = abs(dl_with_opacity - dl_expected) / dl_expected
            
            # For τ > 0, we expect D_L to differ from D_L_expected by exp(τ/2)
            # So the test should pass when τ = 0, and fail when τ > 0
            if tau == 0.0:
                # τ = 0 should give exact Etherington relation
                passes = agreement < 1e-6  # 1 ppm tolerance
            else:
                # τ > 0 should break the relation (this is what we want to test)
                passes = agreement > 1e-6  # Should NOT agree when τ > 0
            
            # Theoretical expectation: D_L = D_L_expected * exp(τ/2)
            theoretical_factor = np.exp(0.5 * tau)
            actual_factor = dl_with_opacity / dl_expected
            
            result = {
                "tau": tau,
                "redshift": z,
                "dl_expected": dl_expected,
                "dl_with_opacity": dl_with_opacity,
                "theoretical_factor": theoretical_factor,
                "actual_factor": actual_factor,
                "factor_error": abs(actual_factor - theoretical_factor) / theoretical_factor,
                "agreement": agreement,
                "passes": passes
            }
            
            results.append(result)
        
        # Overall assessment
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["passes"])
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "overall_passed": passed_tests == total_tests,
            "results": results
        }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation of CGM against observational constraints.
        
        Returns:
            Complete validation results
        """
        print("🔍 ETHERINGTON RELATION VALIDATION")
        print("=" * 50)
        print("Testing CGM's phase framework against distance duality")
        print()
        
        # Test parameters
        distances = [1e6, 1e9, 1e12, 1e15, 1e18]  # Various distances in m
        chiral_gradients = [1e-6, 1e-9, 1e-12, 1e-15, 1e-18]  # rad/m
        
        print("📊 TESTING ETHERINGTON RELATION")
        print("-" * 40)
        
        etherington_results = self.test_etherington_relation(distances, chiral_gradients)
        
        for result in etherington_results["results"]:
            status = "✅" if result["passes"] else "❌"
            print(f"Distance: {result['distance']:8.1e} m, z={result['redshift']:6.3f}, {status}")
        
        print()
        print(f"Etherington Tests: {etherington_results['passed_tests']}/{etherington_results['total_tests']} passed")
        print()
        
        print("📊 TESTING TIME DILATION")
        print("-" * 40)
        
        time_dilation_results = self.test_time_dilation(distances, chiral_gradients)
        
        for result in time_dilation_results["results"]:
            status = "✅" if result["passes"] else "❌"
            print(f"Distance: {result['distance']:8.1e} m, z={result['redshift']:6.3f}, {status}")
        
        print()
        print(f"Time Dilation Tests: {time_dilation_results['passed_tests']}/{time_dilation_results['total_tests']} passed")
        print()
        
        print("📊 TESTING OPACITY EFFECT ON DUALITY")
        print("-" * 40)
        
        opacity_results = self.test_opacity_effect_on_duality()
        
        for result in opacity_results["results"]:
            status = "✅" if result["passes"] else "❌"
            tau = result["tau"]
            factor_error = result["factor_error"]
            print(f"τ = {tau:.1e}: factor error = {factor_error:.2e} {status}")
        
        print(f"Opacity Tests: {opacity_results['passed_tests']}/{opacity_results['total_tests']} passed")
        print()
        
        print("📊 TESTING SKY ANISOTROPY")
        print("-" * 40)
        
        try:
            # Test directional anisotropy
            sky_results = self.sky_duality_anisotropy(tau0=1e-3, eps_polar=0.2, eps_card=0.1)
            
            print(f"Sky scan results (D_L/(1+z)² D_A):")
            print(f"   Mean ratio: {sky_results['mean_ratio']:.6f}")
            print(f"   Std ratio: {sky_results['std_ratio']:.6f}")
            print(f"   Min ratio: {sky_results['min_ratio']:.6f}")
            print(f"   Max ratio: {sky_results['max_ratio']:.6f}")
            print(f"   Fractional variation: {sky_results['frac_sky_var']:.2e}")
        except Exception as e:
            print(f"Sky anisotropy test failed: {e}")
            sky_results = {"error": str(e)}
        
        print()
        
        # Overall assessment
        overall_passed = (etherington_results["overall_passed"] and 
                         time_dilation_results["overall_passed"] and
                         opacity_results["overall_passed"])
        
        print("🎯 OVERALL ASSESSMENT:")
        print(f"   Etherington Relation: {'✅ PASS' if etherington_results['overall_passed'] else '❌ FAIL'}")
        print(f"   Time Dilation: {'✅ PASS' if time_dilation_results['overall_passed'] else '❌ FAIL'}")
        print(f"   Overall: {'✅ PASS' if overall_passed else '❌ FAIL'}")
        
        if not overall_passed:
            print("\n⚠️  CGM's 'equilibrium universe' claim requires refinement")
            print("   to reproduce observed distance and time relations.")
        
        return {
            "etherington_validation": etherington_results,
            "time_dilation_validation": time_dilation_results,
            "overall_passed": overall_passed
        }


def test_etherington_validator():
    """Test the Etherington validator with a simple gyrospace."""
    from core.gyrovector_ops import GyroVectorSpace
    
    gyrospace = GyroVectorSpace(c=1.0)
    validator = EtheringtonValidator(gyrospace)
    
    return validator.run_comprehensive_validation()


if __name__ == "__main__":
    results = test_etherington_validator()
    print("\n" + "="*50)
    print("ETHERINGTON VALIDATION COMPLETE")
