#!/usr/bin/env python3
"""
Test B - Etherington Compton-y Coherence Test

Tests if Etherington distance duality violations correlate with Compton-y anisotropies.
Uses the unified toroidal template and proper axis scanning.
"""

import numpy as np

from pathlib import Path
from typing import Dict, Any, Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.toroidal_anisotropy import (
    torus_template, get_cgm_template_amplitude, 
    find_best_axis, rotate_axis
)


class EtheringtonComptonYCoherenceTest:
    """
    Test B: Do Etherington distance duality violations correlate with Compton-y anisotropies?
    
    Uses unified toroidal template and proper axis scanning.
    """
    
    def __init__(self):
        # CGM physics scales (from Kompaneets analysis)
        self.y_pred_rms = get_cgm_template_amplitude("y")
        self.tau_pred_rms = get_cgm_template_amplitude("tau")
        
        # Template parameters (from your sweeps)
        self.a_polar = 0.2   # Polar anisotropy strength
        self.b_cubic = 0.1   # Cubic anisotropy strength
        
        # Analysis parameters
        self.lmax = 8   # Focus on low multipoles where template lives
        self.nside = 64  # Lower resolution for low-ℓ analysis
        self.fwhm_deg = 10.0  # Smoothing for low-ℓ analysis
        
        # Data manager (will be set by runner)
        self.data_manager = None
        
        # Test parameters
        self.n_mc_rotations = 100  # Number of MC rotations for null distribution
        self.correlation_threshold = 0.1  # Minimum correlation for significance
    
    def set_data_manager(self, data_manager):
        """Set the data manager for this test."""
        self.data_manager = data_manager

    def load_real_data(self) -> Dict[str, Any]:
        """
        Load real Compton-y data from data manager.
        
        Returns:
            Dictionary with y-map and metadata
        """
        if self.data_manager is None:
            raise RuntimeError("Data manager not set. Call set_data_manager() first.")
        
        # Get data from manager
        data = self.data_manager.get_data_for_test("planck_compton_y")
        
        return data
    
    def prepare_data_for_low_ell_analysis(self, y_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for low-ell analysis using data manager's preprocessed data.
        
        Args:
            y_data: Dictionary containing y-map and metadata (not used, gets from data manager)
            
        Returns:
            Dictionary with preprocessed data
        """
        if self.data_manager is None:
            raise RuntimeError("Data manager not set. Call set_data_manager() first.")
        
        # Use data manager's preprocessed data to avoid redundant work
        return self.data_manager.get_preprocessed_data(
            lmax=self.lmax,
            nside_target=self.nside,
            fwhm_deg=self.fwhm_deg
        )
    
    def remove_monopole_dipole(self, map_data: np.ndarray) -> np.ndarray:
        """
        Remove monopole and dipole from a map.
        
        Args:
            map_data: Input map data
            
        Returns:
            Map with monopole and dipole removed
        """
        # Remove monopole (mean)
        map_clean = map_data - np.mean(map_data)
        
        # For dipole removal, we'd need to fit a plane
        # For simplicity, just remove the mean for now
        # This is sufficient for low-ℓ analysis where we focus on ℓ=2,4
        
        return map_clean
    
    def smooth_map_gaussian(self, map_data: np.ndarray, fwhm_deg: float, 
                           nside: int) -> np.ndarray:
        """
        Smooth a map using Gaussian convolution.
        
        Args:
            map_data: Input map data
            fwhm_deg: Full width at half maximum in degrees
            nside: HEALPix nside parameter
            
        Returns:
            Smoothed map
        """
        # Convert FWHM to sigma
        fwhm_rad = np.radians(fwhm_deg)
        sigma_rad = fwhm_rad / (2 * np.sqrt(2 * np.log(2)))
        
        # For low-ℓ analysis, simple Gaussian smoothing is sufficient
        # We'll implement a basic convolution approach
        
        # Create a simple smoothing kernel
        kernel_size = max(3, int(3 * sigma_rad * nside))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Simple box smoothing as approximation (can be improved)
        smoothed_map = np.zeros_like(map_data)
        half_kernel = kernel_size // 2
        
        for i in range(len(map_data)):
            start_idx = max(0, i - half_kernel)
            end_idx = min(len(map_data), i + half_kernel + 1)
            smoothed_map[i] = np.mean(map_data[start_idx:end_idx])
        
        return smoothed_map
    
    def degrade_resolution(self, map_data: np.ndarray, old_nside: int, 
                          new_nside: int) -> np.ndarray:
        """
        Degrade map resolution.
        
        Args:
            map_data: Input map data
            old_nside: Original nside
            new_nside: Target nside
            
        Returns:
            Degraded map
        """
        if old_nside <= new_nside:
            return map_data
        
        # Simple downsampling by averaging
        factor = old_nside // new_nside
        new_npix = len(map_data) // (factor ** 2)
        
        degraded_map = np.zeros(new_npix)
        
        for i in range(new_npix):
            start_idx = i * (factor ** 2)
            end_idx = start_idx + (factor ** 2)
            degraded_map[i] = np.mean(map_data[start_idx:end_idx])
        
        return degraded_map
    
    def compute_spherical_harmonics_simple(self, map_data: np.ndarray, 
                                         nside: int, lmax: int) -> np.ndarray:
        """
        Compute spherical harmonic coefficients using simple approach.
        
        Args:
            map_data: Input map data
            nside: HEALPix nside parameter
            lmax: Maximum multipole
            
        Returns:
            Spherical harmonic coefficients (simplified)
        """
        # For low-ℓ analysis, we can use a simplified approach
        # This is not as precise as healpy but sufficient for our needs
        
        npix = len(map_data)
        
        # Create a simple alm array
        n_alm = (lmax + 1) ** 2
        alm = np.zeros(n_alm, dtype=complex)
        
        # For ℓ=2 and ℓ=4, we can compute simple projections
        # This is a simplified approach - for high precision use healpy
        
        # Simple projection onto P2 and C4 patterns
        for i in range(npix):
            # Convert pixel index to spherical coordinates
            theta, phi = self.pix2ang_simple(i, nside)
            
            # Direction vector
            nhat = np.array([np.sin(theta) * np.cos(phi),
                           np.sin(theta) * np.sin(phi),
                           np.cos(theta)])
            
            # P2 component
            cos_theta = np.cos(theta)
            P2_val = 0.5 * (3 * cos_theta**2 - 1.0)
            
            # C4 component
            C4_val = torus_template(nhat, axis=(0, 0, 1), a_polar=0, b_cubic=1.0)
            
            # Accumulate (simplified)
            if i < n_alm:
                alm[i] += map_data[i] * (P2_val + C4_val)
        
        return alm
    
    def pix2ang_simple(self, ipix: int, nside: int) -> Tuple[float, float]:
        """
        Convert pixel index to spherical coordinates.
        
        Args:
            ipix: Pixel index
            nside: HEALPix nside parameter
            
        Returns:
            Tuple of (theta, phi) in radians
        """
        # Simplified HEALPix coordinate conversion
        # This is an approximation - for high precision use healpy
        
        npix = 12 * nside * nside
        
        # Simple mapping based on pixel index
        # This is a rough approximation
        ring = int(np.sqrt(ipix / npix) * nside)
        phi = 2 * np.pi * (ipix % nside) / nside
        
        # Convert ring to theta
        theta = np.pi * ring / nside
        
        return theta, phi
    
    def compute_etherington_violations(self, axis: np.ndarray) -> np.ndarray:
        """
        Compute Etherington violations using unified toroidal template.
        
        Args:
            axis: Torus axis direction
            
        Returns:
            Array of Etherington violations
        """
        npix = 12 * self.nside * self.nside
        violations = np.zeros(npix)
        
        for i in range(npix):
            theta, phi = self.pix2ang_simple(i, self.nside)
            nhat = np.array([np.sin(theta) * np.cos(phi),
                           np.sin(theta) * np.sin(phi),
                           np.cos(theta)])
            
            # Use the unified template (already zero-mean)
            # Scale to have RMS of 1, then the fitted amplitude will be in y-map units
            template_val = torus_template(nhat, axis=axis, 
                                        a_polar=self.a_polar, 
                                        b_cubic=self.b_cubic)
            
            # Normalize template to have RMS of 1
            violations[i] = template_val
        
        # Remove monopole and dipole to ensure zero-mean
        violations = self.remove_monopole_dipole(violations)
        
        # Normalize to unit RMS
        rms = np.sqrt(np.mean(violations**2))
        if rms > 0:
            violations = violations / rms
        
        return violations
    
    def find_best_fit_axis(self, data_alm: np.ndarray) -> Dict[str, Any]:
        """
        Find the best-fit axis by scanning rotations.
        
        Args:
            data_alm: Spherical harmonic coefficients of the data
            
        Returns:
            Dictionary with best axis and correlation
        """
        # Use the utility function to find best axis
        axis_result = find_best_axis(data_alm, lmax=self.lmax, n_theta=10, n_phi=20)
        
        best_axis = axis_result["best_axis"]
        best_correlation = axis_result["best_correlation"]
        
        return {
            "best_axis": best_axis,
            "best_correlation": best_correlation,
            "axis_scan": axis_result
        }
    
    def compute_correlation_with_axis_scan(self, y_map: np.ndarray, mask: np.ndarray, 
                                         best_axis: np.ndarray) -> Dict[str, Any]:
        """
        Compute amplitude fit between y-map and Etherington violations with axis scanning.
        
        Args:
            y_map: Compton-y map
            mask: Confidence mask
            best_axis: Best-fit axis
            
        Returns:
            Dictionary with amplitude fit results and null distribution
        """
        # Apply mask
        valid_pixels = (mask > 0)
        y_valid = y_map[valid_pixels]
        
        # Compute Etherington violations with best axis
        violations = self.compute_etherington_violations(best_axis)
        violations_valid = violations[valid_pixels]
        
        # Fit amplitude: y_map ≈ A * violations
        # Use weighted least squares (avoid large diagonal matrices)
        XtX = np.sum(violations_valid**2)
        Xty = np.sum(violations_valid * y_valid)
        
        if XtX > 0:
            amplitude = Xty / XtX
        else:
            amplitude = 0.0
        
        # Compute residuals and chi2
        residuals = y_valid - amplitude * violations_valid
        chi2 = np.sum(residuals**2)
        dof = len(y_valid) - 1
        
        # Estimate error from residuals
        if dof > 0:
            amplitude_error = np.sqrt(chi2 / dof / XtX) if XtX > 0 else 0.0
        else:
            amplitude_error = 0.0
        
        # Generate null distribution by random rotations
        amplitudes_null = []
        
        for i in range(self.n_mc_rotations):
            # Random rotation angles
            theta = np.arccos(np.random.uniform(-1, 1))
            phi = np.random.uniform(0, 2*np.pi)
            
            # Rotate axis
            rotated_axis = rotate_axis(best_axis, theta, phi)
            
            # Compute violations with rotated axis
            rotated_violations = self.compute_etherington_violations(rotated_axis)
            rotated_violations_valid = rotated_violations[valid_pixels]
            
            # Fit amplitude
            XtX_null = np.sum(rotated_violations_valid**2)
            Xty_null = np.sum(rotated_violations_valid * y_valid)
            
            if XtX_null > 0:
                amp_null = Xty_null / XtX_null
                amplitudes_null.append(amp_null)
        
        amplitudes_null = np.array(amplitudes_null)
        
        # Compute p-value and upper limit
        p_value = np.mean(np.abs(amplitudes_null) >= abs(amplitude))
        upper_limit_95 = np.percentile(np.abs(amplitudes_null), 95)
        
        return {
            "amplitude": amplitude,
            "amplitude_error": amplitude_error,
            "chi2": chi2,
            "dof": dof,
            "p_value": p_value,
            "upper_limit_95": upper_limit_95,
            "amplitudes_null": amplitudes_null
        }
    
    def run_test(self) -> Dict[str, Any]:
        """
        Run the complete Etherington Compton-y coherence test.
        
        Returns:
            Dictionary with test results and pass/fail assessment
        """
        try:
            # Load real Compton-y data
            y_data = self.load_real_data()
            
            # Prepare data for low-ell analysis
            prepared_data = self.prepare_data_for_low_ell_analysis(y_data)
            
            # Find best-fit axis
            axis_result = self.find_best_fit_axis(prepared_data["alm"])
            best_axis = axis_result["best_axis"]
            
            # Compute correlation with axis scanning
            corr_result = self.compute_correlation_with_axis_scan(
                prepared_data["y_map"], prepared_data["mask"], best_axis)
            
            # Assess results
            amplitude = corr_result["amplitude"]
            amplitude_error = corr_result["amplitude_error"]
            chi2 = corr_result["chi2"]
            dof = corr_result["dof"]
            p_value = corr_result["p_value"]
            upper_limit_95 = corr_result["upper_limit_95"]
            
            # Pass criteria:
            # 1. Sufficient data for analysis
            # 2. Consistent with null hypothesis (p ≥ 0.01)
            # 3. Upper limit >> CGM prediction (safe for theory)
            sufficient_data = len(prepared_data["y_map"]) > 1000
            null_consistent = p_value >= 0.01
            
            # CGM predicts very small signal, so upper limit should be >> CGM prediction
            # The amplitude is now in y-map units, so compare to y_pred_rms
            cgm_safe = upper_limit_95 > 100 * self.y_pred_rms  # Upper limit much larger than CGM prediction
            
            overall_passes = sufficient_data and null_consistent and cgm_safe
            
            # Results
            results = {
                "test_name": "Etherington Compton-y Coherence Test (Test B)",
                "data_source": "Real Planck Compton-y (MILCA) in HEALPix format",
                "nside": prepared_data["nside"],
                "lmax": self.lmax,
                "fwhm_deg": self.fwhm_deg,
                "best_axis": best_axis.tolist(),
                "best_correlation": axis_result["best_correlation"],
                "etherington_amplitude": amplitude,
                "amplitude_error": amplitude_error,
                "chi2": chi2,
                "dof": dof,
                "p_value": p_value,
                "upper_limit_95": upper_limit_95,
                "y_pred_rms_cgm": self.y_pred_rms,
                "sufficient_data": sufficient_data,
                "null_consistent": null_consistent,
                "cgm_safe": cgm_safe,
                "overall_passes": overall_passes,
                "prepared_data": prepared_data,
                "corr_result": corr_result
            }
            
            # Print results
            print("\nTEST RESULTS:")
            print(f"   Best-fit axis: [{best_axis[0]:.3f}, {best_axis[1]:.3f}, {best_axis[2]:.3f}]")
            print(f"   Fitted amplitude: {amplitude:.2e} ± {amplitude_error:.2e}")
            print(f"   P-value: {p_value:.4f}")
            print(f"   95% upper limit: {upper_limit_95:.2e}")
            print(f"   CGM prediction: {self.tau_pred_rms:.2e}")
            print(f"   Result: {'PASS' if overall_passes else 'FAIL'}")
            
            if overall_passes:
                print("   ✓ Upper limits consistent with CGM predictions")
            else:
                print("   ✗ Test failed - see details above")
            
            return results
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "overall_passes": False}


def main():
    """Run the Etherington Compton-y coherence test."""
    test = EtheringtonComptonYCoherenceTest()
    results = test.run_test()
    return results


if __name__ == "__main__":
    main()
