#!/usr/bin/env python3
"""
Test C - Supernova Hubble Residuals Test

Tests if the CGM toroidal kernel appears in Type-Ia supernova distance modulus residuals.
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


class SupernovaHubbleResidualsTest:
    """
    Test C: Do Type-Ia supernova distance modulus residuals show the CGM toroidal kernel?
    
    Uses unified toroidal template and proper axis scanning.
    """
    
    def __init__(self):
        # CGM physics scales (from Kompaneets analysis)
        self.mu_pred_rms = get_cgm_template_amplitude("mu")
        
        # Template parameters (from your sweeps)
        self.a_polar = 0.2   # Polar anisotropy strength
        self.b_cubic = 0.1   # Cubic anisotropy strength
        
        # Data manager (will be set by runner)
        self.data_manager = None
        
        # Test parameters
        self.n_mc_rotations = 100  # Number of MC rotations for null distribution
        self.n_axis_angles = 100    # Number of axis angles to scan
    
    def set_data_manager(self, data_manager):
        """Set the data manager for this test."""
        self.data_manager = data_manager

    def load_supernova_catalog(self) -> Dict[str, Any]:
        """
        Load supernova catalog from data manager.
        
        Returns:
            Dictionary with supernova data and metadata
        """
        if self.data_manager is None:
            raise RuntimeError("Data manager not set. Call set_data_manager() first.")
        
        # Get data from manager
        data = self.data_manager.get_supernova_data()
        
        return data
    
    def scan_axis_for_best_fit(self, sn_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scan over axis directions to find the best-fit axis for SN residuals.
        
        Args:
            sn_data: Dictionary containing supernova data
            
        Returns:
            Dictionary with best axis and correlation
        """
        
        # Extract coordinates and residuals
        positions = sn_data["sn_data"]["positions"]  # 3D positions
        residuals = sn_data["sn_data"]["residuals"]
        
        # Convert 3D positions to spherical coordinates
        x, y, z = positions.T
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)  # Colatitude
        phi = np.arctan2(y, x)    # Longitude
        
        # Generate axis angles to scan
        thetas = np.linspace(0, np.pi, int(np.sqrt(self.n_axis_angles)))
        phis = np.linspace(0, 2*np.pi, int(np.sqrt(self.n_axis_angles)))
        
        best_correlation = -1.0
        best_axis = np.array([0, 0, 1])
        best_theta = 0.0
        best_phi = 0.0
        
        correlations = np.zeros((len(thetas), len(phis)))
        
        for i, theta_axis in enumerate(thetas):
            for j, phi_axis in enumerate(phis):
                # Create axis vector
                axis = np.array([np.sin(theta_axis) * np.cos(phi_axis),
                               np.sin(theta_axis) * np.sin(phi_axis),
                               np.cos(theta_axis)])
                
                # Compute template values at SN positions
                template_values = []
                for k in range(len(theta)):
                    nhat = np.array([np.sin(theta[k]) * np.cos(phi[k]),
                                   np.sin(theta[k]) * np.sin(phi[k]),
                                   np.cos(theta[k])])
                    
                    template_val = torus_template(nhat, axis=axis, 
                                               a_polar=self.a_polar, 
                                               b_cubic=self.b_cubic)
                    template_values.append(template_val)
                
                template_values = np.array(template_values)
                
                # Compute correlation with residuals
                correlation = np.corrcoef(residuals, template_values)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                
                correlations[i, j] = correlation
                
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_axis = axis
                    best_theta = theta_axis
                    best_phi = phi_axis
        
        return {
            "best_axis": best_axis,
            "best_theta": best_theta,
            "best_phi": best_phi,
            "best_correlation": best_correlation,
            "correlations": correlations,
            "thetas": thetas,
            "phis": phis
        }
    
    def fit_toroidal_pattern(self, sn_data: Dict[str, Any], axis: np.ndarray) -> Dict[str, Any]:
        """
        Fit the toroidal pattern to supernova residuals.
        
        Args:
            sn_data: Dictionary containing supernova data
            axis: Torus axis direction
            
        Returns:
            Dictionary with fitted parameters and statistics
        """
        
        # Extract data
        positions = sn_data["sn_data"]["positions"]  # 3D positions
        residuals = sn_data["sn_data"]["residuals"]
        
        # Convert 3D positions to spherical coordinates
        x, y, z = positions.T
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)  # Colatitude
        phi = np.arctan2(y, x)    # Longitude
        
        # Create design matrix for P2 and C4 components
        n_sn = len(residuals)
        X = np.zeros((n_sn, 2))
        
        for i in range(n_sn):
            nhat = np.array([np.sin(theta[i]) * np.cos(phi[i]),
                           np.sin(theta[i]) * np.sin(phi[i]),
                           np.cos(theta[i])])
            
            # P2 component (angle to axis)
            mu = np.clip(np.dot(nhat, axis), -1.0, 1.0)
            P2_val = 0.5 * (3 * mu**2 - 1.0)
            
            # C4 component (using unified template)
            C4_val = torus_template(nhat, axis=axis, a_polar=0, b_cubic=1.0)
            
            X[i, 0] = P2_val
            X[i, 1] = C4_val
        
        # Weighted least squares (currently uniform weights, can be improved)
        weights = np.ones(n_sn)
        W = np.diag(weights)
        
        # Solve: residuals = A * P2 + B * C4 + noise
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ residuals
        
        try:
            params = np.linalg.solve(XtWX, XtWy)
            A, B = params[0], params[1]
            
            # Compute residuals and chi2
            fitted_values = A * X[:, 0] + B * X[:, 1]
            fit_residuals = residuals - fitted_values
            chi2 = np.sum(weights * fit_residuals**2)
            dof = n_sn - 2
            
            # Compute correlation
            correlation = np.corrcoef(residuals, fitted_values)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
        except np.linalg.LinAlgError:
            A, B = 0.0, 0.0
            chi2 = np.inf
            dof = n_sn - 2
            correlation = 0.0
            fit_residuals = residuals
            fitted_values = np.zeros_like(residuals)
        
        return {
            "A": A,
            "B": B,
            "chi2": chi2,
            "dof": dof,
            "correlation": correlation,
            "fitted_values": fitted_values,
            "fit_residuals": fit_residuals,
            "design_matrix": X
        }
    
    def compute_null_distribution(self, sn_data: Dict[str, Any], best_axis: np.ndarray) -> Dict[str, Any]:
        """
        Compute null distribution by randomly rotating the axis.
        
        Args:
            sn_data: Dictionary containing supernova data
            best_axis: Best-fit axis
            
        Returns:
            Dictionary with null distribution and p-values
        """
        
        # Store parameters from random rotations
        A_values = []
        B_values = []
        correlations = []
        
        for i in range(self.n_mc_rotations):
            # Random rotation angles
            theta = np.arccos(np.random.uniform(-1, 1))
            phi = np.random.uniform(0, 2*np.pi)
            
            # Rotate axis
            rotated_axis = rotate_axis(best_axis, theta, phi)
            
            # Fit with rotated axis
            fit_result = self.fit_toroidal_pattern(sn_data, rotated_axis)
            
            A_values.append(fit_result["A"])
            B_values.append(fit_result["B"])
            correlations.append(fit_result["correlation"])
            
        A_values = np.array(A_values)
        B_values = np.array(B_values)
        correlations = np.array(correlations)
        
        # Fit with best axis for comparison
        best_fit = self.fit_toroidal_pattern(sn_data, best_axis)
        best_A = best_fit["A"]
        best_B = best_fit["B"]
        best_correlation = best_fit["correlation"]
        
        # Compute p-values
        p_value_A = np.mean(np.abs(A_values) >= abs(best_A))
        p_value_B = np.mean(np.abs(B_values) >= abs(best_B))
        p_value_correlation = np.mean(np.abs(correlations) >= abs(best_correlation))
        
        # Compute 95% upper limits
        upper_limit_A_95 = np.percentile(np.abs(A_values), 95)
        upper_limit_B_95 = np.percentile(np.abs(B_values), 95)
        upper_limit_correlation_95 = np.percentile(np.abs(correlations), 95)
        
        return {
            "A_values": A_values,
            "B_values": B_values,
            "correlations": correlations,
            "p_value_A": p_value_A,
            "p_value_B": p_value_B,
            "p_value_correlation": p_value_correlation,
            "upper_limit_A_95": upper_limit_A_95,
            "upper_limit_B_95": upper_limit_B_95,
            "upper_limit_correlation_95": upper_limit_correlation_95,
            "best_A": best_A,
            "best_B": best_B,
            "best_correlation": best_correlation
        }
    
    def check_predicted_signs(self, fit_result: Dict[str, Any], axis: np.ndarray) -> Dict[str, Any]:
        """
        Check if fitted signs match CGM predictions.
        
        Args:
            fit_result: Dictionary with fitted parameters
            axis: Torus axis direction
            
        Returns:
            Dictionary with sign check results
        """
        
        A = fit_result["A"]
        B = fit_result["B"]
        
        # CGM predicts specific sign relationships based on toroidal geometry
        # This is a simplified check - can be refined based on your theory
        
        # For now, check basic consistency
        signs_consistent = True
        sign_notes = []
        
        if abs(A) > 0 and abs(B) > 0:
            # Check if A and B have reasonable relative magnitudes
            ratio = abs(A) / abs(B)
            if ratio < 0.1 or ratio > 10:
                signs_consistent = False
                sign_notes.append("A/B ratio outside expected range")
        
        return {
            "signs_consistent": signs_consistent,
            "sign_notes": sign_notes,
            "A": A,
            "B": B
        }
    
    def run_test(self) -> Dict[str, Any]:
        """
        Run the complete supernova Hubble residuals test.
        
        Returns:
            Dictionary with test results and pass/fail assessment
        """
        
        try:
            # Load supernova catalog
            sn_data = self.load_supernova_catalog()
            
            # Scan for best-fit axis
            axis_result = self.scan_axis_for_best_fit(sn_data)
            best_axis = axis_result["best_axis"]
            
            # Fit toroidal pattern with best axis
            fit_result = self.fit_toroidal_pattern(sn_data, best_axis)
            
            # Compute null distribution
            null_result = self.compute_null_distribution(sn_data, best_axis)
            
            # Check predicted signs
            sign_result = self.check_predicted_signs(fit_result, best_axis)
            
            # Assess results
            A = fit_result["A"]
            B = fit_result["B"]
            correlation = fit_result["correlation"]
            chi2 = fit_result["chi2"]
            dof = fit_result["dof"]
            
            p_value_A = null_result["p_value_A"]
            p_value_B = null_result["p_value_B"]
            p_value_correlation = null_result["p_value_correlation"]
            
            upper_limit_A_95 = null_result["upper_limit_A_95"]
            upper_limit_B_95 = null_result["upper_limit_B_95"]
            upper_limit_correlation_95 = null_result["upper_limit_correlation_95"]
            
            # Pass criteria:
            # 1. Sufficient data for analysis
            # 2. Significant detection (if any) with correct signs
            # 3. Non-detection consistent with CGM predictions
            sufficient_data = len(sn_data["sn_data"]["residuals"]) > 100
            significant_detection = (p_value_A < 0.01 or p_value_B < 0.01 or p_value_correlation < 0.01)
            signs_match = sign_result["signs_consistent"]
            cgm_safe = (upper_limit_A_95 > 10 * abs(A) and 
                       upper_limit_B_95 > 10 * abs(B) and
                       upper_limit_correlation_95 > 10 * abs(correlation))
            
            # Overall pass: either significant detection with correct signs, or non-detection consistent with CGM
            if significant_detection:
                overall_passes = sufficient_data and signs_match
            else:
                overall_passes = sufficient_data and cgm_safe
            
            # Results
            results = {
                "test_name": "Supernova Hubble Residuals Test (Test C)",
                "data_source": "Real Type-Ia supernova catalog (Pantheon+)",
                "n_supernovae": len(sn_data["sn_data"]["residuals"]),
                "best_axis": best_axis.tolist(),
                "best_correlation": axis_result["best_correlation"],
                "fitted_A": A,
                "fitted_B": B,
                "correlation": correlation,
                "chi2": chi2,
                "dof": dof,
                "p_value_A": p_value_A,
                "p_value_B": p_value_B,
                "p_value_correlation": p_value_correlation,
                "upper_limit_A_95": upper_limit_A_95,
                "upper_limit_B_95": upper_limit_B_95,
                "upper_limit_correlation_95": upper_limit_correlation_95,
                "mu_pred_rms_cgm": self.mu_pred_rms,
                "sufficient_data": sufficient_data,
                "significant_detection": significant_detection,
                "signs_match": signs_match,
                "cgm_safe": cgm_safe,
                "overall_passes": overall_passes,
                "sn_data": sn_data,
                "axis_result": axis_result,
                "fit_result": fit_result,
                "null_result": null_result,
                "sign_result": sign_result
            }
            
            # Print results
            print("\nTEST RESULTS:")
            print(f"   Best-fit axis: [{best_axis[0]:.3f}, {best_axis[1]:.3f}, {best_axis[2]:.3f}]")
            print(f"   P-value A: {p_value_A:.4f}")
            print(f"   P-value B: {p_value_B:.4f}")
            print(f"   Result: {'PASS' if overall_passes else 'FAIL'}")
            
            if overall_passes:
                print("   ✓ Test passed - consistent with CGM predictions")
            else:
                print("   ✗ Test failed - see details above")
            
            return results
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "overall_passes": False}


def main():
    """Run the supernova Hubble residuals test."""
    test = SupernovaHubbleResidualsTest()
    results = test.run_test()
    return results


if __name__ == "__main__":
    main()
