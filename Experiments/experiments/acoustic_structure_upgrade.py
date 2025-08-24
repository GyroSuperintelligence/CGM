#!/usr/bin/env python3
"""
Acoustic Structure Upgrade: Geometric ℓ Mapping

Implements the assistant's guide requirement to replace manual "mode index" → ℓ 
mapping with standard geometric relations to confront the first-peak location.

This defines:
- Sound speed c_s ≈ c/√(3(1+R))
- Visibility window width (decoupling episode)
- Coherence length r_s = ∫ c_s dt over visibility window
- Peak mapping: ℓ_n ≈ nπ D_A/r_s with odd/even modulation
- Check if first peak lands near ℓ ≈ 220 without tuning both D_A and r_s freely
"""

import numpy as np
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


class AcousticStructureUpgrade:
    """
    Upgrades acoustic structure analysis with geometric ℓ mapping.
    """
    
    def __init__(self):
        # Physical constants
        self.c = 2.99792458e8  # m/s
        self.G = 6.67430e-11   # m³/(kg⋅s²)
        self.H0 = 70.0         # km/s/Mpc
        self.H0_SI = self.H0 * 1000 / (3.086e22)  # 1/s
        
        # Cosmological parameters (flat ΛCDM)
        self.Omega_m = 0.3
        self.Omega_Lambda = 0.7
        self.Omega_b = 0.022
        self.Omega_gamma = 5.38e-5  # Radiation density
        
        # Baryon-to-photon ratio
        self.eta = 6.1e-10
        
        # CMB temperature
        self.T_CMB = 2.725  # K
        
        # Recombination parameters
        self.z_rec = 1100
        self.z_drag = 1060  # Baryon drag epoch
        
    def sound_speed(self, z: float) -> float:
        """
        Sound speed in the photon-baryon fluid.
        
        Args:
            z: Redshift
            
        Returns:
            Sound speed in m/s
        """
        # Baryon-to-photon ratio evolution
        R = self._baryon_to_photon_ratio(z)
        
        # Sound speed: c_s = c/√(3(1+R))
        c_s = self.c / np.sqrt(3 * (1 + R))
        
        return c_s
    
    def _baryon_to_photon_ratio(self, z: float) -> float:
        """
        Baryon-to-photon ratio as function of redshift.
        
        Args:
            z: Redshift
            
        Returns:
            R = (3ρ_b)/(4ρ_γ)
        """
        # At recombination, R ≈ 0.6
        # Evolution: R ∝ (1+z)⁻¹
        R_rec = 0.6
        R = R_rec * (1 + self.z_rec) / (1 + z)
        
        return R
    
    def visibility_function(self, z: float) -> float:
        """
        Visibility function (probability of last scattering).
        
        Args:
            z: Redshift
            
        Returns:
            Visibility function value
        """
        # Simplified visibility function (Gaussian approximation)
        z_peak = self.z_rec
        sigma_z = 100  # Width of decoupling episode
        
        # Gaussian visibility function
        visibility = np.exp(-0.5 * ((z - z_peak) / sigma_z)**2)
        
        return visibility
    
    def coherence_length(self, z_min: float, z_max: float) -> float:
        """
        Coherence length (sound horizon) over visibility window.
        
        Args:
            z_min: Minimum redshift of visibility window
            z_max: Maximum redshift of visibility window
            
        Returns:
            Coherence length in Mpc
        """
        def integrand(z):
            # Integrand: c_s(z) / H(z) (no extra 1+z factor)
            H_z = self._hubble_parameter(z)
            c_s = self.sound_speed(z)
            return c_s / H_z
        
        # Integrate over visibility window
        r_s, _ = quad(integrand, z_min, z_max)
        
        # Convert to Mpc
        r_s_Mpc = r_s * self.c / (3.086e22)
        
        return r_s_Mpc
    
    def _hubble_parameter(self, z: float) -> float:
        """
        Hubble parameter as function of redshift.
        
        Args:
            z: Redshift
            
        Returns:
            Hubble parameter in 1/s
        """
        # H(z) = H₀ √[Ω_m(1+z)³ + Ω_Λ + Ω_γ(1+z)⁴]
        H_z = self.H0_SI * np.sqrt(
            self.Omega_m * (1 + z)**3 + 
            self.Omega_Lambda + 
            self.Omega_gamma * (1 + z)**4
        )
        
        return H_z
    
    def angular_diameter_distance(self, z: float) -> float:
        """
        Angular diameter distance.
        
        Args:
            z: Redshift
            
        Returns:
            Angular diameter distance in Mpc
        """
        def integrand(z_prime):
            # Integrand: 1 / H(z')
            H_z = self._hubble_parameter(z_prime)
            return 1 / H_z
        
        # Integrate from 0 to z
        D_H, _ = quad(integrand, 0, z)
        
        # Angular diameter distance: D_A = D_H / (1+z)
        D_A = D_H * self.c / (1 + z) / (3.086e22)
        
        return D_A
    
    def peak_multipoles(self, z: float, r_s: float, n_peaks: int = 3) -> List[float]:
        """
        Calculate peak multipoles using geometric relation.
        
        Args:
            z: Redshift of observation
            r_s: Sound horizon (coherence length) in Mpc
            n_peaks: Number of peaks to calculate
            
        Returns:
            List of peak multipoles
        """
        # Angular diameter distance at redshift z
        D_A = self.angular_diameter_distance(z)
        
        # Peak multipoles: ℓ_n ≈ nπ D_A/r_s
        ell_peaks = []
        
        for n in range(1, n_peaks + 1):
            ell_n = n * np.pi * D_A / r_s
            ell_peaks.append(ell_n)
        
        return ell_peaks
    
    def visibility_weighted_peaks(self, z: float, r_s: float, 
                                n_peaks: int = 3) -> List[float]:
        """
        Calculate visibility-weighted peak multipoles with odd/even modulation.
        
        Args:
            z: Redshift of observation
            r_s: Sound horizon in Mpc
            n_peaks: Number of peaks to calculate
            
        Returns:
            List of visibility-weighted peak multipoles
        """
        # Base peak multipoles
        base_peaks = self.peak_multipoles(z, r_s, n_peaks)
        
        # Visibility weighting and odd/even modulation
        weighted_peaks = []
        
        for i, ell_base in enumerate(base_peaks):
            n = i + 1
            
            # Visibility weighting (stronger for first peak)
            visibility_weight = np.exp(-(n - 1) / 2)
            
            # Odd/even modulation
            if n % 2 == 1:  # Odd peaks (n=1,3,5...)
                modulation = 1.0  # No modulation for odd peaks
            else:  # Even peaks (n=2,4,6...)
                modulation = 0.8  # Suppression for even peaks
            
            # Apply both effects
            ell_weighted = ell_base * visibility_weight * modulation
            weighted_peaks.append(ell_weighted)
        
        return weighted_peaks
    
    def confront_first_peak(self, z_obs: float = 0.0) -> Dict[str, Any]:
        """
        Confront the first peak location without tuning both D_A and r_s freely.
        
        Args:
            z_obs: Redshift of observation (default: today)
            
        Returns:
            Dictionary with confrontation results
        """
        print("🎯 CONFRONTING FIRST PEAK LOCATION")
        print("=" * 60)
        print(f"Observation redshift: z = {z_obs}")
        print()
        
        # Define visibility window (decoupling episode)
        z_min = self.z_rec - 200  # Start of decoupling
        z_max = self.z_rec + 200  # End of decoupling
        
        print(f"Visibility window: z = {z_min:.0f} to {z_max:.0f}")
        print(f"Decoupling episode width: {z_max - z_min:.0f}")
        print()
        
        # Calculate coherence length
        r_s = self.coherence_length(z_min, z_max)
        print(f"Sound horizon (coherence length): r_s = {r_s:.2f} Mpc")
        
        # Calculate angular diameter distance at last scattering (not observation)
        z_src = self.z_rec
        D_A = self.angular_diameter_distance(z_src)
        print(f"Angular diameter distance (at z_rec): D_A = {D_A:.2f} Mpc")
        print()
        
        # Calculate peak multipoles at last scattering
        ell_peaks = self.peak_multipoles(z_src, r_s, n_peaks=3)
        ell_weighted = self.visibility_weighted_peaks(z_src, r_s, n_peaks=3)
        
        print("📊 PEAK MULTIPOLES:")
        print("-" * 40)
        
        for i, (ell_base, ell_w) in enumerate(zip(ell_peaks, ell_weighted)):
            n = i + 1
            print(f"   Peak {n}:")
            print(f"      Base: ℓ = {ell_base:.1f}")
            print(f"      Weighted: ℓ = {ell_w:.1f}")
        
        # Focus on first peak
        ell_first = ell_weighted[0]
        ell_target = 220  # Target first peak location
        
        print(f"\n🎯 FIRST PEAK CONFRONTATION:")
        print(f"   Predicted: ℓ₁ = {ell_first:.1f}")
        print(f"   Observed:  ℓ₁ ≈ {ell_target}")
        
        # Check agreement
        agreement = abs(ell_first - ell_target) / ell_target
        within_20_percent = agreement < 0.2
        
        print(f"   Agreement: {agreement:.1%}")
        print(f"   Within 20%: {'✅ YES' if within_20_percent else '❌ NO'}")
        
        # Parameter tension analysis
        print(f"\n🔍 PARAMETER TENSION ANALYSIS:")
        print(f"   Fixed parameters:")
        print(f"      - Ω_m = {self.Omega_m}")
        print(f"      - Ω_Λ = {self.Omega_Lambda}")
        print(f"      - H₀ = {self.H0} km/s/Mpc")
        print(f"      - z_rec = {self.z_rec}")
        print()
        print(f"   Derived parameters:")
        print(f"      - r_s = {r_s:.2f} Mpc (from cosmology)")
        print(f"      - D_A = {D_A:.2f} Mpc (from cosmology)")
        print()
        
        if within_20_percent:
            print("✅ SUCCESS: First peak location is consistent with cosmology")
            print("   No parameter tuning needed - both D_A and r_s are")
            print("   determined by standard cosmological parameters")
        else:
            print("⚠️  TENSION: First peak location disagrees with cosmology")
            print("   This suggests either:")
            print("   1. Cosmological parameters need adjustment")
            print("   2. Additional physics in decoupling epoch")
            print("   3. Systematic error in peak measurement")
        
        return {
            "z_obs": z_obs,
            "r_s": r_s,
            "D_A": D_A,
            "ell_first": ell_first,
            "ell_target": ell_target,
            "agreement": agreement,
            "within_20_percent": within_20_percent,
            "ell_peaks": ell_peaks,
            "ell_weighted": ell_weighted
        }
    
    def parameter_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Analyze sensitivity of first peak location to cosmological parameters.
        """
        print("\n🔬 PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 60)
        
        # Base parameters
        base_params = {
            "Omega_m": self.Omega_m,
            "Omega_Lambda": self.Omega_Lambda,
            "H0": self.H0,
            "z_rec": self.z_rec
        }
        
        # Parameter variations
        variations = {
            "Omega_m": [0.25, 0.30, 0.35],
            "Omega_Lambda": [0.65, 0.70, 0.75],
            "H0": [65, 70, 75],
            "z_rec": [1050, 1100, 1150]
        }
        
        sensitivity_results = {}
        
        for param_name, param_values in variations.items():
            print(f"\n📊 {param_name.upper()} SENSITIVITY:")
            print("-" * 30)
            
            ell_values = []
            
            for param_value in param_values:
                # Temporarily modify parameter
                original_value = getattr(self, param_name)
                setattr(self, param_name, param_value)
                
                # Recalculate first peak
                result = self.confront_first_peak()
                ell_values.append(result["ell_first"])
                
                # Restore original value
                setattr(self, param_name, original_value)
            
            # Calculate sensitivity
            ell_range = max(ell_values) - min(ell_values)
            sensitivity = ell_range / 220  # Relative to target
            
            print(f"   Range: {min(ell_values):.1f} to {max(ell_values):.1f}")
            print(f"   Sensitivity: {sensitivity:.1%}")
            
            sensitivity_results[param_name] = {
                "values": param_values,
                "ell_values": ell_values,
                "sensitivity": sensitivity
            }
        
        # Identify most sensitive parameter
        most_sensitive = max(sensitivity_results.keys(), 
                           key=lambda k: sensitivity_results[k]["sensitivity"])
        
        print(f"\n🎯 MOST SENSITIVE PARAMETER: {most_sensitive.upper()}")
        print(f"   Sensitivity: {sensitivity_results[most_sensitive]['sensitivity']:.1%}")
        
        return sensitivity_results
    
    def plot_acoustic_peaks(self, z_obs: float = 0.0):
        """
        Plot acoustic peak structure.
        
        Args:
            z_obs: Redshift of observation
        """
        # Calculate peak structure
        result = self.confront_first_peak(z_obs)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Peak multipoles
        n_peaks = len(result["ell_peaks"])
        peak_numbers = np.arange(1, n_peaks + 1)
        
        ax1.plot(peak_numbers, result["ell_peaks"], 'o-', label='Base peaks', linewidth=2, markersize=8)
        ax1.plot(peak_numbers, result["ell_weighted"], 's-', label='Visibility-weighted', linewidth=2, markersize=8)
        ax1.axhline(y=220, color='red', linestyle='--', label='Target ℓ₁ ≈ 220')
        ax1.set_xlabel('Peak number')
        ax1.set_ylabel('Multipole ℓ')
        ax1.set_title('Acoustic Peak Structure')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Visibility function
        z_range = np.linspace(self.z_rec - 300, self.z_rec + 300, 1000)
        visibility = [self.visibility_function(z) for z in z_range]
        
        ax2.plot(z_range, visibility, 'b-', linewidth=2)
        ax2.axvline(x=self.z_rec, color='red', linestyle='--', label=f'z_rec = {self.z_rec}')
        ax2.set_xlabel('Redshift z')
        ax2.set_ylabel('Visibility function')
        ax2.set_title('Decoupling Epoch Visibility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


def main():
    """Demonstrate acoustic structure upgrade with geometric ℓ mapping."""
    analyzer = AcousticStructureUpgrade()
    
    print("🔬 ACOUSTIC STRUCTURE UPGRADE")
    print("=" * 60)
    print("Replacing manual mode index → ℓ mapping with")
    print("standard geometric relations to confront first-peak location")
    print()
    
    # Confront first peak location
    result = analyzer.confront_first_peak()
    
    # Parameter sensitivity analysis
    sensitivity = analyzer.parameter_sensitivity_analysis()
    
    # Summary
    print("\n🎯 IMPLEMENTATION SUMMARY")
    print("=" * 60)
    print("✅ Replaced manual ℓ mapping with geometric relations")
    print("✅ Defined sound speed: c_s ≈ c/√(3(1+R))")
    print("✅ Integrated coherence length over visibility window")
    print("✅ Mapped peaks by: ℓ_n ≈ nπ D_A/r_s")
    print("✅ Applied visibility weighting and odd/even modulation")
    print("✅ Confronted first peak location without parameter tuning")
    
    if result["within_20_percent"]:
        print("\n🎉 SUCCESS: First peak consistent with cosmology!")
        print("   No parameter tension - standard model works")
    else:
        print("\n⚠️  TENSION: First peak disagrees with cosmology")
        print("   Parameter refinement or new physics needed")
    
    return result, sensitivity


if __name__ == "__main__":
    main()
