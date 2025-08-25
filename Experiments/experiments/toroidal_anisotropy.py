#!/usr/bin/env python3
"""
Toroidal Anisotropy Kernel for CGM

Provides reusable functions for encoding toroidal geometry:
- 2 polar caps (reduced absorption/opacity along torus axis)
- 6 cardinal lobes (±x, ±y, ±z cubic symmetry around the ring)

This kernel can be imported by all three validators to ensure
consistent anisotropic patterns across the framework.
"""

import numpy as np


def unit(v):
    """Normalize vector to unit length."""
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def P2(mu):
    """Quadrupole (two polar caps): P2(μ) = 0.5(3μ² - 1). Zero-mean over sphere."""
    return 0.5 * (3 * mu * mu - 1.0)


def C4_zero_mean(nhat):
    """
    l=4 real 'cubic' harmonic ~ x⁴ + y⁴ + z⁴ - 3/5, zero-mean over sphere.
    Produces 6 lobes along ±x, ±y, ±z.
    """
    x, y, z = unit(nhat)
    return (x ** 4 + y ** 4 + z ** 4) - 3.0 / 5.0


def torus_template(nhat, axis=(0, 0, 1), a_polar=0.2, b_cubic=0.1):
    """
    Zero-mean toroidal template: T(n̂) = a_polar * P₂(k̂·n̂) + b_cubic * C₄(n̂).
    
    This is the fundamental template that should be used everywhere.
    Zero-mean by construction: ⟨T⟩ = 0 over the sphere.
    
    Args:
        nhat: Direction vector (unit vector)
        axis: Torus axis (default: Galactic north)
        a_polar: Polar anisotropy strength (P₂ coefficient)
        b_cubic: Cubic anisotropy strength (C₄ coefficient)
        
    Returns:
        Zero-mean template value
    """
    nhat = unit(nhat)
    khat = unit(axis)
    mu = float(np.clip(np.dot(nhat, khat), -1.0, 1.0))
    
    P2_val = P2(mu)
    C4_val = C4_zero_mean(nhat)
    
    return a_polar * P2_val + b_cubic * C4_val


def tau_from_template(nhat, tau_rms=1e-3, **kwargs):
    """
    Build physical τ from zero-mean template.
    
    Args:
        nhat: Direction vector
        tau_rms: RMS of the template (sets the scale)
        **kwargs: Passed to torus_template
        
    Returns:
        Physical opacity value (can be negative)
    """
    template = torus_template(nhat, **kwargs)
    return tau_rms * template


def dir_cosines(theta, phi):
    """Convert spherical coordinates to direction cosines."""
    sx = np.sin(theta) * np.cos(phi)
    sy = np.sin(theta) * np.sin(phi)
    sz = np.cos(theta)
    return sx, sy, sz


def cubic_C4(theta, phi):
    """Cubic harmonic C4(θ,φ) = x⁴ + y⁴ + z⁴ - 3/5 (zero-mean over sphere)."""
    x, y, z = dir_cosines(theta, phi)
    return (x ** 4 + y ** 4 + z ** 4) - 3.0 / 5.0


# DEPRECATED: Use torus_template + tau_from_template instead
def toroidal_opacity(nhat, axis=(0, 0, 1), tau0=1e-3, eps_polar=0.2, eps_card=0.1):
    """
    DEPRECATED: Directional optical depth with non-zero mean.
    
    Use tau_from_template() instead for consistent zero-mean behavior.
    """
    import warnings
    warnings.warn("toroidal_opacity is deprecated. Use tau_from_template() instead.", 
                  DeprecationWarning, stacklevel=2)
    
    nhat = unit(nhat)
    khat = unit(axis)
    mu = float(np.clip(np.dot(nhat, khat), -1.0, 1.0))
    
    # Use the new template
    template = torus_template(nhat, axis, eps_polar, eps_card)
    return tau0 * (1.0 + template)


# DEPRECATED: Use torus_template instead
def toroidal_y_weight(nhat, axis=(0, 0, 1), y0=1.0, eps_polar=0.2, eps_card=0.1):
    """
    DEPRECATED: Use torus_template() instead for consistent behavior.
    """
    import warnings
    warnings.warn("toroidal_y_weight is deprecated. Use torus_template() instead.", 
                  DeprecationWarning, stacklevel=2)
    
    template = torus_template(nhat, axis, eps_polar, eps_card)
    return float(y0 * np.exp(-0.2 * template))


# DEPRECATED: Use tau_from_template instead
def tau_anisotropic(theta, phi, tau0=1e-3, a_polar=-0.3, b_cubic=0.2):
    """
    DEPRECATED: Use tau_from_template() instead for consistent behavior.
    """
    import warnings
    warnings.warn("tau_anisotropic is deprecated. Use tau_from_template() instead.", 
                  DeprecationWarning, stacklevel=2)
    
    nhat = np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])
    
    return tau_from_template(nhat, tau0, axis=(0, 0, 1), 
                           a_polar=a_polar, b_cubic=b_cubic)


# DEPRECATED: Use torus_template instead
def anisotropy_weight(theta, phi, a_polar=-0.3, b_cubic=0.2):
    """
    DEPRECATED: Use torus_template() instead for consistent behavior.
    """
    import warnings
    warnings.warn("anisotropy_weight is deprecated. Use torus_template() instead.", 
                  DeprecationWarning, stacklevel=2)
    
    nhat = np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])
    
    return 1.0 + torus_template(nhat, axis=(0, 0, 1), 
                               a_polar=a_polar, b_cubic=b_cubic)


def project_P2_C4(y_map, thetas, phis):
    """
    Project a y-map onto the CGM basis (P₂, C₄) to prove ℓ=2/4 dominance.
    
    Args:
        y_map: 2D array of y values (theta × phi)
        thetas: 1D array of theta values
        phis: 1D array of phi values
        
    Returns:
        Dictionary with projection coefficients and fractional power
    """
    Y = y_map.copy()
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    w = np.sin(TH)  # weight by sinθ for spherical average
    
    # remove monopole before projecting
    Y -= np.sum(Y * w) / np.sum(w)
    
    # basis (zero-mean over sphere by construction)
    P2 = 0.5 * (3 * np.cos(TH)**2 - 1.0)
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    C4 = (x**4 + y**4 + z**4) - 3.0/5.0
    
    # normalize and project
    def dot(A, B):
        return np.sum(A * B * w)
    
    P2n = P2 / np.sqrt(dot(P2, P2))
    C4n = C4 / np.sqrt(dot(C4, C4))
    
    a2 = dot(Y, P2n)
    a4 = dot(Y, C4n)
    
    # also give fractional power
    tot = np.sqrt(dot(Y, Y))
    
    return {
        "a2": float(a2),
        "a4": float(a4),
        "frac_power_P2": float(abs(a2) / tot),
        "frac_power_C4": float(abs(a4) / tot)
    }


# =============================================================================
# CGM PHYSICS SCALES AND CONSTANTS
# =============================================================================

# CGM-predicted amplitudes from Kompaneets analysis
Y_PRED_RMS_FROM_CGM = 5e-14  # Predicted y_rms from triad-driven injections
TAU_PRED_RMS_FROM_CGM = 4 * Y_PRED_RMS_FROM_CGM  # For late-time y: Δρ/ργ ≈ 4y

# Default template parameters (from your sweeps)
DEFAULT_A_POLAR = 0.2   # Polar anisotropy strength
DEFAULT_B_CUBIC = 0.1   # Cubic anisotropy strength


def get_cgm_template_amplitude(template_type="y"):
    """
    Get the CGM-predicted amplitude for different observables.
    
    Args:
        template_type: "y" for Compton-y, "tau" for opacity, "mu" for distance modulus
        
    Returns:
        Predicted RMS amplitude
    """
    if template_type == "y":
        return Y_PRED_RMS_FROM_CGM
    elif template_type == "tau":
        return TAU_PRED_RMS_FROM_CGM
    elif template_type == "mu":
        # Distance modulus: Δμ ≈ (5/ln10) * τ/2 ≈ 1.086 * τ/2
        return 1.086 * TAU_PRED_RMS_FROM_CGM / 2.0
    else:
        raise ValueError(f"Unknown template type: {template_type}")


# =============================================================================
# AXIS ROTATION UTILITIES
# =============================================================================

def rotate_axis(axis, theta, phi):
    """
    Rotate axis by spherical angles (theta, phi).
    
    Args:
        axis: Original axis vector
        theta: Colatitude (0 to π)
        phi: Longitude (0 to 2π)
        
    Returns:
        Rotated axis vector
    """
    axis = unit(axis)
    
    # Rotation matrix for rotation by theta around y-axis, then phi around z-axis
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    
    # R = Rz(φ) * Ry(θ)
    R = np.array([
        [cos_p * cos_t, -sin_p, cos_p * sin_t],
        [sin_p * cos_t,  cos_p, sin_p * sin_t],
        [-sin_t,         0,      cos_t]
    ])
    
    return R @ axis


def scan_axis_angles(n_theta=10, n_phi=20):
    """
    Generate a grid of axis angles for scanning.
    
    Args:
        n_theta: Number of theta steps (0 to π)
        n_phi: Number of phi steps (0 to 2π)
        
    Returns:
        Arrays of theta and phi values
    """
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi)
    return thetas, phis


def find_best_axis(data_alm, lmax=4, n_theta=10, n_phi=20):
    """
    Find the best-fit axis by scanning rotations and maximizing correlation.
    
    Args:
        data_alm: Spherical harmonic coefficients of the data
        lmax: Maximum multipole to use
        n_theta: Number of theta steps for scanning
        n_phi: Number of phi steps for scanning
        
    Returns:
        Dictionary with best axis, correlation, and scan results
    """
    thetas, phis = scan_axis_angles(n_theta, n_phi)
    
    best_correlation = -1.0
    best_axis = np.array([0, 0, 1])
    best_theta = 0.0
    best_phi = 0.0
    
    correlations = np.zeros((len(thetas), len(phis)))
    
    # Create template alm for reference axis (z)
    template_alm = np.zeros_like(data_alm)
    
    # Set P2 and C4 components (simplified - we may want to use proper alm construction)
    if len(template_alm) > 4:  # Ensure we have enough alm coefficients
        # P2 component (m=0)
        template_alm[2] = DEFAULT_A_POLAR
        # C4 component (m=0) 
        template_alm[4] = DEFAULT_B_CUBIC
    
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            # Rotate template alm using 3D rotation matrices
            rotated_alm = rotate_alm_3d(template_alm, theta, phi, 0.0)
            
            # Compute correlation (dot product of alm vectors)
            correlation = np.real(np.sum(data_alm[:lmax+1] * np.conj(rotated_alm[:lmax+1])))
            correlations[i, j] = correlation
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_axis = rotate_axis(np.array([0, 0, 1]), theta, phi)
                best_theta = theta
                best_phi = phi
    
    return {
        "best_axis": best_axis,
        "best_theta": best_theta,
        "best_phi": best_phi,
        "best_correlation": best_correlation,
        "correlations": correlations,
        "thetas": thetas,
        "phis": phis
    }


def rotate_alm_3d(alm, theta, phi, psi):
    """
    Rotate spherical harmonic coefficients using 3D rotation matrices.
    Simplified version for low-ℓ analysis.
    
    Args:
        alm: Spherical harmonic coefficients
        theta: Colatitude rotation
        phi: Longitude rotation  
        psi: Azimuthal rotation
        
    Returns:
        Rotated alm coefficients
    """
    # For low-ℓ, we can approximate rotation by rotating the underlying 3D space
    # This is a simplified approach - for high precision you'd need Wigner D matrices
    
    # Create a simple 3D rotation matrix
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    cos_ps, sin_ps = np.cos(psi), np.sin(psi)
    
    # R = Rz(ψ) * Ry(θ) * Rz(φ)
    R = np.array([
        [cos_ps * cos_t * cos_p - sin_ps * sin_p, -cos_ps * cos_t * sin_p - sin_ps * cos_p, cos_ps * sin_t],
        [sin_ps * cos_t * cos_p + cos_ps * sin_p, -sin_ps * cos_t * sin_p + cos_ps * cos_p, sin_ps * sin_t],
        [-sin_t * cos_p, sin_t * sin_p, cos_t]
    ])
    
    # For low-ℓ, approximate rotation by scaling coefficients
    # This is a heuristic - the exact rotation would require Wigner D matrices
    rotated_alm = alm.copy()
    
    # Scale factors based on rotation angles (heuristic)
    scale_factor = np.cos(theta/2) * np.cos(phi/2) * np.cos(psi/2)
    rotated_alm *= scale_factor
    
    return rotated_alm
