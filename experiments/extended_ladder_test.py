#!/usr/bin/env python3
"""
Extended Ladder Test for CGM Interference Analysis

This script tests for ladder peaks at high ℓ values (ℓ=222-370) that require
high lmax and longer runtime. This is a separate script to avoid bloating
the main analysis suite.
"""

import numpy as np
import healpy as hp  # pyright: ignore[reportMissingImports]
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from cgm_cmb_data_analysis import (
    CGMDataManager,
    CGMThresholds,
    Config,
    compute_ladder_comb_statistic,
)


@dataclass
class ExtendedConfig:
    """Configuration for extended ladder testing."""

    nside: int = 512  # Higher resolution for extended ladder
    lmax: int = 400  # High lmax to capture ℓ=370
    fwhm_deg: float = 0.0  # No smoothing for ladder
    mask_apod_fwhm: float = 3.0
    base_seed: int = 42
    n_mc: int = 256  # Monte Carlo budget for null distribution
    memory_axis: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.memory_axis is None:
            self.memory_axis = np.array([-0.070, -0.662, 0.745])


def test_extended_ladder(
    data_manager: CGMDataManager, config: ExtendedConfig
) -> Dict[str, Any]:
    """
    Test for extended ladder peaks at high ℓ values.

    Args:
        data_manager: Data manager for Planck data
        config: Extended ladder configuration

    Returns:
        Dict with extended ladder test results
    """
    print("Testing extended ladder peaks (ℓ=222-370)...")
    print(f"  Configuration: nside={config.nside}, lmax={config.lmax}")

    # Get Planck data at high resolution
    data = data_manager.get_planck_data(
        nside=config.nside,
        lmax=config.lmax,
        fwhm_deg=config.fwhm_deg,
        fast_preprocess=True,
        production_mode=True,
        mask_apod_fwhm=config.mask_apod_fwhm,
    )

    # Extended ladder peaks
    peaks_extended = [222, 259, 296, 333, 370]

    # Compute power spectrum
    cl_full = hp.anafast(data["y_map"] * data["mask"], lmax=config.lmax)

    # Test extended ladder with comb statistic
    comb_extended = compute_ladder_comb_statistic(
        cl_full, peaks=peaks_extended, sigma_l=2.0, null_var=None
    )
    extended_signal = comb_extended["signal"]

    print(f"  Extended ladder signal: {extended_signal:.3f}")

    # Null distribution for extended ladder
    print("  Computing extended ladder null distribution...")
    rng = np.random.default_rng(config.base_seed)

    # Build null variance at each ℓ for weights
    ladder_null_vals = []
    for i in range(config.n_mc):
        alm_rand = data["alm"].copy()
        phases = rng.uniform(0, 2 * np.pi, len(alm_rand))
        alm_rand = np.abs(alm_rand) * np.exp(1j * phases)
        map_rand = hp.alm2map(alm_rand, data["nside"], lmax=config.lmax)
        cl_rand = hp.anafast(map_rand * data["mask"], lmax=config.lmax)
        ladder_null_vals.append(cl_rand)

    ladder_null_vals = np.array(ladder_null_vals)  # (n_mc, lmax+1)
    null_var = np.var(ladder_null_vals, axis=0)

    # Extended ladder null test
    extended_null_matched = []
    for i in range(config.n_mc):
        cl_rand = ladder_null_vals[i]
        comb_extended_null = compute_ladder_comb_statistic(
            cl_rand, peaks=peaks_extended, sigma_l=2.0, null_var=null_var
        )
        extended_null_matched.append(comb_extended_null["signal"])

    extended_null_matched = np.array(extended_null_matched)
    extended_p = (np.sum(extended_null_matched >= extended_signal) + 1.0) / (
        extended_null_matched.size + 1.0
    )

    # Compute Z-score
    extended_z = (extended_signal - np.mean(extended_null_matched)) / (
        np.std(extended_null_matched) + 1e-30
    )

    print(f"  Extended ladder p-value: {extended_p:.4f}")
    print(f"  Extended ladder Z-score: {extended_z:.2f}")

    # Individual peak analysis
    peak_analysis = {}
    for ell in peaks_extended:
        if ell < len(cl_full):
            peak_power = cl_full[ell]
            # Compare to local background (average of neighboring bins)
            local_bg = np.mean(cl_full[max(0, ell - 5) : min(len(cl_full), ell + 6)])
            enhancement = peak_power / (local_bg + 1e-30)
            peak_analysis[ell] = {
                "power": peak_power,
                "local_background": local_bg,
                "enhancement": enhancement,
            }
            print(f"    ℓ={ell}: power={peak_power:.3e}, enhancement={enhancement:.3f}")

    return {
        "extended_signal": extended_signal,
        "extended_p": extended_p,
        "extended_z": extended_z,
        "peaks_tested": peaks_extended,
        "peak_analysis": peak_analysis,
        "null_distribution": extended_null_matched.tolist(),
    }


def main():
    """Main function for extended ladder testing."""
    print("=" * 60)
    print("EXTENDED LADDER TEST")
    print("=" * 60)
    print("Testing for ladder peaks at ℓ=222, 259, 296, 333, 370")
    print("Note: This requires high lmax and longer runtime")

    # Initialize components
    data_manager = CGMDataManager()
    config = ExtendedConfig()

    # Print configuration
    print("\nConfiguration:")
    print(f"  nside: {config.nside}")
    print(f"  lmax: {config.lmax}")
    print(f"  fwhm: {config.fwhm_deg}°")
    print(f"  MC budget: {config.n_mc}")
    print(f"  RNG seed: {config.base_seed}")

    # Run extended ladder test
    results = test_extended_ladder(data_manager, config)

    # Summary
    print("\n" + "=" * 60)
    print("EXTENDED LADDER RESULTS")
    print("=" * 60)
    print(f"Extended ladder signal: {results['extended_signal']:.3f}")
    print(f"Extended ladder p-value: {results['extended_p']:.4f}")
    print(f"Extended ladder Z-score: {results['extended_z']:.2f}")

    if results["extended_p"] < 0.05:
        print("✓ SIGNIFICANT: Extended ladder peaks detected")
    elif results["extended_p"] < 0.1:
        print("✓ SUGGESTIVE: Extended ladder peaks may be present")
    else:
        print("✗ NOT SIGNIFICANT: No evidence for extended ladder peaks")

    print(f"\nPeaks tested: {results['peaks_tested']}")
    print("Individual peak enhancements:")
    for ell, analysis in results["peak_analysis"].items():
        print(f"  ℓ={ell}: {analysis['enhancement']:.3f}x background")


if __name__ == "__main__":
    main()
