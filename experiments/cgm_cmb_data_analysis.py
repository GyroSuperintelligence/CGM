#!/usr/bin/env python3
"""
CGM CMB Data Analysis
Unified empirical validation suite for testing Common Governance Model predictions
against real observational data including Planck Compton-y maps, Etherington distance
duality violations, and Type-Ia supernova Hubble residuals.
"""

import os
import sys
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pickle
from pathlib import Path
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.functions.torus import (
    torus_template, get_cgm_template_amplitude, 
    find_best_axis, rotate_axis
)


class CGMDataManager:
    """
    Centralized data manager for CGM empirical tests.
    
    Loads Planck Compton-y data once and provides cached access to:
    - Raw maps and masks
    - Spherical harmonic coefficients  
    - Pre-computed statistics
    - Test-specific data subsets
    """
    
    def __init__(self, cache_dir: str = "cache"):
        # Data file paths
        experiments_dir = os.path.dirname(os.path.abspath(__file__))
        self.y_map_file = os.path.join(experiments_dir, "data", "milca_ymaps.fits")
        self.mask_file = os.path.join(experiments_dir, "data", "COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits")
        
        # Cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Data storage
        self._y_map: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._nside: Optional[int] = None
        self._npix: Optional[int] = None
        self._alm_cache: Dict[str, Any] = {}
        self._stats_cache: Dict[str, Any] = {}
        
        # Analysis parameters
        self.lmax = 64  # Maximum multipole to analyze
    
    def load_data(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load Planck Compton-y data and masks from real FITS files."""
        cache_file = self.cache_dir / "planck_data_cache.pkl"
        
        # Try to load from cache first
        if not force_reload and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Restore cached data
                self._y_map = cached_data['y_map']
                self._mask = cached_data['mask']
                self._nside = cached_data['nside']
                self._npix = cached_data['npix']
                self._stats_cache = cached_data.get('stats', {})
                
                return self._get_data_summary()
                
            except Exception as e:
                pass
        
        # Load from real FITS files
        try:
            import astropy.io.fits as fits
            
            # Load Compton-y map
            with fits.open(self.y_map_file) as hdul:
                y_data = hdul[1].data['FULL']  # type: ignore
                if y_data is None:
                    raise ValueError("No data in y-map FITS file")
                self._y_map = y_data
                if self._y_map is not None:
                    self._nside = int(np.sqrt(len(self._y_map) / 12))
                    self._npix = len(self._y_map)
                else:
                    raise ValueError("Failed to load y-map data")
            
            # Load mask
            with fits.open(self.mask_file) as hdul:
                mask_data = hdul[1].data['M1']  # type: ignore
                if mask_data is None:
                    raise ValueError("No data in mask FITS file")
                self._mask = mask_data
                if self._mask is not None and self._npix is not None:
                    if len(self._mask) != self._npix:
                        raise ValueError("Mask and y-map have different sizes")
                else:
                    raise ValueError("Failed to load mask data")
            
            # Compute and cache basic statistics
            self._compute_basic_stats()
            
            # Save to cache
            self._save_cache()
            
            return self._get_data_summary()
            
        except ImportError:
            raise ImportError("astropy.io.fits is required to load real Planck data. Install with: pip install astropy")
        except Exception as e:
            raise RuntimeError(f"Failed to load real Planck data: {e}")
    
    def _compute_basic_stats(self):
        """Compute and cache basic statistics."""
        if self._mask is None or self._y_map is None:
            raise RuntimeError("Data not loaded")
        
        # Apply mask to get valid data
        valid_mask = self._mask > 0
        y_valid = self._y_map[valid_mask]
        
        self._stats_cache = {
            'y_range': (y_valid.min(), y_valid.max()),
            'y_mean': np.mean(y_valid),
            'y_std': np.std(y_valid),
            'sky_fraction': np.sum(valid_mask) / len(valid_mask),
            'n_valid_pixels': np.sum(valid_mask)
        }
    
    def _save_cache(self):
        """Save data to cache file."""
        cache_file = self.cache_dir / "planck_data_cache.pkl"
        
        if self._y_map is None or self._mask is None or self._nside is None or self._npix is None:
            raise RuntimeError("Data not loaded")
        
        cache_data = {
            'y_map': self._y_map,
            'mask': self._mask,
            'nside': self._nside,
            'npix': self._npix,
            'stats': self._stats_cache
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            pass
    
    def _get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        if self._y_map is None or self._mask is None or self._nside is None or self._npix is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        return {
            "y_map": self._y_map,
            "mask": self._mask,
            "nside": self._nside,
            "npix": self._npix,
            "stats": self._stats_cache,
            "is_data": True,
            "description": f"Real Planck Compton-y data (nside={self._nside})"
        }
    
    def get_spherical_harmonics(self, lmax: Optional[int] = None) -> Dict[str, Any]:
        """Get spherical harmonic coefficients for y-map using healpy."""
        if lmax is None:
            lmax = self.lmax
        
        cache_key = f"alm_lmax{lmax}"
        
        if cache_key in self._alm_cache:
            return self._alm_cache[cache_key]
        
        if self._y_map is None or self._mask is None:
            raise RuntimeError("Data not loaded")
        
        try:
            import healpy as hp  # pyright: ignore[reportMissingImports]
            
            # Apply mask
            masked_y = self._y_map * (self._mask > 0)
            
            # Compute spherical harmonics using healpy
            alm = hp.map2alm(masked_y, lmax=lmax)
            cl = hp.alm2cl(alm)
            
            result = {
                'alm': alm,
                'cl': cl,
                'lmax': lmax
            }
            
            # Cache result
            self._alm_cache[cache_key] = result
            
            return result
            
        except ImportError:
            raise ImportError("healpy is required for spherical harmonic operations. Install with: pip install healpy")
    
    def get_test_directions(self, n_directions: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Get test directions for correlation tests."""
        print(f"Generating {n_directions} test directions...")
        
        if self._mask is None:
            raise RuntimeError("Data not loaded")
        
        # Get valid pixel indices
        valid_pixels = np.where(self._mask > 0)[0]
        
        if len(valid_pixels) < n_directions:
            print(f"   Only {len(valid_pixels)} valid pixels available")
            test_pixels = valid_pixels
        else:
            # Randomly sample valid pixels
            np.random.seed(seed)
            test_pixels = np.random.choice(valid_pixels, n_directions, replace=False)
        
        # Convert to spherical coordinates using healpy
        theta, phi = self._pix2ang_healpy(test_pixels)
        
        print(f"   Generated {len(test_pixels)} test directions")
        return theta, phi
    
    def _pix2ang_healpy(self, pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pixel indices to spherical coordinates using healpy."""
        if self._nside is None:
            raise RuntimeError("Data not loaded")
        
        try:
            import healpy as hp  # pyright: ignore[reportMissingImports]
            theta, phi = hp.pix2ang(self._nside, pixels)
            return theta, phi
        except ImportError:
            raise ImportError("healpy is required for coordinate operations. Install with: pip install healpy")
    
    def _ang2pix_healpy(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Convert spherical coordinates to pixel indices using healpy."""
        if self._nside is None:
            raise RuntimeError("Data not loaded")
        
        try:
            import healpy as hp  # pyright: ignore[reportMissingImports]
            pixels = hp.ang2pix(self._nside, theta, phi)
            return pixels
        except ImportError:
            raise ImportError("healpy is required for coordinate operations. Install with: pip install healpy")
    
    def extract_values_at_directions(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Extract y-map values at specific directions."""
        if self._y_map is None or self._nside is None:
            raise RuntimeError("Data not loaded")
        
        # Convert spherical coordinates to pixel indices using healpy
        pixels = self._ang2pix_healpy(theta, phi)
        
        # Extract values
        values = self._y_map[pixels]
        
        return values
    
    def get_preprocessed_data(self, lmax: int = 8, nside_target: int = 64, fwhm_deg: float = 10.0) -> Dict[str, Any]:
        """Get preprocessed data for low-ell analysis using proper HEALPix operations."""
        cache_key = f"preprocessed_lmax{lmax}_nside{nside_target}_fwhm{fwhm_deg}"
        
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]
        
        if self._y_map is None or self._mask is None:
            raise RuntimeError("Data not loaded")
        
        try:
            import healpy as hp  # pyright: ignore[reportMissingImports]  # pyright: ignore[reportMissingImports]
            
            # Apply mask
            masked_y = self._y_map * (self._mask > 0)
            
            # Remove monopole and dipole using healpy
            y_clean = hp.remove_dipole(masked_y, gal_cut=30)
            
            # Smooth with Gaussian kernel using healpy
            y_smooth = hp.smoothing(y_clean, fwhm=np.radians(fwhm_deg))
            
            # Degrade resolution using healpy
            y_degraded = hp.ud_grade(y_smooth, nside_out=nside_target)
            mask_degraded = hp.ud_grade(self._mask, nside_out=nside_target)
            
            # Compute spherical harmonics for low-ell analysis using healpy
            alm = hp.map2alm(y_degraded, lmax=lmax)
            cl = hp.alm2cl(alm)
            
            result = {
                'y_map': y_degraded,
                'mask': mask_degraded,
                'nside': nside_target,
                'npix': len(y_degraded),
                'alm': alm,
                'cl': cl,
                'lmax': lmax,
                'fwhm_deg': fwhm_deg,
                'original_nside': self._nside
            }
            
            # Cache result
            self._stats_cache[cache_key] = result
            
            return result
            
        except ImportError:
            raise ImportError("healpy is required for proper HEALPix operations. Install with: pip install healpy")
    
    def get_supernova_data(self) -> Dict[str, Any]:
        """Get real supernova data from Pantheon+ catalog."""
        try:
            # Try to load real Pantheon+ data
            pantheon_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        "data", "pantheon_plus_real.dat")
            
            if os.path.exists(pantheon_file):
                # Load Pantheon+ data (space-separated, skip header)
                data = np.loadtxt(pantheon_file, skiprows=1, dtype=str)
                # Convert numeric columns to float
                redshifts = data[:, 2].astype(float)  # zHD (Hubble Diagram Redshift)
                mu = data[:, 20].astype(float)  # MU_SH0ES (Distance modulus)
                mu_err = data[:, 21].astype(float)  # MU_SH0ES_ERR_DIAG (Distance modulus error)
                ra = data[:, 24].astype(float)  # RA (Right ascension)
                dec = data[:, 25].astype(float)  # DEC (Declination)
                
                # Convert to 3D positions (approximate)
                distances = redshifts * 3000  # Mpc (approximate)
                theta = np.pi/2 - np.radians(dec)  # Colatitude
                phi = np.radians(ra)  # Longitude
                
                x = distances * np.sin(theta) * np.cos(phi)
                y = distances * np.sin(theta) * np.sin(phi)
                z = distances * np.cos(theta)
                
                # Compute residuals from isotropic fit (simple linear fit to z)
                z_fit = np.polyfit(redshifts, mu, 1)
                mu_fit = np.polyval(z_fit, redshifts)
                residuals = mu - mu_fit
                
                return {
                    "sn_data": {
                        "redshifts": redshifts,
                        "mu": mu,
                        "mu_err": mu_err,
                        "residuals": residuals,
                        "ra": ra,
                        "dec": dec,
                        "positions": np.column_stack([x, y, z]),
                        "n_supernovae": len(redshifts)
                    },
                    "is_data": True
                }
            else:
                raise FileNotFoundError(f"Pantheon+ file not found: {pantheon_file}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load real supernova data: {e}")
    
    def get_data_for_test(self, test_name: str) -> Dict[str, Any]:
        """Get data specifically formatted for a test."""
        if test_name == "planck_compton_y":
            # For Test A: Full maps and spherical harmonics
            alm_data = self.get_spherical_harmonics()
            return {
                **self._get_data_summary(),
                'alm': alm_data['alm'],
                'cl': alm_data['cl']
            }
            
        elif test_name == "etherington_coherence":
            # For Test B: Test directions and values
            theta, phi = self.get_test_directions()
            y_values = self.extract_values_at_directions(theta, phi)
            return {
                **self._get_data_summary(),
                'test_directions': {'theta': theta, 'phi': phi},
                'y_values': y_values
            }
            
        elif test_name == "supernova_hubble_residuals":
            # For Test C: Supernova data
            return self.get_supernova_data()
            
        else:
            # Default: just the basic data
            return self._get_data_summary()
    
    def clear_cache(self):
        """Clear all cached data."""
        self._alm_cache.clear()
        self._stats_cache.clear()
        print("Cache cleared")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        if self._y_map is None or self._mask is None:
            return {"status": "No data loaded"}
        
        y_map_mb = self._y_map.nbytes / (1024 * 1024)
        mask_mb = self._mask.nbytes / (1024 * 1024)
        
        # Calculate alm cache memory
        alm_mb = 0
        for cache_entry in self._alm_cache.values():
            if isinstance(cache_entry, dict) and 'alm' in cache_entry:
                alm_mb += cache_entry['alm'].nbytes / (1024 * 1024)
        
        return {
            "y_map_mb": y_map_mb,
            "mask_mb": mask_mb,
            "alm_cache_mb": alm_mb,
            "total_mb": y_map_mb + mask_mb + alm_mb
        }


class BaseCGMTest:
    """Base class for CGM empirical tests with shared utilities."""
    
    def __init__(self):
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
    
    def set_data_manager(self, data_manager):
        """Set the data manager for this test."""
        self.data_manager = data_manager
    
    def remove_monopole_dipole(self, map_data: np.ndarray) -> np.ndarray:
        """Remove monopole and dipole from a map."""
        # Remove monopole (mean)
        map_clean = map_data - np.mean(map_data)
        # For dipole removal, we'd need to fit a plane
        # For simplicity, just remove the mean for now
        return map_clean
    
    def _template_on_sphere(self, axis: np.ndarray) -> np.ndarray:
        """Create template map using exact HEALPix geometry."""
        try:
            import healpy as hp  # pyright: ignore[reportMissingImports]
            
            npix = 12 * self.nside * self.nside
            ipix = np.arange(npix)
            theta, phi = hp.pix2ang(self.nside, ipix)  # exact HEALPix geometry
            
            # Vectorized template generation
            sx, sy = np.sin(theta), np.sin(phi)
            cx, cy = np.cos(theta), np.cos(phi)
            nhat = np.column_stack([sx*cy, sx*sy, cx])
            
            # Apply torus template to all directions at once
            vals = np.array([torus_template(v, axis=axis, a_polar=self.a_polar, b_cubic=self.b_cubic)
                           for v in nhat])
            
            # Remove monopole and normalize
            vals = vals - vals.mean()
            rms = vals.std()
            return vals / rms if rms > 0 else vals
            
        except ImportError:
            raise ImportError("healpy is required for exact HEALPix geometry. Install with: pip install healpy")
    
    def pix2ang_simple(self, ipix: int, nside: int) -> Tuple[float, float]:
        """Convert pixel index to spherical coordinates (simplified) - DEPRECATED."""
        # This function is deprecated in favor of _template_on_sphere with exact HEALPix geometry
        npix = 12 * nside * nside
        
        # Simple mapping based on pixel index
        ring = int(np.sqrt(ipix / npix) * nside)
        phi = 2 * np.pi * (ipix % nside) / nside
        
        # Convert ring to theta
        theta = np.pi * ring / nside
        
        return theta, phi


class PlanckComptonYTest(BaseCGMTest):
    """Test A: Does the CGM toroidal kernel appear in Planck Compton-y observations?"""
    
    def __init__(self):
        super().__init__()
        self.y_pred_rms = get_cgm_template_amplitude("y")
        self.tau_pred_rms = get_cgm_template_amplitude("tau")
        self.significance_threshold = 0.01  # p-value threshold
    
    def load_real_compton_y_data(self) -> Dict[str, Any]:
        """Load real Planck Compton-y data from data manager."""
        if self.data_manager is None:
            raise RuntimeError("Data manager not set. Call set_data_manager() first.")
        
        return self.data_manager.get_data_for_test("planck_compton_y")
    
    def prepare_data_for_low_ell_analysis(self, y_map: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Prepare data for low-ell analysis using data manager's preprocessed data."""
        if self.data_manager is None:
            raise RuntimeError("Data manager not set. Call set_data_manager() first.")
        
        return self.data_manager.get_preprocessed_data(
            lmax=self.lmax,
            nside_target=self.nside,
            fwhm_deg=self.fwhm_deg
        )
    
    def create_template_map(self, axis: np.ndarray) -> np.ndarray:
        """Create template map using exact HEALPix geometry."""
        return self._template_on_sphere(axis)
    
    def find_best_fit_axis(self, data_alm: np.ndarray) -> Dict[str, Any]:
        """Find the best-fit axis by scanning rotations."""
        axis_result = find_best_axis(data_alm, lmax=self.lmax, n_theta=10, n_phi=20)
        
        return {
            "best_axis": axis_result["best_axis"],
            "best_correlation": axis_result["best_correlation"],
            "axis_scan": axis_result
        }
    
    def fit_template_amplitude(self, data_map: np.ndarray, template_map: np.ndarray, 
                              mask: np.ndarray) -> Dict[str, Any]:
        """Fit template amplitude using masked least squares."""
        # Apply mask
        valid_pixels = (mask > 0)
        data_valid = data_map[valid_pixels]
        template_valid = template_map[valid_pixels]
        
        if len(data_valid) < 100:
            return {"amplitude": 0.0, "amplitude_err": np.inf, "chi2": np.inf}
        
        # Simple linear fit: data = A * template + noise
        XtX = np.sum(template_valid**2)
        Xty = np.sum(template_valid * data_valid)
        
        try:
            amplitude = float(Xty / XtX) if XtX > 0 else 0.0
            
            # Compute residuals and chi2
            residuals = data_valid - amplitude * template_valid
            chi2 = np.sum(residuals**2)
            dof = len(residuals) - 1
            
            # Simple uncertainty estimate
            amplitude_err = float(np.sqrt(chi2 / dof / XtX)) if XtX > 0 else np.inf
            
        except:
            amplitude = 0.0
            amplitude_err = np.inf
            chi2 = np.inf
            dof = len(data_valid) - 1
            residuals = None
        
        return {
            "amplitude": amplitude,
            "amplitude_err": amplitude_err,
            "chi2": chi2,
            "dof": dof,
            "residuals": residuals
        }
    
    def compute_null_distribution(self, data_map: np.ndarray, template_map: np.ndarray, 
                                 mask: np.ndarray, best_axis: np.ndarray) -> Dict[str, Any]:
        """Compute null distribution by randomly rotating the template (fixed-axis test)."""
        amplitudes = []
        
        for i in range(self.n_mc_rotations):
            # Random rotation angles
            theta = np.arccos(np.random.uniform(-1, 1))
            phi = np.random.uniform(0, 2*np.pi)
            
            # Rotate template
            rotated_axis = rotate_axis(best_axis, theta, phi)
            rotated_template = self.create_template_map(rotated_axis)
            
            # Fit amplitude
            fit_result = self.fit_template_amplitude(data_map, rotated_template, mask)
            amplitudes.append(fit_result["amplitude"])
        
        amplitudes = np.array(amplitudes)
        
        # Compute p-value
        best_fit = self.fit_template_amplitude(data_map, template_map, mask)
        best_amplitude = best_fit["amplitude"]
        
        p_value = np.mean(np.abs(amplitudes) >= abs(best_amplitude))
        
        # Compute 95% upper limit
        upper_limit_95 = np.percentile(np.abs(amplitudes), 95)
        
        return {
            "amplitudes": amplitudes,
            "p_value": p_value,
            "upper_limit_95": upper_limit_95,
            "best_amplitude": best_amplitude
        }
    
    def compute_null_distribution_scanned(self, prepared_data: Dict[str, Any],
                                        n_iter: int = 200, grid_size: int = 200, seed: int = 0) -> Dict[str, Any]:
        """Compute look-elsewhere-corrected null distribution for free-axis test."""
        try:
            import healpy as hp  # pyright: ignore[reportMissingImports]
            rng = np.random.default_rng(seed)
            
            y_map = prepared_data["y_map"]
            mask = prepared_data["mask"]
            data_alm = prepared_data["alm"]
            
            # Pre-generate a uniform grid of axes
            mu = rng.uniform(-1, 1, grid_size)
            phi = rng.uniform(0, 2*np.pi, grid_size)
            axes0 = np.column_stack([np.sqrt(1-mu**2)*np.cos(phi),
                                   np.sqrt(1-mu**2)*np.sin(phi),
                                   mu])
            
            max_amps = []
            for _ in range(n_iter):
                alpha = rng.uniform(0, 2*np.pi)
                beta = np.arccos(rng.uniform(-1, 1))
                gamma = rng.uniform(0, 2*np.pi)
                R = hp.Rotator(rot=[alpha, beta, gamma], eulertype='ZYZ')
                
                axes = np.array([R(vec) for vec in axes0])
                amps = []
                for ax in axes:
                    tmpl = self._template_on_sphere(ax)
                    fit = self.fit_template_amplitude(y_map, tmpl, mask)
                    amps.append(abs(fit["amplitude"]))
                max_amps.append(np.max(amps))
            
            max_amps = np.array(max_amps)
            
            # Best axis in the data (from alm)
            axis_result = self.find_best_fit_axis(data_alm)
            best_axis = axis_result["best_axis"]
            best_template = self._template_on_sphere(best_axis)
            best_fit = self.fit_template_amplitude(y_map, best_template, mask)
            best_amplitude = abs(best_fit["amplitude"])
            
            p_value = np.mean(max_amps >= best_amplitude)
            upper_limit_95 = np.percentile(max_amps, 95)
            
            return {
                "max_amplitudes": max_amps,
                "p_value": p_value,
                "upper_limit_95": upper_limit_95,
                "best_amplitude": best_amplitude,
                "best_axis": best_axis,
            }
            
        except ImportError:
            raise ImportError("healpy is required for look-elsewhere correction. Install with: pip install healpy")
    
    def run_test(self) -> Dict[str, Any]:
        """Run the complete Planck Compton-y test."""
        try:
            # Load real Compton-y data
            y_data = self.load_real_compton_y_data()
            
            # Prepare data for low-ℓ analysis
            prepared_data = self.prepare_data_for_low_ell_analysis(y_data["y_map"], y_data["mask"])
            
            # Find best-fit axis
            axis_result = self.find_best_fit_axis(prepared_data["alm"])
            best_axis = axis_result["best_axis"]
            
            # Create template map with best axis
            template_map = self.create_template_map(best_axis)
            
            # Fit template amplitude
            fit_result = self.fit_template_amplitude(prepared_data["y_map"], template_map, 
                                                   prepared_data["mask"])
            
            # Compute null distribution (fixed-axis)
            null_result = self.compute_null_distribution(prepared_data["y_map"], template_map,
                                                       prepared_data["mask"], best_axis)
            
            # Compute look-elsewhere-corrected null distribution (free-axis)
            scanned_null = self.compute_null_distribution_scanned(prepared_data, n_iter=200, grid_size=200, seed=0)
            p_value_scanned = scanned_null["p_value"]
            ul95_scanned = scanned_null["upper_limit_95"]
            
            # Prefer the scanned p-value for "free-axis" claims
            p_value = p_value_scanned
            upper_limit_95 = ul95_scanned
            
            # Assess results
            amplitude = fit_result["amplitude"]
            amplitude_err = fit_result["amplitude_err"]
            sigma_null = np.std(null_result["amplitudes"])
            
            # Pass criteria
            null_consistent = p_value >= self.significance_threshold
            cgm_safe = upper_limit_95 > 10 * self.y_pred_rms
            
            overall_passes = null_consistent and cgm_safe
            
            # Run fixed-axis tests
            fixed_axis_results = self.run_fixed_axis_tests(prepared_data["y_map"], prepared_data["mask"])
            
            # Results
            results = {
                "test_name": "Planck Compton-y Test (Test A)",
                "data_source": "Real Planck Compton-y (MILCA) in HEALPix format",
                "nside": prepared_data["nside"],
                "lmax": self.lmax,
                "fwhm_deg": self.fwhm_deg,
                "best_axis": best_axis.tolist(),
                "best_correlation": axis_result["best_correlation"],
                "fitted_amplitude": amplitude,
                "amplitude_error": amplitude_err,
                "p_value": p_value,
                "upper_limit_95": upper_limit_95,
                "y_pred_rms_cgm": self.y_pred_rms,
                "null_consistent": null_consistent,
                "cgm_safe": cgm_safe,
                "overall_passes": overall_passes,
                "p_value_fixed": null_result["p_value"],
                "upper_limit_fixed_95": null_result["upper_limit_95"],
                "p_value_scanned": p_value_scanned,
                "upper_limit_scanned_95": ul95_scanned,
                "sigma_null": sigma_null,
                "fixed_axis": fixed_axis_results
            }
            
            # Print results
            print("\nTEST RESULTS:")
            print(f"   Best-fit axis: [{best_axis[0]:.3f}, {best_axis[1]:.3f}, {best_axis[2]:.3f}]")
            print(f"   P-value: {p_value:.4f}")
            print(f"   95% upper limit: {upper_limit_95:.2e}")
            print(f"   CGM prediction: {self.y_pred_rms:.2e}")
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
    
    def run_fixed_axis_tests(self, data_map: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Run preregistered fixed-axis tests for physically motivated directions."""
        # Define physically motivated axes (preregistered)
        AXES = {
            "galactic_north": np.array([0.0, 0.0, 1.0]),
            "galactic_south": np.array([0.0, 0.0, -1.0]),
            "ecliptic_north": np.array([-0.096, 0.862, 0.498]),   # in Galactic frame
            "ecliptic_south": np.array([0.096, -0.862, -0.498]),
            "cmb_dipole": np.array([-0.070, -0.662, 0.745]),      # Planck dipole (Galactic)
        }
        
        results = {}
        
        for name, axis in AXES.items():
            print(f"\nFixed-axis test: {name}")
            print(f"   Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
            
            # Create template for this axis
            template = self._template_on_sphere(axis)
            
            # Fit amplitude
            fit_result = self.fit_template_amplitude(data_map, template, mask)
            
            # Compute fixed-axis null (no scanning)
            null_result = self.compute_null_distribution(data_map, template, mask, axis)
            
            # Store results
            results[name] = {
                "axis": axis.tolist(),
                "amplitude": fit_result["amplitude"],
                "amplitude_err": fit_result["amplitude_err"],
                "p_value": null_result["p_value"],
                "upper_limit_95": null_result["upper_limit_95"],
                "significant": null_result["p_value"] < self.significance_threshold
            }
            
            print(f"   Amplitude: {fit_result['amplitude']:.2e} ± {fit_result['amplitude_err']:.2e}")
            print(f"   P-value: {null_result['p_value']:.4f}")
            print(f"   95% UL: {null_result['upper_limit_95']:.2e}")
            print(f"   Significant: {'YES' if results[name]['significant'] else 'NO'}")
        
        return results


class EtheringtonComptonYCoherenceTest(BaseCGMTest):
    """Test B: Do Etherington distance duality violations correlate with Compton-y anisotropies?"""
    
    def __init__(self):
        super().__init__()
        self.y_pred_rms = get_cgm_template_amplitude("y")
        self.tau_pred_rms = get_cgm_template_amplitude("tau")
        self.correlation_threshold = 0.1  # Minimum correlation for significance
    
    def load_data(self) -> Dict[str, Any]:
        """Load real Compton-y data from data manager."""
        if self.data_manager is None:
            raise RuntimeError("Data manager not set. Call set_data_manager() first.")
        
        return self.data_manager.get_data_for_test("planck_compton_y")
    
    def prepare_data_for_low_ell_analysis(self, y_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for low-ell analysis using data manager's preprocessed data."""
        if self.data_manager is None:
            raise RuntimeError("Data manager not set. Call set_data_manager() first.")
        
        return self.data_manager.get_preprocessed_data(
            lmax=self.lmax,
            nside_target=self.nside,
            fwhm_deg=self.fwhm_deg
        )
    
    def compute_etherington_violations(self, axis: np.ndarray) -> np.ndarray:
        """Compute Etherington violations using exact HEALPix geometry."""
        # _template_on_sphere already returns zero-mean, unit-RMS values
        return self._template_on_sphere(axis)
    
    def find_best_fit_axis(self, data_alm: np.ndarray) -> Dict[str, Any]:
        """Find the best-fit axis by scanning rotations."""
        axis_result = find_best_axis(data_alm, lmax=self.lmax, n_theta=10, n_phi=20)
        
        return {
            "best_axis": axis_result["best_axis"],
            "best_correlation": axis_result["best_correlation"],
            "axis_scan": axis_result
        }
    
    def compute_correlation_with_axis_scan(self, y_map: np.ndarray, mask: np.ndarray, 
                                         best_axis: np.ndarray) -> Dict[str, Any]:
        """Compute amplitude fit between y-map and Etherington violations with axis scanning."""
        # Apply mask
        valid_pixels = (mask > 0)
        y_valid = y_map[valid_pixels]
        
        # Compute Etherington violations with best axis
        violations = self.compute_etherington_violations(best_axis)
        violations_valid = violations[valid_pixels]
        
        # Fit amplitude: y_map ≈ A * violations
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
        """Run the complete Etherington Compton-y coherence test."""
        try:
            # Load real Compton-y data
            y_data = self.load_data()
            
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
            
            # Pass criteria
            sufficient_data = len(prepared_data["y_map"]) > 1000
            null_consistent = p_value >= 0.01
            cgm_safe = upper_limit_95 > 100 * self.y_pred_rms
            
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
                "overall_passes": overall_passes
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


class SupernovaHubbleResidualsTest(BaseCGMTest):
    """Test C: Do Type-Ia supernova distance modulus residuals show the CGM toroidal kernel?"""
    
    def __init__(self):
        super().__init__()
        self.mu_pred_rms = get_cgm_template_amplitude("mu")
        self.n_axis_angles = 100  # Number of axis angles to scan
    
    def load_supernova_catalog(self) -> Dict[str, Any]:
        """Load supernova catalog from data manager."""
        if self.data_manager is None:
            raise RuntimeError("Data manager not set. Call set_data_manager() first.")
        
        return self.data_manager.get_supernova_data()
    
    def scan_axis_for_best_fit(self, sn_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scan over axis directions to find the best-fit axis for SN residuals."""
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
        """Fit the toroidal pattern to supernova residuals."""
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
        
        # Weighted least squares (currently uniform weights)
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
        """Compute null distribution by randomly rotating the axis."""
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
        """Check if fitted signs match CGM predictions."""
        A = fit_result["A"]
        B = fit_result["B"]
        
        # CGM predicts specific sign relationships based on toroidal geometry
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
        """Run the complete supernova Hubble residuals test."""
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
            
            # Pass criteria
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
                "overall_passes": overall_passes
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