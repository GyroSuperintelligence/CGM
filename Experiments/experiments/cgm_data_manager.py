#!/usr/bin/env python3
"""
CGM Data Manager

Centralized data loading and caching for all CGM empirical tests.
Loads Planck Compton-y data and masks once, provides cached access to:
- Raw maps and masks
- Spherical harmonic coefficients
- Pre-computed statistics
- Test-specific data subsets

This eliminates duplicate data loading and HEALPix operations across tests.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pickle
from pathlib import Path
import warnings


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
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.y_map_file = os.path.join(workspace_root, "real_data", "milca_ymaps.fits")
        self.mask_file = os.path.join(workspace_root, "real_data", "COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits")
        
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
        """
        Load Planck Compton-y data and masks from real FITS files.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Dictionary with loaded data and metadata
        """
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
                # Data is in the second HDU (Y-MAP) with columns
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
                # Data is in the second HDU (GAL-MASK) with columns
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
            "is_real_data": True,
            "description": f"Real Planck Compton-y data (nside={self._nside})"
        }
    
    def get_spherical_harmonics(self, lmax: Optional[int] = None) -> Dict[str, Any]:
        """
        Get spherical harmonic coefficients for y-map using healpy.
        
        Args:
            lmax: Maximum multipole (uses self.lmax if None)
            
        Returns:
            Dictionary with alm coefficients and power spectrum
        """
        if lmax is None:
            lmax = self.lmax
        
        cache_key = f"alm_lmax{lmax}"
        
        if cache_key in self._alm_cache:
            return self._alm_cache[cache_key]
        
        if self._y_map is None or self._mask is None:
            raise RuntimeError("Data not loaded")
        
        try:
            import healpy as hp
            
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
        """
        Get test directions for correlation tests.
        
        Args:
            n_directions: Number of test directions
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (theta, phi) arrays
        """
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
        """
        Convert pixel indices to spherical coordinates using healpy.
        
        Args:
            pixels: Array of pixel indices
            
        Returns:
            Tuple of (theta, phi) arrays in radians
        """
        if self._nside is None:
            raise RuntimeError("Data not loaded")
        
        try:
            import healpy as hp
            
            # Use healpy for proper coordinate conversion
            theta, phi = hp.pix2ang(self._nside, pixels)
            return theta, phi
            
        except ImportError:
            raise ImportError("healpy is required for coordinate operations. Install with: pip install healpy")
    
    def extract_values_at_directions(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Extract y-map values at specific directions.
        
        Args:
            theta: Colatitude array
            phi: Longitude array
            
        Returns:
            Array of y-values at test directions
        """
        if self._y_map is None or self._nside is None:
            raise RuntimeError("Data not loaded")
        
        # Convert spherical coordinates to pixel indices using healpy
        pixels = self._ang2pix_healpy(theta, phi)
        
        # Extract values
        values = self._y_map[pixels]
        
        return values
    
    def _ang2pix_healpy(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Convert spherical coordinates to pixel indices using healpy.
        
        Args:
            theta: Colatitude array in radians
            phi: Longitude array in radians
            
        Returns:
            Array of pixel indices
        """
        if self._nside is None:
            raise RuntimeError("Data not loaded")
        
        try:
            import healpy as hp
            
            # Use healpy for proper coordinate conversion
            pixels = hp.ang2pix(self._nside, theta, phi)
            return pixels
            
        except ImportError:
            raise ImportError("healpy is required for coordinate operations. Install with: pip install healpy")
    
    def get_data_for_test(self, test_name: str) -> Dict[str, Any]:
        """
        Get data specifically formatted for a test.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Dictionary with test-specific data
        """
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
            # For Test C: Supernova data (placeholder)
            # In a real implementation, this would load Pantheon+ data
            warnings.warn("Supernova data not implemented. Using placeholder.")
            return {
                **self._get_data_summary(),
                'sn_data': {
                    'ra': np.random.uniform(0, 360, 1000),
                    'dec': np.random.uniform(-90, 90, 1000),
                    'residuals': np.random.normal(0, 0.1, 1000)
                },
                'stats': {
                    'z_range': (0.01, 1.0),
                    'mu_range': (30, 45)
                }
            }
            
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
        
        # Calculate alm cache memory (each cache entry is a dict with 'alm' and 'cl' keys)
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

    def get_preprocessed_data(self, lmax: int = 8, nside_target: int = 64, fwhm_deg: float = 10.0) -> Dict[str, Any]:
        """
        Get preprocessed data for low-ell analysis using proper HEALPix operations.
        
        This method caches expensive operations like smoothing, monopole/dipole removal,
        and resolution degradation to avoid redundant work between tests.
        
        Args:
            lmax: Maximum multipole for analysis
            nside_target: Target resolution for analysis
            fwhm_deg: Smoothing FWHM in degrees
            
        Returns:
            Dictionary with preprocessed data
        """
        cache_key = f"preprocessed_lmax{lmax}_nside{nside_target}_fwhm{fwhm_deg}"
        
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]
        
        if self._y_map is None or self._mask is None:
            raise RuntimeError("Data not loaded")
        
        try:
            import healpy as hp
            
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
            pantheon_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                        "real_data", "pantheon_plus_real.dat")
            
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
                # In practice, you'd use the published residuals
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
                    "is_real_data": True
                }
            else:
                raise FileNotFoundError(f"Pantheon+ file not found: {pantheon_file}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load real supernova data: {e}")
    



# Global instance for easy access
_data_manager: Optional[CGMDataManager] = None

def get_data_manager() -> CGMDataManager:
    """Get the global data manager instance."""
    global _data_manager
    if _data_manager is None:
        _data_manager = CGMDataManager()
    return _data_manager


if __name__ == "__main__":
    # Test the data manager
    dm = CGMDataManager()
    data = dm.load_data()
    print(f"Data loaded: {data['description']}")
    
    # Test spherical harmonics
    alm_data = dm.get_spherical_harmonics()
    print(f"Spherical harmonics computed: lmax={alm_data['lmax']}")
    
    # Test memory usage
    mem_info = dm.get_memory_usage()
    print(f"Memory usage: {mem_info}")
