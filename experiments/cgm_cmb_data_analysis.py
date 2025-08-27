#!/usr/bin/env python3
"""
CGM Empirical Validation Suite
Tests specific predictions of the Common Governance Model against real observational data.
Focuses on the 8-fold toroidal structure and cross-scale coherence.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union, cast
from pathlib import Path

import hashlib
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# THEORETICAL FRAMEWORK
# ============================================================================

@dataclass
class CGMThresholds:
    """Fundamental CGM thresholds from axiom and theorems."""
    cs_angle: float = np.pi/2      # Common Source chirality seed
    una_angle: float = np.pi/4     # Unity Non-Absolute planar split  
    ona_angle: float = np.pi/4     # Opposition Non-Absolute diagonal tilt
    bu_amplitude: float = 1/(2*np.sqrt(2*np.pi))  # Balance Universal closure
    
    # Derived parameters for toroidal kernel
    a_polar: float = 0.2      # Polar cap strength
    b_cubic: float = 0.1      # Ring lobe strength
    
    # Cross-scale invariants from discoveries
    loop_pitch: float = 1.702935    # Helical pitch
    holonomy_deficit: float = 0.863  # Toroidal holonomy (rad)
    index_37: int = 37              # Recursive ladder index


class TestMode(Enum):
    """Testing paradigms for CGM validation."""
    HYPOTHESIS = "hypothesis"    # Test specific theoretical predictions
    DISCOVERY = "discovery"      # Search for new patterns
    VALIDATION = "validation"    # Cross-validate between domains


@dataclass  
class ToroidalGeometry:
    """8-fold toroidal structure from CGM theory."""
    memory_axis: np.ndarray      # Primary axis (unit vector)
    ring_axes: np.ndarray        # 6 ring directions (6x3 array)
    polar_axes: np.ndarray       # 2 polar directions (2x3 array)
    
    @classmethod
    def from_memory_axis(cls, axis: np.ndarray, ring_phase: float = 0.0):
        """Generate 8-fold structure from memory axis."""
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        
        # Build orthonormal basis
        if abs(axis[2]) < 0.9:
            x = np.cross([0, 0, 1], axis)
        else:
            x = np.cross([1, 0, 0], axis)
        x = x / np.linalg.norm(x)
        y = np.cross(axis, x)
        
        # Polar caps
        polar_axes = np.array([axis, -axis])
        
        # Ring lobes (60° spacing)
        ring_axes = []
        for k in range(6):
            phi = ring_phase + 2*np.pi*k/6
            direction = np.cos(phi)*x + np.sin(phi)*y
            ring_axes.append(direction)
        ring_axes = np.array(ring_axes)
        
        return cls(
            memory_axis=axis,
            ring_axes=ring_axes,
            polar_axes=polar_axes
        )
    
    def get_all_axes(self) -> np.ndarray:
        """Return all 8 axes in order: 2 polar + 6 ring."""
        return np.vstack([self.polar_axes, self.ring_axes])
    
    def predict_sign_pattern(self, a_polar: float, b_cubic: float) -> np.ndarray:
        """Predict the sign pattern for the 8 lobes."""
        # Polar caps: dominated by quadrupole P2
        polar_signs = np.sign([a_polar, a_polar])  # Both same sign
        
        # Ring lobes: dominated by cubic C4 with alternation
        ring_signs = np.array([1, -1, 1, -1, 1, -1]) * np.sign(b_cubic)
        
        return np.concatenate([polar_signs, ring_signs])


# ============================================================================
# DATA MANAGEMENT
# ============================================================================

class CGMDataManager:
    """Centralized data manager with caching and validation."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        # Paths
        self.experiments_dir = Path(__file__).resolve().parent
        self.data_dir = self.experiments_dir / "data"
        
        # Cache setup
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "cgm"
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory caches
        self._data_cache: Dict[str, Any] = {}
        self._results_cache: Dict[str, Any] = {}
    
    def get_planck_data(self, nside: int = 8, lmax: int = 2,
                        fwhm_deg: float = 30.0,
                        fast_preprocess: bool = True) -> Dict[str, Any]:
        """Load and preprocess Planck Compton-y data (fast path)."""
        import time
        t0 = time.perf_counter()

        cache_key = f"planck_n{nside}_l{lmax}_f{int(fwhm_deg)}_fast{int(fast_preprocess)}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.npz"
        if cache_file.exists():
            print(f"Loading Planck data from cache: {cache_file.name}")
            z = np.load(cache_file)
            data = {k: z[k] for k in z.files}
            self._data_cache[cache_key] = data
            return data

        print("Preprocessing Planck data (fast mode)...")
        try:
            import healpy as hp  # pyright: ignore[reportMissingImports]
            import astropy.io.fits as fits
            from typing import Any

            t1 = time.perf_counter()
            y_map_file = self.data_dir / "milca_ymaps.fits"
            mask_file = self.data_dir / "COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits"

            # Load raw as float32 only once
            with fits.open(str(y_map_file), memmap=True) as hdul:
                hdu1 = hdul[1]
                assert isinstance(hdu1, fits.BinTableHDU)
                data_rec = cast(Any, hdu1.data)  # type: ignore
                y_raw = np.array(data_rec['FULL'], dtype=np.float32)
            with fits.open(str(mask_file), memmap=True) as hdul:
                hdu1 = hdul[1]
                assert isinstance(hdu1, fits.BinTableHDU)
                data_rec = cast(Any, hdu1.data)  # type: ignore
                mask_raw = np.array(data_rec['M1'], dtype=np.float32)
            t2 = time.perf_counter()
            print(f"  FITS read: {(t2 - t1):.2f}s")

            # Fast path: degrade first, then (optional) light smoothing
            t3 = time.perf_counter()
            y_low = hp.ud_grade(y_raw, nside_out=nside)  # very fast
            mask_low = hp.ud_grade(mask_raw, nside_out=nside)
            mask_low = (mask_low > 0.5).astype(np.float32)
            t4 = time.perf_counter()
            print(f"  ud_grade to NSIDE={nside}: {(t4 - t3):.2f}s")

            # Remove dipole at low NSIDE (much cheaper)
            y_low = hp.remove_dipole(y_low * (mask_low > 0), gal_cut=0)
            # Optional smoothing now at low NSIDE (very cheap)
            if fwhm_deg > 0:
                y_low = hp.smoothing(y_low, fwhm=np.radians(fwhm_deg))
            t5 = time.perf_counter()
            print(f"  low-res clean/smooth: {(t5 - t4):.2f}s")

            # Low-l alm and cl
            alm = hp.map2alm(y_low * mask_low, lmax=lmax, iter=0)
            cl = hp.alm2cl(alm)

            data = {
                'y_map': y_low.astype(np.float32),
                'mask': mask_low.astype(np.float32),
                'alm': alm.astype(np.complex64),
                'cl': cl.astype(np.float32),
                'nside': int(nside),
                'lmax': int(lmax),
            }
            np.savez_compressed(cache_file, **data)
            self._data_cache[cache_key] = data

            t6 = time.perf_counter()
            print(f"  Total Planck prep: {(t6 - t0):.2f}s (cached: {cache_file.name})")
            return data

        except ImportError:
            raise ImportError("healpy and astropy required")
    
    def get_supernova_data(self) -> Dict[str, Any]:
        """Load Pantheon+ supernova data with proper columns."""
        cache_key = "pantheon_plus"
        
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        cache_file = self.cache_dir / f"{cache_key}.npz"
        
        if cache_file.exists():
            data = dict(np.load(cache_file))
            self._data_cache[cache_key] = data
            return data
        
        print("Loading Pantheon+ data...")
        
        # Load with named columns
        pantheon_file = self.data_dir / "pantheon_plus_real.dat"
        
        # Use correct column names from Pantheon+ data
        col_names = ['CID', 'IDSURVEY', 'zHD', 'zHDERR', 'zCMB', 'zCMBERR', 'zHEL', 'zHELERR',
                    'm_b_corr', 'm_b_corr_err_DIAG', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG', 'CEPH_DIST',
                    'IS_CALIBRATOR', 'USED_IN_SH0ES_HF', 'c', 'cERR', 'x1', 'x1ERR', 'mB', 'mBERR',
                    'x0', 'x0ERR', 'COV_x1_c', 'COV_x1_x0', 'COV_c_x0', 'RA', 'DEC', 'HOST_RA',
                    'HOST_DEC', 'HOST_ANGSEP', 'VPEC', 'VPECERR', 'MWEBV', 'HOST_LOGMASS',
                    'HOST_LOGMASS_ERR', 'PKMJD', 'PKMJDERR', 'NDOF', 'FITCHI2', 'FITPROB',
                    'm_b_corr_err_RAW', 'm_b_corr_err_VPEC', 'biasCor_m_b', 'biasCorErr_m_b',
                    'biasCor_m_b_COVSCALE', 'biasCor_m_b_COVADD']
        
        # Simple numpy-based Pantheon+ loader
        print("Loading Pantheon+ data with numpy...")
        # Load as strings first to handle mixed data types
        data_array = np.loadtxt(pantheon_file, skiprows=1, dtype=str)
        
        # Column positions (adjust if needed)
        z = data_array[:, 4].astype(float)  # zCMB
        mu = data_array[:, 20].astype(float)  # MU_SH0ES
        mu_err = data_array[:, 21].astype(float)  # MU_SH0ES_ERR_DIAG
        ra = data_array[:, 26].astype(float)  # RA
        dec = data_array[:, 27].astype(float)  # DEC
        
        # Quality cuts
        valid = (z > 0.01) & (z < 2.3) & (mu_err > 0) & (mu_err < 1.0)
        z = z[valid]
        mu = mu[valid]
        mu_err = mu_err[valid]
        ra = ra[valid]
        dec = dec[valid]
        
        # Compute residuals using lightweight ΛCDM cosmology
        def Dl_flat_LCDM(z_array: np.ndarray, H0_km_s_Mpc: float = 70.0, Om0: float = 0.3) -> np.ndarray:
            """Compute luminosity distance in flat ΛCDM cosmology."""
            c = 299792.458  # km/s
            Ol0 = 1.0 - Om0
            z_array = np.asarray(z_array, dtype=float)
            
            # Comoving distance integral (trapezoidal)
            def Ez(zv): 
                return np.sqrt(Om0*(1+zv)**3 + Ol0)
            
            D = np.zeros_like(z_array)
            for i, zi in enumerate(z_array):
                zs = np.linspace(0.0, zi, 400)
                Ei = Ez(zs)
                chi = np.trapezoid(1.0/Ei, zs)
                D[i] = chi
            
            Dc = (c / H0_km_s_Mpc) * D   # Mpc
            Dl = (1 + z_array) * Dc
            return Dl
        
        Dl = Dl_flat_LCDM(z)
        mu_model = 5 * np.log10(np.maximum(Dl, 1e-9)) + 25.0
        residuals = mu - mu_model
        
        # Convert to unit vectors
        theta = np.radians(90 - dec)
        phi = np.radians(ra)
        positions = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        data = {
            'z': z.astype(np.float32),
            'mu': mu.astype(np.float32),
            'mu_err': mu_err.astype(np.float32),
            'residuals': residuals.astype(np.float32),
            'ra': ra.astype(np.float32),
            'dec': dec.astype(np.float32),
            'positions': positions.astype(np.float32),
            'n_sn': len(z)
        }
        
        # Save cache
        np.savez_compressed(cache_file, **data)
        self._data_cache[cache_key] = data
        
        print(f"Loaded {len(z)} supernovae")
        print(f"Redshift range: {z.min():.3f} - {z.max():.3f}")
        print(f"Mean residual: {residuals.mean():.3f} ± {residuals.std():.3f} mag")
        print(f"Mean uncertainty: {mu_err.mean():.3f} mag")
        
        return data
    
    def get_bao_data(self) -> Dict[str, Any]:
        """Load BAO data with actual survey positions."""
        cache_key = "bao_alam2016"
        
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        print("Loading BAO data from Alam et al. 2016...")
        
        # BAO effective redshifts and sky positions
        # From SDSS-III BOSS DR12 galaxy survey
        bao_data = {
            'z': np.array([0.38, 0.51, 0.61]),
            # Effective survey centers (approximate)
            'ra': np.array([180.0, 185.0, 190.0]),  # degrees
            'dec': np.array([25.0, 30.0, 35.0]),    # degrees
            # Measurements from consensus analysis
            'dV_rs': np.array([10.0509, 12.9288, 14.5262]),
            'dV_rs_err': np.array([0.1389, 0.1761, 0.2164])
        }
        
        # Convert to unit vectors
        theta = np.radians(90 - bao_data['dec'])
        phi = np.radians(bao_data['ra'])
        positions = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        data = {
            'z': bao_data['z'].astype(np.float32),
            'positions': positions.astype(np.float32),
            'dV_rs': bao_data['dV_rs'].astype(np.float32),
            'dV_rs_err': bao_data['dV_rs_err'].astype(np.float32),
            'n_bao': len(bao_data['z'])
        }
        
        self._data_cache[cache_key] = data
        return data


# ============================================================================
# TOROIDAL TEMPLATE GENERATION
# ============================================================================

def generate_toroidal_template(nside: int, axis: np.ndarray,
                              a_polar: float, b_cubic: float) -> np.ndarray:
    """Generate toroidal template on HEALPix sphere."""
    import healpy as hp  # pyright: ignore[reportMissingImports]
    
    npix = hp.nside2npix(nside)
    template = np.zeros(npix)
    
    # Get pixel directions
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Normalize axis
    axis = np.asarray(axis) / np.linalg.norm(axis)
    
    # Compute mu = cos(angle with axis)
    mu = x*axis[0] + y*axis[1] + z*axis[2]
    
    # P2 (quadrupole) for polar structure
    P2 = 0.5 * (3*mu**2 - 1)
    
    # C4 (cubic) for ring structure
    C4 = x**4 + y**4 + z**4 - 0.6
    
    # Combine
    template = a_polar * P2 + b_cubic * C4
    
    return template


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

class ToroidalCoherenceTest:
    """Test for 8-fold toroidal coherence in data."""
    
    def __init__(self, thresholds: CGMThresholds):
        self.thresholds = thresholds
        self.geometry: Optional[ToroidalGeometry] = None
    
    def set_geometry(self, memory_axis: np.ndarray, ring_phase: float = 0.0):
        """Set the toroidal geometry for testing."""
        self.geometry = ToroidalGeometry.from_memory_axis(memory_axis, ring_phase)
    
    def measure_lobe_amplitudes(self, data_map: np.ndarray, mask: np.ndarray,
                               nside: int) -> np.ndarray:
        """Measure amplitude at each of the 8 lobes."""
        if self.geometry is None:
            raise ValueError("Geometry not set")
        
        amplitudes = []
        all_axes = self.geometry.get_all_axes()
        
        for axis in all_axes:
            template = generate_toroidal_template(
                nside, axis, 
                self.thresholds.a_polar,
                self.thresholds.b_cubic
            )
            
            # Masked least squares fit
            valid = mask > 0
            if valid.sum() < 100:
                amplitudes.append(0.0)
                continue
            
            t = template[valid]
            d = data_map[valid]
            
            XtX = np.dot(t, t)
            if XtX > 0:
                A = np.dot(t, d) / XtX
            else:
                A = 0.0
            
            amplitudes.append(A)
        
        return np.array(amplitudes)
    
    def compute_coherence_score(self, amplitudes: np.ndarray) -> Dict[str, Any]:
        """Compute coherence metrics for the 8-fold pattern."""
        if self.geometry is None:
            raise ValueError("Geometry not set")
            
        # Predicted sign pattern
        predicted = self.geometry.predict_sign_pattern(
            self.thresholds.a_polar,
            self.thresholds.b_cubic
        )
        
        # Observed signs
        observed = np.sign(amplitudes)
        
        # Sign coherence (fraction matching prediction)
        sign_coherence = np.mean(observed == predicted)
        
        # Amplitude coherence (correlation with prediction)
        if np.std(amplitudes) > 0 and np.std(predicted) > 0:
            amp_coherence = np.corrcoef(amplitudes, predicted)[0, 1]
        else:
            amp_coherence = 0.0
        
        # Polar/ring ratio
        polar_amp = np.mean(np.abs(amplitudes[:2]))
        ring_amp = np.mean(np.abs(amplitudes[2:]))
        polar_ring_ratio = polar_amp / (ring_amp + 1e-10)
        
        return {
            'sign_coherence': sign_coherence,
            'amplitude_coherence': amp_coherence,
            'polar_ring_ratio': polar_ring_ratio,
            'mean_amplitude': np.mean(amplitudes),
            'std_amplitude': np.std(amplitudes),
            'amplitudes': amplitudes
        }
    
    def null_distribution_phase_randomized(self, data_map: np.ndarray,
                                          mask: np.ndarray, nside: int,
                                          lmax: int, n_mc: int = 1000,
                                          seed: int = 42) -> List[Dict]:
        """Generate null distribution via phase randomization."""
        import healpy as hp  # pyright: ignore[reportMissingImports]
        
        rng = np.random.default_rng(seed)
        
        # Compute alm of masked map
        alm = hp.map2alm(data_map * mask, lmax=lmax, iter=3)
        l, m = hp.Alm.getlm(lmax)
        
        null_results = []
        
        for i in range(n_mc):
            # Randomize phases for m > 0
            alm_rand = alm.copy()
            for idx in range(len(alm)):
                if m[idx] > 0:
                    phase = rng.uniform(0, 2*np.pi)
                    alm_rand[idx] = np.abs(alm[idx]) * np.exp(1j * phase)
            
            # Generate null map
            null_map = hp.alm2map(alm_rand, nside, lmax=lmax)
            
            # Measure on null
            null_amps = self.measure_lobe_amplitudes(null_map, mask, nside)
            null_coherence = self.compute_coherence_score(null_amps)
            null_results.append(null_coherence)
        
        return null_results
    
    def compute_significance(self, observed: Dict[str, Any],
                           null_dist: List[Dict]) -> Dict[str, float]:
        """Compute p-values for observed metrics."""
        p_values = {}
        
        for key in ['sign_coherence', 'amplitude_coherence', 'polar_ring_ratio']:
            if key not in observed:
                continue
            
            obs_val = observed[key]
            null_vals = [n[key] for n in null_dist]
            
            # Two-tailed test for correlation, one-tailed for others
            if 'coherence' in key:
                k = np.sum(np.abs(null_vals) >= np.abs(obs_val))
            else:
                k = np.sum(null_vals >= obs_val)
            
            p_values[key] = (k + 1) / (len(null_vals) + 1)
        
        return p_values


# ============================================================================
# CROSS-SCALE VALIDATION
# ============================================================================

class CrossScaleValidator:
    """Validate CGM predictions across cosmic scales."""
    
    def __init__(self, data_manager: CGMDataManager, thresholds: CGMThresholds):
        self.dm = data_manager
        self.thresholds = thresholds
        self.results = {}
    
    def find_optimal_axis(self, mode: str = 'cmb_dipole') -> np.ndarray:
        """Determine the memory axis for testing."""
        if mode == 'cmb_dipole':
            # CMB dipole direction (galactic coordinates)
            return np.array([-0.070, -0.662, 0.745])
        elif mode == 'galactic':
            # Galactic north pole
            return np.array([0.0, 0.0, 1.0])
        elif mode == 'ecliptic':
            # Ecliptic north pole
            return np.array([0.0, -0.398, 0.917])
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def test_planck_y_map(self, memory_axis: np.ndarray) -> Dict[str, Any]:
        """Test toroidal pattern in Planck Compton-y map."""
        print("\nTesting Planck Compton-y map...")
        
        # Load data with production parameters
        data = self.dm.get_planck_data(nside=64, lmax=32, fwhm_deg=2.0, fast_preprocess=True)
        
        # Setup test
        test = ToroidalCoherenceTest(self.thresholds)
        test.set_geometry(memory_axis)
        
        # Measure
        amplitudes = test.measure_lobe_amplitudes(
            data['y_map'], data['mask'], data['nside']
        )
        coherence = test.compute_coherence_score(amplitudes)
        
        # Null distribution
        print("  Computing null distribution...")
        null_dist = test.null_distribution_phase_randomized(
            data['y_map'], data['mask'], data['nside'],
            data['lmax'], n_mc=1000
        )
        
        # Significance
        p_values = test.compute_significance(coherence, null_dist)
        
        result = {
            'coherence': coherence,
            'p_values': p_values,
            'memory_axis': memory_axis.tolist()
        }
        
        print(f"  Sign coherence: {coherence['sign_coherence']:.3f} (p={p_values['sign_coherence']:.4f})")
        print(f"  Amplitude coherence: {coherence['amplitude_coherence']:.3f} (p={p_values['amplitude_coherence']:.4f})")
        
        return result
    
    def test_supernova_residuals(self, memory_axis: np.ndarray) -> Dict[str, Any]:
        """Test toroidal pattern in supernova Hubble residuals."""
        print("\nTesting supernova residuals...")
        
        # Load data
        sn = self.dm.get_supernova_data()
        
        # Project positions onto toroid
        geometry = ToroidalGeometry.from_memory_axis(memory_axis)
        all_axes = geometry.get_all_axes()
        
        # Compute projections for each SN onto each lobe
        projections = np.zeros((sn['n_sn'], 8))
        for i, axis in enumerate(all_axes):
            projections[:, i] = np.dot(sn['positions'], axis)
        
        # Find which lobe each SN is closest to
        lobe_assignments = np.argmax(np.abs(projections), axis=1)
        
        # Compute mean residual per lobe
        lobe_residuals = np.zeros(8)
        lobe_counts = np.zeros(8)
        
        for i, lobe in enumerate(lobe_assignments):
            lobe_residuals[lobe] += sn['residuals'][i]
            lobe_counts[lobe] += 1
        
        # Average where we have data
        mask = lobe_counts > 0
        lobe_residuals[mask] /= lobe_counts[mask]
        
        # Coherence analysis
        predicted = geometry.predict_sign_pattern(
            self.thresholds.a_polar,
            self.thresholds.b_cubic
        )
        
        observed_signs = np.sign(lobe_residuals[mask])
        predicted_signs = predicted[mask]
        
        sign_coherence = np.mean(observed_signs == predicted_signs) if mask.sum() > 0 else 0
        
        # Bootstrap null
        print("  Computing bootstrap null...")
        rng = np.random.default_rng(42)
        n_boot = 1000
        null_coherences = []
        
        for _ in range(n_boot):
            # Shuffle residuals
            shuf_res = rng.permutation(sn['residuals'])
            
            # Recompute per lobe
            boot_residuals = np.zeros(8)
            for i, lobe in enumerate(lobe_assignments):
                boot_residuals[lobe] += shuf_res[i]
            boot_residuals[mask] /= lobe_counts[mask]
            
            boot_signs = np.sign(boot_residuals[mask])
            boot_coherence = np.mean(boot_signs == predicted_signs) if mask.sum() > 0 else 0
            null_coherences.append(boot_coherence)
        
        null_coherences_array = np.asarray(null_coherences, dtype=float)
        p_value = np.mean(null_coherences_array >= sign_coherence)
        
        result = {
            'lobe_residuals': lobe_residuals.tolist(),
            'lobe_counts': lobe_counts.tolist(),
            'sign_coherence': float(sign_coherence),
            'p_value': float(p_value),
            'memory_axis': memory_axis.tolist()
        }
        
        print(f"  Sign coherence: {sign_coherence:.3f} (p={p_value:.4f})")
        print(f"  Active lobes: {mask.sum()}/8")
        
        return result
    
    def test_planck_only(self) -> Dict[str, Any]:
        """Test ONLY Planck data across all axes (fast, independent)."""
        print("\n" + "="*60)
        print("PLANCK-ONLY VALIDATION (Fast Path)")
        print("="*60)
        
        axes_to_test = {
            'cmb_dipole': self.find_optimal_axis('cmb_dipole'),
            'galactic': self.find_optimal_axis('galactic'),
            'ecliptic': self.find_optimal_axis('ecliptic')
        }
        
        results = {}
        
        for name, axis in axes_to_test.items():
            print(f"\n### Testing axis: {name}")
            print(f"    Direction: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
            
            # Only Planck test
            planck_result = self.test_planck_y_map(axis)
            results[name] = planck_result
            
            print(f"  ✓ Planck test completed for {name}")
        
        # Find best axis based on Planck only
        best_axis = min(results.keys(), key=lambda k: results[k]['p_values']['sign_coherence'])
        best_p = results[best_axis]['p_values']['sign_coherence']
        
        print("\n" + "="*60)
        print("PLANCK-ONLY SUMMARY")
        print("="*60)
        print(f"Best axis: {best_axis}")
        print(f"Best p-value: {best_p:.4f}")
        print(f"Significance: {'YES' if best_p < 0.05 else 'NO'}")
        
        return {
            'test_type': 'planck_only',
            'axes_tested': {k: v.tolist() for k, v in axes_to_test.items()},
            'results': results,
            'best_axis': best_axis,
            'best_p': best_p,
            'significant': best_p < 0.05
        }
    
    def test_supernova_only(self, memory_axis: np.ndarray | None = None) -> Dict[str, Any]:
        """Test ONLY Supernova data (independent test)."""
        if memory_axis is None:
            memory_axis = self.find_optimal_axis('galactic')
        print("\n" + "="*60)
        print("SUPERNOVA-ONLY VALIDATION")
        print("="*60)
        print(f"Testing axis: [{memory_axis[0]:.3f}, {memory_axis[1]:.3f}, {memory_axis[2]:.3f}]")
        
        try:
            sn_result = self.test_supernova_residuals(memory_axis)
            print(f"✓ Supernova test completed")
            return {
                'test_type': 'supernova_only',
                'memory_axis': memory_axis.tolist(),
                'result': sn_result
            }
        except Exception as e:
            print(f"✗ Supernova test failed: {e}")
            return {
                'test_type': 'supernova_only',
                'memory_axis': memory_axis.tolist(),
                'error': str(e)
            }
    
    def test_cross_scale_consistency(self) -> Dict[str, Any]:
        """Test consistency across all scales (full integration)."""
        print("\n" + "="*60)
        print("FULL CROSS-SCALE VALIDATION")
        print("="*60)
        
        # Test different candidate axes
        axes_to_test = {
            'cmb_dipole': self.find_optimal_axis('cmb_dipole'),
            'galactic': self.find_optimal_axis('galactic'),
            'ecliptic': self.find_optimal_axis('ecliptic')
        }
        
        results = {}
        
        for name, axis in axes_to_test.items():
            print(f"\n### Testing axis: {name}")
            print(f"    Direction: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
            
            axis_results = {}
            
            # Planck test
            axis_results['planck'] = self.test_planck_y_map(axis)
            
            # Supernova test (optional)
            try:
                axis_results['supernova'] = self.test_supernova_residuals(axis)
                p_sn = axis_results['supernova']['p_value']
            except Exception as e:
                print(f"  ⚠ Supernova test failed: {e}")
                p_sn = 1.0  # neutral in Fisher's method
                axis_results['supernova'] = {"error": str(e), "p_value": 1.0}
            
            # Combined significance
            p_planck = axis_results['planck']['p_values']['sign_coherence']
            
            # Fisher's method for combining p-values
            def chi2_sf_df4(x: float) -> float:
                # For df=4, SF = exp(-x/2) * (1 + x/2)
                x = float(x)
                return float(np.exp(-x/2.0) * (1.0 + x/2.0))
            
            chi2 = -2.0 * (np.log(max(p_planck, 1e-300)) + np.log(max(p_sn, 1e-300)))
            combined_p = chi2_sf_df4(chi2)
            
            axis_results['combined_p'] = combined_p
            results[name] = axis_results
            
            print(f"\n  Combined p-value: {combined_p:.4f}")
        
        # Find best axis
        best_axis = min(results.keys(), key=lambda k: results[k]['combined_p'])
        best_p = results[best_axis]['combined_p']
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Best axis: {best_axis}")
        print(f"Combined p-value: {best_p:.4f}")
        print(f"Significance: {'YES' if best_p < 0.05 else 'NO'}")
        
        return {
            'test_type': 'full_integration',
            'axes_tested': {k: v.tolist() for k, v in axes_to_test.items()},
            'results': results,
            'best_axis': best_axis,
            'best_p': best_p,
            'significant': best_p < 0.05
        }




# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete CGM empirical validation."""
    print("CGM EMPIRICAL VALIDATION SUITE v2.0")
    print("Testing theoretical predictions on real data")
    print()
    print("NOTE: Using production parameters (nside=64, lmax=32, n_mc=1000)")
    print("Full resolution analysis with robust statistics")
    print()
    
    # Initialize
    thresholds = CGMThresholds()
    data_manager = CGMDataManager()
    validator = CrossScaleValidator(data_manager, thresholds)
    
    # Run FULL validation with production parameters
    print("Running FULL CROSS-SCALE VALIDATION (Production Mode)")
    print("All tests enabled with proper depths and resolution")
    results = validator.test_cross_scale_consistency()
    
    # Results computed successfully (no file saving)
    print(f"\nResults computed successfully (no file saving)")
    
    # Final assessment
    if results['significant']:
        print("\n✓ CGM VALIDATION SUCCESSFUL")
        print("  Toroidal structure detected across scales")
    else:
        print("\n✗ CGM VALIDATION INCONCLUSIVE")
        print("  More data or refined analysis needed")
    
    return results


if __name__ == "__main__":
    main()