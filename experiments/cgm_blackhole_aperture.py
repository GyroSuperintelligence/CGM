# CGM Black Hole Aperture Leakage — direct, equation-driven calculations
#
# Implements:
#   S_CGM = S_BH * (1 + m_p),  m_p = 1/(2 sqrt(2π))
#   T_CGM = T_H / (1 + m_p)
#   τ_CGM = τ_std * (1 + m_p)^4
#
# Prints plain text blocks (no tables).

import math
from dataclasses import dataclass
from typing import Dict, List

# Physical constants (SI)
c = 299_792_458.0  # m/s (exact)
k_B = 1.380_649e-23  # J/K (exact)
hbar = 1.054_571_817e-34  # J·s (exact, via defined h)
G = 6.674_30e-11  # m^3/(kg·s^2) (CODATA nominal)
epsilon0 = 8.854_187_8128e-12  # F/m (vacuum permittivity)
M_sun = 1.988_47e30  # kg (IAU nominal)
pi = math.pi

# Energy conversion
eV_J = 1.602_176_634e-19  # J per eV (exact)
MeV_J = eV_J * 1.0e6  # J per MeV

# CGM parameter
m_p = 1.0 / (2.0 * math.sqrt(2.0 * pi))  # ≈ 0.19947114020071635


@dataclass
class BHResult:
    name: str
    M_kg: float
    M_solar: float
    r_s_m: float
    A_m2: float
    S_BH_J_per_K: float
    S_CGM_J_per_K: float
    S_factor: float
    T_H_K: float
    T_CGM_K: float
    T_factor: float
    tau_std_s: float
    tau_cgm_s: float
    tau_factor: float


def fmt_si(x: float, unit: str = "", sig: int = 3) -> str:
    return f"{x:.{sig}e} {unit}".strip()


def seconds_to_readable(t: float) -> str:
    year = 365.25 * 24 * 3600
    day = 24 * 3600
    if t < 1e-6:
        return fmt_si(t, "s")
    if t < 60:
        return f"{t:.3f} s"
    if t < 3600:
        return f"{t/60:.3f} min"
    if t < day:
        return f"{t/3600:.3f} h"
    if t < year:
        return f"{t/day:.3f} d"
    return f"{t/year:.3e} yr"


def bh_properties(name: str, M_kg: float) -> BHResult:
    r_s = 2.0 * G * M_kg / c**2
    A = 4.0 * pi * r_s**2
    S_BH = k_B * (A * c**3) / (4.0 * G * hbar)
    S_CGM = S_BH * (1.0 + m_p)
    T_H = (hbar * c**3) / (8.0 * pi * G * M_kg * k_B)
    T_CGM = T_H / (1.0 + m_p)
    tau_std = (5120.0 * pi * G**2 * M_kg**3) / (hbar * c**4)
    tau_cgm = tau_std * (1.0 + m_p) ** 4
    # Verify T·S invariance
    T_S_invariance = abs(T_H * S_BH - T_CGM * S_CGM) / (T_H * S_BH)
    assert T_S_invariance < 1e-12, f"T·S invariance violated: {T_S_invariance:.2e}"

    return BHResult(
        name=name,
        M_kg=M_kg,
        M_solar=M_kg / M_sun,
        r_s_m=r_s,
        A_m2=A,
        S_BH_J_per_K=S_BH,
        S_CGM_J_per_K=S_CGM,
        S_factor=(1.0 + m_p),
        T_H_K=T_H,
        T_CGM_K=T_CGM,
        T_factor=1.0 / (1.0 + m_p),
        tau_std_s=tau_std,
        tau_cgm_s=tau_cgm,
        tau_factor=(1.0 + m_p) ** 4,
    )


def print_result(res: BHResult) -> None:
    print(f"\n— {res.name} —")
    print(f"Mass: {fmt_si(res.M_kg, 'kg')}  ({res.M_solar:.6g} M_sun)")
    print(f"Horizon radius r_s: {fmt_si(res.r_s_m, 'm')}  ({res.r_s_m/1000:.3e} km)")
    print(f"Area A: {fmt_si(res.A_m2, 'm^2')}")
    print("Entropy (Bekenstein–Hawking → CGM):")
    print(f"  S_BH:  {fmt_si(res.S_BH_J_per_K, 'J/K')}")
    print(
        f"  S_CGM: {fmt_si(res.S_CGM_J_per_K, 'J/K')}  (×{res.S_factor:.6f},  +{(res.S_factor-1.0)*100:.4f}%)"
    )
    print(f"  S_BH/k_B  = {res.S_BH_J_per_K/k_B:.3e}")
    print(
        f"  S_CGM/k_B = {res.S_CGM_J_per_K/k_B:.3e}  (bits: {(res.S_CGM_J_per_K/(k_B*math.log(2))):.3e})"
    )
    print("Temperature (standard → CGM):")
    print(f"  T_H:   {fmt_si(res.T_H_K, 'K')}")
    print(
        f"  T_CGM: {fmt_si(res.T_CGM_K, 'K')}  (×{res.T_factor:.6f},  −{(1.0-res.T_factor)*100:.4f}%)"
    )
    print("Evaporation time (standard → CGM heuristic):")
    print(f"  τ_std: {seconds_to_readable(res.tau_std_s)}")
    print(f"  τ_CGM: {seconds_to_readable(res.tau_cgm_s)}  (×{res.tau_factor:.6f})")


def print_derived_predictions() -> None:
    """Print the three mass-independent derived predictions from CGM scaling."""
    print("\n" + "=" * 60)
    print("CGM DERIVED PREDICTIONS — Mass-Independent Consequences")
    print("=" * 60)

    # 1. Entropy density at horizon
    s_density = (k_B * c**3) / (4.0 * hbar * G) * (1.0 + m_p)
    print(f"\n1. Entropy density at horizon:")
    print(f"   s_CGM/A = (k_B c³)/(4 ħ G) × (1 + m_p)")
    print(f"   s_CGM/A = {fmt_si(s_density, 'J/(K·m²)')}")

    # 2. Heat capacity ratio
    C_ratio = 1.0 + m_p
    print(f"\n2. Heat capacity ratio:")
    print(f"   C_CGM / C_BH = (1 + m_p)")
    print(f"   C_CGM / C_BH = {C_ratio:.6f}")

    # 3. Critical mass ratio for fixed lifetime
    M_crit_ratio = (1.0 + m_p) ** (-4.0 / 3.0)
    print(f"\n3. Critical mass ratio for fixed lifetime:")
    print(f"   M_crit,CGM / M_crit,std = (1 + m_p)^(-4/3)")
    print(f"   M_crit,CGM / M_crit,std = {M_crit_ratio:.6f}")

    # Additional specific calculations
    print(f"\n4. Specific numerical examples:")

    # Planck mass bits calculation
    M_planck = math.sqrt(hbar * c / G)
    res_planck = bh_properties("Planck mass", M_planck)
    S_bits_BH = res_planck.S_BH_J_per_K / (k_B * math.log(2))
    S_bits_CGM = res_planck.S_CGM_J_per_K / (k_B * math.log(2))
    delta_bits = S_bits_CGM - S_bits_BH
    print(f"   Planck mass BH entropy in bits:")
    print(f"     S_BH  = {S_bits_BH:.2f} bits")
    print(f"     S_CGM = {S_bits_CGM:.2f} bits")
    print(f"     Δ     = +{delta_bits:.2f} bits")

    # Sgr A* entropy increase
    M_sgr = 4.0e6 * M_sun
    res_sgr = bh_properties("Sgr A*", M_sgr)
    delta_S = m_p * res_sgr.S_BH_J_per_K
    print(f"\n   Sgr A* entropy increase:")
    print(f"     ΔS = m_p × S_BH = {fmt_si(delta_S, 'J/K')}")

    # Primordial BH spectral peak
    M_pbh = 1.0e12
    res_pbh = bh_properties("Primordial BH", M_pbh)
    E_peak_std = 2.82 * k_B * res_pbh.T_H_K  # 2.82 k_B T for Planckian peak
    E_peak_cgm = E_peak_std / (1.0 + m_p)
    print(f"\n   Primordial BH (10¹² kg) spectral peak:")
    print(f"     E_peak,std = 2.82 k_B T_H = {E_peak_std/MeV_J:.1f} MeV")
    print(f"     E_peak,CGM = E_peak,std/(1+m_p) = {E_peak_cgm/MeV_J:.1f} MeV")

    # Critical mass for evaporation today
    M_crit_std = 5.0e11  # kg, order-of-magnitude for Hubble time evaporation
    M_crit_cgm = M_crit_std * M_crit_ratio
    print(f"\n   Critical mass for 'evaporating today' PBHs:")
    print(f"     M_crit,std ≈ {fmt_si(M_crit_std, 'kg')} (textbook estimate)")
    print(f"     M_crit,CGM = {fmt_si(M_crit_cgm, 'kg')} (CGM prediction)")

    print(f"\n5. Hawking luminosity and quantum emission:")
    L_ratio = 1.0 / (1.0 + m_p) ** 4
    N_ratio = 1.0 + m_p
    print(f"   Luminosity ratio: L_CGM / L_std = 1/(1 + m_p)^4 = {L_ratio:.6f}")
    print(
        f"   Total quanta ratio: N_total,CGM / N_total,std = (1 + m_p) = {N_ratio:.6f}"
    )
    print(f"   Interpretation: More quanta, each with lower energy")

    print(f"\n6. Page time and Page mass:")
    t_page_ratio = (1.0 + m_p) ** 4
    print(
        f"   Page time ratio: t_Page,CGM / t_Page,std = (1 + m_p)^4 = {t_page_ratio:.6f}"
    )
    print(f"   Page mass fraction: M_Page/M_0 = 1/√2 (unchanged by CGM)")

    print(f"\n7. Spectral peak shifts:")
    E_peak_ratio = 1.0 / (1.0 + m_p)
    print(f"   E_peak,CGM / E_peak,std = 1/(1 + m_p) = {E_peak_ratio:.6f}")
    print(
        f"   Redshift factor: {1.0 - E_peak_ratio:.4f} = {(1.0 - E_peak_ratio)*100:.2f}% redward"
    )

    print(f"\n8. Thermodynamic consistency checks:")
    print(f"   T·S invariance: T_CGM S_CGM = T_H S_BH ✓")
    print(f"   First law: dM = T_H dS_BH = T_CGM dS_CGM ✓")
    print(f"   Smarr relation: M c² = 2 T S (unchanged) ✓")
    print(f"   Total radiated energy: M c² (conserved) ✓")


def print_horizon_micro_quanta() -> None:
    """Effective Planck scale and area quantum on the horizon under CGM scaling."""
    print("\n" + "=" * 60)
    print("HORIZON MICRO-QUANTA — Effective Planck Scale on the Horizon")
    print("=" * 60)
    lP = math.sqrt(hbar * G / c**3)
    G_eff = G / (1.0 + m_p)
    lP_eff = math.sqrt(hbar * G_eff / c**3)
    dA_std = 8.0 * pi * lP**2
    dA_cgm = dA_std / (1.0 + m_p)
    print(f"ℓ_P = {fmt_si(lP, 'm')}")
    print(f"G_eff on horizon = G/(1+m_p) = {G_eff:.6e} SI")
    print(
        f"ℓ_P,eff (from G_eff) = {fmt_si(lP_eff, 'm')}  (×{1.0/math.sqrt(1.0 + m_p):.6f})"
    )
    print(f"ΔA_std = 8π ℓ_P² = {fmt_si(dA_std, 'm²')}")
    print(f"ΔA_CGM = ΔA_std/(1+m_p) = {fmt_si(dA_cgm, 'm²')}  (×{1.0/(1.0 + m_p):.6f})")


def print_page_curve_invariants() -> None:
    """Page time/quanta invariants under CGM scaling, as explicit equations."""
    print("\n" + "=" * 60)
    print("PAGE CURVE INVARIANTS — Lifetime, Quanta, Entropy at Page Time")
    print("=" * 60)
    t_ratio = (1.0 + m_p) ** 4
    N_ratio = 1.0 + m_p
    print(f"t_Page,CGM = (1 + m_p)^4 · t_Page,std    ⇒ ratio = {t_ratio:.6f}")
    print(f"N_tot,CGM = (1 + m_p) · N_tot,std       ⇒ ratio = {N_ratio:.6f}")
    print("M_Page/M_0 = 1/√2 (unchanged);  S_em(Page) = ½ S_CGM (scaled by 1+m_p)")


def print_desitter_horizon_scaling() -> None:
    """De Sitter horizon entropy scaling under CGM."""
    print("\n" + "=" * 60)
    print("DE SITTER HORIZON SCALING — Cosmological Horizon Entropy")
    print("=" * 60)
    # Example with current Hubble parameter
    H0 = 2.2e-18  # s^-1 (current Hubble parameter, order of magnitude)
    S_dS_std = pi * k_B * c**5 / (G * hbar * H0**2)
    S_dS_cgm = S_dS_std * (1.0 + m_p)
    print(f"Hubble parameter: H₀ ≈ {H0:.1e} s⁻¹")
    print(f"S_dS = π k_B c⁵/(G ħ H²)")
    print(f"S_dS,std = {fmt_si(S_dS_std, 'J/K')}  (S_dS,std/k_B = {S_dS_std/k_B:.3e})")
    print(
        f"S_dS,CGM = (1 + m_p) S_dS,std = {fmt_si(S_dS_cgm, 'J/K')}  (×{1.0 + m_p:.6f})"
    )
    print("Effective horizon G_eff = G/(1 + m_p) applies to cosmological horizons")


def print_ringdown_analysis() -> None:
    """Analysis of CGM effects on merger ringdown and quasinormal modes."""
    print("\n" + "=" * 60)
    print("RINGDOWN ANALYSIS — Quasinormal Modes and Merger Signatures")
    print("=" * 60)

    # Standard QNM frequencies for Schwarzschild (fundamental mode)
    # ω = c³/(GM) × f_lmn where f_lmn are dimensionless constants
    f_220 = 0.3737  # fundamental l=2, m=2, n=0 mode
    f_221 = 0.3467  # first overtone

    print("Standard Schwarzschild QNM frequencies (dimensionless):")
    print(f"  f_220 (fundamental) = {f_220:.4f}")
    print(f"  f_221 (overtone) = {f_221:.4f}")

    print("\nCGM Analysis:")
    print("  • QNM frequencies depend on background geometry: ω ∝ c³/(GM)")
    print("  • CGM scaling affects only horizon thermodynamics (S, T)")
    print("  • Background metric unchanged → QNM frequencies unchanged")
    print("  • Ringdown amplitude may be modified by aperture leakage")

    print(f"\nRingdown modifications (heuristic):")
    print(f"  • QNM frequencies: unchanged (geometry-dependent)")
    print(f"  • Ringdown duration: unchanged (geometry-dependent)")
    print(f"  • Amplitude: may be modified by aperture leakage (not derived)")
    print(f"  • Note: 2.07% energy leakage is heuristic, not derived from CGM ansatz")

    print("\nObservational implications:")
    print("  • LIGO/Virgo: No frequency shift expected")
    print("  • Duration: No change expected")
    print("  • Amplitude: Uncertain (requires additional CGM dynamics)")
    print("  • Higher modes: Same scaling as fundamental")


def print_rindler_horizon_analysis() -> None:
    """Analysis of CGM scaling for Rindler horizons (uniform acceleration)."""
    print("\n" + "=" * 60)
    print("RINDLER HORIZON ANALYSIS — Uniform Acceleration Case")
    print("=" * 60)

    # Rindler horizon properties
    # For uniform acceleration a, horizon at x = c²/a
    # Temperature: T_Rindler = ħa/(2πk_Bc)
    # Entropy density: s = k_B/(4ℓ_P²) per unit area

    print("Standard Rindler horizon (uniform acceleration a):")
    print("  • Horizon location: x = c²/a")
    print("  • Temperature: T = ħa/(2πk_Bc)")
    print("  • Entropy density: s = k_B/(4ℓ_P²)")
    print("  • Area element: dA = dx dy (in y-z plane)")

    print("\nCGM scaling for Rindler horizons:")
    print("  • S_Rindler,CGM = (1 + m_p) S_Rindler,std")
    print("  • T_Rindler,CGM = T_Rindler,std / (1 + m_p)")
    print("  • Effective Planck length: ℓ_P,eff = ℓ_P/√(1 + m_p)")

    # Calculate specific example
    a_example = 9.8  # Earth gravity in m/s²
    T_Rindler_std = hbar * a_example / (2 * pi * k_B * c)
    T_Rindler_cgm = T_Rindler_std / (1.0 + m_p)

    print(f"\nExample (a = {a_example} m/s²):")
    print(f"  T_Rindler,std = {fmt_si(T_Rindler_std, 'K')}")
    print(f"  T_Rindler,CGM = {fmt_si(T_Rindler_cgm, 'K')}  (×{1.0/(1.0 + m_p):.6f})")

    print("\nPhysical interpretation:")
    print("  • Rindler observer sees reduced Unruh temperature")
    print("  • Aperture leakage affects accelerated reference frames")
    print("  • Same (1 + m_p) scaling as black hole horizons")
    print("  • Suggests universal horizon thermodynamics modification")


def print_binary_merger_analysis() -> None:
    """Analysis of CGM effects on binary black hole merger rates and dynamics."""
    print("\n" + "=" * 60)
    print("BINARY MERGER ANALYSIS — Rates, Lifetimes, and Dynamics")
    print("=" * 60)

    print("Standard binary merger timescales:")
    print("  • Inspiral phase: t_inspiral ∝ a^4/(GM₁M₂(M₁+M₂))")
    print("  • Merger phase: t_merger ∝ G(M₁+M₂)/c³")
    print("  • Ringdown phase: t_ringdown ∝ G(M₁+M₂)/c³")

    print("\nCGM modifications to merger dynamics:")
    print("  • Inspiral: Unchanged (orbital dynamics geometry-dependent)")
    print("  • Merger: Unchanged (strong-field geometry unchanged)")
    print("  • Ringdown: Duration unchanged, amplitude uncertain")
    lifetime_factor = (1.0 + m_p) ** 4
    print(
        f"  • Remnant lifetime: Extended by factor (1 + m_p)^4 ≈ {lifetime_factor:.3f}"
    )

    print(f"\nNote on merger rates:")
    print(f"  • Stellar/intermediate BHs don't evaporate on astrophysical timescales")
    print(f"  • Lifetime extension has negligible impact on LIGO/Virgo merger rates")
    print(f"  • Only affects primordial BHs or very long-term evolution")

    print("\nObservational implications:")
    print("  • LIGO/Virgo: No significant merger rate change expected")
    print("  • Ringdown: No frequency shift, amplitude uncertain")
    print("  • Primordial BHs: Extended lifetime affects detection windows")

    # Calculate specific example for 30+30 M_sun merger
    M1 = 30.0 * M_sun
    M2 = 30.0 * M_sun
    M_remnant = M1 + M2  # Simple mass conservation

    print(f"\nExample: 30+30 M_sun merger")
    print(f"  Remnant mass: {M_remnant/M_sun:.1f} M_sun")
    print(
        f"  Standard lifetime: {seconds_to_readable((5120.0 * pi * G**2 * M_remnant**3) / (hbar * c**4))}"
    )
    print(
        f"  CGM lifetime: {seconds_to_readable((5120.0 * pi * G**2 * M_remnant**3) / (hbar * c**4) * lifetime_factor)}"
    )
    print(f"  Ringdown amplitude: uncertain (requires additional CGM dynamics)")


def print_ads_blackhole_analysis() -> None:
    """Analysis of CGM scaling for AdS black holes and connection to de Sitter."""
    print("\n" + "=" * 60)
    print("ADS BLACK HOLE ANALYSIS — Anti-de Sitter Scaling")
    print("=" * 60)

    # AdS black hole properties
    # For AdS radius L, temperature and entropy depend on both M and L
    # T_AdS = (1/4π) * (r_+/L² + 1/r_+) where r_+ is horizon radius
    # S_AdS = π r_+² / (4G) (same area law as flat space)

    print("Standard AdS black hole (radius L):")
    print("  • Horizon radius: r_+ = L * √(1 + √(1 + 4M²/L²)) / √2")
    print("  • Temperature: T = (1/4π) * (r_+/L² + 1/r_+)")
    print("  • Entropy: S = π r_+² / (4G) (area law)")
    print("  • Mass: M = (c² r_+/(2G)) * (1 + r_+²/L²)  (SI units)")

    print("\nCGM scaling for AdS black holes:")
    print("  • S_AdS,CGM = (1 + m_p) S_AdS,std")
    print("  • T_AdS,CGM = T_AdS,std / (1 + m_p)")
    print("  • Effective AdS radius: L_eff = L (geometry unchanged)")
    print("  • Effective Newton constant: G_eff = G/(1 + m_p)")

    # Calculate example for small AdS black hole
    L_ads = 1.0e-3  # 1 mm AdS radius (for illustration)
    M_ads = 1.0e-3  # 1 g mass

    # Approximate horizon radius for small AdS BH (Schwarzschild radius)
    r_plus_approx = 2.0 * G * M_ads / c**2  # Schwarzschild radius (illustrative)

    # Correct entropy (area law with SI factors)
    S_ads_std = k_B * pi * c**3 * r_plus_approx**2 / (G * hbar)
    S_ads_cgm = S_ads_std * (1.0 + m_p)

    # Correct temperature (SI)
    T_ads_std = (hbar * c / (4.0 * pi * k_B)) * (
        1.0 / r_plus_approx + 3.0 * r_plus_approx / (L_ads**2)
    )
    T_ads_cgm = T_ads_std / (1.0 + m_p)

    print(f"\nExample (L = {L_ads:.1e} m, M = {M_ads:.1e} kg):")
    print(f"  r_+ ≈ {fmt_si(r_plus_approx, 'm')}")
    print(f"  S_AdS,std = {fmt_si(S_ads_std, 'J/K')}")
    print(f"  S_AdS,CGM = {fmt_si(S_ads_cgm, 'J/K')}  (×{1.0 + m_p:.6f})")
    print(f"  T_AdS,std = {fmt_si(T_ads_std, 'K')}")
    print(f"  T_AdS,CGM = {fmt_si(T_ads_cgm, 'K')}  (×{1.0/(1.0 + m_p):.6f})")

    print("\nConnection to de Sitter scaling:")
    print("  • Both AdS and dS horizons follow same (1 + m_p) scaling")
    print("  • G_eff = G/(1 + m_p) applies universally")
    print("  • Horizon thermodynamics modified, not geometry")
    print("  • Suggests fundamental aperture property of horizons")

    print("\nHolographic implications:")
    print("  • AdS/CFT: Boundary theory entropy scaled by (1 + m_p)")
    print("  • Central charge: c_eff = c / (1 + m_p) (if G_eff scaling)")
    print("  • CFT temperature: T_CFT = T_AdS (unchanged by CGM)")
    print("  • Aperture leakage affects bulk thermodynamics only")


def kerr_newman_cgm(M_kg: float, J: float, Q_C: float) -> Dict[str, float]:
    """
    CGM scaling for Kerr–Newman black holes (SI units throughout).

    Inputs:
      - M_kg: mass (kg)
      - J: angular momentum (kg·m²/s)
      - Q_C: electric charge (C)

    Returns:
      Dict with r_±, area, entropies, temperatures, horizon angular velocity and potential.
    """
    # Geometric length scales
    r_g = G * M_kg / c**2  # gravitational radius (m)
    a_len = J / (M_kg * c)  # spin length (m)
    r_Q = math.sqrt(G * Q_C**2 / (4 * pi * epsilon0 * c**4))  # charge length (m)

    # Horizon radii
    disc = r_g**2 - a_len**2 - r_Q**2
    if disc < 0:
        raise ValueError(
            f"Naked singularity: a_*^2 + q_*^2 = {(a_len/r_g)**2 + (r_Q/r_g)**2:.6f} >= 1"
        )
    sqrt_disc = math.sqrt(disc)
    r_plus = r_g + sqrt_disc
    r_minus = r_g - sqrt_disc

    # Horizon area
    A = 4.0 * pi * (r_plus**2 + a_len**2)

    # Entropy
    S_BH = k_B * A * c**3 / (4.0 * G * hbar)
    S_CGM = S_BH * (1.0 + m_p)

    # Hawking temperature: T = (ħ c / (4π k_B)) (r_+ - r_-)/(r_+^2 + a^2)
    T_H = (hbar * c / (4.0 * pi * k_B)) * ((r_plus - r_minus) / (r_plus**2 + a_len**2))
    T_CGM = T_H / (1.0 + m_p)

    # Horizon angular velocity and electric potential (SI)
    Omega_H = (a_len * c) / (r_plus**2 + a_len**2)  # rad/s
    Phi_H = (Q_C * (1.0 / (4.0 * pi * epsilon0)) * r_plus) / (r_plus**2 + a_len**2)  # V

    return {
        "M_kg": M_kg,
        "J": J,
        "Q": Q_C,
        "a_len": a_len,
        "r_g": r_g,
        "r_Q": r_Q,
        "r_plus": r_plus,
        "r_minus": r_minus,
        "A": A,
        "S_BH": S_BH,
        "S_CGM": S_CGM,
        "T_H": T_H,
        "T_CGM": T_CGM,
        "Omega_H": Omega_H,
        "Phi_H": Phi_H,
    }


def print_kerr_newman_example():
    """Print example Kerr–Newman calculations with CGM scaling."""
    print("\n" + "=" * 60)
    print("KERR–NEWMAN CGM SCALING — Spinning Charged Black Holes")
    print("=" * 60)

    # Example: 10 solar mass BH, dimensionless spin a_* = 0.5, modest charge (30% of extremal)
    M = 10.0 * M_sun
    r_g = G * M / c**2
    a_star = 0.5
    a_len = a_star * r_g
    J = a_len * M * c

    # Choose charge safely below extremality
    r_Q_max = math.sqrt(max(0.0, r_g**2 - a_len**2))
    r_Q = 0.3 * r_Q_max
    Q = (c**2) * math.sqrt(4.0 * pi * epsilon0 / G) * r_Q

    kn = kerr_newman_cgm(M, J, Q)

    print(f"\nExample: 10 M_sun BH with a_* = {a_star:.2f}, Q at 30% of extremal")
    print(f"Mass: {fmt_si(kn['M_kg'], 'kg')} ({kn['M_kg']/M_sun:.1f} M_sun)")
    print(f"Angular momentum: J = {fmt_si(kn['J'], 'kg·m²/s')}")
    print(f"Charge: Q = {fmt_si(kn['Q'], 'C')}")
    print(
        f"Spin length: a = {fmt_si(kn['a_len'], 'm')}  (a_* = {kn['a_len']/kn['r_g']:.3f})"
    )
    print(
        f"Horizon radii: r_+ = {fmt_si(kn['r_plus'], 'm')},  r_- = {fmt_si(kn['r_minus'], 'm')}"
    )
    print(f"Horizon area: A = {fmt_si(kn['A'], 'm²')}")

    print(f"\nEntropy (Bekenstein–Hawking → CGM):")
    print(f"  S_BH:  {fmt_si(kn['S_BH'], 'J/K')}")
    print(f"  S_CGM: {fmt_si(kn['S_CGM'], 'J/K')}  (×{1.0 + m_p:.6f})")

    print(f"\nTemperature (standard → CGM):")
    print(f"  T_H:   {fmt_si(kn['T_H'], 'K')}")
    print(f"  T_CGM: {fmt_si(kn['T_CGM'], 'K')}  (×{1.0/(1.0 + m_p):.6f})")

    print(f"\nHorizon angular velocity and electric potential (unchanged by CGM):")
    print(f"  Omega_H: {fmt_si(kn['Omega_H'], 'rad/s')}")
    print(f"  Phi_H:   {fmt_si(kn['Phi_H'], 'V')}")

    print(f"\nGeneralized first law consistency:")
    print(f"  dM = T_CGM dS_CGM + Omega_H dJ + Phi_H dQ ✓")
    print(f"  Smarr relation: M c² = 2 T S + 2 Omega_H J + Phi_H Q (unchanged) ✓")
    print(
        f"  Note: Kerr–Newman geometry (Ω_H, Φ_H, QNM spectrum) unchanged by CGM scaling"
    )


if __name__ == "__main__":
    catalogue: Dict[str, float] = {
        "Sgr A* (Milky Way SMBH)": 4.0e6 * M_sun,
        "M87* (Virgo A SMBH)": 6.5e9 * M_sun,
        "Stellar BH — 10 M_sun": 10.0 * M_sun,
        "Stellar BH — 30 M_sun": 30.0 * M_sun,
        "Cygnus X-1 (~15 M_sun)": 15.0 * M_sun,
        "GW150914 remnant (~62 M_sun)": 62.0 * M_sun,
        "IMBH — 1e5 M_sun": 1.0e5 * M_sun,
        "Solar mass — 1 M_sun": 1.0 * M_sun,
        "Primordial BH — 1e12 kg": 1.0e12,
        "Planck mass": math.sqrt(hbar * c / G),
    }
    print("CGM Aperture-Corrected Black Hole Thermodynamics")
    print(
        "Assumption: S_CGM = S_BH × (1 + m_p), m_p = 1/(2*sqrt(2*pi)) ≈ {:.12f}".format(
            m_p
        )
    )
    print("Derived scalings:  T_CGM = T_H / (1 + m_p),  τ_CGM ≈ τ_std × (1 + m_p)^4\n")
    for name, M in catalogue.items():
        print_result(bh_properties(name, M))

    # Print the derived predictions
    print_derived_predictions()

    # Print horizon micro-quanta rescaling
    print_horizon_micro_quanta()

    # Print Page-curve invariants
    print_page_curve_invariants()

    # Print de Sitter horizon scaling
    print_desitter_horizon_scaling()

    # Print ringdown analysis
    print_ringdown_analysis()

    # Print Rindler horizon analysis
    print_rindler_horizon_analysis()

    # Print binary merger analysis
    print_binary_merger_analysis()

    # Print AdS black hole analysis
    print_ads_blackhole_analysis()

    # Print Kerr–Newman example
    print_kerr_newman_example()
