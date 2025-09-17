#!/usr/bin/env python3
"""
CGM Energy Scale Calculations - Stage Thresholds, Actions, and Energies

Calculate the fundamental energy scales for CS, UNA, ONA, BU stages
based on CGM theory with aperture parameter m_p.

Based on the clean hierarchy:
- CS: Top scale (ToE/Planck sector) 
- UNA + ONA: GUT sector (parallel constraints)
- BU: Dual/IR endpoint (fixed point)
"""

import numpy as np
import math
from typing import Dict, Tuple


def calculate_stage_thresholds() -> Dict[str, float]:
    """
    Calculate the stage thresholds as specified:
    - CS: s_p = pi/2
    - UNA: u_p = cos(pi/4) = 1/sqrt(2)  
    - ONA: o_p = pi/4
    - BU: m_p = 1/(2*sqrt(2*pi)) (aperture/closure parameter)
    """
    s_p = math.pi / 2
    u_p = math.cos(math.pi / 4)  # = 1/sqrt(2)
    o_p = math.pi / 4
    m_p = 1 / (2 * math.sqrt(2 * math.pi))
    
    return {
        'CS': s_p,
        'UNA': u_p, 
        'ONA': o_p,
        'BU': m_p
    }


def calculate_stage_actions(thresholds: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate stage actions using the simple map:
    - S_CS = s_p / m_p
    - S_UNA = u_p / m_p  
    - S_ONA = o_p / m_p
    - S_BU = m_p (fixed point)
    """
    m_p = thresholds['BU']
    
    actions = {
        'CS': thresholds['CS'] / m_p,
        'UNA': thresholds['UNA'] / m_p,
        'ONA': thresholds['ONA'] / m_p,
        'BU': m_p
    }
    
    return actions


def calculate_gut_action(actions: Dict[str, float], eta: float = 1.0) -> float:
    """
    Calculate GUT action as parallel constraints (UNA + ONA + CS memory):
    1/S_GUT = 1/S_UNA + 1/S_ONA + eta/S_CS
    
    This models UNA (rotations) and ONA (translations) as complementary
    constraints on the same helical path, with optional CS memory weight.
    
    Args:
        actions: Dictionary of stage actions
        eta: CS memory weight (default 1.0)
    """
    s_gut_inv = (1/actions['UNA'] + 1/actions['ONA'] + eta/actions['CS'])
    s_gut = 1 / s_gut_inv
    
    return s_gut


def calculate_duality_map(actions: Dict[str, float], m_p: float) -> Dict[str, float]:
    """
    Calculate duality map around BU fixed point:
    D(S) = m_p^2 / S
    
    BU is a fixed point: D(m_p) = m_p
    """
    m_p_squared = m_p ** 2
    
    duality = {}
    for stage, action in actions.items():
        if stage == 'BU':
            duality[stage] = m_p  # Fixed point
        else:
            duality[stage] = m_p_squared / action
    
    return duality


def calculate_energies(actions: Dict[str, float], s_gut: float, scale_A: float = 1.0) -> Dict[str, float]:
    """
    Calculate energies using single global constant A:
    E_stage = A x S_stage
    """
    energies = {}
    for stage, action in actions.items():
        energies[stage] = scale_A * action
    
    # Add GUT energy
    energies['GUT'] = scale_A * s_gut
    
    return energies


def calculate_energy_ratios(energies: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate energy ratios relative to CS (anchor-free):
    """
    e_cs = energies['CS']
    
    ratios = {}
    for stage, energy in energies.items():
        if stage != 'CS':
            ratios[f'{stage}/CS'] = energy / e_cs
    
    return ratios


def anchor_by(stage: str, e_value: float, actions: Dict[str, float], s_gut: float) -> Tuple[float, Dict[str, float]]:
    """
    Anchor energies by setting a specific stage to a given energy value.
    
    Args:
        stage: Stage to anchor ('CS', 'UNA', 'ONA', 'BU', 'GUT')
        e_value: Energy value in GeV
        actions: Dictionary of stage actions
        s_gut: GUT action value
        
    Returns:
        Tuple of (scale_A, uv_energies)
    """
    if stage == 'GUT':
        scale_A = e_value / s_gut
    else:
        scale_A = e_value / actions[stage]
    
    # Calculate UV energies with this scale
    uv_energies = {}
    for s, action in actions.items():
        uv_energies[s] = scale_A * action
    uv_energies['GUT'] = scale_A * s_gut
    
    return scale_A, uv_energies


def bu_dual_project(uv_energies: Dict[str, float], e_ew: float = 246.0) -> Dict[str, float]:
    """
    Project UV energies to IR energies using BU-centered optical conjugacy.
    
    Uses the optical invariant: E_i^UV x E_i^IR = (E_CS x E_EW)/(4*pi^2)
    This represents one system with two conjugate foci (UV at CS, IR at BU).
    
    Args:
        uv_energies: Dictionary of UV energies
        e_ew: Observed EW energy in GeV (default 246 GeV)
        
    Returns:
        Dictionary of IR energies
    """
    e_cs_uv = uv_energies['CS']
    
    # Optical invariant constant
    C = (e_cs_uv * e_ew) / (4 * math.pi**2)
    
    # IR energies from conjugacy relation
    ir_energies = {}
    for stage, e_uv in uv_energies.items():
        ir_energies[stage] = C / e_uv
    
    return ir_energies


def bu_dual_project_geometry(actions: Dict[str, float], e_ew: float = 246.0) -> Dict[str, float]:
    """
    Alternative IR calculation using pure geometry + EW (no UV anchor needed).
    
    Shows UNA/ONA "optics" directly: E_i^IR = E_EW x S_CS/(4*pi^2 x S_i)
    
    Args:
        actions: Dictionary of stage actions
        e_ew: Observed EW energy in GeV (default 246 GeV)
        
    Returns:
        Dictionary of IR energies
    """
    s_cs = actions['CS']
    
    ir_energies = {}
    for stage, s_i in actions.items():
        ir_energies[stage] = e_ew * (s_cs / (4 * math.pi**2 * s_i))
    
    return ir_energies


def calculate_optical_invariant(uv_energies: Dict[str, float], ir_energies: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate the optical invariant E_i^UV x E_i^IR for each stage.
    
    The invariant should be constant: (E_CS x E_EW)/(4*pi^2)
    """
    invariant = {}
    for stage in uv_energies:
        if stage in ir_energies:
            invariant[stage] = uv_energies[stage] * ir_energies[stage]
    
    return invariant


def calculate_magnification_swap(uv_energies: Dict[str, float], ir_energies: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate magnification swap ratios showing dual effect.
    
    E_UNA^IR/E_ONA^IR = E_ONA^UV/E_UNA^UV = S_ONA/S_UNA
    """
    swap_ratios = {}
    
    # UV ratio
    if 'UNA' in uv_energies and 'ONA' in uv_energies:
        swap_ratios['UV_ONA/UNA'] = uv_energies['ONA'] / uv_energies['UNA']
    
    # IR ratio  
    if 'UNA' in ir_energies and 'ONA' in ir_energies:
        swap_ratios['IR_UNA/ONA'] = ir_energies['UNA'] / ir_energies['ONA']
    
    # They should be equal (magnification swap)
    if 'UV_ONA/UNA' in swap_ratios and 'IR_UNA/ONA' in swap_ratios:
        swap_ratios['swap_verified'] = abs(swap_ratios['UV_ONA/UNA'] - swap_ratios['IR_UNA/ONA']) < 1e-10
    
    return swap_ratios


def main():
    """Main calculation and display function"""
    print("CGM Energy Scale Analysis - BU-Centered Duality")
    print("=" * 60)
    
    # 1) Calculate stage thresholds
    print("\n1. Stage Thresholds:")
    thresholds = calculate_stage_thresholds()
    
    # Define the mathematical expressions for each threshold
    threshold_expressions = {
        'CS': 'pi/2',
        'UNA': 'cos(pi/4) = 1/sqrt(2)',
        'ONA': 'pi/4', 
        'BU': '1/(2*sqrt(2*pi))'
    }
    
    for stage, value in thresholds.items():
        expr = threshold_expressions[stage]
        print(f"   {stage:4s} = {expr:15s} = {value:.10f}")
    
    # 2) Calculate stage actions  
    print("\n2. Stage Actions:")
    actions = calculate_stage_actions(thresholds)
    for stage, action in actions.items():
        print(f"   S_{stage:4s}: {action:.10f}")
    
    # 3) Calculate GUT action (UNA + ONA + CS memory)
    print("\n3. GUT Action (UNA + ONA + CS memory, eta=1):")
    s_gut = calculate_gut_action(actions, eta=1.0)
    print(f"   S_GUT: {s_gut:.10f}")
    print(f"   S_GUT/S_CS: {s_gut/actions['CS']:.6f}")
    
    # 4) Calculate energy ratios (anchor-free)
    print("\n4. Energy Ratios (anchor-free):")
    energies_ratio = calculate_energies(actions, s_gut, scale_A=1.0)
    ratios = calculate_energy_ratios(energies_ratio)
    for ratio_name, ratio_value in ratios.items():
        print(f"   E_{ratio_name:8s}: {ratio_value:.6f}")
    
    # 5) UV Ladder (anchored at CS = Planck scale)
    print("\n5. UV Ladder (anchored at CS = Planck scale):")
    planck_gev = 1.2209e19  # GeV
    scale_A_uv, uv_energies = anchor_by('CS', planck_gev, actions, s_gut)
    
    print(f"   Scale A = {scale_A_uv:.2e} GeV")
    print("   Energies:")
    for stage, energy in uv_energies.items():
        if energy >= 1e9:
            print(f"   E_{stage:4s}: {energy:.2e} GeV = {energy/1e9:.2f} TeV")
        else:
            print(f"   E_{stage:4s}: {energy:.2e} GeV")
    
    # 6) IR Ladder (BU-dual projected to EW scale)
    print("\n6. IR Ladder (BU-dual projected to EW scale):")
    ir_energies = bu_dual_project(uv_energies, e_ew=246.0)
    
    print("   Energies (BU-centered optical conjugacy):")
    for stage, energy in ir_energies.items():
        if energy >= 1e3:
            print(f"   E_{stage:4s}: {energy:.2f} GeV")
        else:
            print(f"   E_{stage:4s}: {energy:.2e} GeV")
    
    # 6b) Alternative IR calculation (pure geometry)
    print("\n6b. IR Ladder (pure geometry + EW, no UV anchor):")
    ir_energies_geom = bu_dual_project_geometry(actions, e_ew=246.0)
    
    print("   Energies (E_i^IR = E_EW x S_CS/(4*pi^2 x S_i)):")
    for stage, energy in ir_energies_geom.items():
        if energy >= 1e3:
            print(f"   E_{stage:4s}: {energy:.2f} GeV")
        else:
            print(f"   E_{stage:4s}: {energy:.2e} GeV")
    
    # 7) Optical invariant analysis
    print("\n7. Optical Invariant Analysis:")
    optical_invariant = calculate_optical_invariant(uv_energies, ir_energies)
    expected_invariant = (uv_energies['CS'] * 246.0) / (4 * math.pi**2)
    
    print(f"   Expected invariant: (E_CS x E_EW)/(4*pi^2) = {expected_invariant:.2e} GeV^2")
    print("   Calculated invariants:")
    for stage, inv in optical_invariant.items():
        print(f"   E_{stage:4s}^UV x E_{stage:4s}^IR = {inv:.2e} GeV^2")
    
    # 8) Magnification swap analysis
    print("\n8. Magnification Swap Analysis:")
    swap_ratios = calculate_magnification_swap(uv_energies, ir_energies)
    
    print("   UNA/ONA ratios (should be equal):")
    print(f"   UV: E_ONA/E_UNA = {swap_ratios.get('UV_ONA/UNA', 'N/A'):.6f}")
    print(f"   IR: E_UNA/E_ONA = {swap_ratios.get('IR_UNA/ONA', 'N/A'):.6f}")
    print(f"   Swap verified: {swap_ratios.get('swap_verified', False)}")
    
    # Theoretical ratio from geometry
    theoretical_ratio = (thresholds['ONA'] / thresholds['UNA'])  # o_p / u_p
    print(f"   Theoretical: o_p/u_p = (pi/4)/(1/sqrt(2)) = {theoretical_ratio:.6f}")
    
    # Calculate the angle theta = arctan(S_ONA/S_UNA)
    theta_rad = math.atan(actions['ONA'] / actions['UNA'])
    theta_deg = math.degrees(theta_rad)
    print(f"   Angle theta = arctan(S_ONA/S_UNA) = {theta_rad:.6f} rad = {theta_deg:.1f} degrees")
    
    # 9) Theoretical predictions verification
    print("\n9. Theoretical Predictions (UV ratios):")
    print(f"   E_UNA/E_CS = 2/(pi*sqrt(2)) ~ {2/(math.pi * math.sqrt(2)):.6f}")
    print(f"   E_ONA/E_CS = 1/2 = {0.5:.6f}")
    print(f"   E_BU/E_CS = (2*m_p^2)/pi ~ {(2 * thresholds['BU']**2) / math.pi:.6f}")
    print(f"   E_GUT/E_CS ~ {s_gut/actions['CS']:.6f}")
    
    # 10) Optical Law and Involution Analysis
    print("\n10. Optical Law and Involution Analysis:")
    print("   Core invariant: E_i^UV x E_i^IR = (E_CS x E_EW)/(4*pi^2)")
    print("   This is an optical conjugacy in energy space (not an illusion!)")
    print("   - Involution: applying conjugacy twice returns original E")
    print("   - Fixed point: E_BU^IR = E_EW (BU is the IR focus)")
    print("   - 4*pi factor: solid-angle normalization making it look like optics")
    
    # Verify involution (apply conjugacy twice)
    print("\n   Involution verification:")
    ir_to_uv = {}
    for stage, e_ir in ir_energies.items():
        ir_to_uv[stage] = expected_invariant / e_ir
    print("   Applying IR->UV conjugacy:")
    for stage in ['CS', 'UNA', 'ONA', 'BU']:
        if stage in ir_to_uv:
            original = uv_energies[stage]
            recovered = ir_to_uv[stage]
            error = abs(original - recovered) / original
            print(f"   E_{stage:4s}: {original:.2e} -> {recovered:.2e} (error: {error:.2e})")
    
    # 11) Why Gravity Appears Weak (Geometric Interpretation)
    print("\n11. Why Gravity Appears Weak (Geometric Interpretation):")
    print("   Standard: alpha_g(E) ~ G E^2 ~ (E/E_CS)^2")
    print("   In CGM framework:")
    print("   - Solid-angle dilution: (4*pi)^(-2) appears in invariant")
    print("   - IR energies demagnified relative to UV")
    print("   - Gravity only looks weak in IR due to BU-focused conjugacy")
    
    # Calculate dimensionless gravity measures
    print("\n   Dimensionless gravity measures:")
    for stage in ['CS', 'UNA', 'ONA', 'BU']:
        if stage in uv_energies and stage in ir_energies:
            e_uv = uv_energies[stage]
            e_ir = ir_energies[stage]
            alpha_g_uv = (e_uv / uv_energies['CS'])**2
            alpha_g_ir = (e_ir / uv_energies['CS'])**2
            suppression = alpha_g_ir / alpha_g_uv
            print(f"   alpha_g^{stage:4s}: UV={alpha_g_uv:.2e}, IR={alpha_g_ir:.2e}, suppression={suppression:.2e}")
    
    # 12) Conjugate foci system summary
    print("\n12. Conjugate Foci System Summary:")
    print("   One system with two conjugate foci:")
    print("   - UV focus: CS (Planck scale)")
    print("   - IR focus: BU (EW scale)")
    print("   - Optical invariant: E_i^UV x E_i^IR = (E_CS x E_EW)/(4*pi^2)")
    print("   - No double anchoring: once CS is set, EW follows from geometry")
    print("   - 4*pi is the geometric normalizer making it look like optical conjugacy")
    
    # 13) GUT Robustness Check (eta scanning)
    print("\n13. GUT Robustness Check (eta scanning):")
    print("   Scanning CS memory weight eta in GUT calculation:")
    eta_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    for eta in eta_values:
        s_gut_eta = calculate_gut_action(actions, eta=eta)
        e_gut_uv_eta = scale_A_uv * s_gut_eta
        e_gut_ir_eta = expected_invariant / e_gut_uv_eta
        print(f"   eta={eta:3.1f}: S_GUT={s_gut_eta:.6f}, E_GUT^UV={e_gut_uv_eta:.2e} GeV, E_GUT^IR={e_gut_ir_eta:.2f} GeV")
    
    # 14) Aperture dependence analysis
    print("\n14. Aperture Dependence (m_p -> 0):")
    print("   As aperture closes (m_p -> 0):")
    print("   - UV stages (CS, UNA, ONA): action -> infinity (hard)")
    print("   - BU stage: action -> 0 (soft)")
    print("   - This captures UV<->IR duality with BU as the only dual")
    print("   - Solid-angle dilution (4*pi)^(-2) explains why gravity appears weak in IR")
    
    return {
        'thresholds': thresholds,
        'actions': actions, 
        'gut_action': s_gut,
        'uv_energies': uv_energies,
        'ir_energies': ir_energies,
        'ir_energies_geom': ir_energies_geom,
        'optical_invariant': optical_invariant,
        'swap_ratios': swap_ratios,
        'ratios': ratios,
        'scale_A_uv': scale_A_uv
    }


def neutrino_mass_analysis(uv_energies, kappa_R=1.0):
    """
    Compute light neutrino masses via type-I seesaw using CGM 48^2 quantisation.
    
    The CGM framework naturally resolves the GUT neutrino mass problem through
    48^2 quantisation of the GUT scale, giving realistic M_R ~ 10^15 GeV.

    Args:
        uv_energies: dictionary of UV energies from main()
        kappa_R: optional O(1) factor for group representation choice
    """
    # Electroweak scale
    v = 246.0  # GeV

    # CGM 48^2 quantisation (the resolution)
    M_R = kappa_R * uv_energies['GUT'] / (48**2)
    print("\n=== Neutrino Mass Analysis ===")
    print(f"Heavy Majorana scale M_R = {M_R:.2e} GeV (CGM 48^2 quantisation)")
    print("This preserves CS hiddenness - correction happens in UNA/ONA sector")

    # Calculate neutrino masses
    print(f"\nNeutrino masses (seesaw with quantised M_R):")
    yukawas = [0.1, 0.3, 1.0, 3.0]
    for y in yukawas:
        # Seesaw: m_nu = (y^2 * v^2) / M_R
        m_nu_GeV = (y**2 * v**2) / M_R
        m_nu_eV = m_nu_GeV * 1e9  # convert GeV -> eV
        print(f"y = {y:<3.1f} -> m_nu ~ {m_nu_eV:.3e} eV")
    
    print(f"\nNote: y ~ 1 gives m_nu ~ 0.06 eV, matching experimental values")


if __name__ == "__main__":
    results = main()

    # Run neutrino mass analysis (CGM 48^2 quantisation)
    neutrino_mass_analysis(results['uv_energies'], kappa_R=1.0)
