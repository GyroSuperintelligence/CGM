#!/usr/bin/env python3
import numpy as np

# Test the toroidal cavity math
alpha = np.pi / 2  # CS chirality seed
m_p = 1.0 / (2.0 * np.sqrt(2.0 * np.pi))  # BU aperture

print(f"alpha: {alpha}")
print(f"m_p: {m_p}")

# The cavity has two hemispheres (like Earth's day/night)
hemisphere_1_phase = np.exp(1j * alpha)  # CS chirality
hemisphere_2_phase = np.exp(-1j * alpha)  # Conjugate

print(f"hemisphere_1_phase: {hemisphere_1_phase}")
print(f"hemisphere_2_phase: {hemisphere_2_phase}")

# Interference creates standing waves
interference = hemisphere_1_phase + hemisphere_2_phase

print(f"interference: {interference}")
print(f"abs(interference): {abs(interference)}")

# The m_p aperture is the "leakage" that prevents total confinement
leakage_factor = m_p  # ~20% escape

print(f"leakage_factor: {leakage_factor}")

# Leakage is independent of standing-wave nodes - m_p provides escape
# Interference modulates but never kills transmission completely
effective_transmission = max(leakage_factor * abs(interference), leakage_factor)

print(f"leakage_factor * abs(interference): {leakage_factor * abs(interference)}")
print(f"effective_transmission: {effective_transmission}")
print(f"effective_transmission (6 decimal places): {effective_transmission:.6f}")
