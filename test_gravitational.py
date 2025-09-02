#!/usr/bin/env python3
"""
Simple test of gravitational coupling calculation
"""

# Test values from our script
final_mean_closure = 3.141592670702
final_surplus = 1.027e-07

# Calculate gravitational coupling
final_alpha_G_hemisphere = (2.0 / final_mean_closure) * (final_surplus ** 2)

print("Gravitational Coupling Test:")
print(f"  final_mean_closure = {final_mean_closure}")
print(f"  final_surplus = {final_surplus}")
print(f"  final_surplus² = {final_surplus ** 2}")
print(f"  2.0 / final_mean_closure = {2.0 / final_mean_closure}")
print(f"  final_alpha_G_hemisphere = {final_alpha_G_hemisphere}")
print(f"  final_alpha_G_hemisphere (scientific) = {final_alpha_G_hemisphere:.3e}")
print(f"  final_alpha_G_hemisphere (fixed) = {final_alpha_G_hemisphere:.12f}")

# Test different format specifiers
print(f"\nFormat tests:")
print(f"  .3e: {final_alpha_G_hemisphere:.3e}")
print(f"  .6e: {final_alpha_G_hemisphere:.6e}")
print(f"  .12f: {final_alpha_G_hemisphere:.12f}")
print(f"  .15f: {final_alpha_G_hemisphere:.15f}")
print(f"  .20f: {final_alpha_G_hemisphere:.20f}")
