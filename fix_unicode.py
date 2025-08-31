#!/usr/bin/env python3
"""
Script to replace all Greek letters and Unicode characters with ASCII equivalents
in cgm_light_analysis.py to fix the encoding issues.
"""

import re

# Define the replacements
replacements = {
    # Greek letters
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "π": "pi",
    "φ": "phi",
    "θ": "theta",
    "σ": "sigma",
    # Subscripts
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    # Superscripts
    "²": "^2",
    "³": "^3",
    # Other Unicode
    "ℓ": "ell",
    "Δ": "Delta",
    "√": "sqrt",
    "×": "×",  # Keep this as is
    "→": "->",
    "†": "dagger",
    # Special characters
    "❌": "ERROR",
    "✅": "PASS",
    "✓": "OK",
}


def fix_unicode_in_file(filename):
    """Replace all Unicode characters in a file with ASCII equivalents."""
    print(f"Fixing Unicode characters in {filename}...")

    # Read the file
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # Count original Unicode characters
    original_count = sum(content.count(char) for char in replacements.keys())
    print(f"Found {original_count} Unicode characters to replace")

    # Apply all replacements
    for unicode_char, ascii_replacement in replacements.items():
        if unicode_char in content:
            count = content.count(unicode_char)
            content = content.replace(unicode_char, ascii_replacement)
            print(
                f"  Replaced {count} instances of '{unicode_char}' with '{ascii_replacement}'"
            )

    # Write back to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Unicode fix complete for {filename}")


if __name__ == "__main__":
    fix_unicode_in_file("cgm_light_analysis.py")
