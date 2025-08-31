#!/usr/bin/env python3
"""
CGM Anchor Constants with Provenance Tracking

This module provides the non-circular anchors required for CGM derivation.
Each anchor carries a provenance tag indicating its source and reliability.

PROVENANCE TAGS:
- "provisional": Placeholder values for testing (forbid SI mapping)
- "external": Measured from external sources without c assumptions
- "derived": Computed from CGM theory (may have c dependencies)

SI MAPPING POLICY:
- Provisional anchors forbid SI mapping
- Only external, non-c anchors allow dimensionful calibration
- Derived anchors may be used for internal consistency checks

Author: CGM Research Team
Date: 2024
"""

from typing import Dict, Any, Literal
from dataclasses import dataclass

ProvenanceType = Literal["provisional", "external", "derived"]


@dataclass
class CGMAnchor:
    """A CGM anchor constant with provenance tracking."""

    value: float
    provenance: ProvenanceType
    description: str
    source: str
    uncertainty: float = 0.0

    def allows_si_mapping(self) -> bool:
        """Check if this anchor allows SI unit mapping."""
        return self.provenance == "external"

    def __str__(self) -> str:
        return f"{self.value:.6f} ({self.provenance}) - {self.description}"


class CGMAnchors:
    """
    Centralized CGM anchor constants with provenance tracking.

    This class ensures all anchors have explicit provenance and
    enforces the SI mapping policy.
    """

    def __init__(self):
        """Initialize CGM anchors with provenance tracking."""

        # CMB Angular Power Spectrum Ratios (completely dimensionless)
        self.cmb_angular_ratios = {
            "C_2": CGMAnchor(
                value=1000.0,  # μK² (approximate)
                provenance="provisional",
                description="CMB multipole power at ell=2",
                source="Placeholder - needs actual Planck data",
                uncertainty=100.0,
            ),
            "C_37": CGMAnchor(
                value=100.0,  # μK² (your enhancement)
                provenance="provisional",
                description="CMB multipole power at ell=37",
                source="Placeholder - needs actual Planck data",
                uncertainty=10.0,
            ),
        }

        # Supernova Rise/Decline Time Ratios (pure counting of days)
        self.supernova_timing = {
            "rise_time_days": CGMAnchor(
                value=19.0,
                provenance="external",
                description="Type Ia supernova rise time to peak",
                source="Observational data - pure counting",
                uncertainty=2.0,
            ),
            "decline_time_days": CGMAnchor(
                value=53.0,
                provenance="external",
                description="Type Ia supernova decline to half brightness",
                source="Observational data - pure counting",
                uncertainty=5.0,
            ),
        }

        # Atomic Hyperfine Frequency Ratios (pure frequency counting)
        self.atomic_frequencies = {
            "f_Cs": CGMAnchor(
                value=9192631770.0,  # Hz
                provenance="external",
                description="Cesium hyperfine frequency",
                source="Frequency counting - no c assumptions",
                uncertainty=0.0,  # Exact by definition
            ),
            "f_Rb": CGMAnchor(
                value=6834682610.0,  # Hz
                provenance="external",
                description="Rubidium hyperfine frequency",
                source="Frequency counting - no c assumptions",
                uncertainty=0.0,  # Exact by definition
            ),
        }

        # P2/C4 Harmonic Ratios (from CMB analysis)
        self.p2_c4_ratios = {
            "observed": CGMAnchor(
                value=8.089,
                provenance="external",
                description="P2/C4 ratio observed from CMB data",
                source="FIRAS CMB analysis - geometric structure",
                uncertainty=0.1,
            ),
            "expected": CGMAnchor(
                value=12.0,
                provenance="derived",
                description="P2/C4 ratio expected from theory",
                source="CGM geometric theory",
                uncertainty=0.5,
            ),
        }

        # Recursive Structure Constants
        self.recursive_constants = {
            "ell_star": CGMAnchor(
                value=37.0,
                provenance="external",
                description="Multipole enhancement pattern",
                source="CMB angular power spectrum analysis",
                uncertainty=0.0,  # Exact pattern
            ),
            "N_star": CGMAnchor(
                value=37.0,
                provenance="derived",
                description="Recursive ladder index",
                source="CGM recursive structure theory",
                uncertainty=0.0,  # Exact from theory
            ),
        }

    def get_anchor(self, category: str, key: str) -> CGMAnchor:
        """Get a specific anchor by category and key."""
        anchor_dict = getattr(self, category, {})
        if key not in anchor_dict:
            raise KeyError(f"Anchor {category}.{key} not found")
        return anchor_dict[key]

    def get_value(self, category: str, key: str) -> float:
        """Get the value of a specific anchor."""
        return self.get_anchor(category, key).value

    def check_si_mapping_allowed(self) -> bool:
        """
        Check if SI mapping is allowed based on anchor provenance.

        Returns:
            True if all critical anchors are external (non-c)
        """
        critical_anchors = [
            self.cmb_angular_ratios["C_2"],
            self.cmb_angular_ratios["C_37"],
            self.supernova_timing["rise_time_days"],
            self.supernova_timing["decline_time_days"],
            self.atomic_frequencies["f_Cs"],
            self.atomic_frequencies["f_Rb"],
        ]

        return all(anchor.allows_si_mapping() for anchor in critical_anchors)

    def get_provenance_summary(self) -> Dict[str, Any]:
        """Get a summary of all anchor provenances."""
        summary = {}

        for category_name in [
            "cmb_angular_ratios",
            "supernova_timing",
            "atomic_frequencies",
            "p2_c4_ratios",
            "recursive_constants",
        ]:
            category = getattr(self, category_name)
            summary[category_name] = {
                key: {
                    "value": anchor.value,
                    "provenance": anchor.provenance,
                    "description": anchor.description,
                    "allows_si": anchor.allows_si_mapping(),
                }
                for key, anchor in category.items()
            }

        summary["si_mapping_allowed"] = self.check_si_mapping_allowed()
        return summary

    def print_provenance_report(self):
        """Print a detailed provenance report for all anchors."""
        print("\n" + "=" * 60)
        print("CGM ANCHOR PROVENANCE REPORT")
        print("=" * 60)

        for category_name in [
            "cmb_angular_ratios",
            "supernova_timing",
            "atomic_frequencies",
            "p2_c4_ratios",
            "recursive_constants",
        ]:
            category = getattr(self, category_name)
            print(f"\n{category_name.replace('_', ' ').title()}:")

            for key, anchor in category.items():
                status = "✓ SI" if anchor.allows_si_mapping() else "✗ No SI"
                print(
                    f"  {key:15}: {anchor.value:10.6f} [{anchor.provenance:10}] {status}"
                )
                print(f"           {anchor.description}")
                print(f"           Source: {anchor.source}")

        print(f"\nSI Mapping Policy:")
        print(
            f"  All critical anchors external: {'✓ ALLOWED' if self.check_si_mapping_allowed() else '✗ FORBIDDEN'}"
        )
        print(
            f"  Current status: {'PROVISIONAL ANCHORS FORBID SI MAPPING' if not self.check_si_mapping_allowed() else 'EXTERNAL ANCHORS ALLOW SI MAPPING'}"
        )

        if not self.check_si_mapping_allowed():
            print(
                f"  Action required: Replace provisional anchors with external measurements"
            )
            print(f"  Critical anchors needing external data:")
            for key, anchor in self.cmb_angular_ratios.items():
                if anchor.provenance == "provisional":
                    print(f"    - {key}: {anchor.description}")


# Global instance for easy access
cgm_anchors = CGMAnchors()

if __name__ == "__main__":
    # Test the anchor system
    cgm_anchors.print_provenance_report()
