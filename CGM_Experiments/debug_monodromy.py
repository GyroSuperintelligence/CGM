#!/usr/bin/env python3
"""
Debug script for monodromy calculations
This helps identify why monodromy calculations are returning zero
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.gyrovector_ops import GyroVectorSpace, RecursivePath
from stages.ona_stage import ONAStage
import numpy as np


def debug_gyration_operations():
    """Debug basic gyration operations"""
    print("=== DEBUGGING GYRATION OPERATIONS ===")

    gyrospace = GyroVectorSpace(c=1.0)
    ona_stage = ONAStage(gyrospace)

    # Test 1: Simple gyration
    print("\n1. Testing basic gyration:")
    u = np.array([0.5, 0, 0])
    v = np.array([0, 0.5, 0])

    gyr = gyrospace.gyration(u, v)
    print(f"Input u: {u}")
    print(f"Input v: {v}")
    print(f"Gyration result: {gyr}")
    print(f"Is identity? {np.allclose(gyr, np.eye(3))}")

    # Test 2: Different vectors
    print("\n2. Testing with different vectors:")
    test_vectors = [
        np.array([0.3, 0.2, 0.1]),
        np.array([0.1, 0.3, 0.2]),
        np.array([0.2, 0.1, 0.3]),
    ]

    for i, vec1 in enumerate(test_vectors):
        for j, vec2 in enumerate(test_vectors):
            if i != j:
                gyr = gyrospace.gyration(vec1, vec2 - vec1)
                is_identity = np.allclose(gyr, np.eye(3))
                print(f"gyr({vec1}, {vec2-vec1}, {vec2}) = identity? {is_identity}")
                if not is_identity:
                    print(f"  Non-identity result: {gyr}")


def debug_monodromy_calculation():
    """Debug the monodromy calculation step by step"""
    print("\n=== DEBUGGING MONODROMY CALCULATION ===")

    gyrospace = GyroVectorSpace(c=1.0)
    ona_stage = ONAStage(gyrospace)

    # Simple triangular loop
    loop_points = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, -1, 0])]

    print("Loop points:")
    for i, point in enumerate(loop_points):
        print(f"  {i}: {point}")

    # Calculate each gyration step by step
    monodromy = np.eye(3)
    print(f"\nStarting monodromy: {monodromy}")

    for i in range(len(loop_points) - 1):
        p1 = loop_points[i]
        p2 = loop_points[i + 1]

        print(f"\nStep {i}: from {p1} to {p2}")

        # Calculate gyration
        gyr = gyrospace.gyration(p1, p2)
        print(f"  Gyration: {gyr}")

        # Check if it's identity
        is_identity = np.allclose(gyr, np.eye(3))
        print(f"  Is identity: {is_identity}")

        # Apply to monodromy
        monodromy = monodromy @ gyr
        print(f"  Updated monodromy: {monodromy}")

    # Close the loop
    p_last = loop_points[-1]
    p_first = loop_points[0]
    final_gyr = gyrospace.gyration(p_last, p_first)
    print(f"\nClosing loop from {p_last} to {p_first}")
    print(f"Final gyration: {final_gyr}")
    print(f"Is identity: {np.allclose(final_gyr, np.eye(3))}")

    monodromy = monodromy @ final_gyr
    print(f"Final monodromy: {monodromy}")

    # Calculate norm
    monodromy_norm = np.linalg.norm(monodromy - np.eye(3))
    print(f"Monodromy norm: {monodromy_norm}")


def debug_simple_case():
    """Test with a very simple case where we know the answer"""
    print("\n=== DEBUGGING SIMPLE KNOWN CASE ===")

    gyrospace = GyroVectorSpace(c=1.0)

    # Two vectors that should produce non-trivial gyration
    a = np.array([0.6, 0.0, 0.0])  # On x-axis
    b = np.array([0.0, 0.6, 0.0])  # On y-axis

    print(f"Vector a: {a}")
    print(f"Vector b: {b}")

    # These should produce a 90-degree rotation around z-axis
    gyr = gyrospace.gyration(a, b)
    print(f"gyr(a, b) = {gyr}")

    is_identity = np.allclose(gyr, np.eye(3))
    print(f"Is identity: {is_identity}")

    if not is_identity:
        print("Success! Non-trivial gyration detected.")
        # Check if it's a rotation matrix
        det = np.linalg.det(gyr)
        print(f"Determinant: {det} (should be ~1 for rotation)")

        # Check if it's orthogonal
        is_orthogonal = np.allclose(gyr @ gyr.T, np.eye(3))
        print(f"Is orthogonal: {is_orthogonal}")


def main():
    """Run all debug tests"""
    print("MONODROMY DEBUGGING SESSION")
    print("=" * 40)

    debug_gyration_operations()
    debug_monodromy_calculation()
    debug_simple_case()

    print("\n" + "=" * 40)
    print("DEBUGGING COMPLETE")


if __name__ == "__main__":
    main()
