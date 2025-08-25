#!/usr/bin/env python3
"""
Test script for True Resonance Fourier Transform
"""

import sys
import os
import numpy as np

# Import directly by appending the directory to the Python path
sys.path.append('/workspaces/quantoniumos/04_RFT_ALGORITHMS')
import true_rft_exact

# Test with small size for quick verification
print("Testing True Resonance Fourier Transform...")
transformer = true_rft_exact.TrueResonanceFourierTransform(N=16)

# Verify perfect reconstruction
results = transformer.verify_perfect_reconstruction()
print(f"\nPerfect reconstruction: {results['perfect_reconstruction']}")
print(f"Relative error: {results['relative_error']:.2e}")
print(f"Energy error: {results['energy_error']:.2e}")

# Verify non-equivalence to DFT
dft_results = transformer.prove_non_equivalence_to_dft()
print(f"\nNon-equivalent to DFT: {dft_results['non_equivalent']}")
print(f"Distinctness score: {dft_results['distinctness_score']:.2%}")

print("\nTest complete!")

# Test with small size for quick verification
print("Testing True Resonance Fourier Transform...")
transformer = TrueResonanceFourierTransform(N=16)

# Verify perfect reconstruction
results = transformer.verify_perfect_reconstruction()
print(f"\nPerfect reconstruction: {results['perfect_reconstruction']}")
print(f"Relative error: {results['relative_error']:.2e}")
print(f"Energy error: {results['energy_error']:.2e}")

# Verify non-equivalence to DFT
dft_results = transformer.prove_non_equivalence_to_dft()
print(f"\nNon-equivalent to DFT: {dft_results['non_equivalent']}")
print(f"Distinctness score: {dft_results['distinctness_score']:.2%}")

print("\nTest complete!")
