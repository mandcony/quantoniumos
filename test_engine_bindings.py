#!/usr/bin/env python3
"""
Test script for True RFT Engine Bindings
"""

import sys
import os

# Import directly by appending the directory to the Python path
sys.path.append('/workspaces/quantoniumos/04_RFT_ALGORITHMS')
from true_rft_engine_bindings import TrueRFTEngine

# Create engine instance
print("Creating TrueRFTEngine instance...")
engine = TrueRFTEngine(size=16)
engine.init()

# Test with a simple signal
import numpy as np
signal = np.random.randn(16) + 1j * np.random.randn(16)

# Forward transform
print("\nApplying forward transform...")
result = engine.compute_rft(signal)
transformed = result["result"]

# Check energy conservation
original_energy = np.linalg.norm(signal) ** 2
transformed_energy = np.linalg.norm(transformed) ** 2
energy_ratio = transformed_energy / original_energy

print(f"Original signal energy: {original_energy:.6f}")
print(f"Transformed signal energy: {transformed_energy:.6f}")
print(f"Energy ratio: {energy_ratio:.6f} (should be close to 1.0)")

# Inverse transform
print("\nApplying inverse transform...")
inverse_result = engine.compute_inverse_rft(transformed)
reconstructed = inverse_result["result"]

# Check reconstruction error
error = np.linalg.norm(signal - reconstructed)
relative_error = error / np.linalg.norm(signal)

print(f"Reconstruction error: {error:.6e}")
print(f"Relative error: {relative_error:.6e} (should be very small)")

print("\nTest complete!")
