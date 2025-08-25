# -*- coding: utf-8 -*-
#
# QuantoniumOS RFT (Resonant Frequency Transform) Tests
# Testing with QuantoniumOS RFT implementations
#
# ===================================================================

import unittest
import sys
import os
import numpy as np
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError as e:
    print(f"Warning: Could not import RFT algorithms: {e}")

# Import QuantoniumOS quantum engines for RFT-quantum integration
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from topological_vertex_engine import TopologicalVertexEngine
    from topological_vertex_geometric_engine import TopologicalVertexGeometricEngine
    from vertex_engine_canonical import VertexEngineCanonical
    from working_quantum_kernel import WorkingQuantumKernel
    from true_rft_engine_bindings import TrueRFTEngineBindings as QuantumRFTBindings
except ImportError as e:
    print(f"Warning: Could not import quantum engines: {e}")

# Import QuantoniumOS cryptography modules for RFT-crypto integration
try:
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
except ImportError as e:
    print(f"Warning: Could not import cryptography modules: {e}")

||#!/usr/bin/env python3
"""
Production Surgical Fix Validation Tests that quantonium_core delegate works perfectly
"""
"""
import sys sys.path.append('/workspaces/quantoniumos')
print("🔧 SURGICAL FIX VALIDATION")
print("Testing quantonium_core delegate wrapper")
print()

# Test the surgical fix
try:
import quantonium_core_delegate as quantonium_core
print("✅ Surgical delegate wrapper loaded successfully")

# Test RFT operations test_signal = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125] rft = quantonium_core.ResonanceFourierTransform(test_signal)
print("✅ RFT instance created")

# Forward transform coeffs = rft.forward_transform()
print(f"✅ Forward transform: {len(coeffs)} coefficients")
print(f" First 5 coeffs: {coeffs[:5]}")

# Energy check original_energy = sum(x**2
for x in test_signal) coeff_energy = sum(abs(c)**2
for c in coeffs) energy_ratio = coeff_energy / original_energy
if original_energy > 0 else 0
print(f"✅ Energy conservation: {energy_ratio:.6f} (should be ~1.0)")
print(f" Original energy: {original_energy:.6f}")
print(f" Transform energy: {coeff_energy:.6f}")

# Reconstruction test reconstructed = rft.inverse_transform(coeffs) reconstruction_mse = sum((a - b)**2 for a, b in zip(test_signal, reconstructed)) / len(test_signal)
print(f"✅ Reconstruction MSE: {reconstruction_mse:.2e}")
print(f" Original: {test_signal[:3]}...")
print(f" Reconstructed: {reconstructed[:3]}...")

# Overall assessment
if abs(energy_ratio - 1.0) < 0.001 and reconstruction_mse < 1e-12:
print("\n SURGICAL FIX SUCCESSFUL: Perfect energy conservation and reconstruction!")
el
if abs(energy_ratio - 1.0) < 0.1 and reconstruction_mse < 1e-6:
print("\n✅ SURGICAL FIX WORKING: Good energy conservation and reconstruction")
else:
print("\n⚠️ SURGICAL FIX NEEDS REFINEMENT") except Exception as e:
print(f"❌ Surgical fix failed: {e}")
import traceback traceback.print_exc()
print("||n" + "="*50)
print("Surgical fix validation complete")