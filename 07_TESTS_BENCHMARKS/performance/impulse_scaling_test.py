# -*- coding: utf-8 -*-
#
# QuantoniumOS Test Suite
# Testing with QuantoniumOS implementations
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

# Import QuantoniumOS quantum engines
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

# Import QuantoniumOS cryptography modules
try:
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
except ImportError as e:
    print(f"Warning: Could not import cryptography modules: {e}")

# Import QuantoniumOS validators
try:
    sys.path.insert(0, '/workspaces/quantoniumos/02_CORE_VALIDATORS')
    from basic_scientific_validator import BasicScientificValidator
    from definitive_quantum_validation import DefinitiveQuantumValidation
    from phd_level_scientific_validator import PhdLevelScientificValidator
    from publication_ready_validation import PublicationReadyValidation
except ImportError as e:
    print(f"Warning: Could not import validators: {e}")

# Import QuantoniumOS running systems
try:
    sys.path.insert(0, '/workspaces/quantoniumos/03_RUNNING_SYSTEMS')
    from app import app
    from main import main
    from quantonium import QuantoniumOS
except ImportError as e:
    print(f"Warning: Could not import running systems: {e}")

"""
SIMPLE SCALING VERIFICATION Test with impulse input [1,0,0,0...] to isolate scaling factor
"""
"""
import sys
import math sys.path.append('/workspaces/quantoniumos/core')
try:
import quantonium_core
print(" IMPULSE SCALING TEST")
print("=" * 30)

# Test with impulse (delta function) N = 16 impulse = [1.0] + [0.0] * (N-1)
print(f"Input: [1, 0, 0, 0, ...] (N={N})")
print(f"Expected output energy: 1.0") rft = quantonium_core.ResonanceFourierTransform(impulse) output = rft.forward_transform()
if isinstance(output[0], complex): energy = sum(abs(x)**2
for x in output)
else: energy = sum(x*x
for x in output)
print(f"Actual output energy: {energy:.6f}")
print(f"Scaling factor applied: {math.sqrt(energy):.6f}")

# Check what scaling was applied
if abs(energy - 1.0) < 1e-6:
print(" PERFECT: No scaling issue")
el
if abs(energy - 1.0/N) < 1e-6:
print(f" CONFIRMED: Forward applies 1/N scaling")
print(f" FIX: Multiply by sqrt{N} = {math.sqrt(N):.3f}")
el
if abs(energy - 1.0/(N*N)) < 1e-6:
print(f" CONFIRMED: Forward applies 1/N^2 scaling")
print(f" FIX: Multiply by {N} = {N}")
else:
print(f" CUSTOM scaling: {energy:.6f}") except Exception as e:
print(f" Test failed: {e}")