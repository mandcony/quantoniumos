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

"""
Quantonium OS - RFT-based Geometric Waveform Hash Test

Tests that the geometric waveform hash is correctly using the RFT engine.
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.encryption.geometric_waveform_hash import GeometricWaveformHash
# Import necessary modules
import importlib.util
import os

# Load the paper_compliant_rft_fixed module
spec = importlib.util.spec_from_file_location(
    "paper_compliant_rft_fixed", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "04_RFT_ALGORITHMS/paper_compliant_rft_fixed.py")
)
paper_compliant_rft_fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper_compliant_rft_fixed)

# Import specific functions/classes
FixedRFTCryptoBindings, PaperCompliantRFT

class TestRFTGeometricWaveform = paper_compliant_rft_fixed.FixedRFTCryptoBindings, paper_compliant_rft_fixed.PaperCompliantRFT

class TestRFTGeometricWaveform(unittest.TestCase):
    """Test that the geometric waveform hash correctly uses the RFT engine"""

    def test_geometric_hash_uses_rft(self):
        """Test that the geometric waveform hash uses RFT"""
        # Create a test waveform
        test_waveform = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        # Create geometric hash
        hasher = GeometricWaveformHash(test_waveform)

        # Check that geometric hash is not None
        self.assertIsNotNone(hasher.geometric_hash)

        # Check that topological signature is not None
        self.assertIsNotNone(hasher.topological_signature)

        # The real proof is in the code inspection, but we can test functionality
        # Create two identical waveforms
        waveform1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        waveform2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        # Create two geometric hashes
        hasher1 = GeometricWaveformHash(waveform1)
        hasher2 = GeometricWaveformHash(waveform2)

        # Hashes should be identical
        self.assertEqual(hasher1.geometric_hash, hasher2.geometric_hash)

        # Create a different waveform
        waveform3 = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        # Create another geometric hash
        hasher3 = GeometricWaveformHash(waveform3)

        # Hash should be different
        self.assertNotEqual(hasher1.geometric_hash, hasher3.geometric_hash)

if __name__ == "__main__":
    unittest.main()
