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
Quantonium OS - Geometric Waveform Hash Test Suite

Tests for the geometric waveform hash module.
"""

import unittest

from core.encryption.geometric_waveform_hash import (
    extract_parameters_from_hash, generate_waveform_hash, verify_waveform_hash)

class TestGeometricWaveformHash(unittest.TestCase):
    """Test cases for geometric waveform hash functions"""

    def test_hash_generation_is_deterministic(self):
        """Test that hash generation is deterministic for the same inputs"""
        # Generate hashes with the same parameters
        hash1 = generate_waveform_hash(0.5, 0.25)
        hash2 = generate_waveform_hash(0.5, 0.25)

        # Should be identical
        self.assertEqual(hash1, hash2)

    def test_different_params_produce_different_hashes(self):
        """Test that different parameters produce different hashes"""
        hash1 = generate_waveform_hash(0.1, 0.2)
        hash2 = generate_waveform_hash(0.1, 0.3)
        hash3 = generate_waveform_hash(0.2, 0.2)

        # All should be different
        self.assertNotEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        self.assertNotEqual(hash2, hash3)

    def test_hash_verification(self):
        """Test that hash verification works correctly"""
        # Generate a hash
        A, phi = 0.75, 0.33
        wave_hash = generate_waveform_hash(A, phi)

        # Verify with correct parameters
        self.assertTrue(verify_waveform_hash(wave_hash, A, phi))

        # Verify with incorrect parameters
        self.assertFalse(verify_waveform_hash(wave_hash, A + 0.01, phi))
        self.assertFalse(verify_waveform_hash(wave_hash, A, phi + 0.01))

    def test_parameter_extraction(self):
        """Test that parameters can be extracted from hash"""
        # Original parameters
        orig_A, orig_phi = 0.42, 0.67

        # Generate hash
        wave_hash = generate_waveform_hash(orig_A, orig_phi)

        # Extract parameters
        extracted_A, extracted_phi = extract_parameters_from_hash(wave_hash)

        # Check if values are not None before assertion
        self.assertIsNotNone(extracted_A)
        self.assertIsNotNone(extracted_phi)

        if extracted_A is not None and extracted_phi is not None:
            # Should match original values
            self.assertAlmostEqual(orig_A, extracted_A, places=4)
            self.assertAlmostEqual(orig_phi, extracted_phi, places=4)

    def test_parameter_extraction_with_invalid_hash(self):
        """Test that parameter extraction can still handle invalid or base64 hashes"""
        # Invalid hash formats - now we attempt to extract usable parameters
        test_hashes = [
            "invalid_hash_format",
            "A0.5_invalid",
            "P0.5_invalid",
            "A0.5_P0.5",  # Missing hash part
            # Base64-like strings should also work now
            "XvN5CZ7+fr2uIEMEaPlQKqMvlGo7Ld+xNMC8dgjDPNeo4GnJzmsOzn6qkQ==",
        ]

        for test_hash in test_hashes:
            A, phi = extract_parameters_from_hash(test_hash)
            # With our updated implementation, we should get valid parameters
            # even from "invalid" or base64 hashes
            self.assertIsNotNone(A)
            self.assertIsNotNone(phi)
            # Values should be in range 0.0-1.0
            self.assertGreaterEqual(A, 0.0)
            self.assertLessEqual(A, 1.0)
            self.assertGreaterEqual(phi, 0.0)
            self.assertLessEqual(phi, 1.0)

    def test_parameter_normalization(self):
        """Test that parameters are normalized to 0.0-1.0 range"""
        # Test with out-of-range values
        hash1 = generate_waveform_hash(1.5, 0.5)
        hash2 = generate_waveform_hash(-0.5, 0.5)

        # Extract parameters - should be normalized
        A1, phi1 = extract_parameters_from_hash(hash1)
        A2, phi2 = extract_parameters_from_hash(hash2)

        # A1 should be capped at 1.0, A2 at 0.0
        self.assertEqual(A1, 1.0)
        self.assertEqual(A2, 0.0)

if __name__ == "__main__":
    unittest.main()
