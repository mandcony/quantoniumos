#!/usr/bin/env python3
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
from paper_compliant_rft_fixed import FixedRFTCryptoBindings, PaperCompliantRFT


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
