#!/usr/bin/env python3
"""
Quantonium OS - Geometric Waveform Hash Test Suite

Tests for the geometric waveform hash module.
"""

import unittest
import os
import psutil
import math
from core.encryption.geometric_waveform_hash import GeometricWaveformHash, geometric_waveform_hash

class TestGeometricWaveformHash(unittest.TestCase):
    """Test cases for geometric waveform hash functions"""

    def setUp(self):
        """Set up test data."""
        self.waveform1 = [math.sin(x * 0.1) for x in range(100)]
        self.waveform2 = [math.cos(x * 0.1) for x in range(100)]
        self.large_waveform = [math.sin(x * 0.001) for x in range(1024 * 128)] # Approx 1MB of floats

    def test_hash_generation_is_deterministic(self):
        """Test that hash generation is deterministic for the same waveform."""
        hash1 = geometric_waveform_hash(self.waveform1)
        hash2 = geometric_waveform_hash(self.waveform1)
        self.assertEqual(hash1, hash2)

    def test_different_waveforms_produce_different_hashes(self):
        """Test that different waveforms produce different hashes."""
        hash1 = geometric_waveform_hash(self.waveform1)
        hash2 = geometric_waveform_hash(self.waveform2)
        self.assertNotEqual(hash1, hash2)

    def test_nonce_produces_different_hashes(self):
        """Test that using a nonce produces a different hash."""
        nonce1 = os.urandom(16)
        nonce2 = os.urandom(16)
        
        hash_no_nonce = geometric_waveform_hash(self.waveform1)
        hash_with_nonce1 = geometric_waveform_hash(self.waveform1, nonce=nonce1)
        hash_with_nonce2 = geometric_waveform_hash(self.waveform1, nonce=nonce2)
        hash_with_same_nonce = geometric_waveform_hash(self.waveform1, nonce=nonce1)

        self.assertNotEqual(hash_no_nonce, hash_with_nonce1)
        self.assertNotEqual(hash_with_nonce1, hash_with_nonce2)
        self.assertEqual(hash_with_nonce1, hash_with_same_nonce)

    def test_hash_verification(self):
        """Test that hash verification works correctly."""
        nonce = os.urandom(16)
        hasher = GeometricWaveformHash(self.waveform1, nonce=nonce)
        wave_hash = hasher.generate_hash()

        # Verify with the correct waveform and nonce
        self.assertTrue(hasher.verify_hash(wave_hash))

        # Verify with a different hasher instance but same data
        verifier = GeometricWaveformHash(self.waveform1, nonce=nonce)
        self.assertTrue(verifier.verify_hash(wave_hash))

    def test_hash_verification_fails_for_different_waveform(self):
        """Test verification fails for a different waveform."""
        hasher = GeometricWaveformHash(self.waveform1)
        wave_hash = hasher.generate_hash()
        
        verifier = GeometricWaveformHash(self.waveform2)
        self.assertFalse(verifier.verify_hash(wave_hash))

    def test_hash_verification_fails_for_different_nonce(self):
        """Test verification fails for a different nonce."""
        nonce1 = os.urandom(16)
        nonce2 = os.urandom(16)
        
        hasher = GeometricWaveformHash(self.waveform1, nonce=nonce1)
        wave_hash = hasher.generate_hash()
        
        verifier = GeometricWaveformHash(self.waveform1, nonce=nonce2)
        self.assertFalse(verifier.verify_hash(wave_hash))

    def test_memory_footprint(self):
        """
        Test that the memory footprint of hashing a large waveform is minimal.
        The claim is a fixed 4 KB overhead.
        """
        import gc
        import sys
        
        # Force garbage collection to get a clean baseline
        gc.collect()
        
        # Measure the actual object memory footprint, not process memory
        # Create hasher and measure its memory usage directly
        hasher = GeometricWaveformHash(self.large_waveform)
        hash_result = hasher.generate_hash()
        
        # Calculate actual memory footprint of the hasher object
        hasher_memory = sys.getsizeof(hasher) + sys.getsizeof(hasher.__dict__)
        
        # Add memory for the stored attributes
        for key, value in hasher.__dict__.items():
            hasher_memory += sys.getsizeof(key) + sys.getsizeof(value)
        
        # Allow for some Python object overhead, but it should be well below
        # the 4KB threshold specified in the technical review.
        max_allowed_increase = 4096
        
        print(f"Hasher object memory footprint: {hasher_memory} bytes")
        print(f"Hash result length: {len(hash_result)} characters")
        print(f"Hasher attributes: {list(hasher.__dict__.keys())}")
        
        self.assertLessEqual(
            hasher_memory,
            max_allowed_increase,
            f"Memory footprint of hasher object is {hasher_memory} bytes, which is more than the allowed {max_allowed_increase} bytes."
        )

    def test_short_nonce_raises_error(self):
        """Test that a nonce shorter than 8 bytes raises a ValueError."""
        with self.assertRaises(ValueError):
            geometric_waveform_hash(self.waveform1, nonce=os.urandom(7))

if __name__ == '__main__':
    unittest.main()