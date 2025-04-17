#!/usr/bin/env python3
"""
Quantonium OS - Geometric Waveform Hash Test Suite

Tests for the geometric waveform hash module.
"""

import unittest
from core.encryption.geometric_waveform_hash import (
    generate_waveform_hash,
    verify_waveform_hash,
    extract_parameters_from_hash
)

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
        
        # Should match original values
        self.assertAlmostEqual(orig_A, extracted_A, places=4)
        self.assertAlmostEqual(orig_phi, extracted_phi, places=4)
    
    def test_parameter_extraction_with_invalid_hash(self):
        """Test that parameter extraction returns None for invalid hash"""
        # Invalid hash formats
        invalid_hashes = [
            "invalid_hash_format",
            "A0.5_invalid",
            "P0.5_invalid",
            "A0.5_P0.5"  # Missing hash part
        ]
        
        for invalid_hash in invalid_hashes:
            A, phi = extract_parameters_from_hash(invalid_hash)
            self.assertIsNone(A)
            self.assertIsNone(phi)
    
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