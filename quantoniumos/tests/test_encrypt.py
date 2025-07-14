#!/usr/bin/env python3
"""
Quantonium OS - Encryption Test Suite

Tests for the resonance encryption module with enhanced security features.
"""

import unittest
import time
import hashlib
import base64
from core.encryption.resonance_encrypt import (
    encrypt,
    encrypt_data,
    decrypt_data,
    resonance_encrypt,
    resonance_decrypt
)
from core.encryption.geometric_waveform_hash import generate_waveform_hash

class TestResonanceEncryption(unittest.TestCase):
    """Test cases for resonance encryption and decryption"""
    
    def test_encrypt_decrypt_roundtrip(self):
        """Test basic encryption and decryption roundtrip"""
        plaintext = "Test quantum message"
        key = "secure-test-key"
        
        # Encrypt data
        encrypted = encrypt_data(plaintext, key)
        
        # Decrypt data
        decrypted = decrypt_data(encrypted, key)
        
        # Check roundtrip
        self.assertEqual(plaintext, decrypted)
    
    def test_encrypt_returns_valid_dict(self):
        """Test that encrypt() returns a dict with correct fields"""
        plaintext = "Test message for dict return"
        key = "secure-test-key"
        
        # Encrypt data
        result = encrypt(plaintext, key)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("ciphertext", result)
        self.assertIn("ts", result)
        self.assertIn("sig", result)
        
        # Verify signature is valid
        sig_data = result["ciphertext"] + str(result["ts"]) + key
        expected_sig = hashlib.sha256(sig_data.encode()).hexdigest()
        self.assertEqual(result["sig"], expected_sig)
    
    def test_invalid_signature_raises_error(self):
        """Test that invalid signatures are properly rejected"""
        plaintext = "Secret data"
        key = "secure-test-key"
        
        # Get amplitude and phase from key
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        A = float(int(key_hash[:8], 16) % 1000) / 1000
        phi = float(int(key_hash[8:16], 16) % 1000) / 1000
        
        # Encrypt with correct parameters
        encrypted = resonance_encrypt(plaintext, A, phi)
        
        # Tamper with signature (first 4 bytes)
        tampered = bytearray(encrypted)
        tampered[0] = (tampered[0] + 1) % 256  # Change first byte
        
        # Attempt to decrypt with tampered signature
        with self.assertRaises(ValueError):
            resonance_decrypt(bytes(tampered), A, phi)
    
    def test_wrong_key_fails_to_decrypt(self):
        """Test that using wrong key fails to decrypt"""
        plaintext = "Sensitive information"
        key1 = "secure-key-one"
        key2 = "secure-key-two"
        
        # Encrypt with key1
        encrypted = encrypt_data(plaintext, key1)
        
        # Attempt to decrypt with key2
        decrypted = decrypt_data(encrypted, key2)
        
        # Should fail and return error message
        self.assertNotEqual(plaintext, decrypted)
        self.assertTrue(decrypted.startswith("Decryption failed"))
    
    def test_encryption_is_deterministic_with_same_token(self):
        """Test that encryption is deterministic given the same inputs and token"""
        plaintext = "Deterministic test"
        key = "fixed-test-key"
        
        # Get amplitude and phase from key
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        A = float(int(key_hash[:8], 16) % 1000) / 1000
        phi = float(int(key_hash[8:16], 16) % 1000) / 1000
        
        # This should not be used in production code - only for testing deterministic behavior
        import unittest.mock
        with unittest.mock.patch('secrets.token_bytes', return_value=b'x' * 32):
            # Encrypt twice with same parameters
            encrypted1 = resonance_encrypt(plaintext, A, phi)
            encrypted2 = resonance_encrypt(plaintext, A, phi)
            
            # Results should be identical
            self.assertEqual(encrypted1, encrypted2)
    
    def test_timestamp_is_recent(self):
        """Test that the timestamp in encrypt() result is recent"""
        plaintext = "Timestamp test"
        key = "secure-test-key"
        
        # Get current time
        before = int(time.time())
        
        # Encrypt data
        result = encrypt(plaintext, key)
        
        # Get time after encryption
        after = int(time.time())
        
        # Timestamp should be between before and after
        self.assertTrue(before <= result["ts"] <= after)
    
    def test_waveform_hash_consistency(self):
        """Test that waveform hash generation is consistent"""
        # Test parameters
        A = 0.5
        phi = 0.75
        
        # Generate hash multiple times
        hash1 = generate_waveform_hash(A, phi)
        hash2 = generate_waveform_hash(A, phi)
        hash3 = generate_waveform_hash(A, phi)
        
        # All should be identical
        self.assertEqual(hash1, hash2)
        self.assertEqual(hash2, hash3)

if __name__ == "__main__":
    unittest.main()