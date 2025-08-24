#!/usr/bin/env python
"""
Encryption Module Tests
======================
Tests for the encryption modules in QuantoniumOS.
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules
from core.encryption import aes, quantum, rsa


class TestAESEncryption(unittest.TestCase):
    """Test AES encryption"""

    def test_encryption_decryption(self):
        """Test that encryption and decryption work correctly"""
        # Test data
        plaintext = b"This is a test message for AES encryption"
        key = b"0123456789abcdef0123456789abcdef"

        # Encrypt
        ciphertext = aes.encrypt(plaintext, key)
        self.assertNotEqual(plaintext, ciphertext)

        # Decrypt
        decrypted = aes.decrypt(ciphertext, key)
        self.assertEqual(plaintext, decrypted)


class TestRSAEncryption(unittest.TestCase):
    """Test RSA encryption"""

    def test_key_generation(self):
        """Test key generation"""
        # Generate keys
        public_key, private_key = rsa.generate_keypair(bits=2048)
        self.assertIsNotNone(public_key)
        self.assertIsNotNone(private_key)

    def test_encryption_decryption(self):
        """Test that encryption and decryption work correctly"""
        # Generate keys
        public_key, private_key = rsa.generate_keypair(bits=2048)

        # Test data
        plaintext = b"This is a test message for RSA encryption"

        # Encrypt
        ciphertext = rsa.encrypt(plaintext, public_key)
        self.assertNotEqual(plaintext, ciphertext)

        # Decrypt
        decrypted = rsa.decrypt(ciphertext, private_key)
        self.assertEqual(plaintext, decrypted)


class TestQuantumEncryption(unittest.TestCase):
    """Test quantum encryption"""

    def test_key_distribution(self):
        """Test quantum key distribution"""
        # Generate key
        key = quantum.generate_key(length=256)
        self.assertEqual(len(key), 256)

    def test_encryption_decryption(self):
        """Test that encryption and decryption work correctly"""
        # Generate key
        key = quantum.generate_key(length=32)

        # Test data
        plaintext = b"This is a test message for quantum encryption"

        # Encrypt
        ciphertext = quantum.encrypt(plaintext, key)
        self.assertNotEqual(plaintext, ciphertext)

        # Decrypt
        decrypted = quantum.decrypt(ciphertext, key)
        self.assertEqual(plaintext, decrypted)


def run_tests():
    """Run all encryption module tests"""
    unittest.main()


if __name__ == "__main__":
    run_tests()
