#!/usr/bin/env python3
"""
Test suite for minimal RFT encryption demo === Ensures the minimal_rft_encrypt_demo.py works correctly in CI environments and provides automated validation of the canonical RFT encryption properties.
"""
"""

import pytest
import sys
import os
import subprocess from pathlib
import Path

# Add project root to path for imports project_root = Path(__file__).parent sys.path.insert(0, str(project_root))

# Import the minimal demo functions directly from minimal_rft_encrypt_demo
import encrypt, decrypt, derive_key, rft_keystream from canonical_true_rft
import PHI, forward_true_rft, inverse_true_rft

class TestMinimalRFTDemo:
"""
"""
    Test cases for minimal RFT encryption demo
"""
"""

    def test_derive_key_deterministic(self):
"""
"""
        Test that key derivation is deterministic
"""
        """ key1 = derive_key("test_password") key2 = derive_key("test_password") assert key1 == key2 assert len(key1) == 32

        # SHA-256 output length
    def test_keystream_deterministic(self): """
        Test that keystream generation is deterministic
"""
"""
        key = b'0' * 32

        # Fixed key for testing ks1 = rft_keystream(0, 16, key) ks2 = rft_keystream(0, 16, key) assert ks1 == ks2 assert len(ks1) == 16
    def test_keystream_different_indices(self):
"""
"""
        Test that different block indices produce different keystreams
"""
"""
        key = b'0' * 32 ks1 = rft_keystream(0, 16, key) ks2 = rft_keystream(1, 16, key) assert ks1 != ks2
    def test_encrypt_decrypt_roundtrip(self):
"""
"""
        Test basic encryption/decryption round trip
"""
        """ passphrase = "test_secret_123" plaintext = "Hello RFT World!" ciphertext = encrypt(passphrase, plaintext) recovered = decrypt(passphrase, ciphertext) assert recovered == plaintext
    def test_encrypt_decrypt_various_lengths(self): """
        Test with different plaintext lengths
"""
        """ passphrase = "consistent_key" test_cases = [ "",

        # Empty string "A",

        # Single character "Short message",

        # Short "A" * 100,

        # Long message crossing block boundaries "Unicode: phi = phi! ",

        # Unicode content ]
        for plaintext in test_cases: ciphertext = encrypt(passphrase, plaintext) recovered = decrypt(passphrase, ciphertext) assert recovered == plaintext, f"Failed for: {plaintext}"
    def test_ciphertext_format(self): """
        Test ciphertext format compliance
"""
        """ ciphertext = encrypt("key", "message")

        # Should start with RFT prefix assert ciphertext.startswith("RFT")

        # Should have block size encoded assert "-" in ciphertext

        # Should be valid hex after the dash parts = ciphertext.split("-", 1) assert len(parts) == 2 hex_part = parts[1] bytes.fromhex(hex_part)

        # Should not
        raise exception
    def test_wrong_password_fails(self): """
        Test that wrong password produces garbage or fails
"""
        """ plaintext = "Secret message" ciphertext = encrypt("correct_password", plaintext)

        # Wrong password should either fail or produce different result
        try: wrong_result = decrypt("wrong_password", ciphertext) assert wrong_result != plaintext

        # If it doesn't fail, result should differ except (UnicodeDecodeError, ValueError): pass

        # Expected failure is also acceptable
    def test_canonical_rft_integration(self): """
        Test that the demo actually uses canonical RFT functions
"""
"""

        # This test verifies the integration works by calling the RFT functions test_vec = [1.0, 2.0, 3.0, 4.0] coeffs = forward_true_rft(test_vec) recovered = inverse_true_rft(coeffs)

        # Should have good reconstruction accuracy for orig, rec in zip(test_vec, recovered[:len(test_vec)]): assert abs(orig - rec) < 1e-10
    def test_golden_ratio_precision(self):
"""
"""
        Test that golden ratio constant has full precision
"""
"""

        # Verify PHI is imported correctly and has full precision expected_phi = (1.0 + 5**0.5) / 2.0 assert abs(PHI - expected_phi) < 1e-15
    def test_cli_interface(self):
"""
"""
        Test the command line interface works
"""
"""

        # Test the script can be called directly result = subprocess.run([ sys.executable, "minimal_rft_encrypt_demo.py", "test_password", "CLI test message" ], capture_output=True, text=True, cwd=project_root) assert result.returncode == 0 output = result.stdout assert "Golden ratio phi =" in output assert "Ciphertext :" in output assert "Round-trip :" in output assert "Match : True" in output

if __name__ == "__main__":

# Allow running directly as script pytest.main([__file__, "-v"])