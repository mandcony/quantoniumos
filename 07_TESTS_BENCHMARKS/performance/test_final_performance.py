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

import os
import secrets
import time
import enhanced_rft_crypto

class FixedRFTCrypto: """
    Wrapper for enhanced RFT crypto with proper padding handling.
"""

    def __init__(self):
        self.engine = enhanced_rft_crypto.PyEnhancedRFTCrypto()
    def encrypt(self, plaintext, key): """
        Encrypt with proper padding handling.
"""

        # Store original length original_len = len(plaintext)

        # Pad to 16-byte boundary padded_len = ((original_len + 15) // 16) * 16 padded_plaintext = plaintext + b'\x00' * (padded_len - original_len)

        # Encrypt the padded data encrypted =
        self.engine.encrypt(padded_plaintext, key)

        # Prepend original length (4 bytes)
        return original_len.to_bytes(4, 'little') + encrypted
    def decrypt(self, ciphertext, key): """
        Decrypt with proper padding removal.
"""

        # Extract original length
        if len(ciphertext) < 4:
        raise ValueError("Invalid ciphertext: too short") original_len = int.from_bytes(ciphertext[:4], 'little') encrypted_data = ciphertext[4:]

        # Decrypt decrypted_padded =
        self.engine.decrypt(encrypted_data, key)

        # Return only original length
        return decrypted_padded[:original_len]
    def test_enhanced_cpp_rft():
        print("=== Enhanced C++ RFT Crypto Performance Test ===")

        # Initialize crypto engine crypto = FixedRFTCrypto()

        # Test basic reversibility key = secrets.token_bytes(32) plaintext = b"Hello, World! This is a test message for RFT encryption."
        print(f"Plaintext: {plaintext}")

        # Encrypt start_time = time.time() encrypted = crypto.encrypt(plaintext, key) encrypt_time = time.time() - start_time
        print(f"Encrypted ({len(encrypted)} bytes): {encrypted[:32].hex()}...")
        print(f"Encryption time: {encrypt_time:.6f} seconds")

        # Decrypt start_time = time.time() decrypted = crypto.decrypt(encrypted, key) decrypt_time = time.time() - start_time
        print(f"Decrypted: {decrypted}")
        print(f"Decryption time: {decrypt_time:.6f} seconds")

        # Check correctness correct = (plaintext == decrypted)
        print(f"Reversibility: {'PASS'
        if correct else 'FAIL'}")
        if not correct:
        print("ERROR: Decryption failed!")
        return False

        # Performance benchmark
        print("\n=== Performance Benchmark ===") test_sizes = [1024, 4096, 16384, 65536] # 1KB to 64KB
        for size in test_sizes: test_data = secrets.token_bytes(size)

        # Encryption benchmark start_time = time.time() iterations = 100
        for _ in range(iterations): crypto.encrypt(test_data, key) encrypt_total = time.time() - start_time encrypt_throughput = (size * iterations) / (encrypt_total * 1024)

        # KB/s
        print(f"Size: {size:5d} bytes | Encrypt: {encrypt_throughput:8.1f} KB/s | Target: 30,000+ KB/s")

        # Cryptographic quality tests
        print("\n=== Cryptographic Quality Tests ===")

        # Avalanche effect test key1 = secrets.token_bytes(32) key2 = bytearray(key1) key2[0] ^= 1

        # Flip one bit test_msg = secrets.token_bytes(64) enc1 = crypto.encrypt(test_msg, key1) enc2 = crypto.encrypt(test_msg, bytes(key2))

        # Remove length headers for comparison enc1_data = enc1[4:] enc2_data = enc2[4:] diff_bits = 0 total_bits = len(enc1_data) * 8
        for i in range(len(enc1_data)): diff_bits += bin(enc1_data[i] ^ enc2_data[i]).count('1') avalanche = diff_bits / total_bits
        print(f"Avalanche effect (1-bit key change): {avalanche:.3f} (target: ~0.5)")

        # Key sensitivity test sensitive_keys = [] base_encryption = crypto.encrypt(test_msg, key1)[4:]

        # Remove header
        for bit_pos in range(min(256, len(key1) * 8)):

        # Test first 256 bits test_key = bytearray(key1) byte_pos = bit_pos // 8 bit_in_byte = bit_pos % 8 test_key[byte_pos] ^= (1 << bit_in_byte) test_encryption = crypto.encrypt(test_msg, bytes(test_key))[4:]

        # Remove header

        # Count different bits diff_bits = 0
        for i in range(len(base_encryption)): diff_bits += bin(base_encryption[i] ^ test_encryption[i]).count('1') sensitivity = diff_bits / (len(base_encryption) * 8) sensitive_keys.append(sensitivity) avg_sensitivity = sum(sensitive_keys) / len(sensitive_keys)
        print(f"Average key sensitivity: {avg_sensitivity:.3f} (target: >0.4)")

        # Overall assessment
        print(f"\n=== Assessment ===")
        print(f"✓ Reversibility: {'PASS'
        if correct else 'FAIL'}")
        print(f"✓ Avalanche effect: {'GOOD' if 0.4 <= avalanche <= 0.6 else 'NEEDS WORK'} ({avalanche:.3f})")
        print(f"✓ Key sensitivity: {'GOOD'
        if avg_sensitivity >= 0.4 else 'NEEDS WORK'} ({avg_sensitivity:.3f})")
        return correct and 0.4 <= avalanche <= 0.6 and avg_sensitivity >= 0.4

if __name__ == "__main__": success = test_enhanced_cpp_rft()
print(f"\nOverall result: {'SUCCESS'
if success else 'NEEDS IMPROVEMENT'}")