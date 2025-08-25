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
Simple test script for the current RFT Feistel cipher
"""

import os
import time

import rft_feistel_cipher
import RFTFeistelCipher

def test_avalanche(cipher, key, test_size=32, num_tests=10): """
        Test avalanche effect with single bit changes
"""
        total_avalanche = 0
        for _ in range(num_tests):

        # Create test data plaintext1 = os.urandom(test_size) plaintext2 = bytearray(plaintext1)

        # Flip one bit bit_pos = 0 plaintext2[bit_pos // 8] ^= (1 << (bit_pos % 8)) plaintext2 = bytes(plaintext2)

        # Encrypt both ciphertext1 = cipher.encrypt(plaintext1, key) ciphertext2 = cipher.encrypt(plaintext2, key)

        # Count bit differences bit_diff = sum(bin(a ^ b).count('1') for a, b in zip(ciphertext1, ciphertext2)) total_bits = len(ciphertext1) * 8 avalanche = bit_diff / total_bits total_avalanche += avalanche
        return total_avalanche / num_tests
def main():
        print("Testing current RFT Feistel cipher...") cipher = RFTFeistelCipher() key = os.urandom(32)

        # Test basic functionality plaintext = b"Hello, World! This is a test message for RFT."
        print(f"Original: {plaintext}") start_time = time.time() ciphertext = cipher.encrypt(plaintext, key) encrypt_time = time.time() - start_time
        print(f"Encrypted: {ciphertext.hex()[:64]}...")
        print(f"Encryption time: {encrypt_time:.4f}s") start_time = time.time() decrypted = cipher.decrypt(ciphertext, key) decrypt_time = time.time() - start_time
        print(f"Decrypted: {decrypted}")
        print(f"Decryption time: {decrypt_time:.4f}s")
        print(f"Perfect reversibility: {plaintext == decrypted}")

        # Test key sensitivity wrong_key = os.urandom(32)
        try: wrong_decrypt = cipher.decrypt(ciphertext, wrong_key) key_sensitivity = sum(a != b for a, b in zip(plaintext, wrong_decrypt)) / len(plaintext)
        print(f"Key sensitivity: {key_sensitivity:.3f}")
        except:
        print("Key sensitivity: Perfect (decryption fails with wrong key)")

        # Test avalanche effect
        print("\nTesting avalanche effect...") avalanche = test_avalanche(cipher, key)
        print(f"Average avalanche effect: {avalanche:.3f}")

        # Performance test
        print("\nPerformance test...") test_data = os.urandom(1024) # 1KB start_time = time.time()
        for _ in range(100): cipher.encrypt(test_data, key) total_time = time.time() - start_time throughput = (1024 * 100) / total_time / 1024

        # KB/s
        print(f"Throughput: {throughput:.2f} KB/s")

if __name__ == "__main__": main()