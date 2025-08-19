#!/usr/bin/env python3
"""
Test script for the enhanced C++ RFT implementation Measures avalanche effect and key sensitivity with native performance
"""

import os
import sys
import time
def main():
        print("Testing Enhanced C++ RFT Feistel cipher (Native Performance)...")
        print("=" * 60)
        try:

        # Import the optimized native module
import enhanced_rft_crypto cipher = enhanced_rft_crypto.PyEnhancedRFTCrypto()
        print("✓ Using native C++ implementation") except ImportError as e:
        print(f"✗ Failed to
import native module: {e}")
        print("Building native module...")

        # Try to build the module
        try:
import subprocess result = subprocess.run([ 'g++', '-shared', '-fPIC', '-O3', '-march=native', '-flto', '-fno-exceptions', '-fno-rtti', '-funroll-loops', 'core/enhanced_rft_crypto_bindings.cpp', '-o', 'enhanced_rft_crypto.so', '-I/usr/include/python3.10',

        # Adjust for your Python version '-lpython3.10' ], capture_output=True, text=True, cwd='/workspaces/quantoniumos')
        if result.returncode == 0:
import enhanced_rft_crypto cipher = enhanced_rft_crypto.EnhancedRFTCrypto()
        print("✓ Successfully built and loaded native module")
        else:
        raise RuntimeError(f"Build failed: {result.stderr}") except Exception as build_error:
        print(f"✗ Build failed: {build_error}")
        print("Falling back to manual test...")
        return key = os.urandom(32)

        # Test basic functionality plaintext = b"Hello, Enhanced RFT! This tests the optimized C++ engine."
        if len(plaintext) % 2 != 0: plaintext += b'\x00'

        # Pad to even length
        print(f"Original: {plaintext}") start_time = time.time() ciphertext = cipher.encrypt(plaintext, key) encrypt_time = time.time() - start_time
        print(f"Encrypted: {ciphertext.hex()[:64]}...")
        print(f"Encryption time: {encrypt_time:.6f}s") start_time = time.time() decrypted = cipher.decrypt(ciphertext, key) decrypt_time = time.time() - start_time
        print(f"Decrypted: {decrypted}")
        print(f"Decryption time: {decrypt_time:.6f}s")
        print(f"Perfect reversibility: {plaintext == decrypted}")

        # Test avalanche effect
        print("\nTesting avalanche effect...") avalanche = test_avalanche(cipher, key)
        print(f"Average avalanche effect: {avalanche:.3f}")

        # Test key sensitivity
        print("\nTesting key sensitivity...") key_sensitivity = test_key_sensitivity(cipher, 32, 5)
        print(f"Key sensitivity: {key_sensitivity:.3f}")

        # Performance test with larger data
        print("\nPerformance test...") test_data = os.urandom(1024 * 10) # 10KB
        if len(test_data) % 2 != 0: test_data += b'\x00' start_time = time.time() iterations = 100
        for _ in range(iterations): cipher.encrypt(test_data, key) total_time = time.time() - start_time throughput = (len(test_data) * iterations) / total_time / 1024

        # KB/s
        print(f"Throughput: {throughput:.2f} KB/s ({len(test_data) * iterations} bytes in {total_time:.3f}s)")

        # Batch performance test
        if available
        try:
        print("\nBatch performance test...") batch_data = [os.urandom(1024) + b'\x00'
        if os.urandom(1024).__len__() % 2 != 0 else os.urandom(1024)
        for _ in range(50)] start_time = time.time() batch_results = cipher.encrypt_batch(batch_data, key) batch_time = time.time() - start_time batch_throughput = (sum(len(d)
        for d in batch_data)) / batch_time / 1024
        print(f"Batch throughput: {batch_throughput:.2f} KB/s")
        except AttributeError:
        print("Batch encryption not available in this build")

        # Analysis
        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS:")
        print(f"✓ Reversibility: {'PASS'
        if plaintext == decrypted else 'FAIL'}")
        print(f"✓ Avalanche: {'EXCELLENT'
        if avalanche > 0.45 else 'GOOD'
        if avalanche > 0.35 else 'NEEDS WORK'} ({avalanche:.3f})")
        print(f"✓ Key Sensitivity: {'EXCELLENT'
        if key_sensitivity > 0.9 else 'GOOD'
        if key_sensitivity > 0.7 else 'NEEDS WORK'} ({key_sensitivity:.3f})")
        print(f"✓ Throughput: {throughput:.1f} KB/s") target_met = avalanche > 0.45 and key_sensitivity > 0.9 and throughput > 25
        print(f"\n ALL TARGETS MET: {'YES'
        if target_met else 'NO'}")
        if target_met:
        print(" Excellent avalanche (~0.50), key sensitivity (~0.95), and performance (>25 KB/s)!")
        else:
        if throughput < 25:
        print(" ⚠️ Performance target not met - check
        if native compilation succeeded")
        else:
        print(" ⚠️ Cryptographic targets not met - algorithm needs adjustment")
def test_avalanche(cipher, key, test_size=32, num_tests=10): """
        Test avalanche effect with single bit changes
"""
        total_avalanche = 0
        for _ in range(num_tests):

        # Create test data plaintext1 = os.urandom(test_size)
        if len(plaintext1) % 2 != 0: plaintext1 += b'\x00' plaintext2 = bytearray(plaintext1)

        # Flip one bit bit_pos = 0 plaintext2[bit_pos // 8] ^= (1 << (bit_pos % 8)) plaintext2 = bytes(plaintext2)

        # Encrypt both ciphertext1 = cipher.encrypt(plaintext1, key) ciphertext2 = cipher.encrypt(plaintext2, key)

        # Count bit differences bit_diff = sum(bin(a ^ b).count('1') for a, b in zip(ciphertext1, ciphertext2)) total_bits = len(ciphertext1) * 8 avalanche = bit_diff / total_bits total_avalanche += avalanche
        return total_avalanche / num_tests
def test_key_sensitivity(cipher, test_size=32, num_tests=5): """
        Test key sensitivity - decryption with wrong key should fail completely
"""
        total_sensitivity = 0
        for _ in range(num_tests): plaintext = os.urandom(test_size)
        if len(plaintext) % 2 != 0: plaintext += b'\x00' key1 = os.urandom(32) key2 = bytearray(key1)

        # Flip one bit in key key2[0] ^= 1 key2 = bytes(key2)

        # Encrypt with first key ciphertext = cipher.encrypt(plaintext, key1)

        # Try to decrypt with wrong key
        try: wrong_decrypted = cipher.decrypt(ciphertext, key2)

        # Count how many bytes are different from original different_bytes = sum(a != b for a, b in zip(plaintext, wrong_decrypted)) sensitivity = different_bytes / len(plaintext) total_sensitivity += sensitivity
        except:

        # Complete failure is perfect sensitivity total_sensitivity += 1.0
        return total_sensitivity / num_tests

if __name__ == "__main__": main()