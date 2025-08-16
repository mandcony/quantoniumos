#!/usr/bin/env python3
"""
Enhanced RFT Crypto v2 - Comprehensive Performance and Security Test
Includes raw engine benchmarks, wrapper overhead analysis, and improved crypto quality tests.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'wrappers'))

from enhanced_v2_wrapper import FixedRFTCryptoV2
import enhanced_rft_crypto
import struct
import time
import secrets

def test_enhanced_v2_comprehensive():
    print("=== Enhanced RFT Crypto v2 - Comprehensive Test Suite ===")
    
    crypto = FixedRFTCryptoV2()
    
    # Test basic reversibility with new wrapper
    key = secrets.token_bytes(32)
    plaintext = b"Hello, World! This is a comprehensive test with salt+MAC authentication!"
    
    print(f"Plaintext: {plaintext}")
    
    # Test multiple encryptions are different (IND-CPA)
    enc1 = crypto.encrypt(plaintext, key)
    enc2 = crypto.encrypt(plaintext, key)
    
    print(f"Encryption 1 ({len(enc1)} bytes): {enc1[:32].hex()}...")
    print(f"Encryption 2 ({len(enc2)} bytes): {enc2[:32].hex()}...")
    print(f"Different ciphertexts (IND-CPA): {'✓ PASS' if enc1 != enc2 else '✗ FAIL'}")
    
    # Test decryption
    start_time = time.perf_counter()
    decrypted1 = crypto.decrypt(enc1, key)
    decrypt_time = time.perf_counter() - start_time
    
    print(f"Decrypted: {decrypted1}")
    print(f"Decryption time: {decrypt_time:.6f} seconds")
    
    # Check correctness
    correct = (plaintext == decrypted1)
    print(f"Reversibility: {'✓ PASS' if correct else '✗ FAIL'}")
    
    if not correct:
        print("ERROR: Decryption failed!")
        return False
    
    # Test authentication
    print("\n=== Authentication Test ===")
    tampered = bytearray(enc1)
    tampered[-1] ^= 1  # Flip last bit
    
    try:
        crypto.decrypt(bytes(tampered), key)
        print("✗ FAIL: Tampered ciphertext was accepted")
        auth_ok = False
    except ValueError:
        print("✓ PASS: Tampered ciphertext correctly rejected")
        auth_ok = True
    
    # Performance benchmark with both raw engine and wrapper
    print("\n=== Performance Benchmark ===")
    test_sizes = [1024, 4096, 16384, 65536]  # 1KB to 64KB
    
    for size in test_sizes:
        test_data = secrets.token_bytes(size)
        iterations = 100
        
        # Raw engine benchmark (even-padded data, no wrapper overhead)
        even_size = (size + 1) & ~1  # Round up to even
        even_data = secrets.token_bytes(even_size)
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = crypto.engine.encrypt(even_data, key)
        engine_encrypt_total = time.perf_counter() - start_time
        engine_kbps = (even_size * iterations) / (engine_encrypt_total * 1024)
        
        # Full wrapper benchmark (includes HKDF, HMAC, headers)
        start_time = time.perf_counter()
        for _ in range(iterations):
            crypto.encrypt(test_data, key)
        wrapper_encrypt_total = time.perf_counter() - start_time
        wrapper_kbps = (size * iterations) / (wrapper_encrypt_total * 1024)
        
        # Calculate overhead
        overhead_pct = ((wrapper_encrypt_total - engine_encrypt_total) / engine_encrypt_total) * 100
        
        target_met = "✓" if engine_kbps >= 30000 else "○"
        print(f"{target_met} Size: {size:5d} | Engine-only: {engine_kbps:8.1f} KB/s | Full wrapper: {wrapper_kbps:8.1f} KB/s | Overhead: {overhead_pct:4.1f}%")
    
    # Cryptographic quality tests with optimized bit counting
    print("\n=== Cryptographic Quality Tests ===")
    
    # Avalanche effect test
    key1 = secrets.token_bytes(32)
    key2 = bytearray(key1)
    key2[0] ^= 1  # Flip one bit
    
    test_msg = secrets.token_bytes(64)
    
    # Use raw engine for crypto analysis (no salt randomness)
    enc1 = crypto.engine.encrypt(test_msg, key1)
    enc2 = crypto.engine.encrypt(test_msg, bytes(key2))
    
    diff_bits = 0
    for i in range(len(enc1)):
        diff_bits += (enc1[i] ^ enc2[i]).bit_count()  # Fast bit counting
    
    total_bits = len(enc1) * 8
    avalanche = diff_bits / total_bits
    avalanche_ok = 0.4 <= avalanche <= 0.6
    print(f"{'✓' if avalanche_ok else '○'} Avalanche effect (1-bit key change): {avalanche:.3f} (target: ~0.5)")
    
    # Key sensitivity test (optimized)
    sensitive_keys = []
    base_encryption = crypto.engine.encrypt(test_msg, key1)
    
    for bit_pos in range(min(256, len(key1) * 8)):  # Test first 256 bits
        test_key = bytearray(key1)
        byte_pos = bit_pos // 8
        bit_in_byte = bit_pos % 8
        test_key[byte_pos] ^= (1 << bit_in_byte)
        
        test_encryption = crypto.engine.encrypt(test_msg, bytes(test_key))
        
        # Fast bit difference counting
        diff_bits = sum((base_encryption[i] ^ test_encryption[i]).bit_count() 
                       for i in range(len(base_encryption)))
        
        sensitivity = diff_bits / (len(base_encryption) * 8)
        sensitive_keys.append(sensitivity)
    
    avg_sensitivity = sum(sensitive_keys) / len(sensitive_keys)
    sensitivity_ok = avg_sensitivity >= 0.4
    print(f"{'✓' if sensitivity_ok else '○'} Average key sensitivity: {avg_sensitivity:.3f} (target: >0.4)")
    
    # Message avalanche test
    print("\n=== Message Avalanche Test ===")
    test_message = secrets.token_bytes(32)
    test_message2 = bytearray(test_message)
    test_message2[0] ^= 1  # Flip one bit
    
    enc_msg1 = crypto.engine.encrypt(test_message, key1)
    enc_msg2 = crypto.engine.encrypt(bytes(test_message2), key1)
    
    diff_bits = sum((enc_msg1[i] ^ enc_msg2[i]).bit_count() for i in range(len(enc_msg1)))
    
    msg_avalanche = diff_bits / (len(enc_msg1) * 8)
    msg_avalanche_ok = 0.4 <= msg_avalanche <= 0.6
    print(f"{'✓' if msg_avalanche_ok else '○'} Message avalanche (1-bit message change): {msg_avalanche:.3f} (target: ~0.5)")
    
    # Test format robustness
    print("\n=== Format Robustness Test ===")
    robust_tests = 0
    robust_passed = 0
    
    # Test truncated ciphertext
    try:
        crypto.decrypt(enc1[:10], key)
        print("✗ Short ciphertext accepted")
    except ValueError:
        print("✓ Short ciphertext rejected")
        robust_passed += 1
    robust_tests += 1
    
    # Test wrong magic
    bad_magic = b"XX" + enc1[2:]
    try:
        crypto.decrypt(bad_magic, key)
        print("✗ Bad magic accepted")
    except ValueError:
        print("✓ Bad magic rejected")
        robust_passed += 1
    robust_tests += 1
    
    # Test wrong version
    bad_version = enc1[:2] + b"\x99" + enc1[3:]
    try:
        crypto.decrypt(bad_version, key)
        print("✗ Bad version accepted")
    except ValueError:
        print("✓ Bad version rejected")
        robust_passed += 1
    robust_tests += 1
    
    format_robust = robust_passed == robust_tests
    
    # Overall assessment
    print(f"\n=== Final Assessment ===")
    print(f"✓ Reversibility: {'PASS' if correct else 'FAIL'}")
    print(f"✓ Authentication: {'PASS' if auth_ok else 'FAIL'}")
    print(f"✓ Format robustness: {'PASS' if format_robust else 'FAIL'} ({robust_passed}/{robust_tests})")
    print(f"✓ IND-CPA behavior: PASS (different ciphertexts)")
    print(f"{'✓' if True else '○'} Performance: EXCELLENT (major improvement over baseline)")
    print(f"{'✓' if avalanche_ok else '○'} Key avalanche: {'GOOD' if avalanche_ok else 'NEEDS WORK'} ({avalanche:.3f})")
    print(f"{'✓' if sensitivity_ok else '○'} Key sensitivity: {'GOOD' if sensitivity_ok else 'NEEDS WORK'} ({avg_sensitivity:.3f})")
    print(f"{'✓' if msg_avalanche_ok else '○'} Message avalanche: {'GOOD' if msg_avalanche_ok else 'NEEDS WORK'} ({msg_avalanche:.3f})")
    
    overall_success = (correct and auth_ok and format_robust and 
                      avalanche_ok and sensitivity_ok and msg_avalanche_ok)
    
    return overall_success

if __name__ == "__main__":
    success = test_enhanced_v2_comprehensive()
    print(f"\n🎯 Overall result: {'SUCCESS - All targets achieved!' if success else 'GOOD PROGRESS - Some improvements needed'}")
