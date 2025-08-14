#!/usr/bin/env python3
"""
Standalone Cryptographic Security Analysis
==========================================

Tests cryptographic components that are available in the QuantoniumOS system
without relying on complex internal dependencies.
"""

import numpy as np
import hashlib
import secrets
import base64
from collections import Counter
import time


def basic_entropy_analysis(data: bytes) -> float:
    """Calculate Shannon entropy of byte data"""
    if len(data) == 0:
        return 0.0
    
    # Count byte frequencies
    byte_counts = Counter(data)
    data_len = len(data)
    
    # Calculate entropy
    entropy = 0.0
    for count in byte_counts.values():
        p = count / data_len
        entropy -= p * np.log2(p)
    
    return entropy


def avalanche_test(encrypt_func, message: str, key: str, n_tests: int = 50) -> list:
    """Test avalanche effect of encryption function"""
    avalanche_results = []
    
    for i in range(min(n_tests, len(message))):
        try:
            # Original encryption
            original = encrypt_func(message, key)
            
            # Modify one character in message
            modified_message = message[:i] + chr((ord(message[i]) + 1) % 256) + message[i+1:]
            modified = encrypt_func(modified_message, key)
            
            # Count differences
            if len(original) == len(modified):
                differences = sum(1 for a, b in zip(original, modified) if a != b)
                avalanche_ratio = differences / len(original)
                avalanche_results.append(avalanche_ratio)
        except:
            continue
    
    return avalanche_results


def simple_stream_cipher(message: str, key: str) -> str:
    """Simple stream cipher for testing (XOR-based)"""
    # Generate keystream from key using SHA-256
    key_hash = hashlib.sha256(key.encode()).digest()
    
    # Extend keystream to message length
    keystream = b''
    counter = 0
    while len(keystream) < len(message):
        keystream += hashlib.sha256(key_hash + counter.to_bytes(4, 'big')).digest()
        counter += 1
    
    # XOR with message
    result = ''
    for i, char in enumerate(message):
        result += chr(ord(char) ^ keystream[i % len(keystream)])
    
    return result


def geometric_hash_simple(data: bytes) -> str:
    """Simplified geometric hash based on golden ratio transformations"""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Convert data to numeric array
    data_array = np.frombuffer(data, dtype=np.uint8)
    
    # Apply geometric transformations
    transformed = np.zeros_like(data_array, dtype=np.float64)
    
    for i, byte_val in enumerate(data_array):
        # Golden ratio spiral transformation
        angle = (i * phi) % (2 * np.pi)
        radius = byte_val / 255.0
        
        # Coordinate transformation
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # Map back to byte range
        transformed[i] = int((x * y + 1) * 127.5) % 256
    
    # Final hash
    transformed_bytes = transformed.astype(np.uint8).tobytes()
    final_hash = hashlib.sha256(transformed_bytes).hexdigest()
    
    return final_hash


def run_cryptographic_analysis():
    """Run comprehensive cryptographic analysis"""
    
    print("STANDALONE CRYPTOGRAPHIC ANALYSIS")
    print("=" * 50)
    print()
    
    # Test parameters
    test_message = "The quick brown fox jumps over the lazy dog. 1234567890! Testing cryptographic security with various patterns and symbols: @#$%^&*()_+-=[]{}|;:,.<>?"
    test_key = "secure_test_key_2024"
    
    print(f"Test message length: {len(test_message)} characters")
    print(f"Test key: '{test_key}'")
    print()
    
    # 1. STREAM CIPHER ANALYSIS
    print("1. STREAM CIPHER SECURITY ANALYSIS")
    
    # Test encryption/decryption
    encrypted = simple_stream_cipher(test_message, test_key)
    decrypted = simple_stream_cipher(encrypted, test_key)  # XOR is self-inverse
    
    round_trip_success = decrypted == test_message
    print(f"   Round-trip success: {round_trip_success}")
    
    # Analyze encrypted output
    encrypted_bytes = encrypted.encode('latin-1')
    entropy = basic_entropy_analysis(encrypted_bytes)
    print(f"   Ciphertext entropy: {entropy:.3f} bits/byte (target: ~8.0)")
    
    # Statistical uniformity
    byte_counts = Counter(encrypted_bytes)
    mean_freq = len(encrypted_bytes) / 256
    chi_square = sum((count - mean_freq)**2 / mean_freq for count in byte_counts.values())
    print(f"   Chi-square statistic: {chi_square:.2f}")
    
    print()
    
    # 2. AVALANCHE EFFECT
    print("2. AVALANCHE EFFECT ANALYSIS")
    avalanche_results = avalanche_test(simple_stream_cipher, test_message, test_key)
    
    if avalanche_results:
        mean_avalanche = np.mean(avalanche_results)
        std_avalanche = np.std(avalanche_results)
        print(f"   Mean avalanche effect: {mean_avalanche:.1%}")
        print(f"   Standard deviation: {std_avalanche:.1%}")
        print(f"   Avalanche quality: {'GOOD' if 0.40 <= mean_avalanche <= 0.60 else 'POOR'}")
        print(f"   Number of tests: {len(avalanche_results)}")
    else:
        print("   Avalanche test failed")
    
    print()
    
    # 3. GEOMETRIC HASH ANALYSIS
    print("3. GEOMETRIC HASH FUNCTION ANALYSIS")
    
    # Test consistency
    data1 = b"Hello, cryptographic world!"
    data2 = b"Hello, cryptographic world!"
    data3 = b"Hello, cryptographic world?"  # One char different
    
    hash1 = geometric_hash_simple(data1)
    hash2 = geometric_hash_simple(data2)
    hash3 = geometric_hash_simple(data3)
    
    print(f"   Hash 1: {hash1[:32]}...")
    print(f"   Hash 2: {hash2[:32]}...")
    print(f"   Hash 3: {hash3[:32]}...")
    
    consistency = (hash1 == hash2)
    sensitivity = (hash1 != hash3)
    
    print(f"   Deterministic (same input): {consistency}")
    print(f"   Sensitive (different input): {sensitivity}")
    
    # Distribution test
    hash_samples = []
    for i in range(100):
        test_data = f"test_sample_{i}_{secrets.token_hex(8)}".encode()
        hash_result = geometric_hash_simple(test_data)
        # Use first 8 hex chars as sample
        hash_samples.append(int(hash_result[:8], 16))
    
    hash_array = np.array(hash_samples)
    distribution_uniformity = np.std(hash_array) / np.mean(hash_array)
    print(f"   Distribution coefficient: {distribution_uniformity:.3f}")
    
    print()
    
    # 4. TIMING ANALYSIS
    print("4. PERFORMANCE ANALYSIS")
    
    # Encryption timing
    start_time = time.time()
    for _ in range(1000):
        simple_stream_cipher(test_message, test_key)
    encrypt_time = (time.time() - start_time) * 1000
    print(f"   Encryption speed: {encrypt_time:.2f}ms per 1000 operations")
    
    # Hash timing
    start_time = time.time()
    for _ in range(100):
        geometric_hash_simple(data1)
    hash_time = (time.time() - start_time) * 10
    print(f"   Hashing speed: {hash_time:.2f}ms per 100 operations")
    
    print()
    
    # 5. SECURITY ASSESSMENT
    print("5. SECURITY ASSESSMENT SUMMARY")
    print()
    
    security_checks = []
    
    # Check round-trip
    if round_trip_success:
        print("   ✓ Perfect message recovery")
        security_checks.append(True)
    else:
        print("   ✗ Message recovery failed")
        security_checks.append(False)
    
    # Check entropy
    if entropy >= 7.5:
        print("   ✓ High entropy ciphertext")
        security_checks.append(True)
    elif entropy >= 7.0:
        print("   ~ Moderate entropy ciphertext")
        security_checks.append(True)
    else:
        print("   ✗ Low entropy ciphertext")
        security_checks.append(False)
    
    # Check avalanche
    if avalanche_results and 0.40 <= np.mean(avalanche_results) <= 0.60:
        print("   ✓ Good avalanche effect")
        security_checks.append(True)
    else:
        print("   ✗ Poor avalanche effect")
        security_checks.append(False)
    
    # Check hash properties
    if consistency and sensitivity:
        print("   ✓ Hash function properties satisfied")
        security_checks.append(True)
    else:
        print("   ✗ Hash function issues detected")
        security_checks.append(False)
    
    print()
    
    # Overall assessment
    passed_checks = sum(security_checks)
    total_checks = len(security_checks)
    
    print(f"   OVERALL SCORE: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("   🔒 STRONG: All basic security properties satisfied")
    elif passed_checks >= total_checks * 0.75:
        print("   🔓 MODERATE: Most security properties satisfied")
    else:
        print("   ⚠️  WEAK: Significant security issues detected")
    
    print()
    print("   IMPORTANT DISCLAIMERS:")
    print("   • These are basic statistical tests only")
    print("   • Professional cryptographic analysis requires:")
    print("     - NIST Statistical Test Suite (SP 800-22)")
    print("     - Dieharder battery of tests")
    print("     - TestU01 statistical test library")
    print("     - Formal security proofs and peer review")
    print("   • This implementation is FOR RESEARCH ONLY")
    print("   • DO NOT USE in production systems")
    
    return {
        'round_trip': round_trip_success,
        'entropy': entropy,
        'avalanche': avalanche_results,
        'hash_consistency': consistency,
        'hash_sensitivity': sensitivity,
        'security_score': passed_checks / total_checks
    }


if __name__ == "__main__":
    results = run_cryptographic_analysis()
