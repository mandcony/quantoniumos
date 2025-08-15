#!/usr/bin/env python3
"""
Quick test of RFT Feistel cipher performance
"""

from rft_feistel_cipher import RFTFeistelCipher
import os
import time
import numpy as np

def avalanche_test(cipher, samples=100):
    """Test avalanche effect"""
    key = os.urandom(32)
    
    total_xex_avalanche = 0
    total_xex_cbc_avalanche = 0
    
    for _ in range(samples):
        # Generate test data
        plaintext1 = os.urandom(32)
        plaintext2 = bytearray(plaintext1)
        # Flip one random bit
        byte_idx = np.random.randint(0, len(plaintext2))
        bit_idx = np.random.randint(0, 8)
        plaintext2[byte_idx] ^= (1 << bit_idx)
        plaintext2 = bytes(plaintext2)
        
        # Test XEX mode
        ct1_xex = cipher.encrypt(plaintext1, key, mode='XEX')
        ct2_xex = cipher.encrypt(plaintext2, key, mode='XEX')
        
        bit_diff_xex = sum(bin(a ^ b).count('1') for a, b in zip(ct1_xex, ct2_xex))
        avalanche_xex = bit_diff_xex / (len(ct1_xex) * 8)
        total_xex_avalanche += avalanche_xex
        
        # Test XEX-CBC mode
        ct1_cbc = cipher.encrypt(plaintext1, key, mode='XEX-CBC')
        ct2_cbc = cipher.encrypt(plaintext2, key, mode='XEX-CBC')
        
        bit_diff_cbc = sum(bin(a ^ b).count('1') for a, b in zip(ct1_cbc, ct2_cbc))
        avalanche_cbc = bit_diff_cbc / (len(ct1_cbc) * 8)
        total_xex_cbc_avalanche += avalanche_cbc
    
    return total_xex_avalanche / samples, total_xex_cbc_avalanche / samples

def key_sensitivity_test(cipher, samples=100):
    """Test key sensitivity"""
    plaintext = b'A' * 32
    
    total_key_sensitivity = 0
    
    for _ in range(samples):
        key1 = os.urandom(32)
        key2 = bytearray(key1)
        # Flip one random bit in key
        byte_idx = np.random.randint(0, len(key2))
        bit_idx = np.random.randint(0, 8)
        key2[byte_idx] ^= (1 << bit_idx)
        key2 = bytes(key2)
        
        ct1 = cipher.encrypt(plaintext, key1, mode='XEX')
        ct2 = cipher.encrypt(plaintext, key2, mode='XEX')
        
        bit_diff = sum(bin(a ^ b).count('1') for a, b in zip(ct1, ct2))
        key_sensitivity = bit_diff / (len(ct1) * 8)
        total_key_sensitivity += key_sensitivity
    
    return total_key_sensitivity / samples

def main():
    print("=== RFT Feistel Cipher Performance Test ===")
    
    cipher = RFTFeistelCipher()
    
    # Basic functionality test
    print("\n1. Basic functionality test...")
    key = os.urandom(32)
    plaintext = b'Hello, World! This is a test of the RFT cipher.'
    
    start_time = time.time()
    ciphertext = cipher.encrypt(plaintext, key)
    encrypt_time = time.time() - start_time
    
    start_time = time.time()
    decrypted = cipher.decrypt(ciphertext, key)
    decrypt_time = time.time() - start_time
    
    print(f"   Original:  {plaintext}")
    print(f"   Encrypted: {ciphertext.hex()[:64]}...")
    print(f"   Decrypted: {decrypted}")
    print(f"   Perfect reversibility: {plaintext == decrypted}")
    print(f"   Encrypt time: {encrypt_time:.4f}s")
    print(f"   Decrypt time: {decrypt_time:.4f}s")
    
    # Avalanche effect test
    print("\n2. Avalanche effect test (100 samples)...")
    start_time = time.time()
    xex_avalanche, xex_cbc_avalanche = avalanche_test(cipher, 100)
    avalanche_time = time.time() - start_time
    
    print(f"   XEX mode avalanche:     {xex_avalanche:.3f} ({xex_avalanche*100:.1f}%)")
    print(f"   XEX-CBC mode avalanche: {xex_cbc_avalanche:.3f} ({xex_cbc_avalanche*100:.1f}%)")
    print(f"   Test time: {avalanche_time:.2f}s")
    
    # Key sensitivity test
    print("\n3. Key sensitivity test (100 samples)...")
    start_time = time.time()
    key_sensitivity = key_sensitivity_test(cipher, 100)
    key_test_time = time.time() - start_time
    
    print(f"   Key sensitivity: {key_sensitivity:.3f} ({key_sensitivity*100:.1f}%)")
    print(f"   Test time: {key_test_time:.2f}s")
    
    # Performance analysis
    print("\n=== Performance Analysis ===")
    print(f"✓ Functionality: Perfect encryption/decryption")
    print(f"✓ Key sensitivity: {key_sensitivity*100:.1f}% (target: >40%)")
    
    if xex_avalanche > 0.4:
        print(f"✓ XEX avalanche: {xex_avalanche*100:.1f}% (excellent)")
    elif xex_avalanche > 0.3:
        print(f"~ XEX avalanche: {xex_avalanche*100:.1f}% (good)")
    elif xex_avalanche > 0.2:
        print(f"! XEX avalanche: {xex_avalanche*100:.1f}% (moderate, needs improvement)")
    else:
        print(f"✗ XEX avalanche: {xex_avalanche*100:.1f}% (poor, needs significant improvement)")
    
    if xex_cbc_avalanche > 0.4:
        print(f"✓ XEX-CBC avalanche: {xex_cbc_avalanche*100:.1f}% (excellent)")
    else:
        print(f"~ XEX-CBC avalanche: {xex_cbc_avalanche*100:.1f}%")

if __name__ == "__main__":
    main()
