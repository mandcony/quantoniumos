#!/usr/bin/env python3
"""
Quick test of assembly-optimized crypto performance
"""

import os
import sys
import time
import secrets

# Set library path
os.environ['LD_LIBRARY_PATH'] = '/workspaces/quantoniumos/ASSEMBLY/compiled'

# Add paths
sys.path.append('./ASSEMBLY/python_bindings')
sys.path.append('./apps')

try:
    from enhanced_rft_crypto import EnhancedRFTCrypto
    
    print("Testing assembly-optimized crypto performance...")
    
    # Initialize
    crypto = EnhancedRFTCrypto(8)
    password = b"test_password_12345"
    
    # Test data
    test_data = secrets.token_bytes(1024)
    
    print(f"Testing with {len(test_data)} bytes...")
    
    # Encrypt
    start_time = time.perf_counter()
    encrypted = crypto.encrypt(test_data, password)
    encrypt_time = time.perf_counter() - start_time
    
    # Decrypt
    start_time = time.perf_counter()
    decrypted = crypto.decrypt(encrypted, password)
    decrypt_time = time.perf_counter() - start_time
    
    # Calculate throughput
    encrypt_mbps = (len(test_data) / encrypt_time) / (1024 * 1024)
    decrypt_mbps = (len(test_data) / decrypt_time) / (1024 * 1024)
    
    print(f"Encrypt: {encrypt_time:.4f}s ({encrypt_mbps:.3f} MB/s)")
    print(f"Decrypt: {decrypt_time:.4f}s ({decrypt_mbps:.3f} MB/s)")
    
    # Verify correctness
    if test_data == decrypted:
        print("✓ Encryption/decryption successful")
        
        # Check if we hit the paper target
        if encrypt_mbps >= 9.0:
            print("✓ PAPER TARGET ACHIEVED!")
        else:
            print(f"Performance gap: {9.2 - encrypt_mbps:.3f} MB/s below target")
    else:
        print("✗ Encryption/decryption failed")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
