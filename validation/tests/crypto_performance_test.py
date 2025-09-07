#!/usr/bin/env python3
"""
Performance comparison between pure Python and assembly-optimized crypto
"""

import os
import sys
import time
import secrets

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'apps'))

# Import both implementations
print("Importing crypto implementations...")

# Pure Python implementation
try:
    from core.enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2
    print("✓ Successfully imported pure Python implementation")
except ImportError as e:
    print(f"✗ Failed to import pure Python implementation: {e}")
    EnhancedRFTCryptoV2 = None

# Don't try to import assembly version as it requires unitary_rft module
# We'll update the validation script instead to use the proper method names
EnhancedRFTCrypto = None

# Test parameters
KEY = secrets.token_bytes(32)
TEST_SIZES = [1024, 4096, 16384]  # Test with different message sizes
ITERATIONS = 3

def run_performance_test(implementation, name):
    """Run performance tests for a given crypto implementation"""
    print(f"\n{'-' * 60}")
    print(f"Testing {name} implementation")
    print(f"{'-' * 60}")
    
    for size in TEST_SIZES:
        # Generate random message
        message = secrets.token_bytes(size)
        
        # Initialize crypto
        crypto = implementation(KEY)
        
        # Different method names for different implementations
        if name == "Pure Python":
            encrypt_method = lambda msg: crypto.encrypt_aead(msg)
            decrypt_method = lambda enc: crypto.decrypt_aead(enc)
        else:
            encrypt_method = crypto.encrypt
            decrypt_method = crypto.decrypt
            
        # Warmup
        encrypted = encrypt_method(message)
        decrypted = decrypt_method(encrypted)
        assert message == decrypted, "Encryption/decryption failed"
        
        # Test encryption
        total_time = 0
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            encrypted = encrypt_method(message)
            end = time.perf_counter()
            total_time += (end - start)
        
        encrypt_time = total_time / ITERATIONS
        encrypt_mbps = (size / encrypt_time) / (1024 * 1024)
        
        # Test decryption
        total_time = 0
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            decrypted = decrypt_method(encrypted)
            end = time.perf_counter()
            total_time += (end - start)
        
        decrypt_time = total_time / ITERATIONS
        decrypt_mbps = (size / decrypt_time) / (1024 * 1024)
        
        print(f"{size:6d} bytes: {encrypt_mbps:.3f} MB/s encrypt, {decrypt_mbps:.3f} MB/s decrypt")
    
    print()

# Run tests
if __name__ == "__main__":
    print("\n=== Crypto Implementation Performance Comparison ===\n")
    
    # Test pure Python implementation
    if EnhancedRFTCryptoV2:
        run_performance_test(EnhancedRFTCryptoV2, "Pure Python")
    
    # Test assembly-optimized implementation
    if EnhancedRFTCrypto:
        run_performance_test(EnhancedRFTCrypto, "Assembly-optimized")
    
    print("Performance testing complete!")
