#!/usr/bin/env python3
"""
Performance test for paper_compliant_rft_fixed.py
"""

import 04_RFT_ALGORITHMS.paper_compliant_rft_fixed as paper_compliant_rft_fixedimport os
import time

def test_performance():
    """Test performance with different data sizes"""
    print('\nTesting performance with different data sizes...')
    print('=' * 70)
    print('{0:>12} | {1:>12} | {2:>12} | {3:>12} | {4:>15}'.format(
        'Size (bytes)', 'Encrypt (ms)', 'Decrypt (ms)', 'KB/s (Enc)', 'Roundtrip Valid'))
    print('-' * 70)

    crypto = paper_compliant_rft_fixed.FixedRFTCryptoBindings()
    crypto.init_engine()
    
    for size in [16, 64, 256, 1024, 4096]:
        # Generate random data
        data = os.urandom(size)
        key = os.urandom(32)
        
        # Time encryption
        start_time = time.time()
        encrypted = crypto.encrypt_block(data, key)
        encryption_time = (time.time() - start_time) * 1000  # ms
        
        # Time decryption
        start_time = time.time()
        decrypted = crypto.decrypt_block(encrypted, key)
        decryption_time = (time.time() - start_time) * 1000  # ms
        
        # Calculate throughput
        throughput = size / (encryption_time / 1000) / 1024  # KB/s
        
        # Validate roundtrip
        valid = (data == decrypted)
        
        print('{0:>12} | {1:>12.2f} | {2:>12.2f} | {3:>12.2f} | {4:>15}'.format(
            size, encryption_time, decryption_time, throughput, str(valid)))
    
    print('=' * 70)

if __name__ == "__main__":
    print("Running Performance Test\n")
    test_performance()
    print("\nTest completed.")
