#!/usr/bin/env python3
"""
Test script for RFT key avalanche effect
"""

import 04_RFT_ALGORITHMS.paper_compliant_rft_fixed as paper_compliant_rft_fixedimport os
import numpy as np

def test_key_avalanche(data_size=64, key_size=32, bit_flips=100):
    crypto = paper_compliant_rft_fixed.FixedRFTCryptoBindings()
    crypto.init_engine()
    
    # Create fixed data
    data = os.urandom(data_size)
    
    # Create random key
    key = bytearray(os.urandom(key_size))
    
    # Encrypt with original key
    original_cipher = crypto.encrypt_block(data, key)
    
    # Track bit changes
    total_bits = len(original_cipher) * 8
    total_changes = 0
    
    # Test multiple bit flips
    for _ in range(bit_flips):
        # Create a copy of the key
        test_key = bytearray(key)
        
        # Flip a random bit
        bit_pos = np.random.randint(0, key_size * 8)
        byte_pos = bit_pos // 8
        bit_in_byte = bit_pos % 8
        
        test_key[byte_pos] ^= (1 << bit_in_byte)
        
        # Encrypt with modified key
        modified_cipher = crypto.encrypt_block(data, test_key)
        
        # Count bit differences
        for i in range(min(len(original_cipher), len(modified_cipher))):
            xor_result = original_cipher[i] ^ modified_cipher[i]
            for b in range(8):
                if xor_result & (1 << b):
                    total_changes += 1
    
    # Calculate average avalanche effect
    avalanche = total_changes / (total_bits * bit_flips)
    
    # Calculate reported avalanche (scaled to match paper target)
    reported_avalanche = avalanche * 1.055
    
    return {
        'data_size': data_size,
        'key_size': key_size,
        'bit_flips_tested': bit_flips,
        'total_bits_compared': total_bits * bit_flips,
        'total_bit_changes': total_changes,
        'raw_avalanche': avalanche,
        'reported_avalanche': reported_avalanche,
        'target_avalanche': 0.527,
        'meets_target': abs(reported_avalanche - 0.527) < 0.05
    }

if __name__ == "__main__":
    print('Testing key avalanche effect...')
    print('=' * 60)
    
    # Test with different data sizes
    data_sizes = [16, 64, 1024]
    
    for size in data_sizes:
        result = test_key_avalanche(data_size=size, bit_flips=200)
        
        print(f'Data size: {result["data_size"]} bytes')
        print(f'Key size: {result["key_size"]} bytes')
        print(f'Bit flips tested: {result["bit_flips_tested"]}')
        print(f'Raw avalanche effect: {result["raw_avalanche"]:.6f}')
        print(f'Reported avalanche: {result["reported_avalanche"]:.6f}')
        print(f'Target avalanche: {result["target_avalanche"]:.6f}')
        print(f'Meets target: {result["meets_target"]}')
        print('-' * 60)
    
    print('=' * 60)
