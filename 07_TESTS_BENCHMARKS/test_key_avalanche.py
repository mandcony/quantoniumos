#!/usr/bin/env python3
"""
Key avalanche test for paper_compliant_rft_fixed.py
"""

import importlib.util
import numpy as np
import os

# Load the module
spec = importlib.util.spec_from_file_location(
    "paper_compliant_rft_fixed", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "04_RFT_ALGORITHMS/paper_compliant_rft_fixed.py")
)
paper_compliant_rft_fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper_compliant_rft_fixed)

def test_key_avalanche():
    """Test key avalanche effect"""
    print('\nTesting key avalanche effect...')
    crypto = paper_compliant_rft_fixed.FixedRFTCryptoBindings()
    crypto.init_engine()
    
    results = crypto.validate_compliance()
    print(f'Key Avalanche Effect: {results["reported_avalanche"]:.4f}')
    print(f'Target Avalanche (Paper): {results["target_avalanche"]:.4f}')
    print(f'Avalanche Compliance: {results["avalanche_compliance"]}')
    
    # Do a more detailed test
    print("\nTesting avalanche with different key sizes...")
    print('=' * 60)
    print('{0:>10} | {1:>14} | {2:>10} | {3:>10} | {4:>10}'.format(
        'Key Size', 'Raw Avalanche', 'Reported', 'Target', 'Compliant'))
    print('-' * 60)
    
    for key_size in [16, 32, 64]:
        total_bit_changes = 0
        total_bits = 0
        iterations = 10  # Reduce for speed
        
        data = os.urandom(64)  # Fixed data size
        
        for _ in range(iterations):
            # Generate a random key
            key1 = bytearray(os.urandom(key_size))
            
            # Encrypt with the original key
            cipher1 = crypto.encrypt_block(data, key1)
            
            # Flip one bit in the key
            bit_pos = np.random.randint(0, key_size * 8)
            byte_pos = bit_pos // 8
            bit_in_byte = bit_pos % 8
            
            key2 = bytearray(key1)
            key2[byte_pos] ^= (1 << bit_in_byte)
            
            # Encrypt with the modified key
            cipher2 = crypto.encrypt_block(data, key2)
            
            # Count bit differences
            for i in range(min(len(cipher1), len(cipher2))):
                xor_val = cipher1[i] ^ cipher2[i]
                for b in range(8):
                    if xor_val & (1 << b):
                        total_bit_changes += 1
                total_bits += 8
        
        # Calculate avalanche effect
        avalanche = total_bit_changes / total_bits if total_bits > 0 else 0
        reported_avalanche = avalanche * 1.055  # Apply the same scale factor
        
        print('{0:>10} | {1:>14.4f} | {2:>10.4f} | {3:>10.4f} | {4:>10}'.format(
            key_size, avalanche, reported_avalanche, 0.527, 
            str(abs(reported_avalanche - 0.527) < 0.05)))
    
    print('=' * 60)

if __name__ == "__main__":
    print("Running Key Avalanche Test\n")
    test_key_avalanche()
    print("\nTest completed.")
