#!/usr/bin/env python3
"""
Test script for paper_compliant_rft_fixed.py
"""

import 04_RFT_ALGORITHMS.paper_compliant_rft_fixed as paper_compliant_rft_fixedimport numpy as np
import os
import time

def test_rft_unitarity():
    """Test the unitarity property of the RFT transform"""
    print('Testing RFT unitarity with different sizes...')
    print('=' * 70)
    print('{0:>6} | {1:>10} | {2:>10} | {3:>12} | {4:>8}'.format(
        'Size', 'Max Error', 'Mean Error', 'Power Ratio', 'Unitary'))
    print('-' * 70)

    for size in [16, 32, 64, 128]:
        rft = paper_compliant_rft_fixed.PaperCompliantRFT(size=size)

        # Generate random signal
        signal = np.random.random(size)

        # Apply transform
        result = rft.transform(signal)

        # Apply inverse transform
        inverse = rft.inverse_transform(result['transformed'])

        # Check if power is preserved (Parseval's theorem)
        input_power = np.sum(np.abs(signal)**2)
        output_power = np.sum(np.abs(result['transformed'])**2)
        power_ratio = output_power / input_power if input_power > 0 else 0

        # Check roundtrip error
        max_error = np.max(np.abs(signal - inverse['signal']))
        mean_error = np.mean(np.abs(signal - inverse['signal']))

        # Check if the transform is unitary
        # (Input and output should have same energy)
        is_unitary = np.isclose(input_power, output_power, rtol=1e-5)

        print('{0:>6} | {1:>10.4e} | {2:>10.4e} | {3:>12.6f} | {4:>8}'.format(
            size, max_error, mean_error, power_ratio, str(is_unitary)))

    print('=' * 70)

def test_encryption_roundtrip():
    """Test basic encryption and decryption roundtrip"""
    print('\nTesting encryption/decryption roundtrip...')
    crypto = paper_compliant_rft_fixed.FixedRFTCryptoBindings()
    crypto.init_engine()
    
    test_data = b'This is a test message for the RFT encryption algorithm'
    key = b'SecretKey123'

    encrypted = crypto.encrypt_block(test_data, key)
    decrypted = crypto.decrypt_block(encrypted, key)

    print(f'Original:  {test_data}')
    print(f'Decrypted: {decrypted}')
    print(f'Roundtrip Successful: {test_data == decrypted}')

def test_key_avalanche():
    """Test key avalanche effect"""
    print('\nTesting key avalanche effect...')
    crypto = paper_compliant_rft_fixed.FixedRFTCryptoBindings()
    crypto.init_engine()
    
    results = crypto.validate_compliance()
    print(f'Key Avalanche Effect: {results["reported_avalanche"]:.4f}')
    print(f'Target Avalanche (Paper): {results["target_avalanche"]:.4f}')
    print(f'Avalanche Compliance: {results["avalanche_compliance"]}')

def test_performance():
    """Test performance with different data sizes"""
    print('\nTesting performance with different data sizes...')
    print('=' * 60)
    print('{0:>12} | {1:>12} | {2:>12} | {3:>15}'.format(
        'Size (bytes)', 'Encrypt (ms)', 'Decrypt (ms)', 'Roundtrip Valid'))
    print('-' * 60)

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
        
        # Validate roundtrip
        valid = (data == decrypted)
        
        print('{0:>12} | {1:>12.2f} | {2:>12.2f} | {3:>15}'.format(
            size, encryption_time, decryption_time, str(valid)))
    
    print('=' * 60)

if __name__ == "__main__":
    print("Running RFT Implementation Tests\n")
    
    # Run all tests
    test_rft_unitarity()
    test_encryption_roundtrip()
    test_key_avalanche()
    test_performance()
    
    print("\nAll tests completed.")
