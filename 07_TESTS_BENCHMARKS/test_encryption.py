#!/usr/bin/env python3
"""
Test script for RFT encryption/decryption
"""

import importlib.util
import os
import json
import time

# Load the module
spec = importlib.util.spec_from_file_location(
    "paper_compliant_rft_fixed", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "04_RFT_ALGORITHMS/paper_compliant_rft_fixed.py")
)
paper_compliant_rft_fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper_compliant_rft_fixed)

def test_encryption_decryption():
    crypto = paper_compliant_rft_fixed.FixedRFTCryptoBindings()
    crypto.init_engine()
    
    # Test with different data types
    test_cases = [
        # Skip empty string as it's not valid for this encryption method
        ('Short string', 'Hello World'),
        ('Long string', 'This is a longer string that spans multiple blocks for encryption'),
        ('Special chars', '!@#$%^&*()_+-=[]{}|;:,.<>?/~`'),
        ('Binary data', os.urandom(100)),
        ('JSON', json.dumps({'name': 'test', 'value': 123, 'items': [1, 2, 3]})),
    ]
    
    results = []
    
    for name, data in test_cases:
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Generate a key
        key = os.urandom(16)
        
        start_time = time.time()
        encrypted = crypto.encrypt_block(data, key)
        encryption_time = time.time() - start_time
        
        start_time = time.time()
        decrypted = crypto.decrypt_block(encrypted, key)
        decryption_time = time.time() - start_time
        
        is_valid = (data == decrypted)
        
        results.append({
            'test_case': name,
            'data_size': len(data),
            'encryption_time_ms': encryption_time * 1000,
            'decryption_time_ms': decryption_time * 1000,
            'encrypted_size': len(encrypted),
            'roundtrip_valid': is_valid
        })
    
    return results

if __name__ == "__main__":
    # Run the test
    print('Testing encryption/decryption with various data types...')
    print('=' * 80)
    print('{0:25} | {1:>9} | {2:>11} | {3:>9} | {4:>9} | {5:>5}'.format(
        'Test Case', 'Size (B)', 'Enc Size (B)', 'Enc (ms)', 'Dec (ms)', 'Valid'))
    print('-' * 80)

    results = test_encryption_decryption()
    for r in results:
        print('{0:25} | {1:>9} | {2:>11} | {3:>9.2f} | {4:>9.2f} | {5:>5}'.format(
            r['test_case'], 
            r['data_size'], 
            r['encrypted_size'], 
            r['encryption_time_ms'], 
            r['decryption_time_ms'], 
            str(r['roundtrip_valid'])
        ))

    print('=' * 80)
