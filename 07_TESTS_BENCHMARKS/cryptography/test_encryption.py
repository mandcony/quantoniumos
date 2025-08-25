# -*- coding: utf-8 -*-
#
# QuantoniumOS Cryptography Tests
# Testing with QuantoniumOS crypto implementations
#
# ===================================================================

import unittest
import sys
import os
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS cryptography modules
try:
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
    from paper_compliant_crypto_bindings import PaperCompliantCrypto
except ImportError:
    # Fallback imports if modules are in different locations
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError:
    pass

# Import QuantoniumOS quantum engines
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from working_quantum_kernel import WorkingQuantumKernel
except ImportError:
    pass

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
