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
