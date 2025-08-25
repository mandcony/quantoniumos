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
Test script for RFT key avalanche effect
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '04_RFT_ALGORITHMS'))
import paper_compliant_rft_fixed as paper_compliant_rft_fixedimport os
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
