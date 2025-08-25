# -*- coding: utf-8 -*-
#
# QuantoniumOS Test Suite
# Testing with QuantoniumOS implementations
#
# ===================================================================

import unittest
import sys
import os
import numpy as np
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError as e:
    print(f"Warning: Could not import RFT algorithms: {e}")

# Import QuantoniumOS quantum engines
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from topological_vertex_engine import TopologicalVertexEngine
    from topological_vertex_geometric_engine import TopologicalVertexGeometricEngine
    from vertex_engine_canonical import VertexEngineCanonical
    from working_quantum_kernel import WorkingQuantumKernel
    from true_rft_engine_bindings import TrueRFTEngineBindings as QuantumRFTBindings
except ImportError as e:
    print(f"Warning: Could not import quantum engines: {e}")

# Import QuantoniumOS cryptography modules
try:
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
except ImportError as e:
    print(f"Warning: Could not import cryptography modules: {e}")

# Import QuantoniumOS validators
try:
    sys.path.insert(0, '/workspaces/quantoniumos/02_CORE_VALIDATORS')
    from basic_scientific_validator import BasicScientificValidator
    from definitive_quantum_validation import DefinitiveQuantumValidation
    from phd_level_scientific_validator import PhdLevelScientificValidator
    from publication_ready_validation import PublicationReadyValidation
except ImportError as e:
    print(f"Warning: Could not import validators: {e}")

# Import QuantoniumOS running systems
try:
    sys.path.insert(0, '/workspaces/quantoniumos/03_RUNNING_SYSTEMS')
    from app import app
    from main import main
    from quantonium import QuantoniumOS
except ImportError as e:
    print(f"Warning: Could not import running systems: {e}")

"""
Performance test for paper_compliant_rft_fixed.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '04_RFT_ALGORITHMS'))
import paper_compliant_rft_fixed as paper_compliant_rft_fixedimport os
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
