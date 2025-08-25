# -*- coding: utf-8 -*-
#
# QuantoniumOS RFT (Resonant Frequency Transform) Tests
# Testing with QuantoniumOS RFT implementations
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

# Import QuantoniumOS quantum engines for RFT-quantum integration
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

# Import QuantoniumOS cryptography modules for RFT-crypto integration
try:
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
except ImportError as e:
    print(f"Warning: Could not import cryptography modules: {e}")

"""
Basic test for paper_compliant_rft_fixed.py
"""

import sys
import os
from importlib import import_module

# Add the root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module
paper_compliant_rft_fixed = import_module("04_RFT_ALGORITHMS.paper_compliant_rft_fixed")

def test_simple_encryption():
    """Test basic encryption and decryption roundtrip"""
    print('\nTesting encryption/decryption roundtrip...')
    crypto = paper_compliant_rft_fixed.PaperCompliantRFT()
    crypto.init_engine()
    
    test_data = b'This is a test message for the RFT encryption algorithm'
    key = b'SecretKey123'

    encrypted = crypto.encrypt_block(test_data, key)
    decrypted = crypto.decrypt_block(encrypted, key)

    print(f'Original:  {test_data}')
    print(f'Decrypted: {decrypted}')
    print(f'Roundtrip Successful: {test_data == decrypted}')

if __name__ == "__main__":
    print("Running Basic RFT Test\n")
    test_simple_encryption()
    print("\nTest completed.")
