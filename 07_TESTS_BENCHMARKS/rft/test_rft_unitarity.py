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
Unitarity test for paper_compliant_rft_fixed.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '04_RFT_ALGORITHMS'))
import paper_compliant_rft_fixed as paper_compliant_rft_fixedimport numpy as np

def test_rft_unitarity():
    """Test the unitarity property of the RFT transform"""
    print('Testing RFT unitarity with different sizes...')
    print('=' * 70)
    print('{0:>6} | {1:>10} | {2:>10} | {3:>12} | {4:>8}'.format(
        'Size', 'Max Error', 'Mean Error', 'Power Ratio', 'Unitary'))
    print('-' * 70)

    for size in [16, 32, 64]:
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

if __name__ == "__main__":
    print("Running RFT Unitarity Test\n")
    test_rft_unitarity()
    print("\nTest completed.")
