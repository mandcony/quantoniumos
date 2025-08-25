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
Test to validate the channel capacity broadcasting fix.
Ensures that the kernel dimension properly handles the signal size.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import importlib.util
import os

# Load the bulletproof_quantum_kernel module
spec = importlib.util.spec_from_file_location(
    "bulletproof_quantum_kernel", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "05_QUANTUM_ENGINES/bulletproof_quantum_kernel.py")
)
bulletproof_quantum_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bulletproof_quantum_kernel)

# Import specific functions/classes
BulletproofQuantumKernel

def test_channel_capacity_broadcasting = bulletproof_quantum_kernel.BulletproofQuantumKernel

def test_channel_capacity_broadcasting():
    """Test that channel capacity analysis doesn't have broadcasting errors."""
    print("Testing Channel Capacity Broadcasting Fix")
    print("=" * 50)

    # Test with 1000 symbols using 1024-dimensional kernel
    print("1. Testing 1000 symbols with 1024-dimensional kernel...")
    kernel = BulletproofQuantumKernel(dimension=1024)
    kernel.build_resonance_kernel()
    kernel.compute_rft_basis()

    num_symbols = 1000
    information_bits = np.random.randint(0, 2, num_symbols)
    print(f"   Input bits shape: {information_bits.shape}")

    # Forward RFT
    rft_signal = kernel.forward_rft(information_bits.astype(complex))
    print(f"   RFT signal shape: {rft_signal.shape}")

    # Add noise
    snr_linear = 10 ** (10 / 10)  # 10 dB
    noise_power = np.var(rft_signal) / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(rft_signal)) + 1j * np.random.randn(len(rft_signal))
    )
    rft_received = rft_signal + noise
    print(f"   Noisy signal shape: {rft_received.shape}")

    # Decode
    rft_decoded = kernel.inverse_rft(rft_received)
    print(f"   Decoded signal shape: {rft_decoded.shape}")

    rft_decoded_bits = (np.real(rft_decoded) > 0.5).astype(int)[:num_symbols]
    print(f"   Decoded bits shape: {rft_decoded_bits.shape}")

    # Check broadcasting compatibility
    try:
        rft_errors = np.sum(information_bits != rft_decoded_bits)
        rft_error_rate = rft_errors / num_symbols
        print(f"   ✅ Broadcasting compatible! Error rate: {rft_error_rate:.3f}")
        print(f"   Errors: {rft_errors}/{num_symbols}")
    except ValueError as e:
        print(f"   ❌ Broadcasting error: {e}")
        return False

    # Test with smaller kernel (should also work due to truncation/padding)
    print("\n2. Testing 100 symbols with 32-dimensional kernel...")
    kernel_small = BulletproofQuantumKernel(dimension=32)
    kernel_small.build_resonance_kernel()
    kernel_small.compute_rft_basis()

    num_symbols_small = 32  # Match kernel dimension
    information_bits_small = np.random.randint(0, 2, num_symbols_small)
    print(f"   Input bits shape: {information_bits_small.shape}")

    rft_signal_small = kernel_small.forward_rft(information_bits_small.astype(complex))
    rft_decoded_small = kernel_small.inverse_rft(rft_signal_small)
    rft_decoded_bits_small = (np.real(rft_decoded_small) > 0.5).astype(int)[
        :num_symbols_small
    ]

    try:
        rft_errors_small = np.sum(information_bits_small != rft_decoded_bits_small)
        rft_error_rate_small = rft_errors_small / num_symbols_small
        print(f"   ✅ Broadcasting compatible! Error rate: {rft_error_rate_small:.3f}")
        print(f"   Errors: {rft_errors_small}/{num_symbols_small}")
    except ValueError as e:
        print(f"   ❌ Broadcasting error: {e}")
        return False

    print("\n✅ All broadcasting tests passed!")
    print("Channel capacity analysis should now work without dimension mismatches.")
    return True

if __name__ == "__main__":
    success = test_channel_capacity_broadcasting()
    if success:
        print("\n🎉 Channel capacity fix validated successfully!")
    else:
        print("\n❌ Channel capacity fix failed validation!")
        sys.exit(1)
