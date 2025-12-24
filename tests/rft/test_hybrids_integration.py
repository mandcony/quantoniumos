# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
"""
Integration tests for RFT Hybrids to ensure they are embedded in the CI suite.
Wraps functionality from benchmarks/test_all_hybrids.py into standard pytest.
"""

import pytest
import numpy as np
from algorithms.rft.hybrids.rft_hybrid_codec import RFTHybridCodec
from algorithms.rft.hybrids.h3_arft_cascade import H3ARFTCascade

def test_rft_hybrid_codec_sanity():
    """Ensure RFTHybridCodec can instantiate and process a signal."""
    codec = RFTHybridCodec()
    
    # Create a simple test signal
    N = 1024
    t = np.linspace(0, 1, N)
    signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(N)
    signal = signal.astype(np.float32)
    
    # Encode
    encoded = codec.encode(signal)
    assert encoded is not None
    assert "bands" in encoded
    assert encoded["numel"] == N
    
    # Decode
    decoded = codec.decode(encoded)
    assert decoded.shape == signal.shape
    
    # Check fidelity (should be decent even with default settings)
    mse = np.mean((signal - decoded)**2)
    assert mse < 0.1, f"Reconstruction error too high: {mse}"

def test_h3_arft_cascade_sanity():
    """Ensure H3ARFTCascade can instantiate and process a signal."""
    cascade = H3ARFTCascade()
    
    # Create a signal with structure (step) and texture (sine)
    N = 1024
    t = np.linspace(0, 1, N)
    structure = np.sign(np.sin(2 * np.pi * 2 * t))
    texture = 0.2 * np.sin(2 * np.pi * 50 * t)
    signal = (structure + texture).astype(np.float32)
    
    # Encode
    result = cascade.encode(signal)
    assert result is not None
    assert result.coefficients is not None
    assert len(result.coefficients) == 2 * N  # Concatenated structure + texture
    
    # Decode
    decoded = cascade.decode(result.coefficients, N)
    assert decoded.shape == signal.shape
    
    # Check fidelity
    mse = np.mean((signal - decoded)**2)
    assert mse < 0.1, f"Reconstruction error too high: {mse}"

def test_hybrid_registry_alignment():
    """Ensure all hybrids in the registry are loadable."""
    # This checks if the imports in benchmarks/test_all_hybrids.py would succeed
    from algorithms.rft.hybrids.rft_hybrid_codec import RFTHybridCodec
    from algorithms.rft.hybrids.h3_arft_cascade import H3ARFTCascade
    from algorithms.rft.hybrids.hybrid_residual_predictor import TinyResidualPredictor
    
    assert RFTHybridCodec is not None
    assert H3ARFTCascade is not None
    assert TinyResidualPredictor is not None
