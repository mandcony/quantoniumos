"""
Quantonium OS - RFT Roundtrip Test Suite

This module tests the bidirectional Resonance Fourier Transform (RFT) capability
by verifying that signals can be round-tripped through the transform and
its inverse with minimal error.
"""

import numpy as np
import pytest
import math
import sys
import os
from typing import List, Tuple

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the RFT functions
from core.encryption.resonance_fourier import (
    resonance_fourier_transform,
    inverse_resonance_fourier_transform,
    perform_rft, 
    perform_irft
)

# Test constants
SAMPLE_COUNT = 32  # Number of test waveforms to generate
MAX_ERROR = 1e-12  # Maximum allowed error in round-trip


def generate_test_waveforms() -> List[List[float]]:
    """Generate a diverse set of test waveforms for round-trip testing"""
    np.random.seed(42)  # Use fixed seed for reproducibility
    test_waveforms = []
    
    # Generate different types of waveforms
    for i in range(SAMPLE_COUNT):
        n = 16 + i % 8  # Varying lengths from 16 to 23
        
        if i % 4 == 0:
            # Sine wave
            freq = 2 + (i % 8)
            waveform = [math.sin(2 * math.pi * freq * t / n) for t in range(n)]
        elif i % 4 == 1:
            # Composite of sine waves
            waveform = [
                0.5 * math.sin(2 * math.pi * 2 * t / n) + 
                0.3 * math.sin(2 * math.pi * 5 * t / n) +
                0.2 * math.sin(2 * math.pi * 10 * t / n)
                for t in range(n)
            ]
        elif i % 4 == 2:
            # Random waveform
            waveform = np.random.rand(n).tolist()
        else:
            # Impulse
            waveform = [0] * n
            waveform[n // 2] = 1.0
            
        test_waveforms.append(waveform)
    
    return test_waveforms


def get_max_error(original: List[float], reconstructed: List[float]) -> float:
    """Calculate the maximum absolute error between two signals"""
    assert len(original) == len(reconstructed), "Signals must have the same length"
    return max(abs(a - b) for a, b in zip(original, reconstructed))


@pytest.mark.parametrize("waveform_index", range(SAMPLE_COUNT))
def test_rft_irft_roundtrip(waveform_index: int):
    """Test that a waveform can be round-tripped through RFT and IRFT with minimal error"""
    # Generate test waveforms
    test_waveforms = generate_test_waveforms()
    original_waveform = test_waveforms[waveform_index]
    
    # Forward transform
    rft_result = resonance_fourier_transform(original_waveform)
    
    # Inverse transform 
    reconstructed_waveform = inverse_resonance_fourier_transform(rft_result)
    
    # Check that the original and reconstructed waveforms match within tolerance
    error = get_max_error(original_waveform, reconstructed_waveform)
    assert error < MAX_ERROR, f"Round-trip error {error} exceeds maximum allowed error {MAX_ERROR}"


def test_api_wrapper_roundtrip():
    """Test that the API wrappers for RFT and IRFT work correctly together"""
    # Generate a simple test waveform
    waveform = [math.sin(2 * math.pi * t / 16) for t in range(16)]
    
    # Use the API wrapper functions
    rft_result = perform_rft(waveform)
    reconstructed = perform_irft(rft_result)
    
    # Verify the result contains expected keys
    assert "amplitude" in rft_result
    assert "phase" in rft_result
    assert "resonance" in rft_result
    
    # Check that reconstructed waveform is roughly similar to original
    # We can't expect exact reconstruction due to the simplified API
    assert len(reconstructed) == len(waveform)
    
    # The overall shape should be preserved (correlation > 0.95)
    correlation = np.corrcoef(waveform, reconstructed)[0, 1]
    assert correlation > 0.95, f"Correlation {correlation} is too low for API wrappers"


if __name__ == "__main__":
    # Run all tests
    print("Testing RFT/IRFT roundtrip...")
    
    # Generate and test all waveforms
    test_waveforms = generate_test_waveforms()
    for i, waveform in enumerate(test_waveforms):
        rft_result = resonance_fourier_transform(waveform)
        reconstructed = inverse_resonance_fourier_transform(rft_result)
        error = get_max_error(waveform, reconstructed)
        print(f"Waveform {i}: Length={len(waveform)}, Error={error:.3e}")
        
        assert error < MAX_ERROR, f"Error {error} exceeds maximum allowed {MAX_ERROR}"
    
    print("All tests passed!")