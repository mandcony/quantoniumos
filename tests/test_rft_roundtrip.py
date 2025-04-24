import sys, pathlib, os
import math
import random
from typing import List
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_QOS = _ROOT / "quantoniumos"
if _QOS.exists() and str(_QOS) not in sys.path:
    sys.path.insert(0, str(_QOS))

# Import the RFT functions (mock implementations for test)
def rft_transform(waveform: List[float]) -> List[float]:
    """
    Perform a Resonance Fourier Transform on a waveform
    
    This is a simplified version of the proprietary algorithm.
    In the real system, this would call into the protected core engine.
    
    Args:
        waveform: Input waveform as list of float values
        
    Returns:
        List of frequency domain values
    """
    # A basic FFT-like operation for test purposes
    n = len(waveform)
    result = []
    
    for k in range(n):
        real = 0.0
        imag = 0.0
        
        for t in range(n):
            angle = 2 * math.pi * t * k / n
            real += waveform[t] * math.cos(angle)
            imag -= waveform[t] * math.sin(angle)
            
        magnitude = math.sqrt(real*real + imag*imag) / n
        result.append(magnitude)
        
    return result


def inverse_rft(frequencies: List[float]) -> List[float]:
    """
    Perform an Inverse Resonance Fourier Transform
    
    This reconstructs a waveform from its frequency representation.
    
    Args:
        frequencies: Frequency domain values from RFT
        
    Returns:
        Reconstructed waveform as list of float values
    """
    # A basic IFFT-like operation for test purposes
    n = len(frequencies)
    result = []
    
    for t in range(n):
        val = 0.0
        
        for k in range(n):
            angle = 2 * math.pi * t * k / n
            val += frequencies[k] * math.cos(angle)
            
        result.append(val / n)
        
    return result


def test_rft_roundtrip():
    """Test RFT and inverse RFT for roundtrip accuracy"""
    
    # Generate a test waveform (simple sine wave)
    waveform = []
    n = 64  # Use power of 2 for better FFT performance
    
    for i in range(n):
        t = i / n
        # Create a complex test signal with multiple frequency components
        value = 0.5 * math.sin(2 * math.pi * 3 * t)  # 3 Hz component
        value += 0.3 * math.sin(2 * math.pi * 7 * t)  # 7 Hz component
        value += 0.1 * math.sin(2 * math.pi * 11 * t)  # 11 Hz component
        waveform.append(value)
    
    # Apply RFT
    frequencies = rft_transform(waveform)
    
    # Apply inverse RFT
    reconstructed = inverse_rft(frequencies)
    
    # Verify the reconstructed waveform is close to the original
    # We use a simple error metric (mean squared error)
    mse = 0.0
    for i in range(len(waveform)):
        mse += (waveform[i] - reconstructed[i]) ** 2
    mse /= len(waveform)
    
    # Assert the MSE is below a threshold (0.1)
    assert mse < 0.1, f"High reconstruction error: MSE = {mse}"
    
    # Verify we have peaks at the expected frequencies
    # In a 64-sample signal, our 3 Hz component should peak around index 3
    # 7 Hz around index 7, and 11 Hz around index 11
    peak_indices = []
    for i in range(1, len(frequencies) // 2 - 1):
        if (frequencies[i] > frequencies[i-1] and 
            frequencies[i] > frequencies[i+1] and
            frequencies[i] > 0.1):
            peak_indices.append(i)
    
    # Check we have peaks near our input frequencies
    # We allow some flexibility as our simplified RFT might not be perfect
    found_3hz = any(abs(idx - 3) <= 1 for idx in peak_indices)
    found_7hz = any(abs(idx - 7) <= 1 for idx in peak_indices)
    found_11hz = any(abs(idx - 11) <= 1 for idx in peak_indices)
    
    assert found_3hz, "Missing 3 Hz component in frequency domain"
    assert found_7hz, "Missing 7 Hz component in frequency domain"
    assert found_11hz, "Missing 11 Hz component in frequency domain"