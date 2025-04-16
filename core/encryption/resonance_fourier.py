"""
Quantonium OS - Resonance Fourier Transform Module

Implements Fourier analysis tools for resonance waveform processing.
"""

import math
import logging
from typing import List, Dict

# Configure logger
logger = logging.getLogger("resonance_fourier_encryption")
logger.setLevel(logging.INFO)

def resonance_fourier_transform(signal):
    """
    Apply Resonance Fourier Transform to a signal.
    Returns frequency-value pairs.
    """
    # Simple FFT implementation if numpy is not available
    n = len(signal)
    result = []
    
    for k in range(n):
        real_sum = 0.0
        imag_sum = 0.0
        for t in range(n):
            angle = 2 * math.pi * t * k / n
            real_sum += signal[t] * math.cos(angle)
            imag_sum -= signal[t] * math.sin(angle)
        
        amplitude = math.sqrt(real_sum**2 + imag_sum**2) / n
        frequency = k / n
        result.append((frequency, complex(real_sum, imag_sum) / n))
    
    return result

def inverse_resonance_fourier_transform(frequency_components):
    """
    Apply Inverse Resonance Fourier Transform to frequency components.
    Input should be frequency-value pairs from resonance_fourier_transform.
    """
    # Extract frequencies and values
    freqs, amps = zip(*frequency_components)
    n = len(amps)
    
    # Perform inverse FFT
    result = []
    for t in range(n):
        value = 0.0
        for k, amp in enumerate(amps):
            angle = 2 * math.pi * t * k / n
            value += amp.real * math.cos(angle) - amp.imag * math.sin(angle)
        result.append(value)
    
    return result

def perform_rft(waveform: List[float]) -> Dict[str, float]:
    """
    Perform Resonance Fourier Transform on the input waveform.
    This is the function expected by the protected module.
    
    Args:
        waveform: Input waveform as a list of floating point values
        
    Returns:
        Dictionary mapping frequency components to their amplitudes
    """
    logger.info(f"Performing RFT on waveform of length {len(waveform)}")
    
    if not waveform:
        logger.warning("Empty waveform provided, returning empty results")
        return {"amplitude": 0.0, "phase": 0.0, "resonance": 0.0}
    
    try:
        # Apply resonance Fourier transform
        spectrum = resonance_fourier_transform(waveform)
        
        # Extract key metrics from the transform
        result = {}
        
        # Get the first 10 frequency components with significant amplitudes
        for i, (freq, complex_val) in enumerate(spectrum[:10]):
            amplitude = abs(complex_val)
            if amplitude > 0.01:  # Only include significant components
                freq_name = f"freq_{i}"
                result[freq_name] = round(amplitude, 4)
        
        # Add overall metrics
        if waveform:
            result["amplitude"] = round(sum(abs(x) for x in waveform) / len(waveform), 4)
            result["phase"] = round(waveform[0], 4)
            result["resonance"] = round(waveform[-1], 4)
        
        logger.info(f"RFT completed with {len(result)} frequency components")
        return result
        
    except Exception as e:
        logger.error(f"RFT processing error: {str(e)}")
        # Return basic metrics on error
        return {
            "amplitude": round(sum(waveform) / len(waveform), 4) if waveform else 0,
            "phase": round(waveform[0], 4) if waveform else 0,
            "resonance": round(waveform[-1], 4) if waveform else 0
        }