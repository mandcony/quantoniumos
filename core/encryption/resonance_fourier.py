"""
Quantonium OS - Resonance Fourier Transform Module

Implements Fourier analysis tools for resonance waveform processing.
"""

import numpy as np

def resonance_fourier_transform(signal):
    """
    Apply Resonance Fourier Transform to a signal.
    Returns frequency-value pairs.
    """
    n = len(signal)
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n)
    return list(zip(freqs, fft_result))

def inverse_resonance_fourier_transform(frequency_components):
    """
    Apply Inverse Resonance Fourier Transform to frequency components.
    Input should be frequency-value pairs from resonance_fourier_transform.
    """
    # Extract frequencies and values
    freqs, amps = zip(*frequency_components)
    # Return the inverse FFT
    return np.fft.ifft(amps).tolist()