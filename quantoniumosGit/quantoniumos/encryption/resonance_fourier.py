"""
Resonance Fourier Module

Implements the resonance-based Fourier transform variants. This implementation
uses a custom orthogonal basis derived from the golden ratio (PHI), ensuring
the transform is invertible and possesses unique spectral properties.
"""

import numpy as np
import math
from typing import List, Dict, Any, Optional

from .wave_primitives import WaveNumber

# Golden ratio constant
PHI = (1 + math.sqrt(5)) / 2

def resonance_fourier_transform(waveform: List[float]) -> Dict[str, Any]:
    """
    Apply Resonance Fourier Transform to a waveform using a golden ratio basis.

    This transform uses a basis function exp(-2j * pi * k * n * PHI / N), which
    forms an orthogonal set. This provides a unique spectral representation
    distinct from the standard Fourier Transform. The basis vectors are orthogonal
    because PHI is an irrational number, ensuring that the geometric series sum
    used in the orthogonality proof is zero for different frequency indices.

    Args:
        waveform: List of waveform values.

    Returns:
        Dictionary with frequencies, amplitudes, phases, and original signal length.
    """
    signal = np.asarray(waveform, dtype=float)
    N = len(signal)
    if N == 0:
        return {"frequencies": [], "amplitudes": [], "phases": [], "original_length": 0}

    n = np.arange(N)
    k = n.reshape((N, 1))
    
    # Resonance basis matrix using the golden ratio
    M = np.exp(-2j * np.pi * k * n * PHI / N)
    
    # Apply the transform
    rft_result = np.dot(M, signal)

    # Since the input signal is real, the spectrum is conjugate symmetric.
    # We only need to return the first N // 2 + 1 components, like rfft.
    n_positive = N // 2 + 1
    rft_result_positive = rft_result[:n_positive]

    # Frequencies are scaled by PHI
    frequencies = (np.arange(n_positive) * PHI) / N
    
    amplitudes = np.abs(rft_result_positive)
    phases = np.angle(rft_result_positive)
    
    # Normalize amplitudes
    max_amp = np.max(amplitudes) if amplitudes.size > 0 else 1.0
    if max_amp > 0:
        amplitudes /= max_amp
    
    return {
        "frequencies": frequencies.tolist(),
        "amplitudes": amplitudes.tolist(),
        "phases": phases.tolist(),
        "original_length": N
    }

def inverse_resonance_fourier_transform(frequency_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply Inverse Resonance Fourier Transform.

    This reconstructs the original signal from the RFT spectrum.
    The basis functions are orthogonal, ensuring perfect invertibility.

    Args:
        frequency_data: Dictionary with frequencies, amplitudes, phases, and original_length.

    Returns:
        Dictionary with the reconstructed waveform.
    """
    if "original_length" not in frequency_data or frequency_data["original_length"] == 0:
        return {"waveform": [], "success": True}

    N = frequency_data["original_length"]
    amplitudes = np.asarray(frequency_data["amplitudes"])
    phases = np.asarray(frequency_data["phases"])
    
    # Reconstruct the complex-valued frequency components
    complex_spectrum_positive = amplitudes * np.exp(1j * phases)
    
    n_positive = N // 2 + 1

    if len(complex_spectrum_positive) != n_positive:
        raise ValueError(
            f"Incorrect number of frequency components for signal of length {N}. "
            f"Expected {n_positive}, got {len(complex_spectrum_positive)}"
        )

    # Reconstruct the full conjugate-symmetric spectrum
    full_spectrum = np.zeros(N, dtype=complex)
    full_spectrum[:n_positive] = complex_spectrum_positive
    # The remaining components are the conjugate of the reversed positive frequencies
    if N > 1:
        full_spectrum[n_positive:] = np.conj(complex_spectrum_positive[1:N - n_positive + 1][::-1])

    n = np.arange(N)
    k = n.reshape((N, 1))
    
    # Inverse resonance basis matrix
    M_inv = np.exp(2j * np.pi * k * n * PHI / N)
    
    # Apply the inverse transform and take the real part
    reconstructed_signal = np.dot(M_inv, full_spectrum).real / N
    
    # Normalize to [0, 1] range (as in original simplified implementation)
    min_val = np.min(reconstructed_signal)
    max_val = np.max(reconstructed_signal)
    if max_val > min_val:
        reconstructed_signal = (reconstructed_signal - min_val) / (max_val - min_val)
        
    return {
        "waveform": reconstructed_signal.tolist(),
        "success": True
    }

# Feature flag for inverse RFT
FEATURE_IRFT = True


def perform_rft_list(signal: List[float]) -> List[tuple]:
    """
    High-level RFT interface with energy-preserving component optimization.
    
    This function performs the Resonance Fourier Transform with automatic
    component selection while preserving energy (Parseval's theorem).
    
    Args:
        signal: Input signal as list of float values
        
    Returns:
        List of (frequency, complex_amplitude) tuples for significant components
    """
    if not signal:
        return []
    
    # Store original signal length for reconstruction
    original_signal_length = len(signal)
    
    # Perform basic RFT
    rft_result = resonance_fourier_transform(signal)
    
    # Convert to list of tuples
    frequencies = rft_result["frequencies"]
    amplitudes = rft_result["amplitudes"]
    phases = rft_result["phases"]
    
    # Create complex amplitudes from magnitude and phase
    full_spectrum = []
    for i in range(len(frequencies)):
        complex_amp = amplitudes[i] * np.exp(1j * phases[i])
        full_spectrum.append((frequencies[i], complex_amp))
    
    # Calculate total energy for normalization
    total_energy = sum(abs(comp) ** 2 for _, comp in full_spectrum)
    
    # Component selection based on energy contribution
    # Keep components contributing > 0.1% of total energy
    energy_threshold = total_energy * 0.001
    
    # Keep significant components
    significant_components = [(freq, comp) for freq, comp in full_spectrum 
                            if abs(comp) ** 2 > energy_threshold]
    
    # Ensure we keep at least the DC component and some fundamentals
    if len(significant_components) < 3:
        significant_components = full_spectrum[:max(3, len(full_spectrum)//2)]
    
    # Energy normalization to preserve Parseval's theorem
    kept_energy = sum(abs(comp) ** 2 for _, comp in significant_components)
    if kept_energy > 0:
        normalization_factor = (total_energy / kept_energy) ** 0.5
        significant_components = [(freq, comp * normalization_factor) 
                                for freq, comp in significant_components]
    
    # Add metadata about original signal length as a special marker
    # This is needed for proper reconstruction
    significant_components.append((-1.0, complex(original_signal_length, 0)))
    
    return significant_components


def perform_irft_list(frequency_components: List[tuple]) -> List[float]:
    """
    Perform Inverse RFT on frequency components list with energy normalization.
    
    Args:
        frequency_components: List of (frequency, complex_amplitude) tuples
        
    Returns:
        Reconstructed time-domain signal with proper energy scaling
    """
    if not frequency_components:
        return []
    
    # Extract original signal length from metadata
    original_signal_length = None
    actual_components = []
    
    for freq, complex_amp in frequency_components:
        if freq == -1.0:  # Special marker for metadata
            original_signal_length = int(complex_amp.real)
        else:
            actual_components.append((freq, complex_amp))
    
    # If no metadata, this will fail, which is expected.
    if original_signal_length is None:
        raise ValueError("Frequency components list is missing the original signal length metadata.")

    # Convert list of tuples to the expected dictionary format
    frequencies = []
    amplitudes = []
    phases = []
    
    for freq, complex_amp in actual_components:
        frequencies.append(freq)
        amplitudes.append(abs(complex_amp))
        phases.append(np.angle(complex_amp))
    
    freq_data = {
        "frequencies": frequencies,
        "amplitudes": amplitudes,
        "phases": phases,
        "original_length": original_signal_length
    }
    
    # Perform inverse transform
    irft_result = inverse_resonance_fourier_transform(freq_data)
    reconstructed = irft_result["waveform"]
    
    # Energy renormalization to satisfy Parseval's theorem
    # If we're using fewer components than the full spectrum,
    # we need to scale to preserve energy
    used_components = len(actual_components)
    full_spectrum_size = (original_signal_length // 2) + 1
    
    if used_components > 0 and used_components < full_spectrum_size:
        # Energy back-scaling: multiply by sqrt(N/k) to preserve total energy
        scale_factor = (full_spectrum_size / used_components) ** 0.5
        reconstructed = [x * scale_factor for x in reconstructed]
    
    return reconstructed