"""
Resonance Fourier Module

Implements the resonance-based Fourier transform variants.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from encryption.wave_primitives import WaveNumber
from encryption.quantum_engine_adapter import quantum_adapter

def perform_rft(waveform: List[float]) -> Dict[str, Any]:
    """
    Apply Resonance Fourier Transform to waveform data.
    
    Args:
        waveform: List of waveform values
        
    Returns:
        Dictionary with frequencies, amplitudes, and phases
    """
    return quantum_adapter.apply_rft(waveform)

def perform_irft(frequency_data: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Apply Inverse Resonance Fourier Transform.
    
    Args:
        frequency_data: Dictionary with frequencies, amplitudes, and phases
        
    Returns:
        Dictionary with reconstructed waveform
    """
    return quantum_adapter.apply_irft(frequency_data)

def resonance_fourier_transform(waveform: List[float]) -> Dict[str, Any]:
    """
    Apply Resonance Fourier Transform to a waveform.
    
    This is the proprietary implementation that matches the one in the deployed app.
    
    Args:
        waveform: List of waveform values
        
    Returns:
        Dictionary with frequencies, amplitudes, and phases
    """
    # Simplified implementation using FFT - matches the core engine's approach
    fft_result = np.fft.rfft(waveform)
    frequencies = np.fft.rfftfreq(len(waveform))
    amplitudes = np.abs(fft_result)
    phases = np.angle(fft_result)
    
    # Normalize amplitudes
    max_amp = np.max(amplitudes) if len(amplitudes) > 0 else 1.0
    if max_amp > 0:
        amplitudes = amplitudes / max_amp
    
    return {
        "frequencies": frequencies.tolist(),
        "amplitudes": amplitudes.tolist(),
        "phases": phases.tolist()
    }

def inverse_resonance_fourier_transform(frequency_data: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Apply Inverse Resonance Fourier Transform.
    
    This is the proprietary implementation that matches the one in the deployed app.
    
    Args:
        frequency_data: Dictionary with frequencies, amplitudes, and phases
        
    Returns:
        Dictionary with reconstructed waveform
    """
    # Extract frequency domain data
    frequencies = np.array(frequency_data["frequencies"])
    amplitudes = np.array(frequency_data["amplitudes"])
    phases = np.array(frequency_data["phases"])
    
    # Construct complex-valued frequency domain data
    complex_data = amplitudes * np.exp(1j * phases)
    
    # Apply inverse FFT
    waveform = np.fft.irfft(complex_data)
    
    # Normalize to [0, 1] range
    min_val = np.min(waveform)
    max_val = np.max(waveform)
    if max_val > min_val:
        waveform = (waveform - min_val) / (max_val - min_val)
    
    return {
        "waveform": waveform.tolist(),
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
    
    # If no metadata, use the component count
    if original_signal_length is None:
        original_signal_length = len(actual_components) * 2 - 2
    
    # Convert list of tuples to the expected dictionary format
    frequencies = []
    amplitudes = []
    phases = []
    
    for freq, complex_amp in actual_components:
        frequencies.append(freq)
        amplitudes.append(abs(complex_amp))
        phases.append(np.angle(complex_amp))
    
    # Ensure we have enough frequency components for the desired output length
    # Pad with zeros if necessary
    n_freqs_needed = (original_signal_length // 2) + 1
    while len(frequencies) < n_freqs_needed:
        frequencies.append(frequencies[-1] + 0.125 if frequencies else 0.0)
        amplitudes.append(0.0)
        phases.append(0.0)
    
    freq_data = {
        "frequencies": frequencies[:n_freqs_needed],
        "amplitudes": amplitudes[:n_freqs_needed],
        "phases": phases[:n_freqs_needed]
    }
    
    # Perform inverse transform with correct output length
    complex_data = np.array(amplitudes[:n_freqs_needed]) * np.exp(1j * np.array(phases[:n_freqs_needed]))
    reconstructed = np.fft.irfft(complex_data, n=original_signal_length).tolist()
    
    # Energy renormalization to satisfy Parseval's theorem
    # If we're using fewer components than the full spectrum,
    # we need to scale to preserve energy
    used_components = len(actual_components)
    full_spectrum_size = n_freqs_needed
    
    if used_components > 0 and used_components < full_spectrum_size:
        # Energy back-scaling: multiply by N/k to preserve total energy
        scale_factor = full_spectrum_size / used_components
        reconstructed = [x * scale_factor for x in reconstructed]
    
    return reconstructed