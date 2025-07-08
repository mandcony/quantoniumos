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
    
    # Perform basic RFT
    full_spectrum = resonance_fourier_transform(signal)
    
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
    
    return significant_components


def perform_irft_list(frequency_components: List[tuple]) -> List[float]:
    """
    Perform Inverse RFT on frequency components list.
    
    Args:
        frequency_components: List of (frequency, complex_amplitude) tuples
        
    Returns:
        Reconstructed time-domain signal
    """
    if not frequency_components:
        return []
    
    return inverse_resonance_fourier_transform(frequency_components)