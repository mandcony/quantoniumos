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

# Feature flag for inverse RFT
FEATURE_IRFT = True