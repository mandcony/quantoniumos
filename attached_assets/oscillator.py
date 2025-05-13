"""
QuantoniumOS Oscillator Module

This module implements a quantum-inspired oscillator for generating waveforms
that can be used by the Q-Wave Composer and related applications.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any

class Oscillator:
    """
    A quantum-inspired oscillator for generating wave patterns with
    specific frequencies, amplitudes, and phase characteristics.
    """
    
    def __init__(self, frequency: float, amplitude: complex, phase: float):
        """
        Initialize a new oscillator with the given parameters.
        
        Args:
            frequency: Frequency in Hz
            amplitude: Complex amplitude (magnitude and direction)
            phase: Initial phase in radians
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.sample_rate = 44100  # Default sample rate
    
    def generate_waveform(self, duration: float = 1.0) -> List[float]:
        """
        Generate a waveform with the oscillator's parameters.
        
        Args:
            duration: Length of the waveform in seconds
            
        Returns:
            List of float values representing the waveform
        """
        num_samples = int(duration * self.sample_rate)
        waveform = []
        
        for i in range(num_samples):
            t = i / self.sample_rate
            # Calculate the value at this time point
            value = abs(self.amplitude) * math.sin(2 * math.pi * self.frequency * t + self.phase)
            waveform.append(value)
            
        return waveform
    
    def generate_complex_waveform(self, duration: float = 1.0) -> List[complex]:
        """
        Generate a complex waveform with both real and imaginary components.
        
        Args:
            duration: Length of the waveform in seconds
            
        Returns:
            List of complex values representing the waveform
        """
        num_samples = int(duration * self.sample_rate)
        waveform = []
        
        for i in range(num_samples):
            t = i / self.sample_rate
            # Phase angle at this time point
            angle = 2 * math.pi * self.frequency * t + self.phase
            # Complex value with both sine and cosine components
            value = self.amplitude * complex(math.cos(angle), math.sin(angle))
            waveform.append(value)
            
        return waveform
    
    def __repr__(self) -> str:
        """String representation of the oscillator."""
        return f"Oscillator(freq={self.frequency:.2f}Hz, amp={abs(self.amplitude):.2f}, phase={self.phase:.2f}rad)"


def validate_oscillator(osc: Oscillator, duration: float = 1.0) -> List[float]:
    """
    Validate an oscillator by generating a test waveform and applying
    quantum-inspired validation checks.
    
    Args:
        osc: Oscillator instance to validate
        duration: Duration of test waveform in seconds
        
    Returns:
        The generated waveform if valid
    """
    # Generate the waveform
    waveform = osc.generate_waveform(duration)
    
    # Apply validation checks:
    # 1. Check for NaN or infinity values
    if any(math.isnan(x) or math.isinf(x) for x in waveform):
        raise ValueError("Waveform contains invalid values (NaN or infinity)")
        
    # 2. Check amplitude is within reasonable bounds
    peak = max(abs(x) for x in waveform)
    if peak > 10.0:  # Arbitrary limit
        raise ValueError(f"Waveform amplitude too high: {peak}")
        
    # 3. Check for zeroes (completely flat sections)
    zero_runs = 0
    max_zero_run = 0
    current_run = 0
    
    for val in waveform:
        if abs(val) < 1e-10:  # Effectively zero
            current_run += 1
        else:
            if current_run > 0:
                zero_runs += 1
                max_zero_run = max(max_zero_run, current_run)
                current_run = 0
    
    # More than half the waveform being zero is suspicious
    if max_zero_run > len(waveform) / 2:
        raise ValueError(f"Waveform contains too many consecutive zero values: {max_zero_run}")
    
    return waveform


def create_modulated_oscillator(carrier_freq: float, mod_freq: float, 
                               mod_index: float = 1.0) -> Oscillator:
    """
    Create a frequency-modulated oscillator.
    
    Args:
        carrier_freq: Carrier frequency in Hz
        mod_freq: Modulation frequency in Hz
        mod_index: Modulation index (depth of modulation)
        
    Returns:
        Configured Oscillator instance
    """
    # For FM, we use a complex amplitude based on the modulation parameters
    amplitude = complex(1.0, mod_index)
    # The phase is initialized based on the modulation frequency
    phase = math.atan2(mod_freq, carrier_freq)
    
    return Oscillator(carrier_freq, amplitude, phase)