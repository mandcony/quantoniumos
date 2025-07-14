"""
Wave Primitives Module

Core wave mathematics for QuantoniumOS.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class WaveNumber:
    """Represents a complex number with amplitude and phase."""
    def __init__(self, amplitude: float = 1.0, phase: float = 0.0):
        self.amplitude = float(amplitude)
        self.phase = float(phase)
        
    def __repr__(self):
        return f"WaveNumber(amplitude={self.amplitude:.3f}, phase={self.phase:.3f})"
        
    def to_complex(self):
        """Convert to Python complex number."""
        import math
        return complex(
            self.amplitude * math.cos(self.phase),
            self.amplitude * math.sin(self.phase)
        )
        
    def scale_amplitude(self, factor: float):
        """Scale the amplitude by the given factor."""
        self.amplitude *= factor
        return self
        
    def rotate_phase(self, delta_phase: float):
        """Rotate the phase by the given amount."""
        self.phase = (self.phase + delta_phase) % (2 * math.pi)
        return self
        
    @staticmethod
    def from_complex(c: complex) -> 'WaveNumber':
        """Create a WaveNumber from a Python complex number."""
        amplitude = abs(c)
        phase = math.atan2(c.imag, c.real) if amplitude > 0 else 0.0
        return WaveNumber(amplitude, phase)

def generate_waveform(length: int = 64, seed: Optional[int] = None) -> List[float]:
    """
    Generate a waveform with the specified length.
    """
    if seed is not None:
        np.random.seed(seed)
    return list(np.random.normal(0, 1, length))

def normalize_waveform(waveform: List[float]) -> List[float]:
    """
    Normalize a waveform to have unit energy.
    """
    energy = sum(x*x for x in waveform)
    if energy == 0:
        return waveform
    scale = 1.0 / math.sqrt(energy)
    return [x * scale for x in waveform]

def compute_energy(waveform: List[float]) -> float:
    """
    Compute the energy of a waveform.
    """
    return sum(x*x for x in waveform)

def compute_cross_correlation(waveform1: List[float], waveform2: List[float]) -> List[float]:
    """
    Compute the cross-correlation between two waveforms.
    """
    n1, n2 = len(waveform1), len(waveform2)
    result = []
    
    for i in range(n1 + n2 - 1):
        sum_val = 0
        for j in range(n1):
            if 0 <= i - j < n2:
                sum_val += waveform1[j] * waveform2[i - j]
        result.append(sum_val)
    
    return result

def compute_resonance(waveform: List[float], freq: float, phase: float = 0.0) -> float:
    """
    Compute the resonance of a waveform with a given frequency and phase.
    """
    n = len(waveform)
    t = np.arange(n)
    carrier = np.sin(2 * math.pi * freq * t / n + phase)
    return abs(sum(w * c for w, c in zip(waveform, carrier))) / n

def filter_resonances(waveform: List[float], min_freq: float, max_freq: float, 
                      num_freqs: int = 10) -> List[Tuple[float, float]]:
    """
    Find resonances in a specific frequency range.
    Returns a list of (frequency, resonance) tuples.
    """
    freqs = np.linspace(min_freq, max_freq, num_freqs)
    result = []
    
    for freq in freqs:
        # Try different phases and take the maximum resonance
        max_res = 0
        for phase in [0, math.pi/4, math.pi/2, 3*math.pi/4]:
            res = compute_resonance(waveform, freq, phase)
            max_res = max(max_res, res)
        
        result.append((freq, max_res))
    
    return result
