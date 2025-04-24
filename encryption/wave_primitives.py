"""
QuantoniumOS - Wave Primitives

Core classes for wave-based representation and manipulation.
These are the building blocks for the patent-protected quantum-inspired algorithms.
"""

import math
import random
import time

class WaveNumber:
    """
    Represents a wave as amplitude and phase, the fundamental unit 
    for all waveform operations in the QuantoniumOS system.
    """
    def __init__(self, amplitude: float, phase: float = 0.0):
        self.amplitude = float(amplitude)
        self.phase = float(phase)

    def __repr__(self):
        return f"WaveNumber(amplitude={self.amplitude:.3f}, phase={self.phase:.3f})"

    def scale_amplitude(self, factor: float):
        """Scale the amplitude by a factor."""
        self.amplitude *= factor

    def shift_phase(self, delta: float):
        """Shift the phase by delta, normalizing to [0, 2π)."""
        self.phase += delta
        self.phase %= 2 * math.pi  # Normalize phase to [0, 2π)

    def to_float(self) -> float:
        """Convert to a single float value (amplitude only)."""
        return self.amplitude
        
    def to_complex(self) -> complex:
        """Convert to complex number representation."""
        return self.amplitude * complex(math.cos(self.phase), math.sin(self.phase))
        
    @classmethod
    def from_complex(cls, z: complex) -> 'WaveNumber':
        """Create a WaveNumber from a complex number."""
        amplitude = abs(z)
        phase = math.atan2(z.imag, z.real) if amplitude > 1e-12 else 0.0
        return cls(amplitude, phase)

def wave_interference(a: WaveNumber, b: WaveNumber) -> WaveNumber:
    """
    Calculate wave interference between two WaveNumbers.
    Uses complex addition to properly account for phase relationships.
    """
    z_a = a.to_complex()
    z_b = b.to_complex()
    z_result = z_a + z_b
    return WaveNumber.from_complex(z_result)

def calculate_coherence(a: WaveNumber, b: WaveNumber) -> float:
    """
    Calculate the coherence (similarity) between two wave numbers.
    Returns a value between 0.0 (completely different) and 1.0 (identical).
    """
    # Phase difference normalized to [0, π]
    phase_diff = abs((a.phase - b.phase) % (2 * math.pi))
    if phase_diff > math.pi:
        phase_diff = 2 * math.pi - phase_diff
    
    # Normalize to [0, 1] where 0 means opposite phase (π difference)
    # and 1 means same phase (0 difference)
    phase_coherence = 1.0 - (phase_diff / math.pi)
    
    # Amplitude similarity
    amp_ratio = min(a.amplitude, b.amplitude) / max(a.amplitude, b.amplitude) if max(a.amplitude, b.amplitude) > 0 else 1.0
    
    # Combined coherence score
    return 0.5 * (phase_coherence + amp_ratio)


def random_phase(min_value: float = 0.0, max_value: float = 2 * math.pi) -> float:
    """
    Generate a random phase value between min_value and max_value.
    
    By default, returns a value between 0 and 2π (full circle).
    Uses a combination of time and random to increase entropy.
    
    Args:
        min_value: Minimum phase value (default: 0.0)
        max_value: Maximum phase value (default: 2π)
        
    Returns:
        Random phase value as a float
    """
    # Seed with current time to ensure different values even in rapid calls
    random.seed(time.time() * 1000000)
    
    # Generate a random value in the specified range
    return min_value + random.random() * (max_value - min_value)