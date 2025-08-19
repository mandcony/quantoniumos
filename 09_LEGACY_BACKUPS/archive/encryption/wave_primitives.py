"""
Wave Primitives - Mathematical constructs for waveform operations

RESEARCH ONLY: This implementation is for educational and research purposes only.
Not intended for production cryptographic applications.

Provides fundamental wave-based mathematical primitives for use
in resonance-based encryption and hashing operations.

WaveNumber class for signal processing computations.
"""

import numpy as np
from typing import Union, List, Optional
import hashlib
import cmath

class WaveNumber:
    """
    Represents a wave number in the signal processing cryptographic system.

    A wave number encodes both magnitude and phase information for use
    in resonance-based encryption and hashing operations.
    """

    def __init__(self, magnitude: float = 1.0, phase: float = 0.0):
        """
        Initialize a wave number with magnitude and phase.

        Args:
            magnitude: The amplitude of the wave (default: 1.0)
            phase: The phase angle in radians (default: 0.0)
        """
        self.magnitude = float(magnitude)
        self.phase = float(phase) % (2 * np.pi)  # Normalize phase to [0, 2π)

        # Compute complex representation
        self.complex = self.magnitude * cmath.exp(1j * self.phase)

    def __add__(self, other):
        """Add two wave numbers."""
        if isinstance(other, WaveNumber):
            result_complex = self.complex + other.complex
            return WaveNumber.from_complex(result_complex)
        else:
            # Add scalar to magnitude
            return WaveNumber(self.magnitude + other, self.phase)

    def __mul__(self, other):
        """Multiply two wave numbers."""
        if isinstance(other, WaveNumber):
            result_complex = self.complex * other.complex
            return WaveNumber.from_complex(result_complex)
        else:
            # Multiply magnitude by scalar
            return WaveNumber(self.magnitude * other, self.phase)

    def __abs__(self):
        """Return the magnitude of the wave number."""
        return self.magnitude

    def __repr__(self):
        return f"WaveNumber(magnitude={self.magnitude:.4f}, phase={self.phase:.4f})"

    @classmethod
    def from_complex(cls, z: complex):
        """Create a WaveNumber from a complex number."""
        magnitude = abs(z)
        phase = cmath.phase(z)
        return cls(magnitude, phase)

    @classmethod
    def from_bytes(cls, data: bytes):
        """Create a WaveNumber from byte data using hash."""
        # Use SHA-256 to convert bytes to deterministic wave parameters
        hash_digest = hashlib.sha256(data).digest()

        # Extract magnitude and phase from hash
        magnitude_bytes = hash_digest[:8]
        phase_bytes = hash_digest[8:16]

        # Convert to float values
        magnitude = int.from_bytes(magnitude_bytes, 'big') / (2**64 - 1) + 0.1
        phase = int.from_bytes(phase_bytes, 'big') / (2**64 - 1) * 2 * np.pi

        return cls(magnitude, phase)

    def conjugate(self):
        """Return the complex conjugate of the wave number."""
        return WaveNumber(self.magnitude, -self.phase)

    def normalize(self):
        """Return a normalized wave number with magnitude 1."""
        if self.magnitude == 0:
            return WaveNumber(0, 0)
        return WaveNumber(1.0, self.phase)

    def to_bytes(self, length: int = 32) -> bytes:
        """Convert wave number to byte representation."""
        # Combine magnitude and phase into deterministic bytes
        mag_normalized = int(self.magnitude * 1000) % 256
        phase_normalized = int(self.phase * 1000) % 256

        # Create repeating pattern based on wave properties
        pattern = bytes([mag_normalized, phase_normalized] * (length // 2))

        # Pad if necessary
        if len(pattern) < length:
            pattern += bytes([0] * (length - len(pattern)))

        return pattern[:length]

def wave_interference(wave1: WaveNumber, wave2: WaveNumber) -> WaveNumber:
    """
    Compute wave interference pattern between two wave numbers.

    Args:
        wave1: First wave number
        wave2: Second wave number

    Returns:
        WaveNumber representing the interference pattern
    """
    # Constructive/destructive interference
    interference_complex = wave1.complex + wave2.complex
    return WaveNumber.from_complex(interference_complex)

def wave_modulation(carrier: WaveNumber, signal: WaveNumber) -> WaveNumber:
    """
    Perform wave modulation for encryption operations.

    Args:
        carrier: Carrier wave
        signal: Signal wave to modulate

    Returns:
        WaveNumber representing the modulated wave
    """
    modulated_complex = carrier.complex * signal.complex
    return WaveNumber.from_complex(modulated_complex)

def generate_wave_sequence(seed: bytes, length: int) -> List[WaveNumber]:
    """
    Generate a sequence of wave numbers from a seed.

    Args:
        seed: Initial seed bytes
        length: Number of wave numbers to generate

    Returns:
        List of WaveNumber objects
    """
    sequence = []
    current_hash = seed

    for i in range(length):
        # Generate next hash in sequence
        hasher = hashlib.sha256()
        hasher.update(current_hash)
        hasher.update(i.to_bytes(4, 'big'))
        current_hash = hasher.digest()

        # Convert to wave number
        wave_num = WaveNumber.from_bytes(current_hash)
        sequence.append(wave_num)

    return sequence
