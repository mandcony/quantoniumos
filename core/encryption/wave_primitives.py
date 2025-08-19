"""
QuantoniumOS - Wave Primitives Module

RESEARCH ONLY: This implementation is for educational and research purposes only.
Not intended for production cryptographic applications.

This module implements the fundamental wave-based primitives for the system.
"""

import math

class WaveNumber:
    """
    A wave-based number representation for signal processing computations.
    Combines amplitude and phase information.
    """
    def __init__(self, amplitude=1.0, phase=0.0):
        self.amplitude = float(amplitude)
        self.phase = float(phase)

    def __repr__(self):
        return f"WaveNumber(amplitude={self.amplitude:.4f}, phase={self.phase:.4f})"

    def __add__(self, other):
        """Add two wave numbers by vector addition in polar form"""
        if not isinstance(other, WaveNumber):
            other = WaveNumber(other, 0)

        # Convert to complex for addition
        z1 = self.to_complex()
        z2 = other.to_complex()
        result = z1 + z2

        # Convert back to amplitude/phase
        return WaveNumber(abs(result), math.atan2(result.imag, result.real))

    def __mul__(self, other):
        """Multiply two wave numbers"""
        if not isinstance(other, WaveNumber):
            other = WaveNumber(other, 0)

        # Multiply amplitudes, add phases
        amplitude = self.amplitude * other.amplitude
        phase = (self.phase + other.phase) % (2 * math.pi)
        return WaveNumber(amplitude, phase)

    def conjugate(self):
        """Return the complex conjugate of this wave number"""
        return WaveNumber(self.amplitude, -self.phase)

    def to_complex(self):
        """Convert to complex number representation"""
        return self.amplitude * (math.cos(self.phase) + 1j * math.sin(self.phase))

    @classmethod
    def from_complex(cls, z):
        """Create a WaveNumber from a complex number"""
        return cls(abs(z), math.atan2(z.imag, z.real))

    def to_bytes(self):
        """Convert to byte representation for serialization"""
        # Pack amplitude and phase as two double-precision floats
        import struct
        return struct.pack('dd', self.amplitude, self.phase)

    @classmethod
    def from_bytes(cls, data):
        """Create a WaveNumber from byte representation"""
        import struct
        amplitude, phase = struct.unpack('dd', data)
        return cls(amplitude, phase)

def calculate_coherence(wave1: WaveNumber, wave2: WaveNumber) -> float:
    """
    Calculate coherence between two wave numbers.
    Returns a value between 0 and 1.
    """
    # Convert to complex numbers
    z1 = wave1.to_complex()
    z2 = wave2.to_complex()

    # Calculate coherence as normalized dot product
    dot_product = (z1 * z2.conjugate())
    norm_product = abs(z1) * abs(z2)

    if norm_product == 0:
        return 0

    return abs(dot_product) / norm_product

# Self-test
if __name__ == "__main__":
    w1 = WaveNumber(1.0, 0.0)  # Amplitude=1, Phase=0
    w2 = WaveNumber(1.0, math.pi/2)  # Amplitude=1, Phase=90°

    # Test addition
    w_sum = w1 + w2
    print(f"w1 + w2 = {w_sum}")

    # Test multiplication
    w_product = w1 * w2
    print(f"w1 * w2 = {w_product}")

    # Test coherence
    coh = calculate_coherence(w1, w2)
    print(f"Coherence between w1 and w2: {coh:.4f}")

    # Test serialization
    w_bytes = w1.to_bytes()
    w_restored = WaveNumber.from_bytes(w_bytes)
    print(f"Original: {w1}, Restored: {w_restored}")
