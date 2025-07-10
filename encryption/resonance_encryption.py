"""
Resonance Encryption Module

Provides symmetric encryption services based on the QuantoniumOS resonance mathematics.
"""

import base64
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from encryption.quantum_engine_adapter import quantum_adapter


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
            self.amplitude * math.cos(self.phase), self.amplitude * math.sin(self.phase)
        )


def resonance_encrypt(plaintext: str, key: str) -> str:
    """
    Encrypt text using resonance-based encryption.

    Args:
        plaintext: Text to encrypt
        key: Encryption key

    Returns:
        Base64-encoded encrypted data
    """
    return quantum_adapter.encrypt(plaintext, key)


def resonance_decrypt(ciphertext: str, key: str) -> str:
    """
    Decrypt text that was encrypted with resonance-based encryption.

    Args:
        ciphertext: Base64-encoded encrypted data
        key: Decryption key

    Returns:
        Decrypted plaintext
    """
    return quantum_adapter.decrypt(ciphertext, key)


def generate_entropy(amount: int = 32) -> bytes:
    """
    Generate quantum-inspired entropy.

    Args:
        amount: Amount of entropy to generate in bytes

    Returns:
        Random bytes
    """
    entropy = quantum_adapter.generate_entropy(amount)
    return base64.b64decode(entropy)


def generate_waveform(length: int = 64, seed: Optional[int] = None) -> List[float]:
    """
    Generate a waveform with the specified length.

    Args:
        length: Number of points in the waveform
        seed: Optional random seed

    Returns:
        List of waveform values
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate a waveform with multiple frequency components
    x = np.linspace(0, 2 * np.pi, length)
    waveform = np.zeros(length)

    # Add several frequency components with random phases
    for i in range(1, 5):
        freq = i * 0.5
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.1, 1.0) / i
        waveform += amplitude * np.sin(freq * x + phase)

    # Normalize to [0, 1] range
    waveform = (waveform - np.min(waveform)) / (np.max(waveform) - np.min(waveform))

    return waveform.tolist()


def apply_rft(waveform: List[float]) -> Dict[str, Any]:
    """
    Apply Resonance Fourier Transform to a waveform.

    Args:
        waveform: List of waveform values

    Returns:
        Dictionary with frequencies, amplitudes, and phases
    """
    return quantum_adapter.apply_rft(waveform)


def apply_irft(frequency_data: Dict[str, List[float]]) -> List[float]:
    """
    Apply Inverse Resonance Fourier Transform.

    Args:
        frequency_data: Dictionary with frequencies, amplitudes, and phases

    Returns:
        Reconstructed waveform
    """
    result = quantum_adapter.apply_irft(frequency_data)
    return result.get("waveform", [])


def calculate_waveform_hash(waveform: List[float]) -> str:
    """
    Calculate a hash of a waveform for container operations.

    Args:
        waveform: List of waveform values

    Returns:
        Hash string
    """
    # Convert to bytes
    waveform_bytes = np.array(waveform, dtype=np.float32).tobytes()

    # Calculate hash
    hash_value = hashlib.sha256(waveform_bytes).hexdigest()

    return hash_value
