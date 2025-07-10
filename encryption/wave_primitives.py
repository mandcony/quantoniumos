"""
Wave Primitives Module

Core wave mathematics for QuantoniumOS.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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


def analyze_waveform(waveform: List[float]) -> Dict[str, Any]:
    """
    Analyze a waveform to extract its properties.

    Args:
        waveform: List of waveform values

    Returns:
        Dictionary with properties like frequency, phase, and amplitude
    """
    # Calculate FFT
    fft_result = np.fft.rfft(waveform)
    frequencies = np.fft.rfftfreq(len(waveform))

    # Find dominant frequency
    amplitudes = np.abs(fft_result)
    max_idx = np.argmax(amplitudes)

    dominant_frequency = frequencies[max_idx]
    dominant_amplitude = amplitudes[max_idx]
    dominant_phase = np.angle(fft_result[max_idx])

    # Calculate energy
    energy = np.sum(np.square(waveform))

    # Calculate mean and stddev
    mean = np.mean(waveform)
    stddev = np.std(waveform)

    return {
        "dominant_frequency": float(dominant_frequency),
        "dominant_amplitude": float(dominant_amplitude),
        "dominant_phase": float(dominant_phase),
        "energy": float(energy),
        "mean": float(mean),
        "stddev": float(stddev),
    }


def resonance_fourier_transform(waveform: List[float]) -> Dict[str, List[float]]:
    """
    Apply Resonance Fourier Transform to a waveform.

    Args:
        waveform: List of waveform values

    Returns:
        Dictionary with frequencies, amplitudes, and phases
    """
    # Calculate FFT
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
        "phases": phases.tolist(),
    }


def inverse_resonance_fourier_transform(
    frequency_data: Dict[str, List[float]],
) -> List[float]:
    """
    Apply Inverse Resonance Fourier Transform.

    Args:
        frequency_data: Dictionary with frequencies, amplitudes, and phases

    Returns:
        Reconstructed waveform
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

    return waveform.tolist()


def generate_waveform_from_key(key: str, length: int = 64) -> List[float]:
    """
    Generate a waveform based on a key string.

    Args:
        key: String to derive waveform from
        length: Number of points in the waveform

    Returns:
        List of waveform values
    """
    import hashlib

    # Generate a seed from the key
    key_bytes = key.encode("utf-8")
    hash_obj = hashlib.sha256(key_bytes)
    hash_bytes = hash_obj.digest()

    # Use first 4 bytes as seed
    seed = int.from_bytes(hash_bytes[:4], byteorder="big")

    # Generate waveform with the seed
    return generate_waveform(length, seed)
