# File: attached_assets/wave_primitives.py

import math
import numpy as np

class WaveNumber:
    """
    A complex-like class that stores amplitude and phase information for wave calculations.
    This class forms the basis of our resonance-based computation framework.
    """
    def __init__(self, amplitude, phase):
        """
        Initialize a WaveNumber with amplitude and phase.
        
        Args:
            amplitude: Amplitude value (typically between 0 and 1)
            phase: Phase in radians (typically between 0 and 2π)
        """
        self.amplitude = float(amplitude)
        self.phase = float(phase)
        
    def __repr__(self):
        return f"WaveNumber(amplitude={self.amplitude:.3f}, phase={self.phase:.3f})"
        
    def __str__(self):
        return f"{self.amplitude:.3f}∠{self.phase:.3f}π"
        
    def __abs__(self):
        return self.amplitude
        
    def __add__(self, other):
        if isinstance(other, WaveNumber):
            # Convert to rectangular coordinates for addition
            a1, p1 = self.amplitude, self.phase
            a2, p2 = other.amplitude, other.phase
            
            # Convert to rectangular
            x1, y1 = a1 * math.cos(p1), a1 * math.sin(p1)
            x2, y2 = a2 * math.cos(p2), a2 * math.sin(p2)
            
            # Add
            x3, y3 = x1 + x2, y1 + y2
            
            # Convert back to polar
            amplitude = math.sqrt(x3*x3 + y3*y3)
            phase = math.atan2(y3, x3)
            
            return WaveNumber(amplitude, phase)
        elif isinstance(other, (int, float)):
            # Treat scalar as WaveNumber with phase 0
            return self + WaveNumber(other, 0)
        else:
            return NotImplemented
            
    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return WaveNumber(other, 0) + self
        return NotImplemented
        
    def __mul__(self, other):
        if isinstance(other, WaveNumber):
            # For wave multiplication, we multiply amplitudes and add phases
            amplitude = self.amplitude * other.amplitude
            phase = (self.phase + other.phase) % (2 * math.pi)
            return WaveNumber(amplitude, phase)
        elif isinstance(other, (int, float)):
            # For scalar multiplication, we just scale the amplitude
            return WaveNumber(self.amplitude * other, self.phase)
        else:
            return NotImplemented
            
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return WaveNumber(self.amplitude * other, self.phase)
        return NotImplemented
        
    def conjugate(self):
        """Return the complex conjugate (keeps amplitude, negates phase)."""
        return WaveNumber(self.amplitude, -self.phase)
        
    def to_complex(self):
        """Convert to Python complex number."""
        real = self.amplitude * math.cos(self.phase)
        imag = self.amplitude * math.sin(self.phase)
        return complex(real, imag)
        
    @classmethod
    def from_complex(cls, z):
        """Create WaveNumber from a complex number."""
        amplitude = abs(z)
        phase = math.atan2(z.imag, z.real)
        return cls(amplitude, phase)
        
    @classmethod
    def from_resonance(cls, resonance_value):
        """Create WaveNumber from a resonance value (0-1)."""
        # Map resonance to reasonable amplitude and phase
        amplitude = 0.5 + resonance_value * 0.5  # 0.5 to 1.0
        phase = resonance_value * math.pi       # 0 to π
        return cls(amplitude, phase)

def rft(waveform, num_frequencies=None):
    """
    Resonance Fourier Transform - Converts time-domain waveform to resonance frequencies.
    
    This is our alternative to FFT that extracts resonant behavior. Unlike FFT,
    which decomposes into sine/cosine components, RFT identifies resonant modes.
    
    Args:
        waveform: List of amplitude values in time domain
        num_frequencies: Number of frequency components to extract (default: auto)
        
    Returns:
        Dictionary with 'frequencies', 'amplitudes', and 'phases' keys
    """
    if num_frequencies is None:
        # Auto-determine based on waveform length
        num_frequencies = min(max(len(waveform) // 2, 3), 10)
    
    waveform = np.array(waveform)
    n = len(waveform)
    
    # Prepare the result arrays
    frequencies = np.linspace(0.1, 0.9, num_frequencies)  # Resonant frequencies 0.1-0.9
    amplitudes = np.zeros(num_frequencies)
    phases = np.zeros(num_frequencies)
    
    # For each frequency, calculate resonance contribution
    for i, freq in enumerate(frequencies):
        # Create time basis
        t = np.linspace(0, 2*np.pi, n)
        
        # Calculate correlation with resonance pattern at this frequency
        cosine_corr = np.sum(waveform * np.cos(freq * t)) / n
        sine_corr = np.sum(waveform * np.sin(freq * t)) / n
        
        # Convert to amplitude and phase
        amplitudes[i] = np.sqrt(cosine_corr**2 + sine_corr**2)
        phases[i] = np.arctan2(sine_corr, cosine_corr)
    
    # Normalize amplitudes
    if np.sum(amplitudes) > 0:
        amplitudes = amplitudes / np.sum(amplitudes)
    
    return {
        'frequencies': frequencies.tolist(),
        'amplitudes': amplitudes.tolist(),
        'phases': phases.tolist()
    }

def irft(frequency_data, num_samples=32):
    """
    Inverse Resonance Fourier Transform - Convert resonance frequencies back to waveform.
    
    Args:
        frequency_data: Dictionary with 'frequencies', 'amplitudes', and 'phases'
        num_samples: Number of samples in output waveform
        
    Returns:
        Waveform as a list of amplitudes
    """
    frequencies = np.array(frequency_data['frequencies'])
    amplitudes = np.array(frequency_data['amplitudes'])
    phases = np.array(frequency_data['phases'])
    
    # Create time basis
    t = np.linspace(0, 2*np.pi, num_samples)
    
    # Initialize output waveform
    waveform = np.zeros(num_samples)
    
    # Sum all frequency components
    for i in range(len(frequencies)):
        waveform += amplitudes[i] * np.cos(frequencies[i] * t + phases[i])
    
    # Normalize to 0-1 range
    waveform = (waveform - np.min(waveform)) / (np.max(waveform) - np.min(waveform) + 1e-10)
    
    return waveform.tolist()
