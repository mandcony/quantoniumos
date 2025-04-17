"""
Quantonium OS - Live Resonance Data Stream

Provides real-time access to resonance data (encryption keystream amplitudes, RFT spectra)
for visualization in the web UI.
"""

import time
import json
import random
import logging
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Generator, Optional, Union

# Configure logger
logger = logging.getLogger("quantonium_stream")
logger.setLevel(logging.INFO)

# Global variables to store the most recent resonance data
last_encrypt_keystream: Optional[List[float]] = None
last_encrypt_phase: Optional[List[float]] = None
last_encrypt_timestamp: Optional[int] = None

# Thread lock for accessing the shared resonance data
lock = threading.Lock()

def update_encrypt_data(keystream: List[float], phase: Optional[List[float]] = None) -> None:
    """
    Update the global resonance data from an encryption operation
    
    Args:
        keystream: Amplitude values from the encryption keystream
        phase: Optional phase values from the encryption process
    """
    global last_encrypt_keystream, last_encrypt_phase, last_encrypt_timestamp
    
    with lock:
        last_encrypt_keystream = keystream.copy()
        last_encrypt_timestamp = int(time.time() * 1000)
        
        if phase is not None:
            last_encrypt_phase = phase.copy()
        else:
            # Generate synthetic phase data if none provided
            last_encrypt_phase = [random.uniform(0, 2 * np.pi) for _ in range(len(keystream))]
            
    logger.debug(f"Updated encrypt resonance data with {len(keystream)} points")

def _generate_synthetic_data(length: int = 64) -> Dict[str, Union[int, List[float]]]:
    """
    Generate synthetic resonance data when no real data is available
    
    Args:
        length: Number of data points to generate
        
    Returns:
        Dictionary with timestamp, amplitudes, and phases
    """
    timestamp = int(time.time() * 1000)
    
    # Generate a synthetic waveform with some harmonic components
    t = np.linspace(0, 2 * np.pi, length)
    base_freq = time.time() % 10  # Slowly changing base frequency
    
    # Create a complex waveform with multiple frequencies
    amp = []
    for i in range(length):
        val = 0.5 * np.sin(t[i] * base_freq) 
        val += 0.3 * np.sin(t[i] * base_freq * 2.1)
        val += 0.15 * np.sin(t[i] * base_freq * 3.7)
        val += 0.05 * np.random.random()  # Add some noise
        amp.append(val)
    
    # Generate corresponding phase values
    phase = [((t[i] * base_freq) % (2 * np.pi)) for i in range(length)]
    
    return {
        "t": timestamp,
        "amp": amp,
        "phase": phase
    }

def get_current_resonance_data(length: int = 64) -> Dict[str, Union[int, List[float]]]:
    """
    Get the current resonance data, either from the last encryption or synthesized
    
    Args:
        length: Desired number of data points (for synthetic data)
        
    Returns:
        Dictionary with timestamp, amplitudes, and phases
    """
    with lock:
        if (last_encrypt_keystream is not None and 
            last_encrypt_timestamp is not None and
            time.time() * 1000 - last_encrypt_timestamp < 10000):  # Data less than 10 seconds old
            
            return {
                "t": last_encrypt_timestamp,
                "amp": last_encrypt_keystream,
                "phase": last_encrypt_phase or []
            }
    
    # If no recent data, generate synthetic data
    return _generate_synthetic_data(length)

def compute_rft_spectra(waveform: List[float], sample_rate: float = 1000.0) -> Dict[str, List[float]]:
    """
    Compute Resonance Fourier Transform spectra from a waveform
    
    Args:
        waveform: Input waveform values
        sample_rate: Assumed sample rate in Hz
        
    Returns:
        Dictionary with frequency and magnitude arrays
    """
    # Use NumPy's FFT for the calculation
    n = len(waveform)
    fft_result = np.fft.rfft(waveform)
    magnitudes = np.abs(fft_result)
    
    # Normalize magnitudes
    if np.max(magnitudes) > 0:
        magnitudes = magnitudes / np.max(magnitudes)
    
    # Calculate frequency bins
    freqs = np.fft.rfftfreq(n, d=1.0/sample_rate)
    
    # Convert to Python lists for JSON serialization
    return {
        "frequencies": freqs.tolist(),
        "magnitudes": magnitudes.tolist()
    }

def resonance_data_generator(interval_ms: int = 100) -> Generator[str, None, None]:
    """
    Generator yielding resonance data as SSE-formatted strings
    
    Args:
        interval_ms: Interval between data points in milliseconds
        
    Yields:
        SSE-formatted data strings
    """
    try:
        while True:
            # Get current resonance data
            data = get_current_resonance_data()
            
            # Format as SSE event
            yield f"data: {json.dumps(data)}\n\n"
            
            # Sleep for the interval
            time.sleep(interval_ms / 1000.0)
    except GeneratorExit:
        logger.info("Resonance data stream closed")