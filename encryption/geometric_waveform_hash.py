"""
QuantoniumOS - Geometric Waveform Hash

Implements the patent-protected geometric waveform hash algorithm.
This is a secure, proprietary algorithm that produces a unique hash
for each container based on its wave properties.
"""

import hashlib
import base64
import time
import logging
from typing import List, Tuple

from secure_core.python_bindings import engine_core
from encryption.wave_primitives import WaveNumber

logger = logging.getLogger("quantonium_api.hash")

def wave_hash(data: bytes) -> str:
    """
    Generate a geometric waveform hash for the provided data.
    
    This is the patent-protected algorithm that generates a unique hash
    that serves as both identifier and key for a container.
    
    Args:
        data: Binary data to hash
        
    Returns:
        64-character hex string representing the waveform hash
    """
    try:
        # Let the engine handle the proprietary algorithm
        return engine_core.wave_hash(data)
    except Exception as e:
        logger.error(f"Error in wave_hash: {e}")
        # Fall back to demo version if engine fails
        return _wave_hash_demo(data)

def _wave_hash_demo(data: bytes) -> str:
    """
    Demonstration version of the wave hash algorithm.
    This is not the proprietary algorithm, just a demonstration.
    
    Args:
        data: Binary data to hash
        
    Returns:
        64-character hex string
    """
    # Start with a normal cryptographic hash
    base_hash = hashlib.sha256(data).hexdigest()
    
    # Convert portions to symbolic waves
    waves = []
    for i in range(0, len(base_hash), 4):
        chunk = base_hash[i:i+4]
        if not chunk:
            continue
        
        # Convert chunk to amplitude and phase
        value = int(chunk, 16)  # Convert hex to int
        amplitude = 0.1 + (value & 0xFF) / 256.0  # Range 0.1-1.1
        phase = ((value >> 8) & 0xFF) / 128.0 * 3.14159  # Range 0-π
        
        waves.append(WaveNumber(amplitude, phase))
    
    # Generate the new hash by chaining wave operations
    result_bytes = bytearray()
    state = waves[0].amplitude
    
    for i, wave in enumerate(waves):
        # Apply a pseudo wave-interference operation
        state = (state + wave.amplitude) * 0.5
        # Shift based on phase
        state = (state + wave.phase / 10.0) % 1.0
        # Add a byte to the result
        result_bytes.append(int(state * 255))
    
    # Ensure 32 bytes (64 hex chars)
    while len(result_bytes) < 32:
        last = result_bytes[-1] if result_bytes else 0
        result_bytes.append((last * 17 + 255) % 256)
    
    # Convert to hex
    return ''.join(f'{b:02x}' for b in result_bytes[:32])

def extract_wave_parameters(hash_value: str) -> Tuple[List[WaveNumber], float]:
    """
    Extract wave parameters from a hash value.
    These parameters can be used to validate a container.
    
    Args:
        hash_value: 64-character hex hash
        
    Returns:
        Tuple of (wave_list, coherence_threshold)
    """
    if len(hash_value) != 64:
        raise ValueError("Hash must be 64 hex characters")
    
    waves = []
    
    # Extract waves from hash
    for i in range(0, len(hash_value), 4):
        chunk = hash_value[i:i+4]
        if not chunk:
            continue
            
        # Convert chunk to amplitude and phase
        value = int(chunk, 16)
        amplitude = 0.1 + (value & 0xFF) / 256.0
        phase = ((value >> 8) & 0xFF) / 128.0 * 3.14159
        
        waves.append(WaveNumber(amplitude, phase))
    
    # Calculate coherence threshold based on hash properties
    byte_sum = sum(int(hash_value[i:i+2], 16) for i in range(0, len(hash_value), 2))
    coherence_threshold = 0.6 + (byte_sum % 20) / 100.0  # Range 0.6-0.8
    
    return waves, coherence_threshold