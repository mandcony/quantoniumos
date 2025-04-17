"""
Quantonium OS - Geometric Waveform Hash Module

Implements waveform-based hash generation and verification for resonance encryption.
"""

import hashlib
import math
import hmac

def generate_waveform_hash(A, phi, precision=6):
    """
    Generate a unique hash from amplitude and phase values.
    
    Args:
        A (float): Amplitude value (0.0-1.0)
        phi (float): Phase value (0.0-1.0)
        precision (int): Precision for hash generation
        
    Returns:
        str: A unique hash representing the waveform parameters
    """
    # Normalize inputs to ensure they're within range
    A = max(0.0, min(1.0, float(A)))
    phi = max(0.0, min(1.0, float(phi)))
    
    # Generate wave pattern samples based on A and phi
    samples = []
    for i in range(64):
        # Calculate sample point using sine wave with phase and amplitude
        t = i / 64.0
        value = A * math.sin(2 * math.pi * t + phi * 2 * math.pi)
        # Round to precision to ensure consistent results
        value = round(value, precision)
        samples.append(value)
    
    # Create a unique representation from the samples
    sample_str = "_".join([str(sample) for sample in samples])
    
    # Create a SHA-256 hash of the sample pattern
    wave_hash = hashlib.sha256(sample_str.encode()).hexdigest()
    
    # Add the parameters as a prefix for better traceability
    param_prefix = f"A{A:.4f}_P{phi:.4f}_"
    
    return param_prefix + wave_hash

def verify_waveform_hash(wave_hash, A, phi):
    """
    Verify that a waveform hash was generated from the given A and phi values.
    
    Args:
        wave_hash (str): Hash to verify
        A (float): Amplitude value to check against
        phi (float): Phase value to check against
        
    Returns:
        bool: True if the hash matches the parameters, False otherwise
    """
    # Generate a new hash with the provided parameters
    expected_hash = generate_waveform_hash(A, phi)
    
    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(wave_hash, expected_hash)

def extract_parameters_from_hash(wave_hash):
    """
    Extract amplitude and phase values from a geometric hash.
    
    Args:
        wave_hash (str): Hash containing amplitude and phase info
        
    Returns:
        tuple: (amplitude, phase) extracted from the hash, or (None, None) if invalid
    """
    try:
        # Hash format: A{amplitude}_P{phase}_{hash}
        parts = wave_hash.split('_')
        if len(parts) < 3 or not parts[0].startswith('A') or not parts[1].startswith('P'):
            return None, None
        
        amplitude = float(parts[0][1:])
        phase = float(parts[1][1:])
        
        return amplitude, phase
    except (ValueError, IndexError):
        return None, None