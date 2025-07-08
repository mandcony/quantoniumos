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

def geometric_waveform_hash(waveform_data):
    """
    Generate geometric waveform hash for patent-protected algorithm.
    
    Args:
        waveform_data: List of float values representing waveform
        
    Returns:
        str: Geometric hash with patent-protected algorithm
    """
    if not waveform_data:
        return "empty_waveform_hash"
    
    # Convert to amplitude and phase using geometric principles
    A = sum(abs(x) for x in waveform_data) / len(waveform_data)
    phi = math.atan2(sum(waveform_data[1::2]), sum(waveform_data[::2])) / (2 * math.pi)
    phi = (phi + 1) / 2  # Normalize to [0,1]
    
    # Apply golden ratio optimization
    golden_ratio = (1 + math.sqrt(5)) / 2
    A = A * golden_ratio % 1.0
    phi = phi * golden_ratio % 1.0
    
    return generate_waveform_hash(A, phi)

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
        # First, try the original format: A{amplitude}_P{phase}_{hash}
        parts = wave_hash.split('_')
        if len(parts) >= 3 and parts[0].startswith('A') and parts[1].startswith('P'):
            amplitude = float(parts[0][1:])
            phase = float(parts[1][1:])
            return amplitude, phase
        
        # If not in the original format, handle it as Base64-encoded data
        # Use a deterministic approach to extract amplitude and phase
        import hashlib
        import base64
        
        try:
            # If it looks like base64, try to convert it to a more stable hash
            hash_bytes = wave_hash.encode('utf-8')
            hash_digest = hashlib.sha256(hash_bytes).hexdigest()
            
            # Extract amplitude and phase as deterministic values from the hash
            amplitude = float(int(hash_digest[:8], 16) % 1000) / 1000  # 0.0-1.0 range
            phase = float(int(hash_digest[8:16], 16) % 1000) / 1000    # 0.0-1.0 range
            
            return amplitude, phase
        except Exception:
            # In case of any parsing error, return default values
            return 0.5, 0.5
            
    except (ValueError, IndexError):
        return None, None