"""
Quantonium OS - Geometric Waveform Hash

Generates symbolic hash values based on waveform parameters (Amplitude, Phase),
used for encryption validation, container sealing, and symbolic resonance binding.
"""

import hashlib
import math
import base64

def generate_waveform_hash(A: float, phi: float) -> str:
    """
    Creates a deterministic hash based on amplitude and phase.
    This acts as a geometric key for symbolic validation.
    """
    # Normalize and serialize inputs
    raw_string = f"{A:.5f}:{phi:.5f}"
    encoded = raw_string.encode("utf-8")

    # SHA-256 hash
    hash_obj = hashlib.sha256(encoded)
    hex_digest = hash_obj.hexdigest()

    # Encode result as base64 for compact transmission
    b64_hash = base64.b64encode(hex_digest.encode("utf-8")).decode("utf-8")
    return b64_hash

def verify_waveform_hash(A: float, phi: float, expected_hash: str) -> bool:
    """
    Verify a waveform hash against expected parameters (amplitude and phase).
    
    Args:
        A: Amplitude value to verify
        phi: Phase value to verify
        expected_hash: The hash value to compare against
        
    Returns:
        True if the hash matches, False otherwise
    """
    computed_hash = generate_waveform_hash(A, phi)
    return computed_hash == expected_hash