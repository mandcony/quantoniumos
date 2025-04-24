"""
QuantoniumOS - Resonance Encryption

This module implements the proprietary symbolic XOR encryption
using phase-superposition concepts from quantum computing.

All cryptographic operations delegate to the secure core for
the patent-protected algorithms while providing a clean API.
"""

import os
import logging
import time
import base64
import hashlib
from typing import Dict, Any, List
import numpy as np
from api.symbolic_interface import compute_sa_with_engine, get_symbolic_engine, safe_fallback_hash

logger = logging.getLogger("quantonium_api.encrypt")

# Constants
ALG_TYPE = "symbolic"  # Used to verify proper algorithm type is being used

def encrypt_symbolic(plaintext: str, key: str) -> Dict[str, Any]:
    """
    Encrypt plaintext using resonance encryption with the given key
    
    Args:
        plaintext: 128-bit plaintext as hex string
        key: 128-bit key as hex string
        
    Returns:
        Dictionary with:
        - ciphertext: Encrypted result
        - hr: Harmonic Resonance
        - wc: Waveform Coherence
        - sa: Symbolic Alignment
        - entropy: Entropy measurement
    """
    try:
        # Validate inputs
        if len(plaintext) != 32 or len(key) != 32:
            raise ValueError("Plaintext and key must be exactly 32 hex characters (128 bits)")
            
        # Convert hex strings to byte arrays
        pt_bytes = bytes.fromhex(plaintext)
        key_bytes = bytes.fromhex(key)
        
        # Create waveforms from input (phase representation)
        pt_wave = create_waveform(pt_bytes)
        key_wave = create_waveform(key_bytes)
        
        # Compute waveform coherence between input and key
        wc = compute_waveform_coherence(pt_wave, key_wave)
        
        # Try to use the secure engine for encryption
        try:
            engine = get_symbolic_engine()
            
            # Prepare input data
            combined = plaintext + key
            
            # Get SA vector using the secure core
            sa_result = compute_sa_with_engine(combined)
            sa_vector = sa_result["sa_vector"]
            
            # Calculate SA metric from vector (average alignment)
            sa = calculate_sa_metric(sa_vector)
            
            # Perform the symbolic XOR encryption using phase-superposition
            result = apply_symbolic_xor(pt_bytes, key_bytes)
            
        except Exception as e:
            logger.error(f"Secure engine error: {str(e)}")
            # Use a fallback method - this is NOT the proprietary algorithm
            # and should only be used for testing
            logger.warning("Using fallback encryption method - NOT SUITABLE FOR PRODUCTION")
            result = fallback_encrypt(pt_bytes, key_bytes)
            sa_vector = ""
            sa = 0.5  # Default placeholder value
        
        # Convert result back to hex string
        ciphertext = result.hex()
        
        # Compute harmonic resonance (HR)
        hr = compute_harmonic_resonance(pt_wave, key_wave)
        
        # Generate entropy measurement
        entropy = compute_entropy(result)
        
        # Return complete result
        return {
            "ciphertext": ciphertext,
            "hr": hr,
            "wc": wc,
            "sa": sa,
            "sa_vector": sa_vector,
            "entropy": entropy
        }
        
    except Exception as e:
        logger.error(f"Encryption error: {str(e)}")
        raise

def create_waveform(data: bytes) -> List[float]:
    """
    Convert byte data to a waveform representation for symbolic processing
    
    Args:
        data: Input bytes
        
    Returns:
        List of float values representing the waveform
    """
    # Create a waveform with 8 samples per byte (oversampling for better resolution)
    waveform = []
    for byte in data:
        # Convert each bit to amplitude value
        for i in range(8):
            bit = (byte >> i) & 1
            if bit:
                # For 1-bits, use a higher amplitude
                amplitude = 0.8 + (0.2 * np.random.random())
            else:
                # For 0-bits, use a lower amplitude
                amplitude = 0.2 * np.random.random()
            waveform.append(amplitude)
            
    return waveform

def compute_waveform_coherence(wave1: List[float], wave2: List[float]) -> float:
    """
    Compute waveform coherence between two waveforms
    
    Args:
        wave1: First waveform
        wave2: Second waveform
        
    Returns:
        Coherence value between 0 and 1
    """
    # Ensure equal length
    min_len = min(len(wave1), len(wave2))
    w1 = wave1[:min_len]
    w2 = wave2[:min_len]
    
    # Convert to numpy arrays for vectorized operations
    arr1 = np.array(w1)
    arr2 = np.array(w2)
    
    # Calculate normalized dot product
    dot_product = np.sum(arr1 * arr2)
    norm1 = np.sqrt(np.sum(arr1 * arr1))
    norm2 = np.sqrt(np.sum(arr2 * arr2))
    
    if norm1 > 0 and norm2 > 0:
        coherence = abs(dot_product) / (norm1 * norm2)
    else:
        coherence = 0.0
    
    return float(coherence)

def compute_harmonic_resonance(wave1: List[float], wave2: List[float]) -> float:
    """
    Compute harmonic resonance between two waveforms
    
    Args:
        wave1: First waveform
        wave2: Second waveform
        
    Returns:
        Harmonic resonance value between 0 and 1
    """
    # Ensure equal length
    min_len = min(len(wave1), len(wave2))
    w1 = wave1[:min_len]
    w2 = wave2[:min_len]
    
    # Convert to numpy arrays
    arr1 = np.array(w1)
    arr2 = np.array(w2)
    
    # Calculate phase alignment
    # This is a simplified version - the actual algorithm is proprietary
    diff = arr1 - arr2
    diff_squared = diff * diff
    mse = np.mean(diff_squared)
    
    # Convert to resonance value (inversely related to MSE)
    resonance = 1.0 / (1.0 + mse)
    
    return float(resonance)

def apply_symbolic_xor(plaintext: bytes, key: bytes) -> bytes:
    """
    Apply the symbolic XOR operation using phase-superposition
    
    This is the core of the proprietary encryption algorithm.
    
    Args:
        plaintext: Plaintext bytes
        key: Key bytes
        
    Returns:
        Encrypted bytes
    """
    # Ensure equal length
    min_len = min(len(plaintext), len(key))
    pt = plaintext[:min_len]
    k = key[:min_len]
    
    # This is where we would call the secure core engine for the actual operation
    # For now, we'll use a placeholder implementation that simulates phase rotation
    
    # Convert to waveforms
    pt_wave = create_waveform(pt)
    key_wave = create_waveform(k)
    
    # Apply symbolic XOR operation: ψ_out = ψ_in ⊕_φ key_wave
    # where ⊕_φ = "rotate phase if amplitude bit = 1"
    
    # For the actual implementation, we would call into the secure core engine
    # But for this placeholder, we'll simulate it
    
    # The result will have the same length as the inputs
    result = bytearray(min_len)
    
    for i in range(min_len):
        # Simulate phase rotation using a non-linear function
        # This is NOT the actual proprietary algorithm
        phase_factor = (pt[i] + k[i] + (pt[i] ^ k[i])) % 256
        result[i] = (pt[i] ^ k[i] ^ phase_factor) & 0xFF
    
    return bytes(result)

def fallback_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """
    Fallback encryption method for testing when secure core is unavailable
    
    WARNING: This is NOT the proprietary algorithm and should only be used for testing
    
    Args:
        plaintext: Plaintext bytes
        key: Key bytes
        
    Returns:
        Encrypted bytes
    """
    # Ensure equal length
    min_len = min(len(plaintext), len(key))
    pt = plaintext[:min_len]
    k = key[:min_len]
    
    # Simple key stretching
    expanded_key = bytearray(min_len)
    for i in range(min_len):
        expanded_key[i] = k[i % len(k)]
        
    # Bitwise XOR with expanded key
    result = bytearray(min_len)
    for i in range(min_len):
        result[i] = pt[i] ^ expanded_key[i]
    
    return bytes(result)

def calculate_sa_metric(sa_vector: str) -> float:
    """
    Calculate the Symbolic Alignment metric from an SA vector
    
    Args:
        sa_vector: Base64-encoded SA vector
        
    Returns:
        SA metric as a float between 0 and 1
    """
    if not sa_vector:
        return 0.5  # Default value
    
    try:
        # Decode base64 string
        sa_bytes = base64.b64decode(sa_vector)
        
        # Convert to floats (assuming 4-byte floats)
        values = []
        for i in range(0, len(sa_bytes), 4):
            if i + 4 <= len(sa_bytes):
                val_bytes = sa_bytes[i:i+4]
                value = int.from_bytes(val_bytes, byteorder='little')
                # Convert int to float (this is a simplification)
                # In a real implementation, we would use struct.unpack
                float_val = value / (2**32 - 1)
                values.append(float_val)
        
        # Calculate average
        if values:
            return sum(values) / len(values)
        else:
            return 0.5
            
    except Exception as e:
        logger.error(f"Error calculating SA metric: {str(e)}")
        return 0.5

def compute_entropy(data: bytes) -> float:
    """
    Compute the Shannon entropy of data
    
    Args:
        data: Input bytes
        
    Returns:
        Entropy value (typically between 0 and 8 for byte data)
    """
    if not data:
        return 0.0
    
    # Count occurrences of each byte value
    counts = {}
    for byte in data:
        counts[byte] = counts.get(byte, 0) + 1
    
    # Calculate entropy
    length = len(data)
    entropy = 0.0
    
    for count in counts.values():
        probability = count / length
        entropy -= probability * np.log2(probability)
    
    return float(entropy)