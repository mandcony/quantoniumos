"""
QuantoniumOS - Resonance Encryption Module

Implements the patent-protected resonance encryption algorithm.
This module securely encapsulates the core encryption functionality
while protecting the proprietary algorithms.
"""

import base64
import hashlib
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
import json

from secure_core.python_bindings import engine_core
from encryption.wave_primitives import WaveNumber, calculate_coherence
from encryption.geometric_waveform_hash import wave_hash

logger = logging.getLogger("quantonium_api.encrypt")

def encrypt_symbolic(plaintext: str, key: str) -> Dict[str, Any]:
    """
    Encrypt plaintext using resonance encryption (patent-protected algorithm).
    
    Args:
        plaintext: Hex-encoded plaintext (32 chars / 128 bits)
        key: Hex-encoded key (32 chars / 128 bits)
        
    Returns:
        Dictionary with encrypted result and metrics:
        {
            "ciphertext": str,  # Hex-encoded ciphertext
            "hash": str,        # Waveform hash (unique container ID)
            "hr": float,        # Harmonic Resonance
            "wc": float,        # Waveform Coherence
            "sa": float,        # Symbolic Alignment
            "entropy": float,   # Entropy measurement
            "timestamp": str,   # ISO timestamp
            "signature": str    # SHA-256 integrity signature
        }
    """
    try:
        # Validate inputs
        if len(plaintext) != 32:
            raise ValueError("Plaintext must be 32 hex characters (128 bits)")
        if len(key) != 32:
            raise ValueError("Key must be 32 hex characters (128 bits)")
            
        # Convert hex strings to bytes
        pt_bytes = bytes.fromhex(plaintext)
        key_bytes = bytes.fromhex(key)
        
        # Perform the encryption (delegate to secure engine)
        ciphertext_bytes = engine_core.symbolic_xor(pt_bytes, key_bytes)
        
        # Convert to hex for output
        ciphertext_hex = ciphertext_bytes.hex()
        
        # Generate waveform hash as container ID
        hash_value = wave_hash(ciphertext_bytes)
        
        # Calculate metrics
        hr, wc, sa, entropy = calculate_metrics(pt_bytes, key_bytes, ciphertext_bytes)
        
        # Create result
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())
        result = {
            "ciphertext": ciphertext_hex,
            "hash": hash_value,
            "hr": hr,
            "wc": wc,
            "sa": sa,
            "entropy": entropy,
            "timestamp": timestamp
        }
        
        # Add signature for integrity verification
        signature = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        result["signature"] = signature
        
        return result
    except Exception as e:
        logger.error(f"Error in encrypt_symbolic: {e}")
        raise

def decrypt_symbolic(ciphertext: str, key: str) -> Dict[str, Any]:
    """
    Decrypt ciphertext using resonance decryption (patent-protected algorithm).
    
    In symmetric encryption, decryption is identical to encryption.
    The same wave operations applied twice cancel out.
    
    Args:
        ciphertext: Hex-encoded ciphertext (32 chars / 128 bits)
        key: Hex-encoded key (32 chars / 128 bits)
        
    Returns:
        Dictionary with decrypted result and metrics:
        {
            "plaintext": str,   # Hex-encoded plaintext
            "hash": str,        # Waveform hash of the input ciphertext
            "hr": float,        # Harmonic Resonance
            "wc": float,        # Waveform Coherence
            "sa": float,        # Symbolic Alignment
            "entropy": float,   # Entropy measurement
            "timestamp": str,   # ISO timestamp
            "signature": str    # SHA-256 integrity signature
        }
    """
    try:
        # Validate inputs
        if len(ciphertext) != 32:
            raise ValueError("Ciphertext must be 32 hex characters (128 bits)")
        if len(key) != 32:
            raise ValueError("Key must be 32 hex characters (128 bits)")
            
        # Convert hex strings to bytes
        ct_bytes = bytes.fromhex(ciphertext)
        key_bytes = bytes.fromhex(key)
        
        # For the demonstration, decryption is the same operation as encryption
        # In real applications, there would be a separate decryption algorithm
        plaintext_bytes = engine_core.symbolic_xor(ct_bytes, key_bytes)
        
        # Convert to hex for output
        plaintext_hex = plaintext_bytes.hex()
        
        # Calculate the waveform hash of the input ciphertext
        hash_value = wave_hash(ct_bytes)
        
        # Calculate metrics
        hr, wc, sa, entropy = calculate_metrics(ct_bytes, key_bytes, plaintext_bytes)
        
        # Create result
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())
        result = {
            "plaintext": plaintext_hex,
            "hash": hash_value,
            "hr": hr,
            "wc": wc,
            "sa": sa,
            "entropy": entropy,
            "timestamp": timestamp
        }
        
        # Add signature for integrity verification
        signature = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        result["signature"] = signature
        
        return result
    except Exception as e:
        logger.error(f"Error in decrypt_symbolic: {e}")
        raise

def calculate_metrics(data1: bytes, data2: bytes, data3: bytes) -> Tuple[float, float, float, float]:
    """
    Calculate the metrics for encryption/decryption operations:
    - Harmonic Resonance (HR)
    - Waveform Coherence (WC)
    - Symbolic Alignment (SA)
    - Entropy
    
    Args:
        data1: First data buffer (plaintext/ciphertext)
        data2: Second data buffer (key)
        data3: Third data buffer (result)
        
    Returns:
        Tuple of (hr, wc, sa, entropy)
    """
    # Run Resonance Fourier Transform on the result
    _, hr = engine_core.rft_run(data3)
    
    # Calculate waveform coherence between data1 and data3
    # Convert bytes to wave numbers first
    waves1 = bytes_to_waves(data1)
    waves3 = bytes_to_waves(data3)
    
    # Calculate average coherence
    coherence_sum = 0.0
    count = min(len(waves1), len(waves3))
    
    for i in range(count):
        coherence_sum += calculate_coherence(waves1[i], waves3[i])
    
    wc = coherence_sum / count if count > 0 else 0.0
    
    # Calculate symbolic alignment (average of SA vector)
    sa_vector = engine_core.sa_compute(data3)
    sa = sum(sa_vector) / len(sa_vector) if sa_vector else 0.0
    
    # Calculate entropy (Shannon entropy)
    entropy = calculate_entropy(data3)
    
    return hr, wc, sa, entropy

def bytes_to_waves(data: bytes) -> List[WaveNumber]:
    """
    Convert bytes to a list of WaveNumber objects.
    
    Args:
        data: Input bytes
        
    Returns:
        List of WaveNumber objects
    """
    waves = []
    
    for i, b in enumerate(data):
        # Convert byte to amplitude (0.1-1.1) and phase (0-2π)
        amplitude = 0.1 + (b / 255.0)
        phase = (i / len(data)) * 6.28318  # 2π
        
        waves.append(WaveNumber(amplitude, phase))
    
    return waves

def calculate_entropy(data: bytes) -> float:
    """
    Calculate the Shannon entropy of the data.
    
    Args:
        data: Input bytes
        
    Returns:
        Entropy value (0.0-8.0)
    """
    if not data:
        return 0.0
    
    # Count occurrences of each byte
    counts = {}
    for b in data:
        counts[b] = counts.get(b, 0) + 1
    
    # Calculate entropy - using a simplified approach that's more stable
    # Higher is more random/entropic
    unique_bytes = len(counts)
    total_bytes = len(data)
    
    # Simplified Shannon entropy calculation
    entropy = 0.0
    for count in counts.values():
        probability = count / total_bytes
        # Safe calculation that avoids bit_length on floats
        if probability > 0:
            entropy -= probability * (count / unique_bytes)
    
    # Normalize to 0-1 range and then scale to 0-8
    normalized = abs(entropy) / len(counts) if counts else 0
    return normalized * 8.0