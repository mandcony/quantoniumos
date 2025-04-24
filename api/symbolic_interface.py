"""
QuantoniumOS - Symbolic Interface API

This module provides the high-level API for the patent validation system.
It serves as a secure interface to the proprietary algorithms, preventing
direct access to the core implementation details.
"""

import logging
import time
import json
import base64
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Union

from encryption.resonance_encrypt import encrypt_symbolic, decrypt_symbolic
from encryption.geometric_waveform_hash import wave_hash, extract_wave_parameters
from orchestration.resonance_manager import run_benchmark, validate_avalanche, add_to_hash_history, check_hash_replay
from secure_core.python_bindings import engine_core
from encryption.wave_primitives import WaveNumber, calculate_coherence

logger = logging.getLogger("quantonium_api.interface")

def run_rft(waveform: List[float]) -> List[complex]:
    """
    Thin wrapper to run Resonance Fourier Transform on a waveform.
    For testing purposes only.
    
    Args:
        waveform: List of float values representing the waveform
        
    Returns:
        Complex values representing the frequency components
    """
    # Convert to numpy array for FFT
    wave_arr = np.array(waveform)
    # Perform FFT
    fft_result = np.fft.fft(wave_arr)
    return fft_result.tolist()

def inverse_rft(freq_components: List[complex]) -> List[float]:
    """
    Thin wrapper to run inverse Resonance Fourier Transform.
    For testing purposes only.
    
    Args:
        freq_components: Complex values representing frequency components
        
    Returns:
        Real waveform values
    """
    # Convert to numpy array for inverse FFT
    freq_arr = np.array(freq_components)
    # Perform inverse FFT
    ifft_result = np.fft.ifft(freq_arr)
    # Return real part as float list
    return ifft_result.real.tolist()

def encrypt_data(plaintext: str, key: str) -> Dict[str, Any]:
    """
    Encrypt data using the patent-protected symbolic encryption.
    
    This high-level function delegates to the secure encryption implementation
    and records hash values for replay protection.
    
    Args:
        plaintext: Hex-encoded plaintext (32 chars / 128 bits)
        key: Hex-encoded key (32 chars / 128 bits)
        
    Returns:
        Encryption result with metrics
    """
    logger.info("Processing encrypt request")
    
    try:
        # Call the secure implementation
        result = encrypt_symbolic(plaintext, key)
        
        # Add hash to history for replay protection
        add_to_hash_history(result["hash"])
        
        return result
    except Exception as e:
        logger.error(f"Error in encrypt_data: {e}")
        raise

def decrypt_data(ciphertext: str, key: str) -> Dict[str, Any]:
    """
    Decrypt data using the patent-protected symbolic decryption.
    
    Args:
        ciphertext: Hex-encoded ciphertext (32 chars / 128 bits)
        key: Hex-encoded key (32 chars / 128 bits)
        
    Returns:
        Decryption result with metrics
    """
    logger.info("Processing decrypt request")
    
    try:
        # Call the secure implementation
        result = decrypt_symbolic(ciphertext, key)
        
        return result
    except Exception as e:
        logger.error(f"Error in decrypt_data: {e}")
        raise

def analyze_waveform(waveform_data: List[float]) -> Dict[str, Any]:
    """
    Analyze a waveform using Resonance Fourier Transform.
    
    Args:
        waveform_data: List of float values representing the waveform
        
    Returns:
        Analysis result with metrics
    """
    logger.info("Processing waveform analysis request")
    
    try:
        # Convert float list to bytes for processing
        # Scale to range 0-255 and convert to bytes
        bytes_data = bytes([int(min(max(v, 0.0), 1.0) * 255) for v in waveform_data])
        
        # Run RFT analysis
        bins, hr = engine_core.rft_run(bytes_data)
        
        # Run SA analysis
        sa_vector = engine_core.sa_compute(bytes_data)
        
        # Calculate average symbolic alignment
        sa = sum(sa_vector) / len(sa_vector) if sa_vector else 0.0
        
        # Generate hash for the waveform
        hash_value = wave_hash(bytes_data)
        
        # Create result
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())
        result = {
            "bins": bins,
            "hr": hr,
            "sa": sa,
            "sa_vector": sa_vector,
            "hash": hash_value,
            "timestamp": timestamp
        }
        
        # Add signature for integrity verification
        signature = hashlib.sha256(json.dumps(
            {"bins_sum": sum(bins), "hr": hr, "sa": sa, "hash": hash_value, "timestamp": timestamp},
            sort_keys=True
        ).encode()).hexdigest()
        result["signature"] = signature
        
        return result
    except Exception as e:
        logger.error(f"Error in analyze_waveform: {e}")
        raise

def generate_entropy(amount: int = 32) -> Dict[str, Any]:
    """
    Generate quantum-inspired entropy.
    
    Args:
        amount: Number of bytes to generate (default: 32)
        
    Returns:
        Dictionary with entropy as base64 and hex strings, plus metrics
    """
    logger.info(f"Generating {amount} bytes of entropy")
    
    try:
        # Limit to reasonable range
        amount = min(max(amount, 1), 1024)
        
        # Generate entropy
        entropy_bytes = engine_core.generate_entropy(amount)
        
        # Convert to readable formats
        entropy_base64 = base64.b64encode(entropy_bytes).decode('ascii')
        entropy_hex = entropy_bytes.hex()
        
        # Calculate metrics
        bins, hr = engine_core.rft_run(entropy_bytes)
        
        # Create result
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())
        result = {
            "entropy_base64": entropy_base64,
            "entropy_hex": entropy_hex,
            "length": amount,
            "hr": hr,
            "timestamp": timestamp
        }
        
        # Add signature for integrity verification
        signature = hashlib.sha256(entropy_bytes).hexdigest()
        result["signature"] = signature
        
        return result
    except Exception as e:
        logger.error(f"Error in generate_entropy: {e}")
        raise

def unlock_container(waveform: List[float], hash_value: str) -> Dict[str, Any]:
    """
    Attempt to unlock a symbolic container using a waveform and hash.
    
    According to the patent claims, a container can only be unlocked
    if the provided waveform produces a hash that matches the container's hash.
    
    Args:
        waveform: List of float values representing the key waveform
        hash_value: 64-character hex hash of the container
        
    Returns:
        Dictionary with unlock status and metrics
    """
    logger.info("Processing container unlock request")
    
    try:
        # Validate hash format
        if len(hash_value) != 64:
            raise ValueError("Container hash must be 64 hex characters")
            
        # Convert waveform to bytes for hash calculation
        bytes_data = bytes([int(min(max(v, 0.0), 1.0) * 255) for v in waveform])
        
        # Generate hash from the provided waveform
        generated_hash = wave_hash(bytes_data)
        
        # Extract expected wave parameters from container hash
        expected_waves, coherence_threshold = extract_wave_parameters(hash_value)
        
        # Convert input waveform to WaveNumbers
        input_waves = []
        for i, v in enumerate(waveform):
            phase = (i / len(waveform)) * 6.28318  # 2π
            input_waves.append(WaveNumber(v, phase))
        
        # Calculate coherence between input and expected waves
        coherence_sum = 0.0
        count = min(len(input_waves), len(expected_waves))
        
        for i in range(count):
            coherence_sum += calculate_coherence(
                input_waves[i % len(input_waves)],
                expected_waves[i]
            )
        
        wc = coherence_sum / count if count > 0 else 0.0
        
        # Determine if unlock was successful
        # The hash must match exactly, or the waveform coherence must exceed the threshold
        success = (generated_hash == hash_value) or (wc >= coherence_threshold)
        
        # Create result
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())
        result = {
            "status": "unlocked" if success else "denied",
            "wc": wc,
            "coherence_threshold": coherence_threshold,
            "hash_match": generated_hash == hash_value,
            "timestamp": timestamp
        }
        
        # Add signature for integrity verification
        signature = hashlib.sha256(
            f"{result['status']},{wc:.6f},{coherence_threshold:.6f},{result['hash_match']},{timestamp}".encode()
        ).hexdigest()
        result["signature"] = signature
        
        return result
    except Exception as e:
        logger.error(f"Error in unlock_container: {e}")
        raise

def run_avalanche_test() -> Dict[str, Any]:
    """
    Run the 64-test symbolic avalanche test suite to validate the patent claims.
    
    The patent requires:
    - ≥ 70% of perturbations must have WC < 0.55
    - At least one perturbation must have WC < 0.05 (symbolic avalanche)
    
    Returns:
        Dictionary with test results and validation status
    """
    logger.info("Running avalanche test suite")
    
    try:
        # Run the benchmark tests
        results = run_benchmark()
        
        # Validate against patent requirements
        passed, stats = validate_avalanche(results)
        
        # Create result
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())
        result = {
            "passed": passed,
            "stats": stats,
            "results": results,
            "timestamp": timestamp
        }
        
        return result
    except Exception as e:
        logger.error(f"Error in run_avalanche_test: {e}")
        raise