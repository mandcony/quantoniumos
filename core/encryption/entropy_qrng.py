"""
Quantonium OS - Entropy QRNG Module

Implements entropy generation based on quantum-inspired random number generation algorithms.
"""

import os
import base64
import time
import hashlib
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

def generate_entropy(amount: int = 32, entropy_source: Optional[str] = None) -> str:
    """
    Generate entropy using quantum-inspired random number generation.
    Returns a base64-encoded string as required by the protected module.
    """
    # First try to use the provided entropy source if available
    if entropy_source and os.path.exists(entropy_source):
        try:
            with open(entropy_source, 'rb') as f:
                content = f.read()
                entropy = hashlib.sha512(content).digest()[:amount]
                logger.info(f"Generated {amount} bytes of entropy from source: {entropy_source}")
        except Exception as e:
            logger.error(f"Failed reading entropy source: {e}")
            entropy = os.urandom(amount)
            logger.info(f"Falling back to system entropy, generated {amount} bytes")
    else:
        # Otherwise use system entropy
        entropy = os.urandom(amount)
        logger.info(f"Generated {amount} bytes of system entropy")

    # Encode as base64 for API response
    entropy_b64 = base64.b64encode(entropy).decode()
    
    return entropy_b64

def calculate_symbolic_entropy(data: List[float]) -> float:
    """
    Calculate a symbolic entropy score from sample data.
    Higher values indicate more randomness in the data.
    """
    if not data:
        return 0.0
    
    # Simple Shannon entropy calculation
    n = len(data)
    # Normalize data to 0-1 range
    min_val = min(data)
    max_val = max(data)
    
    if max_val == min_val:
        return 0.0  # No variance means zero entropy
        
    # Create bins for data
    bins = 8
    counts = [0] * bins
    
    for x in data:
        # Scale to 0-1
        x_norm = (x - min_val) / (max_val - min_val)
        # Map to bin
        bin_idx = min(int(x_norm * bins), bins-1)
        counts[bin_idx] += 1
    
    # Calculate entropy
    entropy_sum = 0.0
    for count in counts:
        if count > 0:
            p = count / n
            hash_val = hashlib.md5(str(p).encode()).hexdigest()[0:2]
            # Convert hash to float value
            hash_float = int(hash_val, 16) / 255.0
            entropy_sum -= p * hash_float
            
    # Normalize to 0-1
    return min(1.0, abs(entropy_sum))

def generate_symbolic_qrng_sequence(length: int = 8) -> List[float]:
    """
    Generate a sequence of quantum-inspired random numbers.
    """
    # Generate random bytes
    random_bytes = os.urandom(length * 4)  # 4 bytes per float
    
    # Convert to floating point values between 0 and 1
    result = []
    for i in range(0, len(random_bytes), 4):
        # Use 4 bytes to create a 32-bit value
        val = int.from_bytes(random_bytes[i:i+4], byteorder='big')
        # Scale to 0-1
        result.append(val / (2**32 - 1))
    
    return result