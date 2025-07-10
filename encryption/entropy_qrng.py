"""
QuantoniumOS - Entropy QRNG

This module implements a quantum-inspired random number generator for entropy
using oscillator drift model as specified in the QuantoniumOS patent.

The core algorithm is:
    dφ = (τ_n ⊕ τ_{n-1}) mod π
where τ is seeded from os.getrandom(16).
"""

import base64
import hashlib
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("quantonium_api.entropy")

# Define entropy source
source = "QUANTUM_OSCILLATOR"  # Used to verify the correct source is used

# Global oscillator state
_oscillator_state = None


def init_oscillator():
    """Initialize the oscillator with secure random seed"""
    global _oscillator_state

    # Get 16 bytes of entropy from the OS
    try:
        seed_bytes = os.urandom(16)
    except Exception as e:
        logger.error(f"Failed to get entropy from OS: {str(e)}")
        # Fallback to secrets module
        seed_bytes = secrets.token_bytes(16)

    # Convert to array of 16 unsigned bytes
    state = np.frombuffer(seed_bytes, dtype=np.uint8)

    # Initialize phase values (τ values) from the seed
    phases = np.zeros(16, dtype=np.float64)
    for i in range(16):
        # Map each byte to a phase in [0, π)
        phases[i] = (state[i] / 255.0) * np.pi

    _oscillator_state = {
        "seed": seed_bytes,
        "state": state,
        "phases": phases,
        "counter": 0,
        "last_update": time.time(),
    }

    logger.info("Quantum oscillator initialized")
    return _oscillator_state


def update_oscillator():
    """
    Update the oscillator state using the drift model:
    dφ = (τ_n ⊕ τ_{n-1}) mod π
    """
    if _oscillator_state is None:
        init_oscillator()

    # Get current state
    state = _oscillator_state["state"]
    phases = _oscillator_state["phases"]
    counter = _oscillator_state["counter"]

    # Update each phase value
    for i in range(16):
        prev_i = (i - 1) % 16

        # Calculate phase change using XOR of adjacent cells
        # and current cell's state value
        xor_val = int(state[i]) ^ int(state[prev_i])

        # Convert to phase change
        d_phase = (xor_val / 255.0) * np.pi

        # Apply drift model: exclusive-or with previous phase
        phases[i] = (phases[i] + d_phase) % np.pi

        # Update state based on phase
        state[i] = int((phases[i] / np.pi) * 255)

    # Update counter and timestamp
    _oscillator_state["counter"] += 1
    _oscillator_state["last_update"] = time.time()

    return _oscillator_state


def sample_entropy(num_bytes: int = 32) -> bytes:
    """
    Sample entropy from the quantum oscillator

    Args:
        num_bytes: Number of bytes to generate

    Returns:
        Random bytes from the quantum oscillator
    """
    if _oscillator_state is None:
        init_oscillator()

    # Number of update iterations needed
    iterations = (num_bytes + 15) // 16

    # Collect entropy
    entropy_bytes = bytearray()

    for _ in range(iterations):
        # Update oscillator
        update_oscillator()

        # Add current state to entropy
        entropy_bytes.extend(_oscillator_state["state"])

    # Truncate to requested size
    return bytes(entropy_bytes[:num_bytes])


def generate_entropy(bits: int = 512) -> Dict[str, Any]:
    """
    Generate entropy and compute Shannon entropy measure

    Args:
        bits: Number of entropy bits to generate

    Returns:
        Dictionary with:
        - entropy: Shannon entropy measure (bits per byte)
        - raw_entropy: Base64-encoded entropy bytes
    """
    # Calculate bytes needed
    bytes_needed = (bits + 7) // 8

    # Generate entropy
    entropy_bytes = sample_entropy(bytes_needed)

    # Compute Shannon entropy measure
    shannon = compute_shannon_entropy(entropy_bytes)

    # Encode as base64 for transport
    b64_entropy = base64.b64encode(entropy_bytes).decode("ascii")

    return {"entropy": shannon, "raw_entropy": b64_entropy, "bits": bits}


def compute_shannon_entropy(data: bytes) -> float:
    """
    Compute Shannon entropy of a byte sequence

    Args:
        data: Input bytes

    Returns:
        Entropy in bits per byte (0.0 to 8.0)
    """
    if not data:
        return 0.0

    # Count frequency of each byte value
    counts = {}
    for byte in data:
        counts[byte] = counts.get(byte, 0) + 1

    # Calculate Shannon entropy
    length = len(data)
    entropy = 0.0

    for count in counts.values():
        probability = count / length
        entropy -= probability * np.log2(probability)

    return float(entropy)


def chi_squared_test(data: bytes) -> float:
    """
    Perform a chi-squared test on the data to measure randomness
    Lower values indicate more randomness.

    Args:
        data: Input bytes

    Returns:
        Chi-squared statistic
    """
    counts = [0] * 256

    # Count occurrences of each byte value
    for byte in data:
        counts[byte] += 1

    # Expected count for uniform distribution
    expected = len(data) / 256

    # Calculate chi-squared statistic
    chi_squared = 0.0
    for count in counts:
        chi_squared += ((count - expected) ** 2) / expected

    return float(chi_squared)


def entropy_drift_test(sample1: bytes, sample2: bytes) -> float:
    """
    Measure the entropy drift between two samples

    Args:
        sample1: First entropy sample
        sample2: Second entropy sample

    Returns:
        Drift percentage (0.0 to 100.0)
    """
    e1 = compute_shannon_entropy(sample1)
    e2 = compute_shannon_entropy(sample2)

    # Compute percentage difference
    avg = (e1 + e2) / 2
    if avg == 0:
        return 0.0

    diff = abs(e1 - e2)
    drift = (diff / avg) * 100.0

    return float(drift)
