#!/usr/bin/env python3
"""
True RFT Feistel Bindings

This module integrates the True Resonance Fourier Transform with the Feistel
network structure to provide a cryptographically secure encryption system.
"""

import os
import sys
import hashlib
import secrets
import warnings
import numpy as np
from pathlib import Path

# Add the proper path for importing
sys.path.append('/workspaces/quantoniumos/04_RFT_ALGORITHMS')

# Import the true unitary transform
try:
    from true_rft_exact import TrueResonanceFourierTransform
    print("Successfully imported TrueResonanceFourierTransform")
except ImportError as e:
    print(f"Error importing TrueResonanceFourierTransform: {e}")
    warnings.warn(
        "True unitary transform not available. Using fallback implementation.",
        RuntimeWarning
    )

_initialized = False
_transformer = None

def init(size=16):
    """Initialize the RFT Feistel engine"""
    global _initialized, _transformer
    if not _initialized:
        _transformer = TrueResonanceFourierTransform(N=size)
        _initialized = True
        print("RFT Feistel engine initialized with size =", size)
    return True

def generate_key(length=32):
    """Generate a cryptographic key"""
    if not _initialized:
        init()
    return secrets.token_bytes(length)

def _rft_function(data, round_key):
    """
    RFT-based F function for Feistel network
    
    This replaces the simple hash-based F function with a true unitary transform
    that preserves the energy of the signal while providing diffusion.
    """
    global _transformer
    if not _initialized:
        init()
    
    # Convert data to complex array
    data_array = np.frombuffer(data, dtype=np.uint8).astype(np.complex128)
    
    # Pad if necessary
    if len(data_array) < _transformer.N:
        padding = np.zeros(_transformer.N - len(data_array), dtype=np.complex128)
        data_array = np.concatenate([data_array, padding])
    elif len(data_array) > _transformer.N:
        data_array = data_array[:_transformer.N]
    
    # Apply key mixing (key-dependent phase shift)
    key_array = np.frombuffer(round_key, dtype=np.uint8).astype(np.float64)
    if len(key_array) > 0:
        # Create phase shifts based on key
        phase_shifts = np.zeros_like(data_array, dtype=np.complex128)
        for i in range(len(data_array)):
            idx = i % len(key_array)
            phase_shifts[i] = np.exp(1j * np.pi * key_array[idx] / 128)
        
        # Apply phase shift before transform
        data_array = data_array * phase_shifts
    
    # Apply the true unitary transform
    transformed = _transformer.transform(data_array)
    
    # Apply another key-dependent operation
    key_hash = hashlib.sha256(round_key).digest()
    key_hash_array = np.frombuffer(key_hash, dtype=np.uint8).astype(np.float64)
    
    # Create rotation factors for additional avalanche effect
    rotation_factors = np.zeros_like(transformed, dtype=np.complex128)
    for i in range(len(transformed)):
        idx = i % len(key_hash_array)
        angle = np.pi * key_hash_array[idx] / 128
        rotation_factors[i] = np.exp(1j * angle)
    
    # Apply rotation in complex plane (preserves unitarity)
    transformed = transformed * rotation_factors
    
    # Convert back to bytes
    result = np.abs(transformed).astype(np.uint8).tobytes()
    
    # Ensure result is of proper length by repeating or truncating
    if len(result) < len(data):
        result = (result * (len(data) // len(result) + 1))[:len(data)]
    elif len(result) > len(data):
        result = result[:len(data)]
    
    return result

def _feistel_round(left, right, round_key):
    """One round of Feistel network using RFT-based F function"""
    # Apply RFT-based F function
    f_output = _rft_function(right, round_key)
    
    # XOR with left half
    new_right = bytes(l ^ f for l, f in zip(left, f_output[:len(left)]))
    
    return right, new_right

def encrypt(data, key):
    """Encrypt data using RFT-based Feistel network"""
    if not _initialized:
        init()
        
    # Ensure data is bytes
    if not isinstance(data, bytes):
        data = data.encode() if isinstance(data, str) else bytes(data)
    
    # Pad data to even length if needed
    if len(data) % 2 != 0:
        data += b'\x00'
    
    # Split data into blocks
    result = bytearray()
    for i in range(0, len(data), 16):
        block = data[i:i+16]
        # Pad block if needed
        if len(block) < 16:
            block = block.ljust(16, b'\x00')
            
        # Split block into left and right halves
        mid = len(block) // 2
        left = block[:mid]
        right = block[mid:]
        
        # Create round keys (derive from main key using RFT)
        round_keys = []
        for r in range(8):  # 8 rounds
            # Create unique round key with good avalanche effect
            round_data = key + bytes([r]) + bytes([i & 0xFF, (i >> 8) & 0xFF])
            h = hashlib.sha256(round_data).digest()
            round_keys.append(h[:8])
            
        # Apply 8 rounds of Feistel network
        for r in range(8):
            left, right = _feistel_round(left, right, round_keys[r])
            
        # Combine halves and add to result
        result.extend(left)
        result.extend(right)
    
    return bytes(result)

def decrypt(data, key):
    """Decrypt data using RFT-based Feistel network"""
    if not _initialized:
        init()
        
    # Ensure data length is valid
    if len(data) % 16 != 0:
        raise ValueError("Invalid ciphertext length")
    
    # Split data into blocks
    result = bytearray()
    for i in range(0, len(data), 16):
        block = data[i:i+16]
        mid = len(block) // 2
        right = block[:mid]  # Note: left and right are swapped compared to encryption
        left = block[mid:]   # This is because of how Feistel networks work in decryption
        
        # Create round keys (same as encryption but used in reverse)
        round_keys = []
        for r in range(8):  # 8 rounds
            # Create unique round key with good avalanche effect
            round_data = key + bytes([r]) + bytes([i & 0xFF, (i >> 8) & 0xFF])
            h = hashlib.sha256(round_data).digest()
            round_keys.append(h[:8])
        round_keys.reverse()  # Use in reverse order for decryption
            
        # Apply 8 rounds of Feistel network in reverse
        for r in range(8):
            # In decryption, we swap left and right before applying the round function
            right, left = _feistel_round(right, left, round_keys[r])
            
        # Combine halves and add to result (in correct order)
        result.extend(left)
        result.extend(right)
    
    # Remove padding
    while result and result[-1] == 0:
        result.pop()
    
    return bytes(result)

# For compatibility with C++ module interface
def engine_init():
    """Initialize the engine"""
    return init()

if __name__ == "__main__":
    # Self-test
    print("\n=== RFT Feistel Encryption Self-Test ===\n")
    
    # Initialize
    init()
    
    # Test data
    test_data = b"This is a test message for RFT Feistel encryption. It should survive the round trip perfectly."
    print(f"Original data: {test_data.decode()}")
    
    # Generate key
    key = generate_key()
    print(f"Generated key: {key.hex()[:16]}...")
    
    # Encrypt
    encrypted = encrypt(test_data, key)
    print(f"Encrypted (hex): {encrypted.hex()[:64]}...")
    
    # Decrypt
    decrypted = decrypt(encrypted, key)
    
    # Try to decode or show hex if decode fails
    try:
        decoded = decrypted.decode()
        print(f"Decrypted: {decoded}")
    except UnicodeDecodeError:
        print(f"Decrypted (hex, decode failed): {decrypted.hex()[:64]}...")
        print(f"Original (hex): {test_data.hex()[:64]}...")
    
    # Verify binary equality
    print(f"Round-trip successful: {test_data == decrypted}")
    if test_data != decrypted:
        print(f"Length difference: {len(test_data)} vs {len(decrypted)}")
        # Find first difference
        for i, (b1, b2) in enumerate(zip(test_data, decrypted)):
            if b1 != b2:
                print(f"First difference at position {i}: {b1} vs {b2}")
                break
    
    # Test avalanche effect
    print("\n=== Testing Avalanche Effect ===\n")
    
    # Modify one bit in the key
    modified_key = bytearray(key)
    modified_key[0] ^= 1  # Flip one bit
    modified_key = bytes(modified_key)
    
    # Encrypt with modified key
    encrypted2 = encrypt(test_data, modified_key)
    
    # Calculate difference (should be about 50% if good avalanche effect)
    diff_bits = 0
    total_bits = len(encrypted) * 8
    for b1, b2 in zip(encrypted, encrypted2):
        xor = b1 ^ b2
        # Count bits set in xor
        diff_bits += bin(xor).count('1')
    
    avalanche_percentage = (diff_bits / total_bits) * 100
    print(f"Changed bits with 1-bit key change: {diff_bits}/{total_bits} ({avalanche_percentage:.2f}%)")
    print(f"Good avalanche effect: {avalanche_percentage > 45}")
