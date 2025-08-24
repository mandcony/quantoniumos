#!/usr/bin/env python3
"""
Fallback Python implementation for minimal_feistel_bindings

This module provides a Python implementation that mimics the interface of the 
C++ minimal_feistel_bindings module. This is used when the C++ module is not available.
"""

import os
import hashlib
import secrets
import warnings

warnings.warn(
    "Using Python fallback for minimal_feistel_bindings. Performance will be significantly reduced.",
    RuntimeWarning
)

_initialized = False

def init():
    """Initialize the Feistel engine"""
    global _initialized
    _initialized = True
    print("Using Python fallback for minimal_feistel_bindings")
    return True

def generate_key(length=32):
    """Generate a cryptographic key"""
    if not _initialized:
        init()
    return secrets.token_bytes(length)

def _feistel_round(left, right, round_key):
    """One round of Feistel network"""
    # Simple F function using hash
    f_output = hashlib.sha256(right + round_key).digest()
    # XOR with left half
    new_right = bytes(l ^ f for l, f in zip(left, f_output[:len(left)]))
    return right, new_right

def encrypt(data, key):
    """Encrypt data using Feistel network"""
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
        
        # Create round keys (derive from main key)
        round_keys = []
        for r in range(8):  # 8 rounds
            round_keys.append(hashlib.sha256(key + bytes([r])).digest()[:8])
            
        # Apply 8 rounds of Feistel network
        for r in range(8):
            left, right = _feistel_round(left, right, round_keys[r])
            
        # Combine halves and add to result
        result.extend(left)
        result.extend(right)
    
    return bytes(result)

def decrypt(data, key):
    """Decrypt data using Feistel network"""
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
        left = block[:mid]
        right = block[mid:]
        
        # Create round keys (same as encryption but used in reverse)
        round_keys = []
        for r in range(8):  # 8 rounds
            round_keys.append(hashlib.sha256(key + bytes([r])).digest()[:8])
        round_keys.reverse()  # Use in reverse order for decryption
            
        # Apply 8 rounds of Feistel network
        for r in range(8):
            left, right = _feistel_round(left, right, round_keys[r])
            
        # Combine halves and add to result
        result.extend(left)
        result.extend(right)
    
    # Remove padding
    while result and result[-1] == 0:
        result.pop()
    
    return bytes(result)
