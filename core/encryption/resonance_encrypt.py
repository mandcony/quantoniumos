"""
Quantonium OS - Resonance Encryption Module

Implements resonance-based data encryption algorithms that utilize amplitude-phase fluctuation patterns.
"""

import random
import string
import time
import base64
import hashlib

def resonance_encrypt(plaintext, A, phi):
    """
    Encrypt plaintext using resonance parameters A (amplitude) and phi (phase).
    """
    if not isinstance(plaintext, str):
        plaintext = str(plaintext)
    
    # Generate key from amplitude and phase
    key = f"{A:.4f}_{phi:.4f}"
    
    # Use key to create a seed for our encryption keystream
    seed = sum(ord(c) for c in key)
    random.seed(seed)
    
    # Generate keystream
    keystream = [random.randint(1, 255) for _ in range(len(plaintext))]
    
    # XOR encryption
    ciphertext_bytes = bytearray([(ord(char) ^ keystream[i]) for i, char in enumerate(plaintext)])
    
    # Add resonance signature
    signature = hashlib.sha256(key.encode()).digest()[:4]
    ciphertext_bytes = signature + ciphertext_bytes
    
    return bytes(ciphertext_bytes)

def resonance_decrypt(encrypted_data, A, phi):
    """
    Decrypt data using resonance parameters A (amplitude) and phi (phase).
    """
    # Generate key from amplitude and phase
    key = f"{A:.4f}_{phi:.4f}"
    
    # Verify signature (first 4 bytes)
    signature = encrypted_data[:4]
    expected_sig = hashlib.sha256(key.encode()).digest()[:4]
    
    if signature != expected_sig:
        raise ValueError("Invalid resonance signature")
    
    # Extract ciphertext
    ciphertext = encrypted_data[4:]
    
    # Use key to recreate the keystream
    seed = sum(ord(c) for c in key)
    random.seed(seed)
    keystream = [random.randint(1, 255) for _ in range(len(ciphertext))]
    
    # XOR decryption
    plaintext = ''.join([chr(byte ^ keystream[i]) for i, byte in enumerate(ciphertext)])
    
    return plaintext

def _perform_resonance_encryption(plaintext: str, key: str) -> str:
    """
    Internal function to perform resonance encryption using a string key.
    """
    # Extract amplitude and phase from key
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    A = float(int(key_hash[:8], 16) % 1000) / 1000
    phi = float(int(key_hash[8:16], 16) % 1000) / 1000
    
    # Encrypt using resonance parameters
    encrypted = resonance_encrypt(plaintext, A, phi)
    
    # Return base64 encoded string
    return base64.b64encode(encrypted).decode("utf-8")

def encrypt(plaintext: str, key: str) -> dict:
    """
    Encrypt plaintext using the resonance encryption algorithm with a string key.
    Returns a dict with ciphertext, timestamp, and processing metrics.
    """
    start_time = time.time()
    encrypted = _perform_resonance_encryption(plaintext, key)
    elapsed = round((time.time() - start_time) * 1000, 2)
    
    return {
        "ciphertext": encrypted,
        "timestamp": int(time.time()),
        "elapsed_ms": elapsed
    }