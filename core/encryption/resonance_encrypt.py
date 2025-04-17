"""
Quantonium OS - Resonance Encryption Module

Implements resonance-based data encryption algorithms that utilize amplitude-phase fluctuation patterns.
"""

import secrets
import string
import time
import base64
import hashlib
import hmac
import logging
from core.encryption.geometric_waveform_hash import generate_waveform_hash, verify_waveform_hash

# Configure logger
logger = logging.getLogger("resonance_encrypt_encryption")
logger.setLevel(logging.INFO)

def resonance_encrypt(plaintext, A, phi):
    """
    Encrypt plaintext using resonance parameters A (amplitude) and phi (phase).
    Uses geometric waveform hash for signature validation.
    """
    if not isinstance(plaintext, str):
        plaintext = str(plaintext)
    
    # Generate the waveform hash for signature validation
    waveform_hash = generate_waveform_hash(A, phi)
    signature = hashlib.sha256(waveform_hash.encode()).digest()[:4]
    
    # Generate key from amplitude and phase
    key = f"{A:.4f}_{phi:.4f}"
    
    # Generate secure random token for keystream
    token = secrets.token_bytes(32)
    
    # Create keystream using SHA-256(key + token)
    keystream_seed = hashlib.sha256(key.encode() + token).digest()
    keystream = []
    
    # Generate keystream bytes that match the plaintext length
    for i in range(0, len(plaintext), 32):
        chunk_seed = hashlib.sha256(keystream_seed + str(i).encode()).digest()
        keystream.extend(chunk_seed)
    
    # Trim keystream to match plaintext length
    keystream = keystream[:len(plaintext)]
    
    # XOR encryption
    ciphertext_bytes = bytearray([(ord(char) ^ keystream[i]) for i, char in enumerate(plaintext)])
    
    # Include the token in the encrypted data for proper decryption
    # Format: [signature (4 bytes)][token (32 bytes)][ciphertext]
    ciphertext_bytes = signature + token + ciphertext_bytes
    
    return bytes(ciphertext_bytes)

def resonance_decrypt(encrypted_data, A, phi):
    """
    Decrypt data using resonance parameters A (amplitude) and phi (phase).
    Uses geometric waveform hash for signature validation.
    """
    # Generate the waveform hash for verification
    waveform_hash = generate_waveform_hash(A, phi)
    
    # Verify signature (first 4 bytes)
    signature = encrypted_data[:4]
    expected_sig = hashlib.sha256(waveform_hash.encode()).digest()[:4]
    
    if not hmac.compare_digest(signature, expected_sig):
        raise ValueError("Invalid resonance signature - waveform parameters do not match")
    
    # Generate key from amplitude and phase
    key = f"{A:.4f}_{phi:.4f}"
    
    # Extract token (next 32 bytes after signature)
    token = encrypted_data[4:36]
    
    # Extract actual ciphertext (after signature and token)
    ciphertext = encrypted_data[36:]
    
    # Create keystream using SHA-256(key + token)
    keystream_seed = hashlib.sha256(key.encode() + token).digest()
    keystream = []
    
    # Generate keystream bytes that match the ciphertext length
    for i in range(0, len(ciphertext), 32):
        chunk_seed = hashlib.sha256(keystream_seed + str(i).encode()).digest()
        keystream.extend(chunk_seed)
    
    # Trim keystream to match ciphertext length
    keystream = keystream[:len(ciphertext)]
    
    # XOR decryption
    plaintext = ''.join([chr(byte ^ keystream[i]) for i, byte in enumerate(ciphertext)])
    
    return plaintext

def encrypt_data(plaintext: str, key: str) -> str:
    """
    Encrypt plaintext using the resonance encryption algorithm.
    Returns just the encrypted string (for use by protected module).
    """
    return _perform_resonance_encryption(plaintext, key)

def decrypt_data(ciphertext: str, key: str) -> str:
    """
    Decrypt ciphertext using the resonance encryption algorithm.
    Returns the original plaintext (for use by protected module).
    """
    try:
        # Decode the base64 ciphertext
        encrypted_bytes = base64.b64decode(ciphertext)
        
        # Extract amplitude and phase from key
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        A = float(int(key_hash[:8], 16) % 1000) / 1000
        phi = float(int(key_hash[8:16], 16) % 1000) / 1000
        
        # Decrypt using resonance parameters
        return resonance_decrypt(encrypted_bytes, A, phi)
    except Exception as e:
        logger.error(f"Decryption error: {str(e)}")
        return f"Decryption failed: {str(e)}"

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
    Returns a dict with ciphertext, timestamp, and signature.
    """
    start_time = time.time()
    
    # Extract amplitude and phase from key
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    A = float(int(key_hash[:8], 16) % 1000) / 1000
    phi = float(int(key_hash[8:16], 16) % 1000) / 1000
    
    # Generate the waveform hash
    waveform_hash = generate_waveform_hash(A, phi)
    
    # Encrypt the data
    encrypted = _perform_resonance_encryption(plaintext, key)
    
    # Generate timestamp
    ts = int(time.time())
    
    # Create signature using SHA-256(ciphertext + ts + key)
    sig_data = encrypted + str(ts) + key
    sig = hashlib.sha256(sig_data.encode()).hexdigest()
    
    return {
        "ciphertext": encrypted,
        "ts": ts,
        "sig": sig
    }