"""
Quantonium OS - Secret Management

This module handles encryption, decryption, and rotation of sensitive secrets
such as JWT signing keys. It ensures that no secrets are stored in plaintext
in the database.
"""

import os
import base64
import secrets
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Environment variable name for the master encryption key
MASTER_KEY_ENV = "QUANTONIUM_MASTER_KEY"
# Salt for key derivation (not a secret)
KEY_DERIVATION_SALT = b'quantonium_salt_3a1b4c2d5e'
# Number of PBKDF2 iterations
PBKDF2_ITERATIONS = 100000

# Cache the encryption key to avoid repeated derivation
_encryption_key = None
_key_rotation_schedule = {}


def get_master_key():
    """
    Get the master encryption key from environment variable.
    If not set, generate a secure random key and advise the user to save it.
    """
    key = os.environ.get(MASTER_KEY_ENV)
    
    if not key:
        # Generate a secure random key
        key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('ascii')
        
        # Display a prominent warning
        print("\n" + "!" * 80)
        print("WARNING: No master encryption key found in environment!")
        print(f"A temporary key has been generated: {key}")
        print(f"Set this in your environment as {MASTER_KEY_ENV}")
        print("Without this key, you will lose access to encrypted secrets on restart!")
        print("!" * 80 + "\n")
        
        # Set in environment for current process
        os.environ[MASTER_KEY_ENV] = key
    
    return key


def derive_encryption_key():
    """
    Derive the encryption key using PBKDF2 with the master key as input.
    This adds an extra layer of security and ensures the key is the right length.
    """
    global _encryption_key
    
    if _encryption_key is not None:
        return _encryption_key
    
    master_key = get_master_key().encode('utf-8')
    
    # Use PBKDF2 to derive a key of appropriate length
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 32 bytes = 256 bits
        salt=KEY_DERIVATION_SALT,
        iterations=PBKDF2_ITERATIONS
    )
    
    derived_key = base64.urlsafe_b64encode(kdf.derive(master_key))
    _encryption_key = derived_key
    
    return derived_key


def encrypt_secret(plaintext):
    """
    Encrypt a secret value for storage.
    
    Args:
        plaintext: The secret text to encrypt (string)
        
    Returns:
        Encrypted and base64-encoded string
    """
    if plaintext is None:
        return None
    
    # Convert string to bytes if necessary
    if isinstance(plaintext, str):
        plaintext = plaintext.encode('utf-8')
    
    # Get encryption key
    key = derive_encryption_key()
    
    # Create cipher and encrypt
    cipher = Fernet(key)
    encrypted = cipher.encrypt(plaintext)
    
    # Return as base64 encoded string
    return base64.urlsafe_b64encode(encrypted).decode('ascii')


def decrypt_secret(encrypted):
    """
    Decrypt a stored secret value.
    
    Args:
        encrypted: The encrypted secret text (base64 string)
        
    Returns:
        Decrypted string or None if decryption fails
    """
    if not encrypted:
        return None
    
    try:
        # Convert from base64 string to bytes
        if isinstance(encrypted, str):
            encrypted = base64.urlsafe_b64decode(encrypted.encode('ascii'))
        
        # Get encryption key
        key = derive_encryption_key()
        
        # Create cipher and decrypt
        cipher = Fernet(key)
        decrypted = cipher.decrypt(encrypted)
        
        # Return as string
        return decrypted.decode('utf-8')
    except Exception as e:
        print(f"Error decrypting secret: {str(e)}")
        return None


def generate_jwt_secret():
    """
    Generate a new secure random JWT signing secret.
    
    Returns:
        A 64-character hex string (32 bytes)
    """
    return secrets.token_hex(32)


def schedule_key_rotation(key_id, days=90):
    """
    Schedule a key for rotation.
    
    Args:
        key_id: The ID of the key to rotate
        days: Number of days until rotation
    """
    global _key_rotation_schedule
    
    rotation_date = datetime.utcnow() + timedelta(days=days)
    _key_rotation_schedule[key_id] = rotation_date


def get_keys_due_for_rotation():
    """
    Get a list of key IDs that are due for rotation.
    
    Returns:
        List of key IDs
    """
    global _key_rotation_schedule
    
    now = datetime.utcnow()
    due_keys = [key_id for key_id, date in _key_rotation_schedule.items() if date <= now]
    
    # Remove rotated keys from schedule
    for key_id in due_keys:
        _key_rotation_schedule.pop(key_id, None)
    
    return due_keys