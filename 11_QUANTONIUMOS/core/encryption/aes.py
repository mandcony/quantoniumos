"""
AES Encryption Module
===================
AES encryption implementation for QuantoniumOS.
"""

import base64
import os

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def encrypt(plaintext, key, iv=None):
    """Encrypt plaintext using AES-256-CBC"""
    if not isinstance(plaintext, bytes):
        plaintext = plaintext.encode("utf-8")

    if iv is None:
        iv = os.urandom(16)

    # Create cipher
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

    # Pad plaintext to block size
    block_size = 16
    padding_length = block_size - (len(plaintext) % block_size)
    plaintext += bytes([padding_length]) * padding_length

    # Encrypt
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()

    # Return IV + ciphertext
    return iv + ciphertext


def decrypt(ciphertext, key):
    """Decrypt ciphertext using AES-256-CBC"""
    # Extract IV from ciphertext
    iv = ciphertext[:16]
    ciphertext = ciphertext[16:]

    # Create cipher
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

    # Decrypt
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    # Remove padding
    padding_length = plaintext[-1]
    plaintext = plaintext[:-padding_length]

    return plaintext


def generate_key(bits=256):
    """Generate a random AES key"""
    return os.urandom(bits // 8)


if __name__ == "__main__":
    # Example usage
    key = generate_key()
    plaintext = b"This is a test message for AES encryption"

    ciphertext = encrypt(plaintext, key)
    decrypted = decrypt(ciphertext, key)

    print(f"Plaintext: {plaintext}")
    print(f"Ciphertext: {base64.b64encode(ciphertext)}")
    print(f"Decrypted: {decrypted}")
