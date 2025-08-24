"""
RSA Encryption Module
===================
RSA encryption implementation for QuantoniumOS.
"""

import base64
import os

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


def generate_keypair(bits=2048):
    """Generate an RSA key pair"""
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=bits, backend=default_backend()
    )

    public_key = private_key.public_key()

    return public_key, private_key


def encrypt(plaintext, public_key):
    """Encrypt plaintext using RSA"""
    if not isinstance(plaintext, bytes):
        plaintext = plaintext.encode("utf-8")

    # Encrypt
    ciphertext = public_key.encrypt(
        plaintext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    return ciphertext


def decrypt(ciphertext, private_key):
    """Decrypt ciphertext using RSA"""
    # Decrypt
    plaintext = private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    return plaintext


def serialize_public_key(public_key):
    """Serialize a public key to bytes"""
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def serialize_private_key(private_key, password=None):
    """Serialize a private key to bytes"""
    if password:
        encryption = serialization.BestAvailableEncryption(password)
    else:
        encryption = serialization.NoEncryption()

    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    )


def deserialize_public_key(data):
    """Deserialize a public key from bytes"""
    return serialization.load_pem_public_key(data, backend=default_backend())


def deserialize_private_key(data, password=None):
    """Deserialize a private key from bytes"""
    return serialization.load_pem_private_key(
        data, password=password, backend=default_backend()
    )


if __name__ == "__main__":
    # Example usage
    public_key, private_key = generate_keypair()
    plaintext = b"This is a test message for RSA encryption"

    ciphertext = encrypt(plaintext, public_key)
    decrypted = decrypt(ciphertext, private_key)

    print(f"Plaintext: {plaintext}")
    print(f"Ciphertext: {base64.b64encode(ciphertext)}")
    print(f"Decrypted: {decrypted}")
