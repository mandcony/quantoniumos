""""""
Simple version of resonance encryption for testing core functionality

RESEARCH ONLY: This implementation is for educational and research purposes only.
Not intended for production cryptographic applications.
""""""

import hashlib
import secrets

def simple_resonance_encrypt(plaintext: str, key: str) -> bytes:
    """"""
    Simplified version of resonance encryption for testing.
    Only implements core functionality without complex diffusion.
    """"""
    # Convert inputs to bytes
    if not isinstance(plaintext, str):
        plaintext = str(plaintext)
    plaintext_bytes = bytearray(plaintext.encode('utf-8', errors='surrogateescape'))

    # Generate key-based values
    key_hash = hashlib.sha256(key.encode()).digest()
    signature = key_hash[:8]  # 8-byte signature
    token = secrets.token_bytes(32)  # 32-byte random token

    # Simple keystream generation
    keystream = hashlib.sha256(key_hash + token).digest()

    # Extend keystream if needed
    while len(keystream) < len(plaintext_bytes):
        keystream += hashlib.sha256(keystream).digest()
    keystream = keystream[:len(plaintext_bytes)]

    # Simple encryption: XOR with keystream
    result = bytearray(len(plaintext_bytes))
    for i in range(len(plaintext_bytes)):
        result[i] = plaintext_bytes[i] ^ keystream[i]

    # Return signature + token + encrypted data
    return signature + token + bytes(result)

def simple_resonance_decrypt(encrypted_data: bytes, key: str) -> str:
    """"""
    Simplified version of resonance decryption for testing.
    """"""
    # Generate key hash
    key_hash = hashlib.sha256(key.encode()).digest()

    # Verify signature
    signature = encrypted_data[:8]
    if signature != key_hash[:8]:
        raise ValueError("Invalid signature")

    # Extract token and ciphertext
    token = encrypted_data[8:40]
    ciphertext = encrypted_data[40:]

    # Generate keystream
    keystream = hashlib.sha256(key_hash + token).digest()

    # Extend keystream if needed
    while len(keystream) < len(ciphertext):
        keystream += hashlib.sha256(keystream).digest()
    keystream = keystream[:len(ciphertext)]

    # Decrypt: XOR with keystream
    result = bytearray(len(ciphertext))
    for i in range(len(ciphertext)):
        result[i] = ciphertext[i] ^ keystream[i]

    # Convert back to string
    return result.decode('utf-8', errors='surrogateescape')

def test_simple_resonance():
    """"""
    Test the simple resonance encryption
    """"""
    print("Testing simple resonance encryption...")

    # Test data
    plaintext = "Hello QuantoniumOS!"
    key = "test_key_123"

    print(f"Original text: {plaintext}")

    # Encrypt
    print("Encrypting...")
    encrypted = simple_resonance_encrypt(plaintext, key)
    print(f"Encrypted length: {len(encrypted)} bytes")

    # Decrypt
    print("Decrypting...")
    decrypted = simple_resonance_decrypt(encrypted, key)
    print(f"Decrypted text: {decrypted}")

    # Verify
    if decrypted == plaintext:
        print("Test passed: encryption/decryption successful")
    else:
        print("Test failed: decrypted text doesn't match original")

if __name__ == "__main__":
    test_simple_resonance()
