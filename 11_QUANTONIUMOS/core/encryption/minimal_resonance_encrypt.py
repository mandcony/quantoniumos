"""
Minimal version of resonance encryption for debugging
"""

import hashlib
import secrets

def minimal_encrypt(plaintext: str, key: str) -> bytes:
    """Minimal encryption for debugging"""
    # Convert to bytes
    data = plaintext.encode('utf-8')

    # Generate key material
    key_hash = hashlib.sha256(key.encode()).digest()
    signature = key_hash[:8]
    token = secrets.token_bytes(32)

    # Simple keystream
    keystream = hashlib.sha256(key_hash + token).digest()
    while len(keystream) < len(data):
        keystream += hashlib.sha256(keystream).digest()
    keystream = keystream[:len(data)]

    # XOR encryption
    result = bytearray(len(data))
    for i in range(len(data)):
        result[i] = data[i] ^ keystream[i]

    return signature + token + bytes(result)

def minimal_decrypt(ciphertext: bytes, key: str) -> str:
    """Minimal decryption for debugging"""
    # Generate key hash
    key_hash = hashlib.sha256(key.encode()).digest()

    # Check signature
    if ciphertext[:8] != key_hash[:8]:
        raise ValueError("Invalid signature")

    # Extract parts
    token = ciphertext[8:40]
    data = ciphertext[40:]

    # Regenerate keystream
    keystream = hashlib.sha256(key_hash + token).digest()
    while len(keystream) < len(data):
        keystream += hashlib.sha256(keystream).digest()
    keystream = keystream[:len(data)]

    # XOR decryption
    result = bytearray(len(data))
    for i in range(len(data)):
        result[i] = data[i] ^ keystream[i]

    return result.decode('utf-8')

def test_minimal():
    """Test minimal encryption"""
    print("Testing minimal encryption...")

    key = "test_key_123"
    test_data = [
        "Hello QuantoniumOS!",
        "Special chars: !@#$%^&*()",
        "Unicode: 你好, привет, สวัสดี"
    ]

    for test_str in test_data:
        print(f"\nTest string: {test_str}")
        print(f"Length: {len(test_str)} chars")

        try:
            encrypted = minimal_encrypt(test_str, key)
            print(f"Encrypted length: {len(encrypted)} bytes")

            decrypted = minimal_decrypt(encrypted, key)
            print(f"Decrypted: {decrypted}")

            if decrypted == test_str:
                print("✓ Success: encryption/decryption worked")
            else:
                print("✗ Failed: decrypted text doesn't match")
                print(f"Original : {test_str}")
                print(f"Decrypted: {decrypted}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_minimal()
