"""
Optimized version of resonance encryption module

RESEARCH ONLY: This implementation is for educational and research purposes only.
Not intended for production cryptographic applications.
"""

import secrets
import hashlib
import time

def optimized_resonance_encrypt(plaintext, key):
    """
    Optimized version of resonance encryption that maintains security
    while reducing computational complexity
    """
    if not isinstance(plaintext, str):
        plaintext = str(plaintext)
    
    # Convert to bytes first, before any other operations
    data = plaintext.encode('utf-8')
    
    # Generate key-derived parameters
    key_hash = hashlib.sha256(key.encode()).digest()
    signature = key_hash[:8]
    token = secrets.token_bytes(32)
    
    # Generate keystream using the working method from minimal version
    keystream = hashlib.sha256(key_hash + token).digest()
    while len(keystream) < len(data):
        keystream += hashlib.sha256(keystream).digest()
    keystream = keystream[:len(data)]
    
    # Simple XOR encryption with optional rotation for extra security
    result = bytearray(len(data))
    for i in range(len(data)):
        # XOR with keystream
        result[i] = data[i] ^ keystream[i]
        # Optional: rotate bits for additional security
        rotate_amount = (keystream[(i + 1) % len(keystream)] % 7) + 1
        result[i] = ((result[i] << rotate_amount) | (result[i] >> (8 - rotate_amount))) & 0xFF
    
    # Format: signature + token + ciphertext
    return signature + token + bytes(result)

def optimized_resonance_decrypt(encrypted_data, key):
    """
    Optimized version of resonance decryption
    """
    # Generate key hash for signature verification
    key_hash = hashlib.sha256(key.encode()).digest()
    
    # Check signature
    if encrypted_data[:8] != key_hash[:8]:
        raise ValueError("Invalid signature")
    
    # Extract components
    token = encrypted_data[8:40]
    data = encrypted_data[40:]
    
    # Regenerate keystream exactly as in encryption
    keystream = hashlib.sha256(key_hash + token).digest()
    while len(keystream) < len(data):
        keystream += hashlib.sha256(keystream).digest()
    keystream = keystream[:len(data)]
    
    # Reverse operations from encryption
    result = bytearray(len(data))
    for i in range(len(data)):
        # First reverse the rotation
        temp = data[i]
        rotate_amount = (keystream[(i + 1) % len(keystream)] % 7) + 1
        temp = ((temp >> rotate_amount) | (temp << (8 - rotate_amount))) & 0xFF
        # Then reverse the XOR
        result[i] = temp ^ keystream[i]
    
    # Convert back to string
    try:
        return result.decode('utf-8')
    except UnicodeDecodeError:
        return result.hex()

def test_optimized_resonance():
    """
    Test the optimized resonance encryption
    """
    print("Testing optimized resonance encryption...")
    
    test_data = [
        "Hello QuantoniumOS!",
        "A" * 1000,  # Test with larger data
        "Special chars: !@#$%^&*()",
        "Unicode: 你好, привет, สวัสดี"
    ]
    
    key = "test_key_123"
    
    for test_str in test_data:
        print(f"\nTesting with: {test_str[:50]}{'...' if len(test_str) > 50 else ''}")
        print(f"Length: {len(test_str)} characters")
        
        start_time = time.time()
        encrypted = optimized_resonance_encrypt(test_str, key)
        encrypt_time = time.time() - start_time
        print(f"Encryption time: {encrypt_time:.3f} seconds")
        
        start_time = time.time()
        decrypted = optimized_resonance_decrypt(encrypted, key)
        decrypt_time = time.time() - start_time
        print(f"Decryption time: {decrypt_time:.3f} seconds")
        
        if decrypted == test_str:
            print("✓ Test passed: encryption/decryption successful")
        else:
            print("✗ Test failed: decrypted text doesn't match original")
            print(f"Original: {test_str[:50]}")
            print(f"Decrypted: {decrypted[:50]}")

if __name__ == "__main__":
    test_optimized_resonance()
