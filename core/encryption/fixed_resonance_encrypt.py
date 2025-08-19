"""
Fixed version of optimized resonance encryption

RESEARCH ONLY: This implementation is for educational and research purposes only.
Not intended for production cryptographic applications.
"""

import secrets
import hashlib
from typing import Union

def generate_keystream(seed: bytes, length: int, max_chunk_size: int = 1024*1024) -> bytes:
    """Generate keystream in chunks for better memory usage"""
    if length > 1024 * 1024 * 100:  # 100MB max
        raise ValueError("Data too large")

    keystream = bytearray()
    chunk_size = min(max_chunk_size, length)
    num_chunks = (length + chunk_size - 1) // chunk_size

    # Generate a fixed salt for this keystream
    stream_salt = hashlib.sha256(seed).digest()

    for i in range(num_chunks):
        # Generate unique chunk seed
        chunk_seed = seed + i.to_bytes(4, 'big')

        # Calculate this chunk's length remaining = length - len(keystream) this_chunk_size = min(chunk_size, remaining) # Generate chunk using fixed salt chunk = hashlib.pbkdf2_hmac( 'sha256', chunk_seed, stream_salt, # Fixed salt based on seed 100, # Iterations this_chunk_size ) keystream.extend(chunk) # Progress for large streams if length > 1024*1024: # 1MB print(f"Generating keystream... {len(keystream)/length*100:.1f}%") return bytes(keystream) def fixed_resonance_encrypt(plaintext: Union[str, bytes], key: str) -> bytes: """ Fixed version of resonance encryption with safety checks """ # Convert input to bytes if isinstance(plaintext, str): data = plaintext.encode('utf-8') else: data = plaintext # Enforce size limits if len(data) > 1024 * 1024 * 50: # 50MB max raise ValueError("Input data too large") # Generate key hash and signature key_hash = hashlib.sha256(key.encode()).digest() signature = key_hash[:8] token = secrets.token_bytes(32) # Generate keystream safely try: keystream = generate_keystream(key_hash + token, len(data)) except Exception as e: raise RuntimeError(f"Keystream generation failed: {str(e)}") # Encrypt data with progress tracking result = bytearray(len(data)) for i in range(len(data)): # XOR with keystream result[i] = data[i] ^ keystream[i] # Rotate bits (with safety check) rotate_amount = (keystream[(i + 1) % len(keystream)] % 7) + 1 result[i] = ((result[i] << rotate_amount) | (result[i] >> (8 - rotate_amount))) & 0xFF # Progress tracking for large inputs if len(data) > 1024*1024 and i % (1024*1024) == 0: # Every 1MB print(f"Encrypting... {i/len(data)*100:.1f}%") return signature + token + bytes(result) def fixed_resonance_decrypt(encrypted_data: bytes, key: str) -> Union[str, bytes]: """ Fixed version of resonance decryption with safety checks """ # Size validation if len(encrypted_data) < 41: # 8 bytes sig + 32 bytes token + at least 1 byte data raise ValueError("Invalid encrypted data size") # Generate key hash for signature verification key_hash = hashlib.sha256(key.encode()).digest() # Check signature if encrypted_data[:8] != key_hash[:8]: raise ValueError("Invalid signature") # Extract components token = encrypted_data[8:40] data = encrypted_data[40:] # Generate keystream safely try: keystream = generate_keystream(key_hash + token, len(data)) except Exception as e: raise RuntimeError(f"Keystream generation failed: {str(e)}") # Decrypt data with progress tracking result = bytearray(len(data)) for i in range(len(data)): # First reverse the rotation temp = data[i] rotate_amount = (keystream[(i + 1) % len(keystream)] % 7) + 1 temp = ((temp >> rotate_amount) | (temp << (8 - rotate_amount))) & 0xFF # Then reverse the XOR result[i] = temp ^ keystream[i] # Progress tracking for large inputs if len(data) > 1024*1024 and i % (1024*1024) == 0: # Every 1MB print(f"Decrypting... {i/len(data)*100:.1f}%") # Convert back to string or return bytes try: return result.decode('utf-8')
    except UnicodeDecodeError:
        return bytes(result)

def test_fixed_resonance(max_size_mb: int = 10):
    """Test the fixed encryption with size limits"""
    print("Testing fixed resonance encryption...")

    test_cases = [
        ("Small text", "Hello QuantoniumOS!"),
        ("Special chars", "!@#$%^&*()"),
        ("Unicode", "你好, привет, สวัสดี"),
        ("1MB test", "A" * (1024 * 1024)),  # 1MB
        ("Binary", secrets.token_bytes(1024))  # 1KB random bytes
    ]

    key = "test_key_123"
    all_passed = True

    for test_name, test_data in test_cases:
        print(f"\nTesting {test_name}:")
        try:
            # Encrypt
            encrypted = fixed_resonance_encrypt(test_data, key)
            print(f"Encrypted size: {len(encrypted)} bytes")

            # Decrypt
            decrypted = fixed_resonance_decrypt(encrypted, key)

            # Verify
            if isinstance(test_data, str):
                success = decrypted == test_data
            else:
                success = decrypted == test_data

            if success:
                print("✓ Success: Data matches")
            else:
                print("✗ Fail: Data mismatch")
                all_passed = False

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            all_passed = False

    return all_passed

if __name__ == "__main__":
    if test_fixed_resonance():
        print("\n✓ All tests passed successfully")
    else:
        print("\n✗ Some tests failed - see above for details")
