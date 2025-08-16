#!/usr/bin/env python3

import enhanced_rft_crypto
import struct

class FixedRFTCryptoV2:
    """Wrapper for enhanced RFT crypto v2 with proper length handling."""
    
    def __init__(self):
        self.engine = enhanced_rft_crypto.PyEnhancedRFTCrypto()
    
    def encrypt(self, plaintext, key):
        """Encrypt with proper length preservation."""
        original_len = len(plaintext)
        
        # The new implementation requires even-length input
        if len(plaintext) % 2 != 0:
            padded_plaintext = plaintext + b'\x00'
        else:
            padded_plaintext = plaintext
        
        # Encrypt the padded data
        encrypted = self.engine.encrypt(padded_plaintext, key)
        
        # Prepend original length (4 bytes)
        return struct.pack('<I', original_len) + encrypted
    
    def decrypt(self, ciphertext, key):
        """Decrypt with proper length restoration."""
        if len(ciphertext) < 4:
            raise ValueError("Invalid ciphertext: too short")
        
        # Extract original length
        original_len = struct.unpack('<I', ciphertext[:4])[0]
        encrypted_data = ciphertext[4:]
        
        # Decrypt
        decrypted_padded = self.engine.decrypt(encrypted_data, key)
        
        # Return only original length
        return decrypted_padded[:original_len]

def test_enhanced_v2():
    print("=== Enhanced RFT Crypto v2 Performance Test ===")
    
    crypto = FixedRFTCryptoV2()
    
    # Test various sizes
    test_cases = [
        b"A",  # 1 byte
        b"AB",  # 2 bytes  
        b"Hello",  # 5 bytes
        b"Hello, World!",  # 13 bytes
        b"Hello, World! This is a test message!",  # 37 bytes
        b"A" * 100,  # 100 bytes
        b"B" * 1000,  # 1000 bytes
    ]
    
    key = b"test_key_32_bytes_long_for_v2_test"
    
    for i, plaintext in enumerate(test_cases):
        print(f"\nTest {i+1}: {len(plaintext)} bytes")
        
        # Encrypt
        encrypted = crypto.encrypt(plaintext, key)
        print(f"Original: {len(plaintext)} bytes")
        print(f"Encrypted: {len(encrypted)} bytes")
        
        # Decrypt
        decrypted = crypto.decrypt(encrypted, key)
        print(f"Decrypted: {len(decrypted)} bytes")
        
        # Check match
        match = (decrypted == plaintext)
        print(f"Match: {'✓' if match else '✗'}")
        
        if not match:
            print(f"Expected: {plaintext[:20]}")
            print(f"Got:      {decrypted[:20]}")
            return False
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    test_enhanced_v2()
