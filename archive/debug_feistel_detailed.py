#!/usr/bin/env python3

import enhanced_rft_crypto

def debug_feistel():
    # Create cipher instance
    cipher = enhanced_rft_crypto.PyEnhancedRFTCrypto()
    
    # Test with simple known data
    key = b"testkey123456789"  # 16 bytes
    plaintext = b"12345678"    # 8 bytes (will be padded to 16)
    
    print("=== Enhanced RFT Feistel Debug ===")
    print(f"Key: {key.hex()}")
    print(f"Plaintext: {plaintext.hex()}")
    
    # Pad to 16 bytes for testing
    padded_plaintext = plaintext + b'\x00' * (16 - len(plaintext))
    print(f"Padded plaintext: {padded_plaintext.hex()}")
    
    # Split into left and right halves
    left_orig = padded_plaintext[:8]
    right_orig = padded_plaintext[8:]
    print(f"Left half: {left_orig.hex()}")
    print(f"Right half: {right_orig.hex()}")
    
    # Encrypt
    encrypted = cipher.encrypt(padded_plaintext, key)
    print(f"Encrypted: {encrypted.hex()}")
    
    # Split encrypted into left and right
    left_enc = encrypted[:8]
    right_enc = encrypted[8:]
    print(f"Encrypted left: {left_enc.hex()}")
    print(f"Encrypted right: {right_enc.hex()}")
    
    # Decrypt
    decrypted = cipher.decrypt(encrypted, key)
    print(f"Decrypted: {decrypted.hex()}")
    
    # Check match
    match = decrypted == padded_plaintext
    print(f"Match: {match}")
    
    if not match:
        print("Mismatch details:")
        for i in range(len(padded_plaintext)):
            if decrypted[i] != padded_plaintext[i]:
                print(f"  Byte {i}: {padded_plaintext[i]:02x} -> {decrypted[i]:02x}")

if __name__ == "__main__":
    debug_feistel()
