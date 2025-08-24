#!/usr/bin/env python3

import secrets

import enhanced_rft_crypto


def test_basic_functionality():
        print("=== Testing Enhanced RFT Crypto v2 ===")

        # Create cipher instance cipher = enhanced_rft_crypto.PyEnhancedRFTCrypto()

        # Test with even-length input (Feistel requirement) key = secrets.token_bytes(32) plaintext = b"Hello, World! This is a test message!" # 37 bytes -> will be padded to 38
        print(f"Original length: {len(plaintext)} bytes")
        print(f"Plaintext: {plaintext}")

        # Encrypt encrypted = cipher.encrypt(plaintext, key)
        print(f"Encrypted length: {len(encrypted)} bytes")
        print(f"Encrypted: {encrypted[:32].hex()}...")

        # Decrypt decrypted = cipher.decrypt(encrypted, key)
        print(f"Decrypted length: {len(decrypted)} bytes")
        print(f"Decrypted: {decrypted}")

        # Check
        if we need to handle padding
        if len(decrypted) == len(plaintext) + 1 and decrypted[:-1] == plaintext:
        print("✓ Perfect match (with padding byte)")
        return True
        el
        if decrypted == plaintext:
        print("✓ Perfect match")
        return True
        else:
        print("✗ Mismatch!")
        print(f"Expected: {plaintext.hex()}")
        print(f"Got: {decrypted.hex()}")
        return False
def test_even_length():
        print("\n=== Testing with even-length input ===") cipher = enhanced_rft_crypto.PyEnhancedRFTCrypto() key = secrets.token_bytes(32) plaintext = b"Hello, World!123" # 16 bytes (even)
        print(f"Plaintext ({len(plaintext)} bytes): {plaintext}") encrypted = cipher.encrypt(plaintext, key) decrypted = cipher.decrypt(encrypted, key)
        print(f"Decrypted ({len(decrypted)} bytes): {decrypted}") match = (decrypted == plaintext)
        print(f"Match: {'✓'
        if match else '✗'}")
        return match

if __name__ == "__main__": test1 = test_basic_functionality() test2 = test_even_length()
if test1 and test2:
print("\n🎉 All basic tests passed!")
else:
print("\n❌ Some tests failed!")