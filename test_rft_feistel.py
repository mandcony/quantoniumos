#!/usr/bin/env python3
"""
Test script for True RFT Feistel Bindings
"""

import sys
import os
from pathlib import Path

# Import the True RFT Feistel Bindings
sys.path.append('/workspaces/quantoniumos/06_CRYPTOGRAPHY')
import true_rft_feistel_bindings

# Initialize the engine
print("Initializing the True RFT Feistel Engine...")
true_rft_feistel_bindings.init()

# Test encryption and decryption
print("\nTesting encryption and decryption...")
test_message = "This is a secret message that needs to be encrypted with True RFT Feistel network."
print(f"Original message: {test_message}")

# Generate a key
key = true_rft_feistel_bindings.generate_key()
print(f"Key (hex): {key.hex()[:20]}...")

# Encrypt the message
encrypted = true_rft_feistel_bindings.encrypt(test_message.encode(), key)
print(f"Encrypted (hex): {encrypted.hex()[:40]}...")

# Decrypt the message
decrypted = true_rft_feistel_bindings.decrypt(encrypted, key)
decrypted_message = decrypted.decode()
print(f"Decrypted message: {decrypted_message}")

# Verify successful round trip
is_successful = test_message == decrypted_message
print(f"Round trip successful: {is_successful}")

# Test avalanche effect
print("\nTesting avalanche effect...")

# Change one bit in the key
modified_key = bytearray(key)
modified_key[0] ^= 1  # Flip one bit
modified_key = bytes(modified_key)
print(f"Modified key (hex): {modified_key.hex()[:20]}...")

# Encrypt with modified key
encrypted2 = true_rft_feistel_bindings.encrypt(test_message.encode(), modified_key)
print(f"Encrypted with modified key (hex): {encrypted2.hex()[:40]}...")

# Calculate bit differences
diff_bits = 0
total_bits = len(encrypted) * 8
for b1, b2 in zip(encrypted, encrypted2):
    xor = b1 ^ b2
    diff_bits += bin(xor).count('1')

avalanche_percentage = (diff_bits / total_bits) * 100
print(f"Bit differences: {diff_bits}/{total_bits} ({avalanche_percentage:.2f}%)")
print(f"Good avalanche effect (>45%): {avalanche_percentage > 45}")

# Try to decrypt with wrong key
print("\nTesting decryption with wrong key...")
try:
    wrong_decrypted = true_rft_feistel_bindings.decrypt(encrypted, modified_key)
    wrong_message = wrong_decrypted.decode(errors='replace')
    print(f"Decrypted with wrong key: {wrong_message[:40]}...")
    print(f"Matches original: {wrong_message == test_message}")
except Exception as e:
    print(f"Error decrypting with wrong key: {e}")

print("\nTest complete!")
