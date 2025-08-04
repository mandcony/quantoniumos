"""
Detailed avalanche test showing bit changes visually
"""
import hashlib
import secrets
import argparse
import sys
from fixed_resonance_encrypt import fixed_resonance_encrypt

def visualize_bit_changes(bytes1, bytes2):
    """Show exactly which bits changed between two byte sequences"""
    changes = []
    for b1, b2 in zip(bytes1, bytes2):
        # Get binary representation of each byte
        bits1 = format(b1, '08b')
        bits2 = format(b2, '08b')
        # Compare bits and mark changes
        diff = ''
        for bit1, bit2 in zip(bits1, bits2):
            if bit1 == bit2:
                diff += '.'  # No change
            else:
                diff += 'X'  # Bit changed
        changes.append((bits1, bits2, diff))
    return changes

def test_single_bit_avalanche(position=0):
    """Test avalanche effect by changing a single bit and showing the changes"""
    print("Detailed Avalanche Effect Analysis")
    print("=================================")
    
    # Original input (using a small message to show changes clearly)
    message = "Test"
    key = "test_key_123"
    
    print(f"Original message: {message}")
    print(f"Testing bit flip at position {position}")
    print()
    
    # Get original encryption
    cipher1 = fixed_resonance_encrypt(message, key)
    
    # Modify one bit in the input
    modified = list(message)
    char_pos = position % len(modified)
    char = ord(modified[char_pos])
    bit_pos = (position // len(modified)) % 8
    char ^= (1 << bit_pos)
    modified[char_pos] = chr(char)
    modified_msg = ''.join(modified)
    
    # Get modified encryption
    cipher2 = fixed_resonance_encrypt(modified_msg, key)
    
    # Compare the actual ciphertext parts (skip signature and token)
    actual_cipher1 = cipher1[40:]
    actual_cipher2 = cipher2[40:]
    
    # Show the changes
    print("Bit changes in first 16 bytes of ciphertext:")
    print("Original vs Modified (X marks changed bits)")
    print("-" * 50)
    
    changes = visualize_bit_changes(actual_cipher1[:16], actual_cipher2[:16])
    
    total_changes = 0
    total_bits = 0
    
    for i, (orig, mod, diff) in enumerate(changes):
        print(f"Byte {i:2d}: {orig} (original)")
        print(f"       {mod} (modified)")
        print(f"       {diff} (changes)")
        print()
        
        total_changes += diff.count('X')
        total_bits += 8
    
    # Calculate statistics for shown bytes
    change_percent = (total_changes / total_bits) * 100
    
    print("Summary:")
    print(f"Total bits changed: {total_changes} out of {total_bits}")
    print(f"Change percentage: {change_percent:.2f}%")
    
    # Calculate statistics for entire ciphertext
    full_changes = sum(bin(a ^ b).count('1') for a, b in zip(actual_cipher1, actual_cipher2))
    full_bits = len(actual_cipher1) * 8
    full_percent = (full_changes / full_bits) * 100
    
    print(f"\nFull ciphertext statistics:")
    print(f"Total bits changed: {full_changes} out of {full_bits}")
    print(f"Overall change percentage: {full_percent:.2f}%")

if __name__ == "__main__":
    # Test with first bit change
    test_single_bit_avalanche(0)
    print("\n" + "=" * 50 + "\n")
    # Test with a different position
    test_single_bit_avalanche(5)
