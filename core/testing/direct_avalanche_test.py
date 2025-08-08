"""
Direct Avalanche Effect Test for Enhanced Cryptographic Primitives

This script tests the avalanche effect of the enhanced cryptographic algorithms
directly, without going through the wrapper classes used in the main test suite.
"""

import os
import random
import struct
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Fix import paths - add the project root to Python's module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Import QuantoniumOS modules
from core.encryption.resonance_encrypt import resonance_encrypt
from core.encryption.geometric_waveform_hash import GeometricWaveformHash

def count_bit_differences(data1: bytes, data2: bytes) -> int:
    """Count the number of different bits between two byte sequences."""
    # Make sure we compare equal-length sequences
    min_len = min(len(data1), len(data2))
    bit_diff = 0
    
    # Compare each byte
    for i in range(min_len):
        # XOR the bytes to get different bits
        xor_result = data1[i] ^ data2[i]
        # Count bits set to 1 in the XOR result
        while xor_result:
            bit_diff += xor_result & 1
            xor_result >>= 1
    
    return bit_diff

def calculate_bit_change_percentage(data1: bytes, data2: bytes) -> float:
    """Calculate the percentage of bits that have changed."""
    min_len = min(len(data1), len(data2))
    total_bits = min_len * 8
    
    if total_bits == 0:
        return 0.0
        
    bit_diff = count_bit_differences(data1, data2)
    return (bit_diff / total_bits) * 100

def test_encryption_avalanche(num_tests: int = 1000) -> Tuple[float, float, float, float, List[float]]:
    """
    Test the avalanche effect of the enhanced resonance encryption algorithm.
    
    Returns:
        Tuple containing mean, std_dev, min, max, and list of bit change percentages
    """
    bit_change_percentages = []
    
    for _ in range(num_tests):
        # Generate random plaintext (32-128 bytes)
        plaintext_len = random.randint(32, 128)
        plaintext = os.urandom(plaintext_len)
        plaintext_str = plaintext.hex()
        
        # Generate random amplitude and phase parameters
        A = random.uniform(0.5, 2.0)  # Amplitude
        phi = random.uniform(0, 2 * np.pi)  # Phase
        
        # Encrypt the plaintext
        ciphertext1 = resonance_encrypt(plaintext_str, A, phi)
        
        # Flip a single bit in the plaintext
        modified_plaintext = bytearray(plaintext)
        byte_index = random.randint(0, len(modified_plaintext) - 1)
        bit_index = random.randint(0, 7)
        modified_plaintext[byte_index] ^= (1 << bit_index)  # Flip one bit
        modified_plaintext_str = modified_plaintext.hex()
        
        # Encrypt the modified plaintext
        ciphertext2 = resonance_encrypt(modified_plaintext_str, A, phi)
        
        # Calculate bit change percentage
        bit_change_percentage = calculate_bit_change_percentage(ciphertext1, ciphertext2)
        bit_change_percentages.append(bit_change_percentage)
    
    # Calculate statistics
    mean_change = np.mean(bit_change_percentages)
    std_dev = np.std(bit_change_percentages)
    min_change = np.min(bit_change_percentages)
    max_change = np.max(bit_change_percentages)
    
    return mean_change, std_dev, min_change, max_change, bit_change_percentages

def test_hash_avalanche(num_tests: int = 1000) -> Tuple[float, float, float, float, List[float]]:
    """
    Test the avalanche effect of the enhanced geometric waveform hash algorithm.
    
    Returns:
        Tuple containing mean, std_dev, min, max, and list of bit change percentages
    """
    bit_change_percentages = []
    
    for _ in range(num_tests):
        # Generate random waveform data
        waveform1 = [random.uniform(-1.0, 1.0) for _ in range(100)]
        
        # Create a hash of the original waveform
        hasher1 = GeometricWaveformHash(waveform=waveform1)
        hash1_full = hasher1.generate_hash()
        digest_part1 = bytes.fromhex(hash1_full.split('_')[-1])  # Convert hex string to raw bytes
        
        # Modify a single value in the waveform by flipping a bit in IEEE-754 representation
        waveform2 = waveform1.copy()
        index = random.randint(0, len(waveform2) - 1)
        
        # Convert float to bytes, flip a random bit, convert back
        original_bytes = struct.pack('<d', waveform2[index])
        byte_array = bytearray(original_bytes)
        
        # Flip a random bit
        byte_index = random.randint(0, 7)
        bit_index = random.randint(0, 7)
        byte_array[byte_index] ^= (1 << bit_index)
        
        # Convert back to float
        waveform2[index] = struct.unpack('<d', bytes(byte_array))[0]
        
        # Create a hash of the modified waveform
        hasher2 = GeometricWaveformHash(waveform=waveform2)
        hash2_full = hasher2.generate_hash()
        digest_part2 = bytes.fromhex(hash2_full.split('_')[-1])  # Convert hex string to raw bytes
        
        # Calculate bit change percentage on raw digest bytes
        bit_change_percentage = calculate_bit_change_percentage(digest_part1, digest_part2)
        bit_change_percentages.append(bit_change_percentage)
    
    # Calculate statistics
    mean_change = np.mean(bit_change_percentages)
    std_dev = np.std(bit_change_percentages)
    min_change = np.min(bit_change_percentages)
    max_change = np.max(bit_change_percentages)
    
    return mean_change, std_dev, min_change, max_change, bit_change_percentages

def plot_histogram(bit_change_percentages: List[float], title: str, filename: str):
    """Plot histogram of bit change percentages."""
    plt.figure(figsize=(10, 6))
    plt.hist(bit_change_percentages, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(50, color='red', linestyle='--', label='Ideal (50%)')
    plt.axvline(np.mean(bit_change_percentages), color='green', linestyle='-', label=f'Mean: {np.mean(bit_change_percentages):.2f}%')
    
    plt.xlabel('Bit Change Percentage')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_dir = os.path.join(project_root, 'test_results', 'direct_avalanche')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

if __name__ == "__main__":
    # Force reload of the geometric waveform hash module to ensure fresh code
    import importlib
    import core.encryption.geometric_waveform_hash as gwh
    importlib.reload(gwh)
    GeometricWaveformHash = gwh.GeometricWaveformHash
    
    print("Running Direct Avalanche Effect Tests on Enhanced Cryptographic Primitives")
    
    # Test enhanced encryption avalanche effect
    print("\nTesting Enhanced Resonance Encryption Avalanche Effect...")
    enc_mean, enc_std, enc_min, enc_max, enc_percentages = test_encryption_avalanche()
    
    print(f"  Mean Bit Change: {enc_mean:.2f}% (Ideal: 50%)")
    print(f"  Std Deviation: {enc_std:.2f}%")
    print(f"  Range: {enc_min:.2f}% - {enc_max:.2f}%")
    print(f"  Passed: {'YES' if 45 <= enc_mean <= 55 and enc_std < 10 else 'NO'}")
    
    # Plot encryption avalanche histogram
    plot_histogram(enc_percentages, 
                  f'Enhanced Resonance Encryption Avalanche Effect (Mean: {enc_mean:.2f}%)', 
                  'enhanced_encryption_avalanche.png')
    
    # Test enhanced hash avalanche effect
    print("\nTesting Enhanced Geometric Waveform Hash Avalanche Effect...")
    hash_mean, hash_std, hash_min, hash_max, hash_percentages = test_hash_avalanche()
    
    print(f"  Mean Bit Change: {hash_mean:.2f}% (Ideal: 50%)")
    print(f"  Std Deviation: {hash_std:.2f}%")
    print(f"  Range: {hash_min:.2f}% - {hash_max:.2f}%")
    print(f"  Passed: {'YES' if 45 <= hash_mean <= 55 and hash_std < 10 else 'NO'}")
    
    # Plot hash avalanche histogram
    plot_histogram(hash_percentages, 
                  f'Enhanced Geometric Waveform Hash Avalanche Effect (Mean: {hash_mean:.2f}%)', 
                  'enhanced_hash_avalanche.png')
    
    print("\nDirect avalanche testing completed. Results saved to test_results/direct_avalanche/")
