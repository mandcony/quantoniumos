""""""
Test diffusion properties of the optimized resonance encryption
""""""
import secrets
import statistics
from optimized_resonance_encrypt import optimized_resonance_encrypt

def count_bit_differences(bytes1, bytes2):
    """"""Count how many bits differ between two byte sequences""""""
    diff_bits = 0
    for b1, b2 in zip(bytes1, bytes2):
        xor = b1 ^ b2
        for bit in range(8):
            if (xor >> bit) & 1:
                diff_bits += 1
    return diff_bits

def test_avalanche_effect(num_tests=1000):
    """"""Test the avalanche effect by changing single bits and measuring diffusion""""""
    print("Testing avalanche effect...")
    print(f"Running {num_tests} tests...")

    # Store bit change percentages
    percentages = []

    key = "test_key_123"
    base_input = b"Test message for avalanche analysis" * 4  # Longer input for better analysis

    for test in range(num_tests):
        if test % 100 == 0 and test > 0:
            print(f"Completed {test} tests...")

        # Encrypt original input
        cipher1 = optimized_resonance_encrypt(base_input, key)

        # Modify a single random bit in the input
        modified = bytearray(base_input)
        byte_pos = secrets.randbelow(len(modified))
        bit_pos = secrets.randbelow(8)
        modified[byte_pos] ^= (1 << bit_pos)

        # Encrypt modified input
        cipher2 = optimized_resonance_encrypt(bytes(modified), key)

        # Compare the ciphertexts (excluding signature and token)
        actual_cipher1 = cipher1[40:]  # Skip signature (8 bytes) and token (32 bytes)
        actual_cipher2 = cipher2[40:]

        # Count bit differences
        diff_bits = count_bit_differences(actual_cipher1, actual_cipher2)
        total_bits = len(actual_cipher1) * 8
        percentage = (diff_bits / total_bits) * 100

        percentages.append(percentage)

    # Calculate statistics
    avg_change = statistics.mean(percentages)
    std_dev = statistics.stdev(percentages)
    min_change = min(percentages)
    max_change = max(percentages)

    print("\nAvalanche Effect Analysis:")
    print("=" * 50)
    print(f"Average bit change: {avg_change:.2f}%")
    print(f"Standard deviation: {std_dev:.2f}%")
    print(f"Minimum bit change: {min_change:.2f}%")
    print(f"Maximum bit change: {max_change:.2f}%")
    print("=" * 50)

    # Evaluate results
    if 45 <= avg_change <= 55 and std_dev < 5:
        print("✓ PASS: Good avalanche effect (close to ideal 50% with low variance)")
    else:
        print("✗ FAIL: Poor avalanche effect (significantly different from ideal 50%)")

if __name__ == "__main__":
    test_avalanche_effect()
