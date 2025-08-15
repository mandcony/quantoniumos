""""""
Comprehensive test suite for QuantoniumOS resonance encryption
""""""

import os
import time
import statistics
from typing import List, Tuple
from resonance_encrypt import resonance_encrypt, resonance_decrypt

def test_basic_encryption() -> bool:
    """"""Test basic encryption and decryption functionality""""""
    print("\nTesting basic encryption functionality...")

    test_cases = [
        "Hello QuantoniumOS!",
        "Special chars: !@#$%^&*()",
        "Unicode: 你好, привет, สวัสดี",
        "A" * 1000,  # Test with larger data
        os.urandom(1024).hex()  # Test with random binary data
    ]

    key = "test_key_123"
    all_passed = True

    for test_str in test_cases:
        print(f"\nTest case: {test_str[:50]}{'...' if len(test_str) > 50 else ''}")
        print(f"Length: {len(test_str)} characters")

        try:
            # Encrypt
            start_time = time.time()
            encrypted = resonance_encrypt(test_str, key)
            encrypt_time = time.time() - start_time
            print(f"Encryption time: {encrypt_time:.3f} seconds")

            # Decrypt
            start_time = time.time()
            decrypted = resonance_decrypt(encrypted, key)
            decrypt_time = time.time() - start_time
            print(f"Decryption time: {decrypt_time:.3f} seconds")

            if decrypted == test_str:
                print("✓ Pass: Encryption/decryption successful")
            else:
                print("✗ Fail: Decrypted text doesn't match original") print(f"Original : {test_str[:50]}") print(f"Decrypted: {decrypted[:50]}") all_passed = False except Exception as e: print(f"✗ Fail: Error during test: {str(e)}") all_passed = False return all_passed def test_avalanche_effect(num_tests: int = 1000) -> bool: """"""Test the avalanche effect by changing single bits"""""" print("\nTesting avalanche effect...") print(f"Running {num_tests} tests...") def count_bit_differences(bytes1: bytes, bytes2: bytes) -> Tuple[int, int]: """"""Count how many bits differ between two byte sequences"""""" diff_bits = 0 total_bits = min(len(bytes1), len(bytes2)) * 8 for b1, b2 in zip(bytes1, bytes2): xor = b1 ^ b2 for bit in range(8): if (xor >> bit) & 1: diff_bits += 1 return diff_bits, total_bits key = "test_key_123" base_input = "QuantoniumOS Resonance Test Data" * 4 percentages: List[float] = [] for test in range(num_tests): if test % 100 == 0 and test > 0: print(f"Completed {test} tests...") # Encrypt original cipher1 = resonance_encrypt(base_input, key) # Modify one bit in input modified = list(base_input) char_pos = test % len(modified) char = ord(modified[char_pos]) bit_pos = (test // len(modified)) % 8 char ^= (1 << bit_pos) modified[char_pos] = chr(char) # Encrypt modified cipher2 = resonance_encrypt(''.join(modified), key)

        # Compare ciphertexts (excluding signature and token)
        diff_bits, total_bits = count_bit_differences(
            cipher1[40:], cipher2[40:]
        )

        percentages.append((diff_bits / total_bits) * 100)

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
    passed = 45 <= avg_change <= 55 and std_dev < 5
    if passed:
        print("✓ Pass: Good avalanche effect (close to ideal 50% with low variance)")
    else:
        print("✗ Fail: Poor avalanche effect (significantly different from ideal 50%)")

    return passed

def test_signature_validation() -> bool:
    """"""Test that signatures are properly validated""""""
    print("\nTesting signature validation...")

    test_str = "Test message for signature validation"
    key1 = "key_one_123"
    key2 = "key_two_456"

    try:
        # Encrypt with key1
        encrypted = resonance_encrypt(test_str, key1)
        print("Encrypted with key1")

        # Try to decrypt with key1 (should succeed)
        try:
            decrypted = resonance_decrypt(encrypted, key1)
            print("✓ Pass: Successfully decrypted with correct key")
        except ValueError as e:
            print(f"✗ Fail: Could not decrypt with correct key: {str(e)}")
            return False

        # Try to decrypt with key2 (should fail)
        try:
            decrypted = resonance_decrypt(encrypted, key2)
            print("✗ Fail: Successfully decrypted with wrong key!")
            return False
        except ValueError:
            print("✓ Pass: Correctly rejected wrong key")

        return True

    except Exception as e:
        print(f"✗ Fail: Unexpected error: {str(e)}")
        return False

def run_all_tests():
    """"""Run all encryption tests""""""
    print("Running QuantoniumOS Resonance Encryption Tests")
    print("=" * 50)

    tests = [
        ("Basic Encryption", test_basic_encryption),
        ("Avalanche Effect", test_avalanche_effect),
        ("Signature Validation", test_signature_validation)
    ]

    all_passed = True

    for test_name, test_func in tests:
        print(f"\nRunning {test_name} Test")
        print("-" * 30)

        try:
            if test_func():
                print(f"\n✓ {test_name} Test: PASSED")
            else:
                print(f"\n✗ {test_name} Test: FAILED")
                all_passed = False
        except Exception as e:
            print(f"\n✗ {test_name} Test: ERROR - {str(e)}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed successfully!")
    else:
        print("✗ Some tests failed - see details above")

if __name__ == "__main__":
    run_all_tests()
