#!/usr/bin/env python3
"""
Debug the exact decryption issue in the C++ engine.
Focus on Feistel structure and whitening order.
"""

import enhanced_rft_crypto_bindings


def test_feistel_structure():
    """Test the Feistel structure with minimal data to isolate the bug."""

    # Initialize engine
    result = enhanced_rft_crypto_bindings.init_engine()
    print(f"✅ Engine init: {result}")

    # Test with 16-byte plaintext (minimal for Feistel)
    password = b"test123"
    salt = b"testsalt"
    key = enhanced_rft_crypto_bindings.generate_key_material(password, salt, 32)
    print(f"✅ Key generated: {len(key)} bytes")

    # Simple 16-byte plaintext
    plaintext = b"AAAAAAAAAAAAAAAA"  # 16 bytes of 'A'
    print(f"📝 Original: {plaintext.hex()}")

    # Encrypt
    encrypted = enhanced_rft_crypto_bindings.encrypt_block(plaintext, key)
    print(f"🔒 Encrypted: {encrypted.hex()}")

    # Decrypt
    decrypted = enhanced_rft_crypto_bindings.decrypt_block(encrypted, key)
    print(f"🔓 Decrypted: {decrypted.hex()}")

    # Compare
    print(f"\n📊 Comparison:")
    print(f"Original:  {plaintext}")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {plaintext == decrypted}")

    # Byte-by-byte analysis
    print(f"\n🔍 Byte-by-byte analysis:")
    for i, (o, d) in enumerate(zip(plaintext, decrypted)):
        print(f"  Byte {i:2d}: orig={o:02x} dec={d:02x} diff={o^d:02x}")

    return plaintext == decrypted


def test_multiple_sizes():
    """Test different block sizes to see if the issue is size-dependent."""

    enhanced_rft_crypto_bindings.init_engine()
    password = b"test123"
    salt = b"testsalt"
    key = enhanced_rft_crypto_bindings.generate_key_material(password, salt, 32)

    test_cases = [
        (b"AA" * 8, "16-byte"),  # 16 bytes
        (b"BB" * 16, "32-byte"),  # 32 bytes
        (b"CC" * 32, "64-byte"),  # 64 bytes
    ]

    results = []
    for plaintext, desc in test_cases:
        print(f"\n🧪 Testing {desc} plaintext...")
        encrypted = enhanced_rft_crypto_bindings.encrypt_block(plaintext, key)
        decrypted = enhanced_rft_crypto_bindings.decrypt_block(encrypted, key)
        match = plaintext == decrypted
        results.append(match)
        print(f"   Match: {match}")
        if not match:
            print(f"   First 16 bytes - Orig: {plaintext[:16].hex()}")
            print(f"   First 16 bytes - Dec:  {decrypted[:16].hex()}")

    return results


if __name__ == "__main__":
    print("🔧 Debugging Decryption Issue\n")

    print("=" * 50)
    print("Test 1: Basic Feistel Structure")
    print("=" * 50)
    basic_result = test_feistel_structure()

    print("\n" + "=" * 50)
    print("Test 2: Multiple Block Sizes")
    print("=" * 50)
    size_results = test_multiple_sizes()

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Basic test passed: {basic_result}")
    print(f"Size tests passed: {size_results}")
    print(f"All tests passed: {basic_result and all(size_results)}")

    if not (basic_result and all(size_results)):
        print("\n❌ DECRYPTION ISSUE CONFIRMED")
        print("   - The C++ Feistel decrypt logic has a bug")
        print("   - Likely whitening order or round sequence issue")
        print("   - Need to fix the C++ implementation")
    else:
        print("\n✅ DECRYPTION WORKING CORRECTLY")
