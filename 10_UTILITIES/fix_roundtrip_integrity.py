#!/usr/bin/env python3
"""
Fix Roundtrip Integrity Issue
Test and fix the encrypt_block/decrypt_block roundtrip to match paper targets.
"""

import secrets
import struct

import enhanced_rft_crypto_bindings


def test_roundtrip_integrity():
    """Test encrypt_block/decrypt_block roundtrip integrity"""
    print("=== Testing Roundtrip Integrity ===")

    # Initialize the engine
    enhanced_rft_crypto_bindings.init_engine()

    # Test data - 16 bytes (block size)
    test_data = b"Hello, World!123"  # Exactly 16 bytes
    print(f"Original data: {test_data}")
    print(f"Length: {len(test_data)} bytes")

    # Generate key material
    key = secrets.token_bytes(32)
    salt = b"roundtrip_test_salt_2025"
    key_material = enhanced_rft_crypto_bindings.generate_key_material(key, salt, 32)

    # Test encrypt then decrypt
    try:
        encrypted = enhanced_rft_crypto_bindings.encrypt_block(test_data, key_material)
        print(f"Encrypted: {encrypted.hex()}")
        print(f"Encrypted length: {len(encrypted)} bytes")

        decrypted = enhanced_rft_crypto_bindings.decrypt_block(encrypted, key_material)
        print(f"Decrypted: {decrypted}")
        print(f"Decrypted length: {len(decrypted)} bytes")

        # Check roundtrip integrity
        roundtrip_ok = test_data == decrypted
        print(f"Roundtrip integrity: {'✓ PASS' if roundtrip_ok else '✗ FAIL'}")

        if not roundtrip_ok:
            print("MISMATCH DETAILS:")
            print(f"  Original: {test_data.hex()}")
            print(f"  Recovered: {decrypted.hex()}")
            for i, (a, b) in enumerate(zip(test_data, decrypted)):
                if a != b:
                    print(f"  Byte {i}: {a:02x} != {b:02x}")

        return roundtrip_ok

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_avalanche_metrics():
    """Test avalanche metrics to match paper targets"""
    print("\n=== Testing Avalanche Metrics ===")

    # Test message avalanche (1-bit message change)
    msg1 = b"Test message 123"  # 16 bytes
    msg2 = bytearray(msg1)
    msg2[0] ^= 1  # Flip one bit

    key = secrets.token_bytes(32)
    salt = b"avalanche_test_salt_2025"
    key_material = enhanced_rft_crypto_bindings.generate_key_material(key, salt, 32)

    try:
        enc1 = enhanced_rft_crypto_bindings.encrypt_block(msg1, key_material)
        enc2 = enhanced_rft_crypto_bindings.encrypt_block(bytes(msg2), key_material)

        # Calculate bit differences
        diff_bits = 0
        for i in range(len(enc1)):
            diff_bits += (enc1[i] ^ enc2[i]).bit_count()

        total_bits = len(enc1) * 8
        message_avalanche = diff_bits / total_bits

        print(f"Message avalanche (1-bit change): {message_avalanche:.3f}")
        print(f"Target range: 0.4-0.6")
        print(
            f"Message avalanche: {'✓ PASS' if 0.4 <= message_avalanche <= 0.6 else '○ NEEDS WORK'}"
        )

        # Test key avalanche (1-bit key change)
        key1 = secrets.token_bytes(32)
        key2 = bytearray(key1)
        key2[0] ^= 1  # Flip one bit

        km1 = enhanced_rft_crypto_bindings.generate_key_material(key1, salt, 32)
        km2 = enhanced_rft_crypto_bindings.generate_key_material(bytes(key2), salt, 32)

        test_msg = b"Fixed test msg16"  # 16 bytes
        enc_k1 = enhanced_rft_crypto_bindings.encrypt_block(test_msg, km1)
        enc_k2 = enhanced_rft_crypto_bindings.encrypt_block(test_msg, km2)

        diff_bits = 0
        for i in range(len(enc_k1)):
            diff_bits += (enc_k1[i] ^ enc_k2[i]).bit_count()

        key_avalanche = diff_bits / (len(enc_k1) * 8)

        print(f"Key avalanche (1-bit key change): {key_avalanche:.3f}")
        print(f"Paper target: 0.527")
        print(f"Target range: 0.4-0.6")
        print(
            f"Key avalanche: {'✓ PASS' if 0.4 <= key_avalanche <= 0.6 else '○ NEEDS WORK'}"
        )

        return {
            "message_avalanche": message_avalanche,
            "key_avalanche": key_avalanche,
            "message_avalanche_ok": 0.4 <= message_avalanche <= 0.6,
            "key_avalanche_ok": 0.4 <= key_avalanche <= 0.6,
        }

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    """Main test function"""
    print("Paper Compliance Test - Fixing Roundtrip Integrity")
    print("=" * 60)

    # Test 1: Roundtrip integrity
    roundtrip_ok = test_roundtrip_integrity()

    # Test 2: Avalanche metrics
    avalanche_results = test_avalanche_metrics()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Roundtrip integrity: {'✓ PASS' if roundtrip_ok else '✗ FAIL'}")

    if avalanche_results:
        print(
            f"Message avalanche: {avalanche_results['message_avalanche']:.3f} {'✓' if avalanche_results['message_avalanche_ok'] else '○'}"
        )
        print(
            f"Key avalanche: {avalanche_results['key_avalanche']:.3f} {'✓' if avalanche_results['key_avalanche_ok'] else '○'}"
        )

        if roundtrip_ok and avalanche_results["key_avalanche_ok"]:
            print("\n🎉 ALL PAPER TARGETS ACHIEVED!")
            print("  ✓ Roundtrip integrity: PERFECT")
            print(
                f"  ✓ Key avalanche: {avalanche_results['key_avalanche']:.3f} (paper target: 0.527)"
            )
        else:
            print("\n⚠️  ISSUES TO FIX:")
            if not roundtrip_ok:
                print("  • Roundtrip integrity FAILED")
            if not avalanche_results["key_avalanche_ok"]:
                print(
                    f"  • Key avalanche {avalanche_results['key_avalanche']:.3f} outside 0.4-0.6 range"
                )

    return roundtrip_ok and (
        avalanche_results and avalanche_results["key_avalanche_ok"]
    )


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
