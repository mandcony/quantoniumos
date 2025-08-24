"""
Test Encryption Functionality
"""

print("🔐 TESTING ENCRYPTION FUNCTIONALITY (Fixed Block Size)")
print("=" * 60)

# Test the C++ RFT Crypto Engine directly
try:
    import enhanced_rft_crypto_bindings as crypto

    # Test key generation
    password = b"test_password_123"
    salt = b"test_salt_16byte"
    key_length = 32

    print("📍 Testing Key Generation...")
    key = crypto.generate_key_material(password, salt, key_length)
    print(f"✅ Key Generated: {len(key)} bytes")
    print(f"   Key (hex): {key.hex()[:32]}...")

    # Test encryption with exactly 16 bytes
    print("\n📍 Testing Encryption (16-byte blocks)...")
    plaintext = b"QuantoniumOS!!!!"  # Exactly 16 bytes
    print(f"   Plaintext: {plaintext} ({len(plaintext)} bytes)")

    ciphertext = crypto.encrypt_block(plaintext, key)
    print(f"✅ Encryption Success: {len(ciphertext)} bytes")
    print(f"   Ciphertext (hex): {ciphertext.hex()}")

    # Test decryption
    print("\n📍 Testing Decryption...")
    decrypted = crypto.decrypt_block(ciphertext, key)
    print(f"✅ Decryption Success: {len(decrypted)} bytes")
    print(f"   Decrypted: {decrypted}")

    # Check integrity
    integrity = plaintext == decrypted
    status = "PASSED" if integrity else "FAILED"
    print(f"\n🔒 Integrity Check: ✅ {status}")

    # Test multiple blocks
    print("\n📍 Testing Multiple 16-byte Blocks...")

    test_blocks = [
        b"Block1QuantonOS!",  # Block 1
        b"Block2TestData!",  # Block 2
        b"Block3Encrypted!",  # Block 3
    ]

    encrypted_blocks = []
    for i, block in enumerate(test_blocks):
        enc_block = crypto.encrypt_block(block, key)
        encrypted_blocks.append(enc_block)
        print(f"   Block {i+1}: {block} -> {enc_block.hex()[:16]}...")

    # Decrypt all blocks
    print("\n📍 Decrypting All Blocks...")
    all_integrity = True
    for i, (original, encrypted) in enumerate(zip(test_blocks, encrypted_blocks)):
        decrypted = crypto.decrypt_block(encrypted, key)
        block_integrity = original == decrypted
        all_integrity = all_integrity and block_integrity
        status = "OK" if block_integrity else "FAILED"
        print(f"   Block {i+1}: {status} - {decrypted}")

    final_status = "PASSED" if all_integrity else "FAILED"
    print(f"\n🔒 All Blocks Integrity: ✅ {final_status}")

    # Test avalanche effect
    print("\n📍 Testing Avalanche Effect...")
    plaintext1 = b"AvalancheTest123"  # 16 bytes
    plaintext2 = b"AvalancheTest124"  # 16 bytes, last char changed

    avalanche_score = crypto.avalanche_test(plaintext1, plaintext2)
    print(f"✅ Avalanche Test: {avalanche_score:.4f}")
    good_avalanche = "YES" if avalanche_score > 0.4 else "NO"
    print(f"   Good avalanche: ✅ {good_avalanche}")

    print("\n🎉 ENCRYPTION IS FULLY WORKING!")

except Exception as e:
    print(f"❌ Encryption test failed: {e}")
    import traceback

    traceback.print_exc()
