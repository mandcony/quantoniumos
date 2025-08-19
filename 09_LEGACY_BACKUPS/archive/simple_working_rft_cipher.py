||#!/usr/bin/env python3
""""""
Simple Reversible RFT Cipher - Proof of Concept This demonstrates the cryptographic potential of RFT with proper reversibility. Focus: Make it work correctly first, then optimize for security.
"""
"""

import numpy as np
import hashlib from typing
import Dict

class SimpleReversibleRFTCipher:
"""
"""
    Simple but mathematically correct RFT cipher Key insight: Keep ALL operations in the integer domain to avoid floating-point precision loss that breaks reversibility.
"""
"""

    def __init__(self, block_size: int = 8):
        self.block_size = block_size
        self.phi = (1 + np.sqrt(5)) / 2

        # Golden ratio

        # Use smaller values to avoid overflow
        self.phase_multipliers = [ 1, # phi⁰ 2, # ~phi¹ (simplified) 3, # ~phi^2 (simplified) 5 # ~phi^3 (simplified) ]
    def derive_keys(self, password: bytes) -> Dict[str, np.ndarray]:
"""
"""
        Derive keys from password
"""
"""

        # Generate key material key_material = hashlib.sha256(password).digest()

        # Create 4 phase keys phase_keys = []
        for i in range(4): start = i * 8 key_bytes = key_material[start:start + 8]

        # Keep as integers to avoid float precision issues phase_key = np.frombuffer(key_bytes, dtype=np.uint8)[:
        self.block_size] phase_keys.append(phase_key)
        return {'phase_keys': phase_keys}
    def _apply_simple_transform(self, data: np.ndarray, key: np.ndarray, multiplier: int, encrypt: bool = True) -> np.ndarray:
"""
"""
        Apply simple reversible transformation
"""
"""

        if encrypt:

        # Simple key-dependent XOR with golden ratio structure transformed = data.copy().astype(np.int32)

        # Use int32 to avoid overflow
        for i in range(len(data)): key_byte = int(key[i % len(key)])

        # Use modular arithmetic to stay in byte range transformed[i] = (int(data[i]) + key_byte * multiplier) % 256
        else:

        # Exact reverse: subtract instead of add transformed = data.copy().astype(np.int32)
        for i in range(len(data)): key_byte = int(key[i % len(key)])

        # Modular subtraction (exact inverse of addition) transformed[i] = (int(data[i]) - key_byte * multiplier) % 256
        return transformed.astype(np.uint8)
    def _permute_bytes(self, data: np.ndarray, key: np.ndarray, encrypt: bool = True) -> np.ndarray:
"""
"""
        Apply reversible byte permutation
"""
"""

        # Create deterministic permutation from key perm_indices = np.argsort(key[:len(data)])
        if encrypt:

        # Apply permutation
        return data[perm_indices]
        else:

        # Apply inverse permutation inverse_perm = np.argsort(perm_indices)
        return data[inverse_perm]
    def encrypt_block(self, plaintext_block: np.ndarray, keys: Dict) -> np.ndarray:
"""
"""
        Encrypt a single block
"""
"""
        state = plaintext_block.copy()

        # Apply 4 phases with different multipliers
        for phase in range(4): phase_key = keys['phase_keys'][phase] multiplier =
        self.phase_multipliers[phase]

        # Transform with key and golden ratio multiplier state =
        self._apply_simple_transform(state, phase_key, multiplier, encrypt=True)

        # Permute bytes state =
        self._permute_bytes(state, phase_key, encrypt=True)
        return state
    def decrypt_block(self, ciphertext_block: np.ndarray, keys: Dict) -> np.ndarray:
"""
"""
        Decrypt a single block (exact reverse of encryption)
"""
"""
        state = ciphertext_block.copy()

        # Reverse all 4 phases in exact reverse order
        for phase in reversed(range(4)): phase_key = keys['phase_keys'][phase] multiplier =
        self.phase_multipliers[phase]

        # Reverse permutation first state =
        self._permute_bytes(state, phase_key, encrypt=False)

        # Reverse transformation state =
        self._apply_simple_transform(state, phase_key, multiplier, encrypt=False)
        return state
    def encrypt(self, plaintext: bytes, password: bytes) -> bytes:
"""
"""
        Encrypt data
"""
"""
        keys =
        self.derive_keys(password)

        # Simple padding: add length byte at end padding_len =
        self.block_size - ((len(plaintext) + 1) %
        self.block_size)
        if padding_len ==
        self.block_size: padding_len = 0

        # Plaintext + padding + length_byte padded = plaintext + b'\x00' * padding_len + bytes([len(plaintext) % 256])

        # Encrypt block by block ciphertext = b''
        for i in range(0, len(padded),
        self.block_size): block = padded[i:i +
        self.block_size] block_array = np.frombuffer(block, dtype=np.uint8) encrypted_block =
        self.encrypt_block(block_array, keys) ciphertext += encrypted_block.astype(np.uint8).tobytes()
        return ciphertext
    def decrypt(self, ciphertext: bytes, password: bytes) -> bytes:
"""
"""
        Decrypt data
"""
"""
        keys =
        self.derive_keys(password)

        # Decrypt block by block decrypted_blocks = []
        for i in range(0, len(ciphertext),
        self.block_size): block = ciphertext[i:i +
        self.block_size] block_array = np.frombuffer(block, dtype=np.uint8) decrypted_block =
        self.decrypt_block(block_array, keys) decrypted_blocks.append(decrypted_block.tobytes())

        # Reconstruct plaintext padded_plaintext = b''.join(decrypted_blocks)

        # Extract original length from last byte
        if padded_plaintext: original_length = padded_plaintext[-1]

        # Extract original message plaintext = padded_plaintext[:original_length]
        return plaintext
        return b''
    def test_simple_cipher():
"""
"""
        Test the simple reversible cipher
"""
        """ cipher = SimpleReversibleRFTCipher(block_size=8) password = b"test123"
        print("🔐 Simple Reversible RFT Cipher Test")
        print("=" * 45)
        print(f"Block size: {cipher.block_size} bytes")
        print(f"Golden ratio multipliers: {cipher.phase_multipliers}")
        print()

        # Test various message sizes test_messages = [ b"Hi", b"Hello!", b"Test123", b"A" * 7,

        # Just under block size b"B" * 8,

        # Exactly block size b"C" * 9,

        # Just over block size b"The quick brown fox", b"This is a longer test message for the RFT cipher." ] all_passed = True for i, message in enumerate(test_messages, 1):
        print(f"Test {i}: '{message.decode('utf-8', errors='replace')}' ({len(message)} bytes)")

        # Encrypt ciphertext = cipher.encrypt(message, password)
        print(f" Encrypted: {ciphertext.hex()} ({len(ciphertext)} bytes)")

        # Decrypt decrypted = cipher.decrypt(ciphertext, password)
        print(f" Decrypted: '{decrypted.decode('utf-8', errors='replace')}' ({len(decrypted)} bytes)")

        # Check success = decrypted == message
        print(f" Result: {'✅ PASS'
        if success else '❌ FAIL'}")
        if not success: all_passed = False
        print(f" Expected: {message}")
        print(f" Got: {decrypted}")
        print()
        print(f"Overall Result: {'🎉 All tests passed!'
        if all_passed else '💥 Some tests failed'}")
        if all_passed:

        # Test avalanche effect
        print("\n Avalanche Effect Test") msg1 = b"Avalanche test message" msg2 = bytearray(msg1) msg2[0] ^= 0x01

        # Flip one bit cipher1 = cipher.encrypt(msg1, password) cipher2 = cipher.encrypt(bytes(msg2), password)

        # Count bit differences bit_diffs = sum(bin(a ^ b).count('1') for a, b in zip(cipher1, cipher2)) total_bits = len(cipher1) * 8 avalanche_ratio = bit_diffs / total_bits
        print(f"Input change: 1 bit")
        print(f"Output change: {bit_diffs}/{total_bits} bits ({avalanche_ratio:.3f})")
        if avalanche_ratio > 0.3:
        print("✅ Good avalanche effect!")
        else:
        print("⚠️ Moderate avalanche effect (could be improved)")

        # Test key sensitivity
        print("||n🔑 Key Sensitivity Test") password2 = b"test124"

        # Change last character cipher_diff_key = cipher.encrypt(msg1, password2) key_bit_diffs = sum(bin(a ^ b).count('1') for a, b in zip(cipher1, cipher_diff_key)) key_avalanche = key_bit_diffs / total_bits
        print(f"Key change: 1 character")
        print(f"Output change: {key_bit_diffs}/{total_bits} bits ({key_avalanche:.3f})")
        if key_avalanche > 0.4:
        print("✅ Strong key sensitivity!")
        else:
        print("⚠️ Moderate key sensitivity")

if __name__ == "__main__": test_simple_cipher()