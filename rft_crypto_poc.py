||#!/usr/bin/env python3
""""""
RFT-Enhanced Cryptographic Proof of Concept

Demonstrates how RFT can be used in cryptography with proper key management
and the 4-phase geometric waveform structure based on golden ratio powers.

This is a proof of concept showing the cryptographic potential, not a production cipher.
""""""

import numpy as np
import hashlib
from typing import Dict, Tuple
import time

class RFTCryptoProofOfConcept:
    """"""
    RFT-based cryptographic proof of concept

    Key innovations:
    1. Secret key derivation from password
    2. 4-phase golden ratio waveform structure
    3. Frequency domain key mixing
    4. Non-linear transformations for avalanche effect
    """"""

    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.block_size = 16

        # 4-phase structure based on phi^n
        self.phase_coeffs = np.array([
            1.0,           # phi^0
            self.phi,      # phi^1
            self.phi**2,   # phi^2
            self.phi**3    # phi^3
        ])

    def derive_key(self, password: bytes, salt: bytes = b"rft_salt") -> Dict[str, np.ndarray]:
        """"""Derive cryptographic keys from password using RFT structure""""""

        # Generate key material
        key_material = hashlib.pbkdf2_hmac('sha256', password, salt, 10000, 128)
        key_bytes = np.frombuffer(key_material, dtype=np.uint8)

        # Split into 4 phase keys + round constants
        phase_keys = []
        for i in range(4):
            phase_key = key_bytes[i*16:(i+1)*16].astype(np.float64) / 255.0
            phase_keys.append(phase_key)

        round_constants = key_bytes[64:80].astype(np.float64) / 255.0
        mixing_key = key_bytes[80:96].astype(np.float64) / 255.0

        return {
            'phase_keys': phase_keys,
            'round_constants': round_constants,
            'mixing_key': mixing_key
        }

    def rft_frequency_mixing(self, data: np.ndarray, key: np.ndarray,
                           phase_coeff: float, encrypt: bool = True) -> np.ndarray:
        """"""Apply RFT-based frequency domain key mixing""""""

        # Convert to frequency domain
        fft_data = np.fft.fft(data.astype(np.complex128))

        if encrypt:
            # Phase 1: Key-dependent frequency permutation
            perm_indices = np.argsort(key)
            fft_data = fft_data[perm_indices]

            # Phase 2: Golden ratio phase modulation
            phase_mod = 2 * np.pi * key * phase_coeff
            fft_data *= np.exp(1j * phase_mod)

            # Phase 3: Resonance coupling
            resonance = np.exp(1j * 2 * np.pi * np.arange(len(data)) * phase_coeff / len(data))
            fft_data *= resonance

        else:
            # Reverse for decryption
            # Reverse Phase 3
            resonance = np.exp(-1j * 2 * np.pi * np.arange(len(data)) * phase_coeff / len(data))
            fft_data *= resonance

            # Reverse Phase 2
            phase_mod = -2 * np.pi * key * phase_coeff
            fft_data *= np.exp(1j * phase_mod)

            # Reverse Phase 1
            perm_indices = np.argsort(key)
            reverse_perm = np.argsort(perm_indices)
            fft_data = fft_data[reverse_perm]

        # Convert back to time domain
        result = np.fft.ifft(fft_data)
        return np.real(result)

    def nonlinear_substitution(self, data: np.ndarray, round_key: np.ndarray,
                             encrypt: bool = True) -> np.ndarray:
        """"""Non-linear byte substitution using golden ratio modulation""""""

        if encrypt:
            # Forward substitution
            mixed = (data + round_key) % 1.0
            # Use golden ratio in sine transformation for non-linearity
            substituted = 0.5 * (1 + np.sin(2 * np.pi * mixed * self.phi))
        else:
            # Reverse substitution
            # Reverse sine transformation
            sin_input = 2 * data - 1
            sin_input = np.clip(sin_input, -1, 1)  # Ensure valid domain
            mixed = np.arcsin(sin_input) / (2 * np.pi * self.phi)
            substituted = (mixed - round_key) % 1.0

        return substituted

    def encrypt_block(self, plaintext_block: np.ndarray, keys: Dict) -> np.ndarray:
        """"""Encrypt a single block using 4-phase RFT structure""""""

        state = plaintext_block.copy()

        # Apply 4 phases with different golden ratio powers
        for phase in range(4):
            phase_key = keys['phase_keys'][phase]
            phase_coeff = self.phase_coeffs[phase]

            # RFT frequency mixing
            state = self.rft_frequency_mixing(state, phase_key, phase_coeff, encrypt=True)

            # Non-linear substitution
            round_key = keys['round_constants']
            state = self.nonlinear_substitution(state, round_key, encrypt=True)

        # Final mixing
        mixing_key = keys['mixing_key']
        state = self.rft_frequency_mixing(state, mixing_key, self.phi, encrypt=True)

        return state

    def decrypt_block(self, ciphertext_block: np.ndarray, keys: Dict) -> np.ndarray:
        """"""Decrypt a single block by reversing encryption""""""

        state = ciphertext_block.copy()

        # Reverse final mixing
        mixing_key = keys['mixing_key']
        state = self.rft_frequency_mixing(state, mixing_key, self.phi, encrypt=False)

        # Reverse 4 phases
        for phase in reversed(range(4)):
            phase_key = keys['phase_keys'][phase]
            phase_coeff = self.phase_coeffs[phase]

            # Reverse non-linear substitution
            round_key = keys['round_constants']
            state = self.nonlinear_substitution(state, round_key, encrypt=False)

            # Reverse RFT frequency mixing
            state = self.rft_frequency_mixing(state, phase_key, phase_coeff, encrypt=False)

        return state

    def encrypt(self, plaintext: bytes, password: bytes) -> bytes:
        """"""Encrypt data using RFT cipher""""""

        keys = self.derive_key(password)

        # Pad to block size
        padding_len = self.block_size - (len(plaintext) % self.block_size)
        if padding_len == self.block_size:
            padding_len = 0
        padded = plaintext + bytes([padding_len] * padding_len)

        ciphertext = b''
        for i in range(0, len(padded), self.block_size):
            block = padded[i:i + self.block_size]
            block_array = np.frombuffer(block, dtype=np.uint8).astype(np.float64) / 255.0

            encrypted_block = self.encrypt_block(block_array, keys)
            block_bytes = (np.clip(encrypted_block, 0, 1) * 255).astype(np.uint8)
            ciphertext += block_bytes.tobytes()

        return ciphertext

    def decrypt(self, ciphertext: bytes, password: bytes) -> bytes:
        """"""Decrypt data using RFT cipher""""""

        keys = self.derive_key(password)

        plaintext = b''
        for i in range(0, len(ciphertext), self.block_size):
            block = ciphertext[i:i + self.block_size]
            block_array = np.frombuffer(block, dtype=np.uint8).astype(np.float64) / 255.0

            decrypted_block = self.decrypt_block(block_array, keys)
            block_bytes = (np.clip(decrypted_block, 0, 1) * 255).astype(np.uint8)
            plaintext += block_bytes.tobytes()

        # Remove padding
        if plaintext:
            padding_len = plaintext[-1]
            if padding_len <= self.block_size:
                plaintext = plaintext[:-padding_len]

        return plaintext

    def F_function(self, rft_output: np.ndarray, phase_key: np.ndarray, round_num: int) -> np.ndarray:
        # 1. Convert the RFT output to 32-bit words for ARX mixing
        words = np.frombuffer(rft_output.tobytes(), dtype=np.uint32).copy()
        
        for i in range(len(words)):
            key_val = np.uint32(phase_key[i % len(phase_key)])
            # ARX mixing: add round constant and key; ensure modulo 2^32 arithmetic
            words[i] = (words[i] + np.uint32(round_num) + key_val) & 0xffffffff
            # Rotate left by a round-dependent amount
            rot = (i + round_num) % 16
            words[i] ^= ((words[i] << rot) | (words[i] >> (32 - rot))) & 0xffffffff
        
        # 2. Convert the mixed words back to a byte array
        mixed_bytes = words.tobytes()
        result = np.frombuffer(mixed_bytes, dtype=np.uint8).copy()
        
        # 3. Apply a round-dependent byte permutation for cross-talk between bytes
        permuted = np.empty_like(result)
        block_size = len(result)
        for idx in range(block_size):
            new_idx = (((idx * (round_num + 1)) + (round_num % block_size))) % block_size
            permuted[new_idx] = result[idx]
        
        return permuted

def test_rft_crypto_poc():
    """"""Test the RFT crypto proof of concept""""""

    cipher = RFTCryptoProofOfConcept()
    password = b"secret_password_123"

    print("🔐 RFT Cryptographic Proof of Concept")
    print("=" * 50)
    print(f"Block size: {cipher.block_size} bytes")
    print(f"Golden ratio phi: {cipher.phi:.6f}")
    print(f"4-phase coefficients: {cipher.phase_coeffs}")
    print()

    # Test messages of different lengths
    test_messages = [
        b"Hello RFT!",
        b"This is a longer test message for RFT encryption.",
        b"A" * 50,  # Repeated pattern
        b"The quick brown fox jumps over the lazy dog."
    ]

    all_tests_passed = True

    for i, message in enumerate(test_messages, 1):
        print(f"Test {i}: {len(message)} bytes")
        print(f"Message: {message}")

        # Encrypt
        start_time = time.time()
        ciphertext = cipher.encrypt(message, password)
        encrypt_time = time.time() - start_time

        print(f"Ciphertext ({len(ciphertext)} bytes): {ciphertext.hex()}")

        # Decrypt
        start_time = time.time()
        decrypted = cipher.decrypt(ciphertext, password)
        decrypt_time = time.time() - start_time

        print(f"Decrypted: {decrypted}")

        # Verify
        success = decrypted == message
        print(f"Success: {'✅' if success else '❌'}")
        print(f"Timing: encrypt {encrypt_time:.4f}s, decrypt {decrypt_time:.4f}s")

        if not success:
            all_tests_passed = False
            print(f"❌ Expected: {message}")
            print(f"❌ Got: {decrypted}")

        print("-" * 30)

    if all_tests_passed:
        print("\n🌊 Testing Avalanche Effect...")

        # Test avalanche with bit flip
        base_msg = b"Avalanche test message!"
        base_cipher = cipher.encrypt(base_msg, password)

        # Flip one bit
        modified_msg = bytearray(base_msg)
        modified_msg[0] ^= 0x01
        modified_cipher = cipher.encrypt(bytes(modified_msg), password)

        # Calculate differences
        bit_changes = sum(bin(a ^ b).count('1') for a, b in zip(base_cipher, modified_cipher))
        total_bits = len(base_cipher) * 8
        avalanche_ratio = bit_changes / total_bits

        print(f"Input change: 1 bit flip")
        print(f"Output changes: {bit_changes}/{total_bits} bits ({avalanche_ratio:.3f})")

        if avalanche_ratio > 0.3:
            print("✅ Good avalanche effect!")
        else:
            print("⚠️ Weak avalanche effect")

        print("\n🔑 Testing Key Sensitivity...")

        # Test with different password
        password2 = b"secret_password_124"  # Last digit changed
        cipher2 = cipher.encrypt(base_msg, password2)

        key_bit_changes = sum(bin(a ^ b).count('1') for a, b in zip(base_cipher, cipher2))
        key_avalanche = key_bit_changes / total_bits

        print(f"Key change: 1 character")
        print(f"Output changes: {key_bit_changes}/{total_bits} bits ({key_avalanche:.3f})")

        if key_avalanche > 0.3:
            print("✅ Good key sensitivity!")
        else:
            print("⚠️ Weak key sensitivity")

    print(f"||n📋 Summary: {'All tests passed! ✅' if all_tests_passed else 'Some tests failed ❌'}")

if __name__ == "__main__":
    test_rft_crypto_poc()
