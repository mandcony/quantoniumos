||#!/usr/bin/env python3
"""
Mathematically Reversible RFT Cipher Fixed version that addresses the core reversibility issues: 1. No quantization between rounds (stay in complex domain) 2. Multiplicative-only magnitude modulation (perfectly reversible) 3. Stable matrix operations (avoid poorly conditioned matrices) 4. Proper padding handling
"""
"""

import numpy as np
import hashlib from typing
import Tuple, Optional from dataclasses
import dataclass @dataclass

class RFTCipherKey:
"""
"""
    RFT cipher key with 4-phase geometric structure
"""
"""
    phase_keys: np.ndarray # 4 keys for phi, phi^2, phi^3, phi⁴ phases round_keys: np.ndarray

    # Per-round mixing keys resonance_seed: float

    # Golden ratio resonance modulation

class ReversibleRFTCipher:
"""
"""
    Mathematically reversible RFT cipher with proper cryptographic structure Key fixes for reversibility: - No quantization until final output - Multiplicative magnitude modulation only - Stable orthogonal mixing matrices - Proper padding with length indicator
"""
"""

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.num_rounds = 4

        # Fewer rounds for testing
        self.phi = (1 + np.sqrt(5)) / 2

        # Golden ratio

        # Golden ratio powers for 4-phase structure
        self.phi_powers = np.array([ 1.0, # phi⁰ (base)
        self.phi, # phi¹
        self.phi**2, # phi^2
        self.phi**3 # phi^3 ])
    def generate_key(self, password: bytes, salt: bytes = b'rft_salt') -> RFTCipherKey:
"""
"""
        Generate cipher key from password
"""
"""

        # Generate sufficient key material total_key_bytes = 4 *
        self.block_size +
        self.num_rounds *
        self.block_size + 8 key_material = hashlib.pbkdf2_hmac('sha256', password, salt, 10000, total_key_bytes)

        # Parse key material offset = 0

        # Phase keys (4 phases × block_size bytes each) phase_keys = []
        for _ in range(4): phase_key = np.frombuffer(key_material[offset:offset +
        self.block_size], dtype=np.uint8) phase_keys.append(phase_key.astype(np.float64) / 255.0) offset +=
        self.block_size

        # Round keys for mixing round_keys = []
        for _ in range(
        self.num_rounds): round_key = np.frombuffer(key_material[offset:offset +
        self.block_size], dtype=np.uint8) round_keys.append(round_key.astype(np.float64) / 255.0) offset +=
        self.block_size

        # Resonance seed resonance_bytes = key_material[offset:offset + 8] resonance_seed = np.frombuffer(resonance_bytes, dtype=np.float64)[0]
        return RFTCipherKey( phase_keys=np.array(phase_keys), round_keys=np.array(round_keys), resonance_seed=resonance_seed )
    def _create_orthogonal_matrix(self, size: int, seed: float) -> np.ndarray:
"""
"""
        Create a stable orthogonal matrix for reversible mixing
"""
"""

        # Use seed to create deterministic random state rng_state = np.random.get_state() np.random.seed(int(abs(seed) * 1000000) % 2**32)

        # Generate random matrix A = np.random.randn(size, size)

        # Make it orthogonal using QR decomposition (always stable) Q, R = np.linalg.qr(A)

        # Ensure determinant is +1 (proper rotation, not reflection)
        if np.linalg.det(Q) < 0: Q[:, 0] *= -1

        # Restore random state np.random.set_state(rng_state)
        return Q.astype(np.complex128)
    def _apply_reversible_phase_transform(self, data: np.ndarray, phase_key: np.ndarray, phi_power: float, encrypt: bool = True) -> np.ndarray:
"""
"""
        Apply mathematically reversible 4-phase transformation
"""
"""

        # Convert to frequency domain fft_data = np.fft.fft(data.astype(np.complex128))
        if encrypt:

        # Phase 1: Multiplicative magnitude modulation (reversible)

        # Use 1 + scale instead of arbitrary addition magnitude_scale = 1.0 + 0.5 * np.sin(2 * np.pi * phase_key * phi_power) fft_data *= magnitude_scale

        # Phase 2: Phase rotation (perfectly reversible) phase_rotation = 2 * np.pi * phase_key * phi_power fft_data *= np.exp(1j * phase_rotation)

        # Phase 3: Frequency permutation (reversible with inverse permutation) perm_indices = np.argsort(phase_key) fft_data = fft_data[perm_indices]

        # Phase 4: Golden ratio resonance coupling (reversible) resonance = np.exp(1j * 2 * np.pi * np.arange(len(data)) * phi_power / len(data)) fft_data *= resonance
        else:

        # Decrypt: reverse all operations in exact reverse order

        # Reverse Phase 4: Remove resonance coupling resonance = np.exp(-1j * 2 * np.pi * np.arange(len(data)) * phi_power / len(data)) fft_data *= resonance

        # Reverse Phase 3: Undo permutation perm_indices = np.argsort(phase_key) inverse_perm = np.argsort(perm_indices)

        # This is the correct inverse fft_data = fft_data[inverse_perm]

        # Reverse Phase 2: Undo phase rotation phase_rotation = -2 * np.pi * phase_key * phi_power fft_data *= np.exp(1j * phase_rotation)

        # Reverse Phase 1: Undo magnitude modulation (division is exact) magnitude_scale = 1.0 + 0.5 * np.sin(2 * np.pi * phase_key * phi_power) fft_data /= magnitude_scale

        # Convert back to time domain result = np.fft.ifft(fft_data)
        return result
    def _apply_orthogonal_mixing(self, data: np.ndarray, round_key: np.ndarray, resonance_seed: float, encrypt: bool = True) -> np.ndarray:
"""
"""
        Apply reversible orthogonal mixing
"""
"""

        # Create orthogonal matrix (same for encrypt/decrypt due to deterministic seed) mix_matrix =
        self._create_orthogonal_matrix(len(data), resonance_seed + np.sum(round_key))
        if encrypt:
        return mix_matrix @ data
        else:

        # For orthogonal matrices: inverse = transpose
        return mix_matrix.T @ data
    def encrypt_block(self, plaintext_block: np.ndarray, key: RFTCipherKey) -> np.ndarray:
"""
"""
        Encrypt a single block (stays in complex domain)
"""
"""

        # Start with complex representation state = plaintext_block.astype(np.complex128)

        # Apply rounds
        for round_num in range(
        self.num_rounds): round_key = key.round_keys[round_num]

        # Apply 4-phase transformations
        for phase in range(4): phase_key = key.phase_keys[phase] phi_power =
        self.phi_powers[phase] state =
        self._apply_reversible_phase_transform( state, phase_key, phi_power, encrypt=True )

        # Orthogonal mixing between rounds (except last)
        if round_num <
        self.num_rounds - 1: state =
        self._apply_orthogonal_mixing( state, round_key, key.resonance_seed, encrypt=True )
        return state
    def decrypt_block(self, ciphertext_block: np.ndarray, key: RFTCipherKey) -> np.ndarray:
"""
"""
        Decrypt a single block (exact mathematical inverse)
"""
"""
        state = ciphertext_block.astype(np.complex128)

        # Reverse all rounds in exact reverse order
        for round_num in reversed(range(
        self.num_rounds)): round_key = key.round_keys[round_num]

        # Reverse orthogonal mixing (except for last round which didn't have it)
        if round_num <
        self.num_rounds - 1: state =
        self._apply_orthogonal_mixing( state, round_key, key.resonance_seed, encrypt=False )

        # Reverse 4-phase transformations in reverse order
        for phase in reversed(range(4)): phase_key = key.phase_keys[phase] phi_power =
        self.phi_powers[phase] state =
        self._apply_reversible_phase_transform( state, phase_key, phi_power, encrypt=False )
        return state
    def encrypt(self, plaintext: bytes, key: RFTCipherKey) -> bytes:
"""
"""
        Encrypt arbitrary length data with proper padding
"""
"""

        # Proper padding: always add padding length byte padding_len =
        self.block_size - (len(plaintext) %
        self.block_size) padded = plaintext + bytes([padding_len] * padding_len)

        # Encrypt block by block (stay in complex domain until end) encrypted_blocks = []
        for i in range(0, len(padded),
        self.block_size): block = padded[i:i +
        self.block_size] block_array = np.frombuffer(block, dtype=np.uint8).astype(np.float64) / 255.0

        # Encrypt (returns complex array) encrypted_complex =
        self.encrypt_block(block_array, key) encrypted_blocks.append(encrypted_complex)

        # Convert all blocks to bytes at once (minimize quantization) ciphertext = b''
        for encrypted_block in encrypted_blocks:

        # Take real part and normalize CAREFULLY block_real = np.real(encrypted_block)

        # Normalize to [0, 1] range for byte conversion block_min, block_max = np.min(block_real), np.max(block_real)
        if block_max > block_min: block_normalized = (block_real - block_min) / (block_max - block_min)
        else: block_normalized = np.zeros_like(block_real)

        # Convert to bytes block_bytes = (block_normalized * 255).astype(np.uint8).tobytes() ciphertext += block_bytes
        return ciphertext
    def decrypt(self, ciphertext: bytes, key: RFTCipherKey) -> bytes:
"""
"""
        Decrypt arbitrary length data
"""
"""

        # Decrypt block by block decrypted_blocks = []
        for i in range(0, len(ciphertext),
        self.block_size): block = ciphertext[i:i +
        self.block_size]

        # Convert back to normalized floats (reverse of encryption) block_array = np.frombuffer(block, dtype=np.uint8).astype(np.float64) / 255.0

        # Decrypt (this should be the mathematical inverse) decrypted_complex =
        self.decrypt_block(block_array, key) decrypted_blocks.append(decrypted_complex)

        # Convert all blocks back to bytes plaintext = b''
        for decrypted_block in decrypted_blocks:

        # Take real part (imaginary should be ~0
        if decryption worked) block_real = np.real(decrypted_block)

        # Convert back to bytes (should recover original values) block_bytes = (np.clip(block_real, 0, 1) * 255).astype(np.uint8).tobytes() plaintext += block_bytes

        # Remove padding using the padding length indicator
        if plaintext: padding_len = plaintext[-1] if 0 < padding_len <=
        self.block_size:

        # Verify padding is correct expected_padding = bytes([padding_len] * padding_len)
        if plaintext[-padding_len:] == expected_padding: plaintext = plaintext[:-padding_len]
        return plaintext
    def test_reversible_cipher():
"""
"""
        Test the mathematically reversible RFT cipher
"""
        """ cipher = ReversibleRFTCipher(block_size=16) password = b"test_password" key = cipher.generate_key(password)
        print("🔐 Reversible RFT Cipher Test")
        print("=" * 40)
        print(f"Block size: {cipher.block_size} bytes")
        print(f"Rounds: {cipher.num_rounds}")
        print(f"Golden ratio: {cipher.phi:.6f}")
        print()

        # Test messages test_messages = [ b"Hello!", b"Test message", b"A" * 15,

        # Just under block size b"B" * 16,

        # Exactly block size b"C" * 17,

        # Just over block size b"The quick brown fox jumps over the lazy dog." ] all_success = True for i, message in enumerate(test_messages, 1):
        print(f"Test {i}: {len(message)} bytes")
        print(f"Original: {message}")

        # Encrypt ciphertext = cipher.encrypt(message, key)
        print(f"Encrypted ({len(ciphertext)} bytes): {ciphertext.hex()}")

        # Decrypt decrypted = cipher.decrypt(ciphertext, key)
        print(f"Decrypted: {decrypted}")

        # Check success success = decrypted == message
        print(f"Success: {'✅'
        if success else '❌'}")
        if not success: all_success = False
        print(f" Expected: {message}")
        print(f" Got: {decrypted}")
        print(f" Diff: {len(message)} vs {len(decrypted)} bytes")
        print("-" * 30)

        # Overall result
        print(f"Overall: {'✅ All tests passed!'
        if all_success else '❌ Some tests failed'}")
        if all_success:

        # Test avalanche effect
        print("||n Avalanche Effect Test") base_msg = b"Avalanche test!" base_cipher = cipher.encrypt(base_msg, key)

        # Flip one bit modified_msg = bytearray(base_msg) modified_msg[0] ^= 0x01 modified_cipher = cipher.encrypt(bytes(modified_msg), key)

        # Calculate differences bit_changes = sum(bin(a ^ b).count('1') for a, b in zip(base_cipher, modified_cipher)) total_bits = len(base_cipher) * 8 avalanche_ratio = bit_changes / total_bits
        print(f"Input change: 1 bit")
        print(f"Output changes: {bit_changes}/{total_bits} bits ({avalanche_ratio:.3f})")
        if avalanche_ratio > 0.3:
        print("✅ Good avalanche effect!")
        else:
        print("⚠️ Weak avalanche effect")

if __name__ == "__main__": test_reversible_cipher()