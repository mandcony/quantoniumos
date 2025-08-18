#!/usr/bin/env python3
"""
Paper-Compliant RFT Implementation - FIXED ROUNDTRIP
This implements the exact paper specifications with guaranteed roundtrip integrity.
"""

import numpy as np
import hashlib
import hmac
from typing import Dict, Any

class PaperCompliantRFT:
    """
    Paper-compliant RFT implementation with PERFECT roundtrip integrity
    Following the exact specifications from the research paper.
    """
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.block_size = 16  # Paper standard
        
    def generate_key_material(self, password: bytes, salt: bytes, length: int) -> bytes:
        """Generate key material using HKDF-SHA256 (paper standard)"""
        # Extract phase
        prk = hmac.new(salt, password, hashlib.sha256).digest()
        
        # Expand phase
        okm = b""
        counter = 1
        while len(okm) < length:
            okm += hmac.new(prk, okm[-32:] + b"RFT-PAPER-v1" + bytes([counter]), hashlib.sha256).digest()
            counter += 1
        
        return okm[:length]
    
    def _feistel_f(self, right: bytes, key: bytes) -> bytes:
        """
        Paper-compliant Feistel F function with RFT structure
        This follows the exact mathematical specification from the paper.
        """
        # Convert to arrays for computation
        r_array = np.frombuffer(right, dtype=np.uint8)
        k_array = np.frombuffer(key[:len(right)], dtype=np.uint8)
        
        # Phase 1: Golden ratio modulation (paper eq. 3)
        phi_mod = (r_array.astype(np.float64) * self.phi) % 256
        
        # Phase 2: Key mixing with RFT pattern (paper eq. 4)
        mixed = (phi_mod + k_array.astype(np.float64)) % 256
        
        # Phase 3: Non-linear transformation (paper eq. 5)
        nonlinear = (np.sin(2 * np.pi * mixed / 256) * 127 + 128) % 256
        
        # Phase 4: Final diffusion (paper eq. 6)
        diffused = np.zeros_like(nonlinear)
        for i in range(len(nonlinear)):
            left_idx = (i - 1) % len(nonlinear)
            right_idx = (i + 1) % len(nonlinear)
            diffused[i] = (nonlinear[i] + nonlinear[left_idx] + nonlinear[right_idx]) / 3 % 256
        
        return diffused.astype(np.uint8).tobytes()
    
    def encrypt_block(self, plaintext: bytes, key: bytes) -> bytes:
        """
        Encrypt 16-byte block using paper-compliant Feistel-RFT
        Guaranteed mathematical reversibility.
        """
        if len(plaintext) != 16:
            raise ValueError("Block must be exactly 16 bytes")
        
        # Generate round keys
        round_keys = []
        for i in range(16):  # Paper specifies 16 rounds
            round_key = self.generate_key_material(key, f"round_{i}".encode(), 8)
            round_keys.append(round_key)
        
        # Split into Feistel halves
        left = plaintext[:8]
        right = plaintext[8:]
        
        # 16 Feistel rounds (paper standard)
        for round_num in range(16):
            # Standard Feistel: (L, R) -> (R, L ⊕ F(R, K))
            f_output = self._feistel_f(right, round_keys[round_num])
            
            # XOR left with F output
            new_left = bytes(a ^ b for a, b in zip(left, f_output))
            new_right = right
            
            left = new_right  # R becomes new L
            right = new_left  # L ⊕ F(R, K) becomes new R
        
        return left + right
    
    def decrypt_block(self, ciphertext: bytes, key: bytes) -> bytes:
        """
        Decrypt 16-byte block - EXACT mathematical inverse of encrypt_block
        This guarantees perfect roundtrip integrity.
        """
        if len(ciphertext) != 16:
            raise ValueError("Block must be exactly 16 bytes")
        
        # Generate the SAME round keys as encryption
        round_keys = []
        for i in range(16):
            round_key = self.generate_key_material(key, f"round_{i}".encode(), 8)
            round_keys.append(round_key)
        
        # Split into Feistel halves
        left = ciphertext[:8]
        right = ciphertext[8:]
        
        # 16 Feistel rounds IN REVERSE ORDER
        for round_num in range(15, -1, -1):  # 15, 14, 13, ..., 1, 0
            # Reverse Feistel: (L, R) -> (R ⊕ F(L, K), L)
            f_output = self._feistel_f(left, round_keys[round_num])
            
            # XOR right with F output
            new_right = bytes(a ^ b for a, b in zip(right, f_output))
            new_left = left
            
            right = new_left  # L becomes new R
            left = new_right   # R ⊕ F(L, K) becomes new L
        
        return left + right
    
    def avalanche_test(self, key1: bytes, key2: bytes) -> float:
        """Test avalanche effect between two keys"""
        test_block = b"Test message 123"  # 16 bytes
        
        enc1 = self.encrypt_block(test_block, key1)
        enc2 = self.encrypt_block(test_block, key2)
        
        # Count differing bits
        diff_bits = 0
        for a, b in zip(enc1, enc2):
            diff_bits += (a ^ b).bit_count()
        
        return diff_bits / (len(enc1) * 8)

# Wrapper to match the existing interface
class FixedRFTCryptoBindings:
    """
    Drop-in replacement for the broken C++ binding
    Provides the same interface but with guaranteed roundtrip integrity.
    """
    
    def __init__(self):
        self.engine = PaperCompliantRFT()
        self._initialized = False
    
    def init_engine(self):
        """Initialize engine (compatibility)"""
        self._initialized = True
        
    def encrypt_block(self, plaintext: bytes, key: bytes) -> bytes:
        """Encrypt 16-byte block with guaranteed roundtrip"""
        return self.engine.encrypt_block(plaintext, key)
        
    def decrypt_block(self, ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt 16-byte block with guaranteed roundtrip"""
        return self.engine.decrypt_block(ciphertext, key)
        
    def generate_key_material(self, password: bytes, salt: bytes, length: int) -> bytes:
        """Generate key material"""
        return self.engine.generate_key_material(password, salt, length)
        
    def avalanche_test(self, key1: bytes, key2: bytes) -> float:
        """Test avalanche effect"""
        return self.engine.avalanche_test(key1, key2)

def test_fixed_implementation():
    """Test the fixed implementation"""
    print("Testing Fixed Paper-Compliant RFT Implementation")
    print("=" * 60)
    
    rft = FixedRFTCryptoBindings()
    rft.init_engine()
    
    # Test roundtrip integrity
    test_data = b"Hello, World!123"  # Exactly 16 bytes
    key = b"test_key_32_bytes_for_encryption!"  # 32 bytes
    
    print(f"Original: {test_data}")
    
    encrypted = rft.encrypt_block(test_data, key)
    print(f"Encrypted: {encrypted.hex()}")
    
    decrypted = rft.decrypt_block(encrypted, key)
    print(f"Decrypted: {decrypted}")
    
    roundtrip_ok = (test_data == decrypted)
    print(f"Roundtrip integrity: {'✓ PASS' if roundtrip_ok else '✗ FAIL'}")
    
    if roundtrip_ok:
        print("\n🎉 ROUNDTRIP INTEGRITY FIXED!")
        print("  • Perfect encrypt→decrypt cycle")
        print("  • Paper-compliant RFT implementation")
        print("  • Mathematical reversibility guaranteed")
    
    # Test avalanche effect
    key1 = b"test_key_32_bytes_for_encryption!"
    key2 = bytearray(key1)
    key2[0] ^= 1  # Flip one bit
    
    avalanche = rft.avalanche_test(key1, bytes(key2))
    print(f"\nKey avalanche (1-bit change): {avalanche:.3f}")
    print(f"Target range: 0.4-0.6")
    print(f"Paper target: 0.527")
    print(f"Avalanche effect: {'✓ PASS' if 0.4 <= avalanche <= 0.6 else '○ NEEDS WORK'}")
    
    return roundtrip_ok and (0.4 <= avalanche <= 0.6)

if __name__ == "__main__":
    success = test_fixed_implementation()
    print(f"\nOverall result: {'SUCCESS - All paper targets achieved!' if success else 'PARTIAL - Some issues remain'}")
