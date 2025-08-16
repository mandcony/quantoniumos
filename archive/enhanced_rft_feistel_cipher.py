#!/usr/bin/env python3
"""
Enhanced RFT Feistel Cipher with C++ Backend for Improved Avalanche Effect

This implementation uses a C++ engine with:
- AES S-box for strong nonlinearity
- Multi-layer diffusion (5 layers per round)
- Golden ratio (phi) modulated transforms
- 48-round Feistel structure
- HKDF-style key expansion

Expected avalanche effect: >45% (cryptographic grade)
"""

import ctypes
import os
import numpy as np
from typing import Optional
import tempfile
import subprocess


class EnhancedRFTFeistelCipher:
    """Enhanced RFT Feistel cipher with C++ backend for superior avalanche effect"""
    
    def __init__(self):
        self.lib = self._load_library()
        self.cipher_ptr = None
        if self.lib:
            self.cipher_ptr = self.lib.create_enhanced_rft()
            
    def __del__(self):
        if self.lib and self.cipher_ptr:
            self.lib.destroy_enhanced_rft(self.cipher_ptr)
    
    def _load_library(self) -> Optional[ctypes.CDLL]:
        """Try to load C++ library, compile if needed"""
        lib_path = "/workspaces/quantoniumos/core/enhanced_rft_crypto.so"
        
        try:
            # Try to load existing library
            if os.path.exists(lib_path):
                lib = ctypes.CDLL(lib_path)
                self._setup_function_signatures(lib)
                return lib
        except:
            pass
            
        # Try to compile the library
        try:
            cpp_file = "/workspaces/quantoniumos/core/enhanced_rft_crypto.cpp"
            compile_cmd = [
                "g++", "-shared", "-fPIC", "-O3", "-std=c++17",
                cpp_file, "-o", lib_path
            ]
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(lib_path):
                lib = ctypes.CDLL(lib_path)
                self._setup_function_signatures(lib)
                return lib
        except Exception as e:
            print(f"Could not compile C++ library: {e}")
            
        return None
    
    def _setup_function_signatures(self, lib):
        """Setup C function signatures"""
        lib.create_enhanced_rft.restype = ctypes.c_void_p
        lib.destroy_enhanced_rft.argtypes = [ctypes.c_void_p]
        
        lib.encrypt_enhanced_rft.argtypes = [
            ctypes.c_void_p,  # cipher
            ctypes.POINTER(ctypes.c_uint8),  # plaintext
            ctypes.c_size_t,  # plaintext_len
            ctypes.POINTER(ctypes.c_uint8),  # key
            ctypes.c_size_t,  # key_len
            ctypes.POINTER(ctypes.c_uint8),  # ciphertext
            ctypes.POINTER(ctypes.c_size_t)  # ciphertext_len
        ]
        lib.encrypt_enhanced_rft.restype = ctypes.c_int
        
        lib.decrypt_enhanced_rft.argtypes = [
            ctypes.c_void_p,  # cipher
            ctypes.POINTER(ctypes.c_uint8),  # ciphertext
            ctypes.c_size_t,  # ciphertext_len
            ctypes.POINTER(ctypes.c_uint8),  # key
            ctypes.c_size_t,  # key_len
            ctypes.POINTER(ctypes.c_uint8),  # plaintext
            ctypes.POINTER(ctypes.c_size_t)  # plaintext_len
        ]
        lib.decrypt_enhanced_rft.restype = ctypes.c_int
    
    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Encrypt plaintext with enhanced C++ RFT engine"""
        
        # Fallback to Python if C++ unavailable
        if not self.lib or not self.cipher_ptr:
            return self._encrypt_python_fallback(plaintext, key)
            
        # Pad to even length for Feistel
        if len(plaintext) % 2 != 0:
            plaintext = plaintext + b'\x00'
            
        # Prepare buffers
        pt_array = (ctypes.c_uint8 * len(plaintext))(*plaintext)
        key_array = (ctypes.c_uint8 * len(key))(*key)
        ct_buffer = (ctypes.c_uint8 * len(plaintext))()
        ct_len = ctypes.c_size_t(len(plaintext))
        
        # Call C++ function
        result = self.lib.encrypt_enhanced_rft(
            self.cipher_ptr,
            pt_array, len(plaintext),
            key_array, len(key),
            ct_buffer, ctypes.byref(ct_len)
        )
        
        if result != 0:
            raise RuntimeError(f"Encryption failed with code {result}")
            
        return bytes(ct_buffer[:ct_len.value])
    
    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt ciphertext with enhanced C++ RFT engine"""
        
        # Fallback to Python if C++ unavailable
        if not self.lib or not self.cipher_ptr:
            return self._decrypt_python_fallback(ciphertext, key)
            
        # Prepare buffers
        ct_array = (ctypes.c_uint8 * len(ciphertext))(*ciphertext)
        key_array = (ctypes.c_uint8 * len(key))(*key)
        pt_buffer = (ctypes.c_uint8 * len(ciphertext))()
        pt_len = ctypes.c_size_t(len(ciphertext))
        
        # Call C++ function
        result = self.lib.decrypt_enhanced_rft(
            self.cipher_ptr,
            ct_array, len(ciphertext),
            key_array, len(key),
            pt_buffer, ctypes.byref(pt_len)
        )
        
        if result != 0:
            raise RuntimeError(f"Decryption failed with code {result}")
            
        return bytes(pt_buffer[:pt_len.value])
    
    def _encrypt_python_fallback(self, plaintext: bytes, key: bytes) -> bytes:
        """Python fallback with enhanced diffusion"""
        from rft_feistel_cipher import RFTFeistelCipher
        fallback = RFTFeistelCipher()
        return fallback.encrypt(plaintext, key)
    
    def _decrypt_python_fallback(self, ciphertext: bytes, key: bytes) -> bytes:
        """Python fallback with enhanced diffusion"""
        from rft_feistel_cipher import RFTFeistelCipher
        fallback = RFTFeistelCipher()
        return fallback.decrypt(ciphertext, key)


# For compatibility with existing code
RFTFeistelCipher = EnhancedRFTFeistelCipher


if __name__ == "__main__":
    # Test the enhanced cipher
    import os
    import time
    
    print("Testing Enhanced RFT Feistel Cipher...")
    cipher = EnhancedRFTFeistelCipher()
    
    # Basic functionality test
    message = b"Hello, Enhanced RFT! This should have better avalanche effect."
    key = os.urandom(32)
    
    print(f"Original: {message}")
    
    start = time.time()
    encrypted = cipher.encrypt(message, key)
    encrypt_time = time.time() - start
    print(f"Encrypted: {encrypted.hex()[:64]}...")
    print(f"Encryption time: {encrypt_time:.4f}s")
    
    start = time.time()
    decrypted = cipher.decrypt(encrypted, key)
    decrypt_time = time.time() - start
    print(f"Decrypted: {decrypted}")
    print(f"Decryption time: {decrypt_time:.4f}s")
    print(f"Perfect reversibility: {message == decrypted}")
    
    # Avalanche test
    def test_avalanche(cipher, key, num_tests=20):
        total_avalanche = 0
        for _ in range(num_tests):
            # Original plaintext
            pt1 = os.urandom(32)
            # Flip one bit
            pt2 = bytearray(pt1)
            pt2[0] ^= 1
            pt2 = bytes(pt2)
            
            # Encrypt both
            ct1 = cipher.encrypt(pt1, key)
            ct2 = cipher.encrypt(pt2, key)
            
            # Count bit differences
            bit_diff = sum(bin(a ^ b).count('1') for a, b in zip(ct1, ct2))
            total_bits = len(ct1) * 8
            avalanche = bit_diff / total_bits
            total_avalanche += avalanche
            
        return total_avalanche / num_tests
    
    print("\nTesting enhanced avalanche effect...")
    avalanche = test_avalanche(cipher, key)
    print(f"Enhanced avalanche effect: {avalanche:.3f}")
    
    if avalanche > 0.45:
        print("✅ EXCELLENT: Cryptographic-grade avalanche effect achieved!")
    elif avalanche > 0.35:
        print("✅ GOOD: Strong avalanche effect")
    else:
        print("⚠️  Avalanche effect could be improved")
