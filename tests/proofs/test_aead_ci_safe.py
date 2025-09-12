#!/usr/bin/env python3
"""
CI-SAFE AEAD COMPLIANCE TEST
============================
Deterministic version with explicit seeding for CI reproducibility.
"""

import numpy as np
import time
import hashlib
import hmac
from typing import Dict, List, Tuple, Optional
import sys
import os

# Set deterministic seeds for CI reproducibility
np.random.seed(42)

# Import the true RFT kernel for cryptographic operations
try:
    from true_rft_kernel import TrueRFTKernel
    USE_REAL_RFT = True
except ImportError:
    print("Warning: Could not import TrueRFTKernel, using mock")
    USE_REAL_RFT = False

class RFTBasedAEAD:
    """CI-safe AEAD implementation using RFT with deterministic behavior."""
    
    def __init__(self, key_size: int = 32):
        self.key_size = key_size
        self.nonce_size = 16
        self.tag_size = 16
        
    def _derive_transform_key(self, key: bytes, nonce: bytes) -> int:
        """Derive RFT size from key and nonce deterministically."""
        # Deterministic key derivation
        hash_input = key + nonce
        hash_output = hashlib.sha256(hash_input).digest()
        
        # Use first 4 bytes to determine RFT size (power of 2, 8-64)
        size_bytes = hash_output[:4]
        size_int = int.from_bytes(size_bytes, 'big')
        
        # Map to valid RFT sizes deterministically
        valid_sizes = [8, 16, 32, 64]
        size_index = size_int % len(valid_sizes)
        return valid_sizes[size_index]
    
    def _apply_rft_transform(self, data: bytes, transform_size: int, forward: bool = True) -> bytes:
        """Apply RFT transform to data."""
        if not USE_REAL_RFT:
            # Mock transformation for CI compatibility
            if forward:
                return hashlib.sha256(data + b"forward").digest()[:len(data)]
            else:
                return hashlib.sha256(data + b"inverse").digest()[:len(data)]
        
        # Convert bytes to complex array
        if len(data) == 0:
            return data
            
        # Pad data to transform_size
        padded_size = ((len(data) + transform_size - 1) // transform_size) * transform_size
        padded_data = data + b'\x00' * (padded_size - len(data))
        
        # Convert to complex array
        complex_data = np.frombuffer(padded_data, dtype=np.uint8).astype(np.complex128)
        
        # Apply RFT transform
        kernel = TrueRFTKernel(transform_size)
        if forward:
            transformed = kernel.forward_transform(complex_data[:transform_size])
        else:
            transformed = kernel.inverse_transform(complex_data[:transform_size])
        
        # Convert back to bytes
        result_data = np.real(transformed).astype(np.uint8).tobytes()
        return result_data[:len(data)]
    
    def encrypt(self, plaintext: bytes, associated_data: bytes, key: bytes, nonce: bytes) -> Tuple[bytes, bytes]:
        """Encrypt plaintext with AEAD."""
        if len(key) != self.key_size:
            raise ValueError(f"Key must be {self.key_size} bytes")
        if len(nonce) != self.nonce_size:
            raise ValueError(f"Nonce must be {self.nonce_size} bytes")
        
        # Derive transform parameters
        transform_size = self._derive_transform_key(key, nonce)
        
        # Apply RFT encryption
        ciphertext = self._apply_rft_transform(plaintext, transform_size, forward=True)
        
        # Generate authentication tag
        tag_input = key + nonce + associated_data + ciphertext
        tag = hmac.new(key, tag_input, hashlib.sha256).digest()[:self.tag_size]
        
        return ciphertext, tag
    
    def decrypt(self, ciphertext: bytes, tag: bytes, associated_data: bytes, key: bytes, nonce: bytes) -> bytes:
        """Decrypt ciphertext with AEAD."""
        if len(key) != self.key_size:
            raise ValueError(f"Key must be {self.key_size} bytes")
        if len(nonce) != self.nonce_size:
            raise ValueError(f"Nonce must be {self.nonce_size} bytes")
        if len(tag) != self.tag_size:
            raise ValueError(f"Tag must be {self.tag_size} bytes")
        
        # Verify authentication tag
        tag_input = key + nonce + associated_data + ciphertext
        expected_tag = hmac.new(key, tag_input, hashlib.sha256).digest()[:self.tag_size]
        
        if not hmac.compare_digest(tag, expected_tag):
            raise ValueError("Authentication tag verification failed")
        
        # Derive transform parameters
        transform_size = self._derive_transform_key(key, nonce)
        
        # Apply RFT decryption
        plaintext = self._apply_rft_transform(ciphertext, transform_size, forward=False)
        
        return plaintext

def generate_deterministic_kats() -> List[Dict]:
    """Generate deterministic Known Answer Tests for CI."""
    kats = []
    
    # Fully deterministic test vectors
    test_cases = [
        {
            'plaintext': b'Hello, World!',
            'associated_data': b'header_info',
            'key': b'0123456789abcdef' * 2,  # 32 bytes
            'nonce': b'nonce_0123456789',     # 16 bytes
            'description': 'Basic test case'
        },
        {
            'plaintext': b'',
            'associated_data': b'',
            'key': b'\x00' * 32,
            'nonce': b'\x00' * 16,
            'description': 'Empty plaintext and AD'
        },
        {
            'plaintext': b'A' * 64,
            'associated_data': b'B' * 32,
            'key': b'\xff' * 32,
            'nonce': b'\xff' * 16,
            'description': 'Long plaintext'
        }
    ]
    
    aead = RFTBasedAEAD()
    
    for test_case in test_cases:
        try:
            ciphertext, tag = aead.encrypt(
                test_case['plaintext'],
                test_case['associated_data'],
                test_case['key'],
                test_case['nonce']
            )
            
            # Test round-trip
            try:
                decrypted = aead.decrypt(
                    ciphertext, tag,
                    test_case['associated_data'],
                    test_case['key'],
                    test_case['nonce']
                )
                round_trip_success = decrypted == test_case['plaintext']
            except Exception:
                round_trip_success = False
            
            kat = {
                'description': test_case['description'],
                'plaintext_hex': test_case['plaintext'].hex(),
                'ciphertext_hex': ciphertext.hex(),
                'tag_hex': tag.hex(),
                'round_trip_success': round_trip_success
            }
            
            kats.append(kat)
            print(f"KAT {test_case['description']}: {'PASS' if round_trip_success else 'FAIL'}")
            
        except Exception as e:
            print(f"KAT generation failed for {test_case['description']}: {e}")
    
    return kats

def test_tamper_detection() -> bool:
    """Test tamper detection with deterministic cases."""
    aead = RFTBasedAEAD()
    
    # Deterministic test case
    plaintext = b'Secret message'
    associated_data = b'public header'
    key = b'secret_key_32_bytes_long_here!XX'  # 32 bytes
    nonce = b'nonce_16_bytes!!'  # 16 bytes
    
    try:
        ciphertext, tag = aead.encrypt(plaintext, associated_data, key, nonce)
        
        # Test tamper detection
        tamper_tests = [
            ("modified ciphertext", ciphertext[:-1] + b'\x00', tag),
            ("modified tag", ciphertext, tag[:-1] + b'\x00'),
            ("modified associated data", ciphertext, tag)
        ]
        
        all_detected = True
        for test_name, test_ct, test_tag in tamper_tests:
            try:
                if test_name == "modified associated data":
                    aead.decrypt(test_ct, test_tag, b'wrong_header', key, nonce)
                else:
                    aead.decrypt(test_ct, test_tag, associated_data, key, nonce)
                print(f"FAIL: {test_name} not detected")
                all_detected = False
            except ValueError:
                print(f"PASS: {test_name} detected")
        
        return all_detected
        
    except Exception as e:
        print(f"Tamper detection test failed: {e}")
        return False

def main():
    """CI-safe AEAD compliance test."""
    print("CI-SAFE AEAD COMPLIANCE TEST")
    print("=" * 50)
    print("Deterministic test with explicit seeding")
    print("=" * 50)
    
    # Generate and test KATs
    print("Generating deterministic Known Answer Tests...")
    kats = generate_deterministic_kats()
    
    kat_successes = sum(1 for kat in kats if kat['round_trip_success'])
    kat_pass = kat_successes == len(kats)
    
    print(f"\nKAT Results: {kat_successes}/{len(kats)} passed")
    
    # Test tamper detection
    print("\nTesting tamper detection...")
    tamper_pass = test_tamper_detection()
    
    # Final result
    overall_pass = kat_pass and tamper_pass
    
    print(f"\n" + "=" * 50)
    print("CI-SAFE AEAD COMPLIANCE RESULTS")
    print("=" * 50)
    print(f"KAT Tests: {'PASS' if kat_pass else 'FAIL'}")
    print(f"Tamper Detection: {'PASS' if tamper_pass else 'FAIL'}")
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
    
    return overall_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
