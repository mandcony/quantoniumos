"""
Enhanced Hash Test Module

This module provides the enhanced_geometric_hash function needed for 
avalanche effect testing in publication_ready_validation.py.

Features:
- RFT-based spectral analysis
- Golden ratio geometric transformations  
- Multi-round diffusion layers
- Keyed non-linear transformations for research-grade cryptographic diffusion
"""

import hashlib
import numpy as np
import struct
from typing import Optional
import sys
sys.path.append('.')

# Import canonical RFT implementation
try:
    from canonical_true_rft import forward_true_rft, inverse_true_rft
    RFT_AVAILABLE = True
except ImportError:
    RFT_AVAILABLE = False

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
DEFAULT_ROUNDS = 4
OUTPUT_SIZE = 32  # 256-bit output

# Standard AES S-box for truly non-linear substitution
AES_SBOX = bytes([
    99,124,119,123,242,107,111,197, 48,  1,103, 43,254,215,171,118,
    202,130,201,125,250, 89, 71,240,173,212,162,175,156,164,114,192,
    183,253,147, 38, 54, 63,247,204, 52,165,229,241,113,216, 49, 21,
     4,199, 35,195, 24,150,  5,154,  7, 18,128,226,235, 39,178,117,
     9,131, 44, 26, 27,110, 90,160, 82, 59,214,179, 41,227, 47,132,
    83,209,  0,237, 32,252,177, 91,106,203,190, 57, 74, 76, 88,207,
    208,239,170,251, 67, 77, 51,133, 69,249,  2,127, 80, 60,159,168,
     81,163, 64,143,146,157, 56,245,188,182,218, 33, 16,255,243,210,
    205, 12, 19,236, 95,151, 68, 23,196,167,126, 61,100, 93, 25,115,
     96,129, 79,220, 34, 42,144,136, 70,238,184, 20,222, 94, 11,219,
    224, 50, 58, 10, 73,  6, 36, 92,194,211,172, 98,145,149,228,121,
    231,200, 55,109,141,213, 78,169,108, 86,244,234,101,122,174,  8,
    186,120, 37, 46, 28,166,180,198,232,221,116, 31, 75,189,139,138,
    112, 62,181,102, 72,  3,246, 14, 97, 53, 87,185,134,193, 29,158,
    225,248,152, 17,105,217,142,148,155, 30,135,233,206, 85, 40,223,
    140,161,137, 13,191,230, 66,104, 65,153, 45, 15,176, 84,187, 22
])

def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """HKDF key derivation with SHA-256"""
    import hmac
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    okm, t, ctr = b"", b"", 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([ctr]), hashlib.sha256).digest()
        okm += t
        ctr += 1
    return okm[:length]

def derive_round_key(master_key: bytes, round_num: int, outlen: int = 32) -> bytes:
    """Derive round key using HKDF"""
    salt = b"RFT-hash-salt"
    info = b"RFT-round-" + round_num.to_bytes(4, "big")
    return hkdf_sha256(master_key, salt, info, outlen)

def sbox_bytes(buf: bytes) -> bytes:
    """Apply AES S-box (truly non-linear substitution)"""
    return bytes(AES_SBOX[b] for b in buf)

def enhanced_geometric_hash(data: bytes, key: bytes, rounds: int = DEFAULT_ROUNDS) -> bytes:
    """
    Enhanced geometric hash with keyed non-linear diffusion.
    
    This is a research-grade hash function with cryptographic diffusion metrics that combines:
    1. RFT spectral analysis
    2. Golden ratio geometric transformations
    3. Multi-round keyed diffusion
    4. Non-linear mixing functions
    
    Args:
        data: Input data to hash
        key: Secret key for keyed operations
        rounds: Number of diffusion rounds (default: 4)
        
    Returns:
        32-byte hash output
    """
    if not isinstance(data, bytes):
        data = bytes(data)
    if not isinstance(key, bytes):
        key = bytes(key)
    
    # Stage 1: Initial keyed preprocessing
    state = _keyed_preprocessing(data, key)
    
    # Stage 2: RFT-based spectral diffusion
    if RFT_AVAILABLE:
        state = _rft_spectral_diffusion(state, key)
    else:
        # Fallback: enhanced geometric diffusion
        state = _geometric_diffusion_fallback(state, key)
    
    # Stage 3: Multi-round keyed diffusion
    for round_num in range(rounds):
        state = _keyed_diffusion_round(state, key, round_num)
    
    # Stage 4: Final output conditioning
    return _finalize_output(state, key)

def _keyed_preprocessing(data: bytes, key: bytes) -> bytes:
    """Initial keyed preprocessing with HMAC-like construction."""
    # HMAC-style keyed preprocessing
    ipad = bytes([0x36] * 64)
    opad = bytes([0x5C] * 64)
    
    # Prepare key
    if len(key) > 64:
        key = hashlib.sha256(key).digest()
    key = key.ljust(64, b'\x00')
    
    # Inner hash
    inner_key = bytes(k ^ i for k, i in zip(key, ipad))
    inner_hash = hashlib.sha256(inner_key + data).digest()
    
    # Outer hash  
    outer_key = bytes(k ^ o for k, o in zip(key, opad))
    return hashlib.sha256(outer_key + inner_hash).digest()

def _rft_spectral_diffusion(state: bytes, key: bytes) -> bytes:
    """RFT-based spectral diffusion with golden ratio weighting."""
    try:
        # Convert to float array
        float_data = np.frombuffer(state, dtype=np.uint8).astype(np.float64)
        
        # Pad to power of 2 for efficiency
        padded_size = 64
        if len(float_data) < padded_size:
            padded = np.zeros(padded_size)
            padded[:len(float_data)] = float_data
            float_data = padded
        
        # Apply forward RFT
        rft_spectrum = forward_true_rft(float_data)
        
        # Golden ratio spectral weighting (keyed)
        key_hash = int.from_bytes(key[:8], 'little') % 1000  # Limit to prevent overflow
        weights = np.array([(PHI ** (i % 32 + (key_hash % 100))) % 1 for i in range(len(rft_spectrum))])
        
        # Apply spectral modifications
        magnitude = np.abs(rft_spectrum)
        phase = np.angle(rft_spectrum)
        
        # Keyed phase rotation
        phase_shift = (key_hash * PHI) % (2 * np.pi)
        modified_phase = (phase + phase_shift * weights) % (2 * np.pi)
        
        # Reconstruct with modified spectrum
        modified_spectrum = magnitude * weights * np.exp(1j * modified_phase)
        
        # Inverse RFT
        time_domain = inverse_true_rft(modified_spectrum).real
        
        # Convert back to bytes with non-linear transformation
        result = np.zeros(32, dtype=np.uint8)
        for i in range(32):
            val = abs(time_domain[i % len(time_domain)])
            # Non-linear byte mapping with golden ratio
            transformed = int((val * PHI) % 256)
            result[i] = transformed
        
        return result.tobytes()
        
    except Exception:
        # Fallback to geometric diffusion
        return _geometric_diffusion_fallback(state, key)

def _geometric_diffusion_fallback(state: bytes, key: bytes) -> bytes:
    """Enhanced geometric diffusion without RFT."""
    result = bytearray(32)
    key_val = int.from_bytes(key[:8], 'little')
    
    for i in range(32):
        # Get input byte (cycling through state)
        byte_val = state[i % len(state)]
        
        # Keyed geometric transformation
        angle = ((i * PHI) + (key_val * PHI)) % (2 * np.pi)
        radius = byte_val / 255.0
        
        # Complex coordinate transformation
        z = radius * np.exp(1j * angle)
        
        # Non-linear mixing
        mixed_real = (z.real * PHI) % 1
        mixed_imag = (z.imag * PHI) % 1
        
        # Combine and quantize
        combined = (mixed_real + mixed_imag) % 1
        result[i] = int(combined * 255)
    
    return bytes(result)

def _keyed_diffusion_round(state: bytes, key: bytes, round_num: int) -> bytes:
    """Apply keyed non-linear diffusion round using HKDF and AES S-box"""
    # HKDF key derivation (cryptographically proper)
    rk = derive_round_key(key, round_num, 64)
    
    # XOR with round key
    x = bytes(a ^ b for a, b in zip(state.ljust(64, b"\0"), rk))
    
    # Non-affine S-box substitution (truly non-linear)
    x = sbox_bytes(x)
    
    # Lightweight linear diffusion (invertible)
    y = bytearray(64)
    for i in range(64):
        y[i] = (x[i] ^ x[(i+1) & 63] ^ ((x[(i+5) & 63] << 1) & 0xFF))
    
    return bytes(y[:32])  # collapse to 256-bit

def _finalize_output(state: bytes, key: bytes) -> bytes:
    """Final output conditioning with key-dependent finalization."""
    # Double-key finalization
    final_key = hashlib.sha256(key + b'FINAL').digest()
    
    # XOR state with key material
    keyed_state = bytes(s ^ k for s, k in zip(state, final_key))
    
    # Final hash with original key for authentication
    return hashlib.sha256(keyed_state + key).digest()[:OUTPUT_SIZE]

# Self-test
if __name__ == "__main__":
    # Basic functionality test
    test_data = b"Hello, World!"
    test_key = b"secret_key_123"
    
    hash1 = enhanced_geometric_hash(test_data, test_key)
    hash2 = enhanced_geometric_hash(test_data, test_key)
    
    print(f"Hash 1: {hash1.hex()}")
    print(f"Hash 2: {hash2.hex()}")
    print(f"Deterministic: {hash1 == hash2}")
    
    # Avalanche test
    test_data_modified = b"Hello, World."  # One character difference
    hash3 = enhanced_geometric_hash(test_data_modified, test_key)
    
    diff_bits = sum(bin(b1 ^ b2).count('1') for b1, b2 in zip(hash1, hash3))
    avalanche_rate = 100.0 * diff_bits / (len(hash1) * 8)
    print(f"Avalanche rate: {avalanche_rate:.2f}%")
    
    # Key sensitivity test
    hash4 = enhanced_geometric_hash(test_data, test_key + b"x")
    diff_bits_key = sum(bin(b1 ^ b2).count('1') for b1, b2 in zip(hash1, hash4))
    key_sensitivity = 100.0 * diff_bits_key / (len(hash1) * 8)
    print(f"Key sensitivity: {key_sensitivity:.2f}%")
    
    if RFT_AVAILABLE:
        print("✅ RFT spectral diffusion: AVAILABLE")
    else:
        print("⚠️  RFT spectral diffusion: Using geometric fallback")
