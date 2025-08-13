#!/usr/bin/env python3
"""
Deterministic Geometric Waveform Hashing

Cryptographically secure, deterministic implementation following the pattern:
1. Map to coordinates deterministically using golden angle
2. Quantize to fixed-point integers (Q32.32) to avoid FP drift
3. Serialize with explicit endianness  
4. Final digest = SHA-256(serialized_fixed_point_bytes)

This ensures bit-for-bit identical outputs across platforms and runs.
"""

import hashlib
import struct
import secrets
from typing import List, Optional

# Q32.32 fixed-point scale
SCALE = 2**32
GOLDEN_RATIO = 1.6180339887498948482  # Exact golden ratio


def geometric_waveform_hash_deterministic(
    xs: List[float],
    *,
    key: Optional[bytes] = None,
    nonce: Optional[bytes] = None,
    hash_length: int = 64
) -> str:
    """
    Generate deterministic geometric waveform hash.
    
    Args:
        xs: Input waveform data
        key: Optional key for keyed mode (must be bytes)
        nonce: Optional nonce for unique outputs (must be bytes)
        hash_length: Length of hex output string
        
    Returns:
        Hex string of specified length
        
    Examples:
        # Deterministic mode - always same output for same input
        hash1 = geometric_waveform_hash_deterministic([1.0, 2.0, 3.0])
        hash2 = geometric_waveform_hash_deterministic([1.0, 2.0, 3.0])
        assert hash1 == hash2
        
        # Keyed mode - different outputs with different keys
        hash_k1 = geometric_waveform_hash_deterministic([1.0, 2.0], key=b"key1")
        hash_k2 = geometric_waveform_hash_deterministic([1.0, 2.0], key=b"key2") 
        assert hash_k1 != hash_k2
    """
    if not xs:
        return "0" * hash_length
    
    N = len(xs)
    
    # 1) Deterministic angles using golden angle - NO RANDOMNESS
    theta0 = 0.0  # Fixed starting angle
    thetas = [(theta0 + k / GOLDEN_RATIO) % 1.0 for k in range(N)]
    
    # 2) Convert to fixed-point coordinates to avoid FP drift
    coords = []
    for x, t in zip(xs, thetas):
        # Quantize to Q32.32 fixed-point with well-defined rounding
        cx = int(round(x * SCALE))
        ct = int(round(t * SCALE))
        coords.append((cx, ct))
    
    # 3) Serialize with explicit endianness
    buf = bytearray()
    
    # Domain separation tag
    buf.extend(b"RFT-GEO-HASH/v1\0")
    
    # Optional key and nonce for keyed mode
    if key is not None:
        buf.extend(b"K:")
        buf.extend(key)
    if nonce is not None:
        buf.extend(b"N:")
        buf.extend(nonce)
    
    # Serialize coordinates as big-endian 64-bit signed integers
    for cx, ct in coords:
        buf.extend(struct.pack(">q", cx))  # big-endian 64-bit signed
        buf.extend(struct.pack(">q", ct))
    
    # 4) Cryptographically secure hash
    digest = hashlib.sha256(buf).digest()
    
    # Convert to hex string of requested length
    hex_str = digest.hex()
    
    # Expand if needed using deterministic method
    while len(hex_str) < hash_length:
        # Deterministically expand using SHA-256 chain
        digest = hashlib.sha256(digest + b"expand").digest()
        hex_str += digest.hex()
    
    return hex_str[:hash_length]


def crypto_secure_random_bytes(n: int) -> bytes:
    """Generate cryptographically secure random bytes."""
    return secrets.token_bytes(n)


def crypto_secure_randrange(max_val: int) -> int:
    """Generate cryptographically secure random integer."""
    return secrets.SystemRandom().randrange(max_val)


def validate_deterministic_hash(test_data: List[float], iterations: int = 100) -> bool:
    """
    Test that hash function is perfectly deterministic.
    
    Returns:
        True if all iterations produce identical hash, False otherwise
    """
    first_hash = geometric_waveform_hash_deterministic(test_data)
    
    for _ in range(iterations - 1):
        current_hash = geometric_waveform_hash_deterministic(test_data)
        if current_hash != first_hash:
            return False
    
    return True


def test_domain_separation():
    """Test that different modes produce different hashes."""
    data = [1.0, 2.0, 3.0, 4.0]
    key = b"test_key"
    nonce = b"test_nonce"
    
    # Test all combinations are different
    hash_plain = geometric_waveform_hash_deterministic(data)
    hash_key = geometric_waveform_hash_deterministic(data, key=key)
    hash_nonce = geometric_waveform_hash_deterministic(data, nonce=nonce)
    hash_both = geometric_waveform_hash_deterministic(data, key=key, nonce=nonce)
    
    hashes = [hash_plain, hash_key, hash_nonce, hash_both]
    
    # All should be different
    return len(set(hashes)) == len(hashes)


def test_keyed_determinism():
    """Test that keyed mode is still deterministic."""
    data = [1.0, 2.0, 3.0]
    key = b"fixed_key"
    nonce = b"fixed_nonce"
    
    hash1 = geometric_waveform_hash_deterministic(data, key=key, nonce=nonce)
    hash2 = geometric_waveform_hash_deterministic(data, key=key, nonce=nonce)
    
    return hash1 == hash2


if __name__ == "__main__":
    # Run validation tests
    print("🔍 Testing Deterministic Geometric Hash")
    print("=" * 45)
    
    test_data = [1.0, 0.5, -0.3, 0.8, 0.2]
    
    # Test 1: Deterministic behavior
    print("Test 1: Deterministic behavior...")
    is_deterministic = validate_deterministic_hash(test_data, 100)
    print(f"✅ PASSED" if is_deterministic else "❌ FAILED")
    
    # Test 2: Domain separation
    print("Test 2: Domain separation...")
    domain_sep_ok = test_domain_separation()
    print(f"✅ PASSED" if domain_sep_ok else "❌ FAILED")
    
    # Test 3: Keyed determinism
    print("Test 3: Keyed determinism...")
    keyed_det_ok = test_keyed_determinism()
    print(f"✅ PASSED" if keyed_det_ok else "❌ FAILED")
    
    # Show sample outputs
    print("\nSample outputs:")
    print(f"Plain:    {geometric_waveform_hash_deterministic(test_data)[:32]}")
    print(f"Keyed:    {geometric_waveform_hash_deterministic(test_data, key=b'key')[:32]}")
    print(f"Nonce:    {geometric_waveform_hash_deterministic(test_data, nonce=b'123')[:32]}")
    
    all_passed = is_deterministic and domain_sep_ok and keyed_det_ok
    print(f"\n🎉 All tests {'PASSED' if all_passed else 'FAILED'}")
