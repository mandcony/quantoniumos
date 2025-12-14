#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
RFT-SIS Cryptographic Hash Test Suite
=====================================

Tests for post-quantum secure hash using RFT + SIS lattice.

USPTO Patent 19/169,399
"""

import pytest
import numpy as np
import hashlib
from algorithms.rft import RFTSISHash


class TestRFTSISHashBasic:
    """Basic functionality tests."""
    
    @pytest.fixture
    def hasher(self):
        return RFTSISHash()
    
    def test_hash_returns_bytes(self, hasher):
        """Hash output is bytes."""
        h = hasher.hash(b"test")
        assert isinstance(h, bytes)
    
    def test_hash_length_256_bits(self, hasher):
        """Hash is 256 bits (32 bytes)."""
        h = hasher.hash(b"test")
        assert len(h) == 32
    
    def test_hash_hex_format(self, hasher):
        """hash_hex returns proper hex string."""
        h = hasher.hash_hex(b"test")
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in '0123456789abcdef' for c in h)
    
    def test_deterministic(self, hasher):
        """Same input always produces same output."""
        for _ in range(10):
            data = np.random.bytes(64)
            h1 = hasher.hash(data)
            h2 = hasher.hash(data)
            assert h1 == h2
    
    def test_different_outputs(self, hasher):
        """Different inputs produce different outputs."""
        inputs = [b"a", b"b", b"aa", b"ab", b"ba", b"bb"]
        hashes = [hasher.hash(x) for x in inputs]
        assert len(set(hashes)) == len(hashes)


class TestAvalancheEffect:
    """Avalanche effect: 1-bit change → ~50% output bits flip."""
    
    @pytest.fixture
    def hasher(self):
        return RFTSISHash()
    
    def test_single_bit_flip(self, hasher):
        """Flipping 1 input bit changes ~50% output bits."""
        flips = []
        
        for _ in range(200):
            data = np.random.bytes(32)
            h1 = hasher.hash(data)
            
            # Flip one random bit
            modified = bytearray(data)
            byte_idx = np.random.randint(32)
            bit_idx = np.random.randint(8)
            modified[byte_idx] ^= (1 << bit_idx)
            h2 = hasher.hash(bytes(modified))
            
            diff = sum(bin(a ^ b).count('1') for a, b in zip(h1, h2))
            flips.append(diff / 256 * 100)
        
        avg = np.mean(flips)
        std = np.std(flips)
        
        assert 45 <= avg <= 55, f"Avalanche {avg:.1f}% not near 50%"
        assert std < 10, f"Avalanche std {std:.1f}% too high"
    
    def test_avalanche_all_positions(self, hasher):
        """Avalanche works for all input byte positions."""
        data = b"A" * 32
        h_orig = hasher.hash(data)
        
        for pos in range(32):
            modified = bytearray(data)
            modified[pos] = ord('B')
            h_mod = hasher.hash(bytes(modified))
            
            diff = sum(bin(a ^ b).count('1') for a, b in zip(h_orig, h_mod))
            pct = diff / 256 * 100
            
            assert 30 <= pct <= 70, f"Position {pos}: avalanche {pct:.1f}%"


class TestCollisionResistance:
    """Hash should be collision-resistant."""
    
    @pytest.fixture
    def hasher(self):
        return RFTSISHash()
    
    def test_no_collisions_sequential(self, hasher):
        """No collisions in sequential integers."""
        hashes = set()
        for i in range(5000):
            h = hasher.hash(i.to_bytes(8, 'big'))
            assert h not in hashes, f"Collision at {i}"
            hashes.add(h)
    
    def test_no_collisions_random(self, hasher):
        """No collisions in random data."""
        hashes = set()
        for _ in range(5000):
            data = np.random.bytes(32)
            h = hasher.hash(data)
            hashes.add(h)
        
        assert len(hashes) == 5000
    
    def test_similar_inputs_different_hashes(self, hasher):
        """Very similar inputs should have very different hashes."""
        base = b"The quick brown fox jumps over the lazy dog"
        h_base = hasher.hash(base)
        
        # Change each character
        for i in range(len(base)):
            modified = bytearray(base)
            modified[i] = (modified[i] + 1) % 256
            h_mod = hasher.hash(bytes(modified))
            
            assert h_mod != h_base


class TestOutputDistribution:
    """Hash output should be uniformly distributed."""
    
    @pytest.fixture
    def hasher(self):
        return RFTSISHash()
    
    def test_byte_uniformity(self, hasher):
        """Each output byte should be roughly uniform."""
        byte_counts = np.zeros((32, 256), dtype=int)
        
        for i in range(2560):
            h = hasher.hash(i.to_bytes(8, 'big'))
            for j, b in enumerate(h):
                byte_counts[j, b] += 1
        
        # Chi-square test for each byte position
        expected = 2560 / 256  # = 10
        for j in range(32):
            chi_sq = np.sum((byte_counts[j] - expected)**2 / expected)
            # 255 df, p=0.01 critical value is ~310
            assert chi_sq < 400, f"Byte {j} not uniform: χ²={chi_sq:.1f}"
    
    def test_bit_balance(self, hasher):
        """Each output bit should be ~50% 0s and ~50% 1s."""
        bit_counts = np.zeros(256, dtype=int)  # 32 bytes * 8 bits
        
        for i in range(1000):
            h = hasher.hash(i.to_bytes(8, 'big'))
            for j, b in enumerate(h):
                for k in range(8):
                    if b & (1 << k):
                        bit_counts[j * 8 + k] += 1
        
        # Each bit should be ~500/1000
        for i, count in enumerate(bit_counts):
            pct = count / 1000 * 100
            assert 40 <= pct <= 60, f"Bit {i}: {pct:.1f}% ones (expected ~50%)"


class TestInputVariety:
    """Hash should work with various input types and sizes."""
    
    @pytest.fixture
    def hasher(self):
        return RFTSISHash()
    
    def test_empty_input(self, hasher):
        """Empty input should produce valid hash."""
        h = hasher.hash(b"")
        assert len(h) == 32
    
    def test_single_byte(self, hasher):
        """Single byte input should work."""
        h = hasher.hash(b"X")
        assert len(h) == 32
    
    def test_large_input(self, hasher):
        """Large inputs should work."""
        data = np.random.bytes(10000)
        h = hasher.hash(data)
        assert len(h) == 32
    
    def test_unicode_string(self, hasher):
        """Unicode strings (as bytes) should work."""
        h = hasher.hash("Hello 世界 🌍".encode('utf-8'))
        assert len(h) == 32


class TestSISParameters:
    """Test SIS lattice parameter configuration."""
    
    def test_default_parameters(self):
        """Default parameters should be secure."""
        hasher = RFTSISHash()
        assert hasher.sis_n == 512
        assert hasher.sis_m == 1024
        assert hasher.sis_q == 3329  # Kyber prime
        assert hasher.sis_beta == 100
    
    def test_custom_parameters(self):
        """Custom parameters should work."""
        hasher = RFTSISHash(sis_n=256, sis_m=512, sis_q=7681, sis_beta=50)
        h = hasher.hash(b"test")
        assert len(h) == 32


class TestComparisonToSHA:
    """Compare RFT-SIS to standard SHA for sanity checking."""
    
    @pytest.fixture
    def hasher(self):
        return RFTSISHash()
    
    def test_different_from_sha256(self, hasher):
        """RFT-SIS output should differ from SHA-256."""
        data = b"test data"
        rft_hash = hasher.hash(data)
        sha_hash = hashlib.sha256(data).digest()
        
        assert rft_hash != sha_hash
    
    def test_comparable_avalanche(self, hasher):
        """Avalanche should be comparable to SHA-256."""
        def measure_avalanche(hash_func, n=100):
            flips = []
            for _ in range(n):
                data = np.random.bytes(32)
                h1 = hash_func(data)
                modified = bytearray(data)
                modified[0] ^= 1
                h2 = hash_func(bytes(modified))
                diff = sum(bin(a ^ b).count('1') for a, b in zip(h1, h2))
                flips.append(diff / 256 * 100)
            return np.mean(flips)
        
        sha_avalanche = measure_avalanche(lambda x: hashlib.sha256(x).digest())
        rft_avalanche = measure_avalanche(lambda x: hasher.hash(x))
        
        # Both should be near 50%
        assert abs(sha_avalanche - 50) < 5
        assert abs(rft_avalanche - 50) < 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
