#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
ANS Codec Regression Tests
===========================

Golden vector tests for the rANS entropy coder.
Ensures deterministic encoding/decoding behavior.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.rft.compression.ans import ans_encode, ans_decode


# Golden test vectors - NEVER CHANGE THESE
GOLDEN_ANS_VECTORS = {
    'simple': {
        'symbols': [0, 1, 2, 3, 4, 5, 6, 7],
        'description': 'Simple ascending sequence',
    },
    'repeated': {
        'symbols': [5] * 20,
        'description': 'Repeated symbol (high redundancy)',
    },
    'binary': {
        'symbols': [0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        'description': 'Binary alphabet',
    },
    'random_8bit': {
        'symbols': [23, 156, 89, 201, 45, 178, 12, 234, 67, 144, 98, 210, 33, 187, 76, 222],
        'description': 'Random 8-bit values',
    },
    'full_alphabet': {
        'symbols': list(range(256)),
        'description': 'All 8-bit symbols',
    },
    'alternating': {
        'symbols': [0, 255, 0, 255, 0, 255, 0, 255],
        'description': 'Alternating extremes',
    },
    'fibonacci': {
        'symbols': [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233],
        'description': 'Fibonacci sequence',
    },
}


class TestANSRoundtrip:
    """Test lossless round-trip encoding/decoding."""
    
    @pytest.mark.parametrize("vector_name", list(GOLDEN_ANS_VECTORS.keys()))
    def test_roundtrip_golden_vectors(self, vector_name: str):
        """Golden vectors should round-trip exactly."""
        vec = GOLDEN_ANS_VECTORS[vector_name]
        symbols = vec['symbols']
        
        # Encode
        encoded, freq_data = ans_encode(symbols)
        
        # Decode
        decoded = ans_decode(encoded, freq_data, len(symbols))
        
        assert decoded == symbols, (
            f"Golden vector '{vector_name}' failed:\n"
            f"  Expected: {symbols}\n"
            f"  Got:      {decoded}"
        )
    
    @pytest.mark.parametrize("n", [10, 100, 1000, 5000])
    def test_roundtrip_random_length(self, n: int):
        """Random symbol sequences of various lengths."""
        rng = np.random.default_rng(42)
        symbols = rng.integers(0, 256, size=n).tolist()
        
        encoded, freq_data = ans_encode(symbols)
        decoded = ans_decode(encoded, freq_data, n)
        
        assert decoded == symbols, f"Random n={n} failed"
    
    @pytest.mark.parametrize("max_val", [1, 7, 15, 31, 63, 127, 255])
    def test_roundtrip_limited_alphabet(self, max_val: int):
        """Test with limited alphabet sizes."""
        rng = np.random.default_rng(42)
        symbols = rng.integers(0, max_val + 1, size=100).tolist()
        
        encoded, freq_data = ans_encode(symbols)
        decoded = ans_decode(encoded, freq_data, 100)
        
        assert decoded == symbols, f"Limited alphabet max_val={max_val} failed"


class TestANSDeterminism:
    """Test that encoding is deterministic."""
    
    def test_deterministic_encoding(self):
        """Same input should produce same output."""
        symbols = [1, 2, 3, 4, 5, 6, 7, 8]
        
        encoded1, freq1 = ans_encode(symbols)
        encoded2, freq2 = ans_encode(symbols)
        
        assert np.array_equal(encoded1, encoded2), "Encoding is non-deterministic!"
    
    def test_consistent_across_instances(self):
        """Multiple encode calls should produce same output."""
        symbols = GOLDEN_ANS_VECTORS['random_8bit']['symbols']
        
        encodings = []
        for _ in range(5):
            encoded, _ = ans_encode(symbols)
            encodings.append(encoded.tolist())
        
        for i, enc in enumerate(encodings[1:], 1):
            assert enc == encodings[0], f"Encoding {i} differs"


class TestANSCompression:
    """Test compression efficiency."""
    
    def test_redundant_data_compresses(self):
        """Highly redundant data should compress well."""
        # All same symbol - maximum redundancy
        symbols = [42] * 1000
        
        encoded, freq_data = ans_encode(symbols)
        
        # Theoretical minimum is log2(1000) ≈ 10 bits ≈ 2 bytes
        # ANS with uniform prior won't be optimal, but should still compress
        original_bytes = 1000  # 1 byte per symbol
        compressed_bytes = len(encoded) * 2  # uint16 = 2 bytes
        ratio = original_bytes / max(compressed_bytes, 1)
        
        assert ratio > 2, f"Poor redundant compression: ratio={ratio:.3f}"
    
    def test_random_data_efficiency(self):
        """Random data should not expand significantly."""
        rng = np.random.default_rng(42)
        symbols = rng.integers(0, 256, size=1000).tolist()
        
        encoded, freq_data = ans_encode(symbols)
        
        # Should be close to 8 bits/symbol for uniform 8-bit random
        bits_per_symbol = len(encoded) * 16 / len(symbols)
        
        # Allow some overhead for ANS state
        assert bits_per_symbol < 10, f"Too much expansion: {bits_per_symbol:.3f} bps"


class TestANSEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_symbol(self):
        """Single symbol should work."""
        symbols = [128]
        
        encoded, freq_data = ans_encode(symbols)
        decoded = ans_decode(encoded, freq_data, 1)
        
        assert decoded == symbols
    
    def test_empty_sequence(self):
        """Empty sequence should work (or gracefully fail)."""
        encoded, freq_data = ans_encode([])
        
        # Either empty or minimal header
        assert len(encoded) < 100  # Reasonable for empty
    
    def test_boundary_values(self):
        """Test symbol boundary values 0 and 255."""
        symbols = [0, 255, 0, 0, 255, 255, 0, 255]
        
        encoded, freq_data = ans_encode(symbols)
        decoded = ans_decode(encoded, freq_data, len(symbols))
        
        assert decoded == symbols
    
    def test_long_sequence(self):
        """Test with very long sequence."""
        rng = np.random.default_rng(42)
        n = 100000
        symbols = rng.integers(0, 256, size=n).tolist()
        
        encoded, freq_data = ans_encode(symbols)
        decoded = ans_decode(encoded, freq_data, n)
        
        assert decoded == symbols, f"Long sequence n={n} failed"


class TestANSPrecision:
    """Test different precision settings."""
    
    @pytest.mark.parametrize("precision", [8, 12, 16])
    def test_precision_roundtrip(self, precision: int):
        """Various precision settings should round-trip."""
        rng = np.random.default_rng(42)
        symbols = rng.integers(0, 256, size=100).tolist()
        
        encoded, freq_data = ans_encode(symbols, precision=precision)
        decoded = ans_decode(encoded, freq_data, 100)
        
        assert decoded == symbols, f"Precision {precision} failed"


class TestANSStateReconstruction:
    """Test that decoder properly reconstructs encoder state."""
    
    def test_incremental_decode(self):
        """Decoding should work incrementally."""
        symbols = list(range(50))
        
        encoded, freq_data = ans_encode(symbols)
        decoded = ans_decode(encoded, freq_data, len(symbols))
        
        for i, expected in enumerate(symbols):
            assert decoded[i] == expected, f"Failed at position {i}: expected {expected}, got {decoded[i]}"
    
    def test_state_independence(self):
        """Different sequences should produce correct outputs."""
        sequences = [
            [0, 0, 0, 0, 0],
            [255, 255, 255, 255, 255],
            [0, 255, 0, 255, 0],
            [1, 2, 3, 4, 5],
        ]
        
        for seq in sequences:
            encoded, freq_data = ans_encode(seq)
            decoded = ans_decode(encoded, freq_data, len(seq))
            
            assert decoded == seq, f"Sequence {seq} failed"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
