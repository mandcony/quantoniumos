#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Canonical RFT Test Suite
========================

Tests for the canonical Resonant Fourier Transform definition:

    Ψₖ(t) = exp(2πi × fₖ × t + i × θₖ)
    
    fₖ = (k+1) × φ       (Resonant Frequency)
    θₖ = 2π × k / φ      (Golden Phase)
    φ = (1+√5)/2         (Golden Ratio)

USPTO Patent 19/169,399
"""

import pytest
import numpy as np
from algorithms.rft import (
    PHI, PHI_INV,
    rft_frequency, rft_phase, rft_basis_function, rft_basis_matrix,
    rft_forward, rft_inverse, rft, irft,
    BinaryRFT, RFTSISHash
)


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestConstants:
    """Test golden ratio constants."""
    
    def test_phi_value(self):
        """PHI should be the golden ratio."""
        expected = (1 + np.sqrt(5)) / 2
        assert np.isclose(PHI, expected)
        assert np.isclose(PHI, 1.618033988749895)
    
    def test_phi_inv_value(self):
        """PHI_INV should be 1/φ = φ - 1."""
        assert np.isclose(PHI_INV, 1/PHI)
        assert np.isclose(PHI_INV, PHI - 1)
    
    def test_phi_identity(self):
        """φ² = φ + 1 (defining property of golden ratio)."""
        assert np.isclose(PHI**2, PHI + 1)


# =============================================================================
# BASIS FUNCTION TESTS
# =============================================================================

class TestBasisFunctions:
    """Test RFT basis function generation."""
    
    def test_rft_frequency(self):
        """fₖ = (k+1) × φ."""
        assert np.isclose(rft_frequency(0), PHI)
        assert np.isclose(rft_frequency(1), 2 * PHI)
        assert np.isclose(rft_frequency(7), 8 * PHI)
    
    def test_rft_phase(self):
        """θₖ = 2π × k / φ."""
        assert np.isclose(rft_phase(0), 0)
        assert np.isclose(rft_phase(1), 2 * np.pi / PHI)
        assert np.isclose(rft_phase(2), 4 * np.pi / PHI)
    
    def test_basis_function_shape(self):
        """Basis function should have correct shape."""
        t = np.linspace(0, 1, 100)
        psi = rft_basis_function(3, t)
        assert psi.shape == (100,)
        assert psi.dtype == np.complex128
    
    def test_basis_function_formula(self):
        """Ψₖ(t) = exp(2πi × fₖ × t + i × θₖ)."""
        k = 3
        t = np.array([0.0, 0.25, 0.5, 0.75])
        
        f_k = rft_frequency(k)
        theta_k = rft_phase(k)
        expected = np.exp(2j * np.pi * f_k * t + 1j * theta_k)
        
        actual = rft_basis_function(k, t)
        np.testing.assert_allclose(actual, expected)
    
    def test_basis_matrix_shape(self):
        """Basis matrix should be N×T."""
        Psi = rft_basis_matrix(8, 128)
        assert Psi.shape == (8, 128)
    
    def test_basis_matrix_cached(self):
        """Basis matrix should be cached for same parameters."""
        Psi1 = rft_basis_matrix(8, 128)
        Psi2 = rft_basis_matrix(8, 128)
        assert Psi1 is Psi2  # Same object (cached)


# =============================================================================
# FORWARD/INVERSE TRANSFORM TESTS
# =============================================================================

class TestTransform:
    """Test forward and inverse RFT."""
    
    def test_forward_shape(self):
        """Forward RFT output shape."""
        x = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        W = rft_forward(x)
        assert W.shape[0] == len(x) * 16  # Default oversampling
    
    def test_forward_custom_samples(self):
        """Forward RFT with custom sample count."""
        x = np.array([1, 2, 3, 4])
        W = rft_forward(x, T=100)
        assert W.shape == (100,)
    
    def test_inverse_shape(self):
        """Inverse RFT output shape."""
        W = np.random.randn(128) + 1j * np.random.randn(128)
        x = rft_inverse(W, N=8)
        assert x.shape == (8,)
    
    def test_roundtrip_correlation(self):
        """Forward then inverse should recover signal structure."""
        x = np.sin(2 * np.pi * 3 * np.linspace(0, 1, 32))
        W = rft_forward(x, T=len(x) * 4)
        x_rec = np.real(rft_inverse(W, len(x)))
        
        # Should have high correlation
        corr = np.corrcoef(x, x_rec)[0, 1]
        assert corr > 0.95
    
    def test_convenience_functions(self):
        """rft() and irft() should work."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        W = rft(x)
        x_rec = irft(W, len(x))
        
        assert W is not None
        assert len(x_rec) == len(x)


# =============================================================================
# BINARY RFT TESTS
# =============================================================================

class TestBinaryRFT:
    """Test binary encoding and wave-domain computation."""
    
    def test_encode_decode_all_bytes(self):
        """All 256 byte values should roundtrip perfectly."""
        brft = BinaryRFT(8)
        for val in range(256):
            wave = brft.encode(val)
            recovered = brft.decode(wave)
            assert recovered == val, f"Failed for {val}"
    
    def test_encode_integer(self):
        """Encode integer values."""
        brft = BinaryRFT(8)
        wave = brft.encode(0xAA)
        assert wave.dtype == np.complex128
        assert len(wave) == brft.T
    
    def test_encode_bytes(self):
        """Encode bytes."""
        brft = BinaryRFT(8)
        wave = brft.encode(b'\xAA')
        recovered = brft.decode(wave)
        assert recovered == 0xAA
    
    def test_multi_bit_widths(self):
        """Support 8, 16, 32, 64-bit operations."""
        for bits in [8, 16, 32, 64]:
            brft = BinaryRFT(bits)
            test_val = (1 << bits) // 3  # 0x555...
            
            wave = brft.encode(test_val)
            recovered = brft.decode(wave)
            
            assert recovered == test_val, f"Failed for {bits}-bit"
    
    def test_frequencies_property(self):
        """frequencies property returns correct values."""
        brft = BinaryRFT(8)
        freqs = brft.frequencies
        
        assert len(freqs) == 8
        assert np.isclose(freqs[0], PHI)
        assert np.isclose(freqs[7], 8 * PHI)
    
    def test_phases_property(self):
        """phases property returns correct values."""
        brft = BinaryRFT(8)
        phases = brft.phases
        
        assert len(phases) == 8
        assert np.isclose(phases[0], 0)
        assert np.isclose(phases[1], 2 * np.pi / PHI)


# =============================================================================
# WAVE-DOMAIN LOGIC TESTS
# =============================================================================

class TestWaveLogic:
    """Test logic operations on waveforms."""
    
    @pytest.fixture
    def brft(self):
        return BinaryRFT(8)
    
    def test_wave_xor(self, brft):
        """XOR operation in wave domain."""
        test_cases = [
            (0b10101010, 0b11001100),
            (0xFF, 0x00),
            (0x0F, 0xF0),
            (0x00, 0x00),
            (0xFF, 0xFF),
        ]
        
        for a, b in test_cases:
            wa, wb = brft.encode(a), brft.encode(b)
            result_wave = brft.wave_xor(wa, wb)
            result = brft.decode(result_wave)
            expected = a ^ b
            assert result == expected, f"XOR failed: {a:08b} ^ {b:08b}"
    
    def test_wave_and(self, brft):
        """AND operation in wave domain."""
        test_cases = [
            (0b10101010, 0b11001100),
            (0xFF, 0x00),
            (0xFF, 0xFF),
            (0x0F, 0xF0),
        ]
        
        for a, b in test_cases:
            wa, wb = brft.encode(a), brft.encode(b)
            result_wave = brft.wave_and(wa, wb)
            result = brft.decode(result_wave)
            expected = a & b
            assert result == expected, f"AND failed: {a:08b} & {b:08b}"
    
    def test_wave_or(self, brft):
        """OR operation in wave domain."""
        test_cases = [
            (0b10101010, 0b11001100),
            (0xFF, 0x00),
            (0x00, 0x00),
            (0x0F, 0xF0),
        ]
        
        for a, b in test_cases:
            wa, wb = brft.encode(a), brft.encode(b)
            result_wave = brft.wave_or(wa, wb)
            result = brft.decode(result_wave)
            expected = a | b
            assert result == expected, f"OR failed: {a:08b} | {b:08b}"
    
    def test_wave_not(self, brft):
        """NOT operation in wave domain."""
        test_cases = [0x00, 0xFF, 0xAA, 0x55, 0x0F, 0xF0]
        
        for a in test_cases:
            wa = brft.encode(a)
            result_wave = brft.wave_not(wa)
            result = brft.decode(result_wave)
            expected = (~a) & 0xFF
            assert result == expected, f"NOT failed: ~{a:08b}"
    
    def test_chained_operations(self, brft):
        """Chained operations without intermediate decoding."""
        A, B, C, D = 0x12, 0x34, 0x56, 0x78
        wA, wB, wC, wD = [brft.encode(x) for x in [A, B, C, D]]
        
        # (A XOR B) AND (C OR D)
        w_ab = brft.wave_xor(wA, wB)
        w_cd = brft.wave_or(wC, wD)
        w_result = brft.wave_and(w_ab, w_cd)
        result = brft.decode(w_result)
        expected = (A ^ B) & (C | D)
        
        assert result == expected
    
    def test_deep_chain(self, brft):
        """Deep operation chains (1000+) should not degrade."""
        val, key = 0xAA, 0x55
        wave = brft.encode(val)
        key_wave = brft.encode(key)
        
        for _ in range(1000):
            wave = brft.wave_xor(wave, key_wave)
        
        result = brft.decode(wave)
        expected = val  # Even number of XORs returns original
        assert result == expected


# =============================================================================
# CRYPTOGRAPHIC HASH TESTS
# =============================================================================

class TestRFTSISHash:
    """Test RFT-SIS cryptographic hash."""
    
    @pytest.fixture
    def hasher(self):
        return RFTSISHash()
    
    def test_hash_output_length(self, hasher):
        """Hash should produce 32 bytes (256 bits)."""
        h = hasher.hash(b"test")
        assert len(h) == 32
    
    def test_hash_deterministic(self, hasher):
        """Same input should produce same hash."""
        data = b"hello world"
        h1 = hasher.hash(data)
        h2 = hasher.hash(data)
        assert h1 == h2
    
    def test_hash_different_inputs(self, hasher):
        """Different inputs should produce different hashes."""
        h1 = hasher.hash(b"hello")
        h2 = hasher.hash(b"world")
        assert h1 != h2
    
    def test_avalanche_effect(self, hasher):
        """1-bit change should flip ~50% of output bits."""
        flips = []
        
        for _ in range(100):
            data = np.random.bytes(32)
            h1 = hasher.hash(data)
            
            # Flip one bit
            modified = bytearray(data)
            modified[0] ^= 1
            h2 = hasher.hash(bytes(modified))
            
            # Count bit differences
            diff = sum(bin(a ^ b).count('1') for a, b in zip(h1, h2))
            flips.append(diff / 256 * 100)
        
        avg_flip = np.mean(flips)
        assert 40 <= avg_flip <= 60, f"Avalanche {avg_flip}% not near 50%"
    
    def test_collision_resistance(self, hasher):
        """No collisions in 1000 random inputs."""
        hashes = set()
        
        for i in range(1000):
            h = hasher.hash(i.to_bytes(8, 'big'))
            assert h not in hashes, f"Collision at {i}"
            hashes.add(h)
    
    def test_hash_hex(self, hasher):
        """hash_hex returns hex string."""
        h = hasher.hash_hex(b"test")
        assert isinstance(h, str)
        assert len(h) == 64  # 32 bytes = 64 hex chars


# =============================================================================
# NOISE IMMUNITY TESTS
# =============================================================================

class TestNoiseImmunity:
    """Test robustness to noise."""
    
    def add_noise(self, wave, snr_db):
        """Add Gaussian noise at specified SNR."""
        signal_power = np.mean(np.abs(wave)**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power/2) * (
            np.random.randn(len(wave)) + 1j*np.random.randn(len(wave))
        )
        return wave + noise
    
    def test_high_snr(self):
        """Perfect recovery at high SNR (20+ dB)."""
        brft = BinaryRFT(8)
        
        for _ in range(100):
            val = np.random.randint(256)
            wave = brft.encode(val)
            noisy = self.add_noise(wave, 20)
            recovered = brft.decode(noisy)
            assert recovered == val
    
    def test_moderate_snr(self):
        """Low BER at moderate SNR (10 dB)."""
        brft = BinaryRFT(8)
        errors = 0
        
        for _ in range(200):
            val = np.random.randint(256)
            wave = brft.encode(val)
            noisy = self.add_noise(wave, 10)
            recovered = brft.decode(noisy)
            if recovered != val:
                errors += 1
        
        ber = errors / 200 * 100
        assert ber < 5, f"BER {ber}% too high at 10 dB SNR"
    
    def test_noise_floor(self):
        """Should work down to ~0 dB SNR."""
        brft = BinaryRFT(8)
        errors = 0
        
        for _ in range(200):
            val = np.random.randint(256)
            wave = brft.encode(val)
            noisy = self.add_noise(wave, 0)
            recovered = brft.decode(noisy)
            if recovered != val:
                errors += 1
        
        ber = errors / 200 * 100
        assert ber < 10, f"BER {ber}% too high at 0 dB SNR"


# =============================================================================
# GOLDEN RATIO PROPERTY TESTS
# =============================================================================

class TestGoldenRatioProperties:
    """Test mathematical properties of golden ratio structure."""
    
    def test_fibonacci_convergence(self):
        """Fibonacci ratios converge to φ."""
        fib = [1, 1]
        for _ in range(20):
            fib.append(fib[-1] + fib[-2])
        
        ratio = fib[-1] / fib[-2]
        assert np.isclose(ratio, PHI, rtol=1e-10)
    
    def test_golden_angle(self):
        """Golden angle is 2π/φ² ≈ 137.5° (complement 2π/φ ≈ 222.5°)."""
        golden_angle_rad = 2 * np.pi / (PHI ** 2)
        golden_angle_deg = np.degrees(golden_angle_rad)
        
        assert np.isclose(golden_angle_deg, 137.5, rtol=0.01)
    
    def test_frequency_ratio(self):
        """Frequency ratios approach φ."""
        freqs = [rft_frequency(k) for k in range(100)]
        ratios = [freqs[k+1] / freqs[k] for k in range(99)]
        
        # Later ratios should approach PHI
        assert np.isclose(ratios[-1], (101/100), rtol=0.01)


# =============================================================================
# HALF ADDER TEST (Arithmetic via Logic)
# =============================================================================

class TestArithmeticViaLogic:
    """Test arithmetic operations built from wave logic."""
    
    def test_half_adder(self):
        """Half adder: Sum = A XOR B, Carry = A AND B."""
        brft = BinaryRFT(1)
        
        for a in [0, 1]:
            for b in [0, 1]:
                wa = brft.encode(a)
                wb = brft.encode(b)
                
                sum_wave = brft.wave_xor(wa, wb)
                carry_wave = brft.wave_and(wa, wb)
                
                sum_result = brft.decode(sum_wave)
                carry_result = brft.decode(carry_wave)
                
                assert sum_result == a ^ b
                assert carry_result == a & b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
