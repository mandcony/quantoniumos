#!/usr/bin/env python3
"""Validation tests for Theorem 10 hybrid Φ-RFT / DCT decomposition."""
from __future__ import annotations

import numpy as np
import pytest

from algorithms.rft.hybrid_basis import PHI, adaptive_hybrid_compress


def _relative_error(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x)
    return float(np.linalg.norm(x - y) / denom) if denom else 0.0


def test_ascii_signal_dct_dominant():
    """ASCII-like sequence should favour the structural (DCT) branch."""
    rng = np.random.default_rng(7)
    text = "The quick brown fox jumps over the lazy dog. " * 8
    ascii_values = np.array([ord(ch) for ch in text], dtype=np.float64)
    ascii_values = ascii_values[:256]
    noise = rng.integers(-3, 4, size=ascii_values.size)
    signal = (ascii_values + noise - np.mean(ascii_values)) / np.std(ascii_values)

    x_struct, x_texture, weights, metadata = adaptive_hybrid_compress(signal, max_iter=6)

    struct_energy = np.linalg.norm(x_struct) ** 2
    texture_energy = np.linalg.norm(x_texture) ** 2
    total_energy = np.linalg.norm(signal) ** 2
    residual_energy = metadata["residual_energy"][-1] if metadata["residual_energy"] else total_energy

    assert weights["dct"] > 0.8
    assert struct_energy > texture_energy
    assert residual_energy / total_energy < 0.05


def test_fibonacci_signal_rft_dominant():
    """Quasi-periodic Fibonacci modulation should favour the Φ-RFT branch."""
    n = 256
    t = np.arange(n)
    signal = np.sin(2 * np.pi * t / PHI) + 0.35 * np.sin(2 * np.pi * t / (PHI ** 2))

    x_struct, x_texture, weights, metadata = adaptive_hybrid_compress(signal, max_iter=6)

    struct_energy = np.linalg.norm(x_struct) ** 2
    texture_energy = np.linalg.norm(x_texture) ** 2
    total_energy = np.linalg.norm(signal) ** 2
    residual_energy = metadata["residual_energy"][-1] if metadata["residual_energy"] else total_energy

    assert weights["rft"] > weights["dct"]
    assert texture_energy > struct_energy
    assert residual_energy / total_energy < 0.08


def test_mixed_signal_hybrid_balanced():
    """Mixed ASCII/Fibonacci signal should allocate energy to both bases."""
    rng = np.random.default_rng(11)
    n = 256
    ascii_segment = rng.integers(60, 90, size=n // 2)
    fib_part = np.arange(n // 2)
    fib_signal = 40 * np.sin(2 * np.pi * fib_part / PHI)
    mixed = np.concatenate([ascii_segment.astype(float), fib_signal])
    mixed = (mixed - np.mean(mixed)) / np.std(mixed)

    x_struct, x_texture, weights, metadata = adaptive_hybrid_compress(mixed, max_iter=6)
    reconstruction = x_struct + x_texture

    struct_ratio = np.linalg.norm(x_struct) ** 2 / np.linalg.norm(mixed) ** 2
    texture_ratio = np.linalg.norm(x_texture) ** 2 / np.linalg.norm(mixed) ** 2

    assert weights["dct"] > 0.4 and weights["rft"] > 0.2
    assert 0.2 < struct_ratio < 0.8
    assert 0.2 < texture_ratio < 0.8
    assert _relative_error(mixed, reconstruction) < 5e-3
