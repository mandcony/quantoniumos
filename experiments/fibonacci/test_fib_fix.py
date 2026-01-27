#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Test that fibonacci overflow is fixed."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PHI = (1.0 + 5.0 ** 0.5) / 2.0

def _fib_ratio_sequence(n: int) -> np.ndarray:
    """Compute F(k+1)/F(k) iteratively - converges to phi rapidly."""
    ratios = np.ones(n, dtype=np.float64)
    if n == 0:
        return ratios
    ratios[0] = 1.0
    prev_ratio = 1.0
    for i in range(1, n):
        new_ratio = 1.0 + 1.0 / prev_ratio
        ratios[i] = new_ratio
        prev_ratio = new_ratio
    return ratios

def _frac(arr: np.ndarray) -> np.ndarray:
    frac, _ = np.modf(arr)
    return np.where(frac < 0.0, frac + 1.0, frac)

def fibonacci_rft_forward(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    k = np.arange(n, dtype=np.float64)
    fib_ratios = _fib_ratio_sequence(n)  # No overflow!
    theta = 2.0 * np.pi * (1.0 + alpha * (fib_ratios - PHI)) * _frac(k / PHI)
    D_fib = np.exp(1j * theta)
    ctheta = np.pi * (k * k / float(n))
    C_sig = np.exp(1j * ctheta)
    X = np.fft.fft(x, norm="ortho")
    return D_fib * (C_sig * X)

# Test with 5560 characters like english_prose
print("Testing fibonacci RFT with n=5560...")
data = np.random.randint(0, 256, 5560).astype(np.float64)

try:
    result = fibonacci_rft_forward(data)
    print(f"SUCCESS! Output shape: {result.shape}, first 3 values: {result[:3]}")
except Exception as e:
    print(f"FAILED: {e}")
