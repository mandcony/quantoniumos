# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Optimized Φ-RFT Implementation with Fused Diagonals
====================================================

Key optimizations:
1. Fuse D_φ and C_σ into single diagonal: E = D_φ ⊙ C_σ
2. Precompute combined phase: θ[k] = 2πβ·frac(k/φ) + πσk²/n
3. Use single complex multiply instead of two
4. Cache phase tables for repeated transforms
5. Optionally use native SIMD backend

Performance target: RFT ≈ FFT × 1.2–1.5 (down from ~5× overhead)

Mathematical equivalence:
  Original: Y = D_φ · (C_σ · FFT(x))  [2 element-wise multiplies]
  Fused:    Y = E · FFT(x)            [1 element-wise multiply]
  
  where E[k] = D_φ[k] · C_σ[k] = exp(i · θ_fused[k])
"""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from functools import lru_cache
from typing import Tuple, Optional

PHI = (1.0 + 5.0 ** 0.5) / 2.0
PHI_INV = PHI - 1.0  # = 1/φ = φ - 1
PI = np.pi
TWO_PI = 2.0 * PI


def _frac(arr: np.ndarray) -> np.ndarray:
    """Fractional part in [0, 1)."""
    return np.mod(arr, 1.0)


@lru_cache(maxsize=32)
def _precompute_fused_phases(
    n: int, 
    beta: float = 1.0, 
    sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute fused phase diagonal E = D_φ · C_σ.
    
    Returns (E_fwd, E_inv) where:
      E_fwd[k] = exp(i · θ_fused[k])
      E_inv[k] = exp(-i · θ_fused[k]) = conj(E_fwd[k])
      
    θ_fused[k] = 2πβ·frac(k/φ) + πσk²/n
    
    Cache hit rate should be high for repeated transforms of same size.
    """
    k = np.arange(n, dtype=np.float64)
    
    # Fused phase: θ = θ_phi + θ_chirp
    # θ_phi = 2πβ·frac(k/φ)
    # θ_chirp = πσk²/n
    theta_phi = TWO_PI * beta * _frac(k * PHI_INV)
    theta_chirp = PI * sigma * (k * k) / n
    theta_fused = theta_phi + theta_chirp
    
    # Single exp call for fused diagonal
    E_fwd = np.exp(1j * theta_fused).astype(np.complex128, copy=False)
    E_inv = np.conj(E_fwd)  # Conjugate for inverse
    
    return E_fwd, E_inv


def rft_forward_optimized(
    x: ArrayLike, 
    *, 
    beta: float = 1.0, 
    sigma: float = 1.0
) -> np.ndarray:
    """
    Optimized forward Φ-RFT with fused diagonal.
    
    Y = E · FFT(x)
    
    where E = D_φ · C_σ (single fused diagonal)
    
    This is mathematically equivalent to the original:
      Y = D_φ · (C_σ · FFT(x))
    
    but uses only one element-wise multiply instead of two.
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    
    # Get cached fused diagonal
    E_fwd, _ = _precompute_fused_phases(n, beta, sigma)
    
    # FFT + single fused multiply
    X = np.fft.fft(x, norm="ortho")
    return E_fwd * X


def rft_inverse_optimized(
    y: ArrayLike, 
    *, 
    beta: float = 1.0, 
    sigma: float = 1.0
) -> np.ndarray:
    """
    Optimized inverse Φ-RFT with fused diagonal.
    
    x = IFFT(E† · Y)
    
    where E† = conj(E) = conj(D_φ · C_σ)
    """
    y = np.asarray(y, dtype=np.complex128)
    n = y.shape[0]
    
    # Get cached fused diagonal (use conjugate for inverse)
    _, E_inv = _precompute_fused_phases(n, beta, sigma)
    
    # Single fused multiply + IFFT
    return np.fft.ifft(E_inv * y, norm="ortho")


# =============================================================================
# Even More Optimized: In-place with preallocated buffers
# =============================================================================

class OptimizedRFTEngine:
    """
    Stateful RFT engine with preallocated buffers for zero-copy transforms.
    
    Usage:
        engine = OptimizedRFTEngine(1024)
        Y = engine.forward(x)  # Reuses internal buffers
        x_rec = engine.inverse(Y)
    """
    
    __slots__ = ('_size', '_beta', '_sigma', '_E_fwd', '_E_inv', '_buffer')
    
    def __init__(
        self, 
        size: int, 
        beta: float = 1.0, 
        sigma: float = 1.0
    ):
        self._size = size
        self._beta = beta
        self._sigma = sigma
        self._E_fwd, self._E_inv = _precompute_fused_phases(size, beta, sigma)
        self._buffer = np.empty(size, dtype=np.complex128)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward transform with minimal allocations."""
        if x.shape[0] != self._size:
            raise ValueError(f"Input size {x.shape[0]} != engine size {self._size}")
        
        # In-place FFT to buffer, then multiply
        np.copyto(self._buffer, x)
        X = np.fft.fft(self._buffer, norm="ortho")
        return self._E_fwd * X
    
    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform with minimal allocations."""
        if y.shape[0] != self._size:
            raise ValueError(f"Input size {y.shape[0]} != engine size {self._size}")
        
        # Multiply then IFFT
        np.multiply(self._E_inv, y, out=self._buffer)
        return np.fft.ifft(self._buffer, norm="ortho")
    
    def forward_inplace(self, x: np.ndarray) -> np.ndarray:
        """Forward transform, modifying input in place."""
        X = np.fft.fft(x, norm="ortho")
        np.multiply(self._E_fwd, X, out=x)
        return x
    
    @property
    def size(self) -> int:
        return self._size


# =============================================================================
# Compatibility wrapper for benchmark
# =============================================================================

def rft_forward(
    x: ArrayLike, 
    *, 
    beta: float = 1.0, 
    sigma: float = 1.0, 
    phi: float = PHI
) -> np.ndarray:
    """
    Drop-in replacement for closed_form_rft.rft_forward.
    
    Note: phi parameter is ignored (always uses golden ratio).
    """
    return rft_forward_optimized(x, beta=beta, sigma=sigma)


def rft_inverse(
    y: ArrayLike, 
    *, 
    beta: float = 1.0, 
    sigma: float = 1.0, 
    phi: float = PHI
) -> np.ndarray:
    """
    Drop-in replacement for closed_form_rft.rft_inverse.
    """
    return rft_inverse_optimized(y, beta=beta, sigma=sigma)


# =============================================================================
# Verification
# =============================================================================

def verify_equivalence(n: int = 1024, trials: int = 5) -> dict:
    """
    Verify that optimized RFT produces identical results to original.
    """
    from algorithms.rft.core.closed_form_rft import (
        rft_forward as rft_forward_orig,
        rft_inverse as rft_inverse_orig,
    )
    
    rng = np.random.default_rng(42)
    errors = []
    
    for _ in range(trials):
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        
        # Original implementation
        Y_orig = rft_forward_orig(x)
        x_rec_orig = rft_inverse_orig(Y_orig)
        
        # Optimized implementation
        Y_opt = rft_forward_optimized(x)
        x_rec_opt = rft_inverse_optimized(Y_opt)
        
        # Compare forward outputs
        fwd_err = np.linalg.norm(Y_orig - Y_opt) / np.linalg.norm(Y_orig)
        
        # Compare round-trip
        rt_err_orig = np.linalg.norm(x - x_rec_orig) / np.linalg.norm(x)
        rt_err_opt = np.linalg.norm(x - x_rec_opt) / np.linalg.norm(x)
        
        errors.append({
            'forward_diff': fwd_err,
            'roundtrip_orig': rt_err_orig,
            'roundtrip_opt': rt_err_opt,
        })
    
    return {
        'n': n,
        'trials': trials,
        'max_forward_diff': max(e['forward_diff'] for e in errors),
        'max_roundtrip_orig': max(e['roundtrip_orig'] for e in errors),
        'max_roundtrip_opt': max(e['roundtrip_opt'] for e in errors),
        'equivalent': all(e['forward_diff'] < 1e-12 for e in errors),  # Allow for FFT accumulation
    }


if __name__ == "__main__":
    # Quick benchmark comparison
    import time
    
    print("=" * 60)
    print("Φ-RFT Optimized vs Original Comparison")
    print("=" * 60)
    
    from algorithms.rft.core.closed_form_rft import (
        rft_forward as rft_forward_orig,
        rft_inverse as rft_inverse_orig,
    )
    
    sizes = [256, 1024, 4096, 16384]
    runs = 100
    
    for n in sizes:
        x = np.random.randn(n).astype(np.complex128)
        
        # Warmup
        for _ in range(10):
            _ = rft_forward_orig(x)
            _ = rft_forward_optimized(x)
        
        # Time original
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = rft_forward_orig(x)
        t_orig = (time.perf_counter() - t0) / runs * 1e6
        
        # Time optimized
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = rft_forward_optimized(x)
        t_opt = (time.perf_counter() - t0) / runs * 1e6
        
        # Time FFT only
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = np.fft.fft(x, norm="ortho")
        t_fft = (time.perf_counter() - t0) / runs * 1e6
        
        speedup = t_orig / t_opt
        ratio_to_fft = t_opt / t_fft
        
        print(f"n={n:5d} | FFT: {t_fft:7.1f}µs | "
              f"Orig: {t_orig:7.1f}µs | Opt: {t_opt:7.1f}µs | "
              f"Speedup: {speedup:.2f}x | RFT/FFT: {ratio_to_fft:.2f}x")
    
    print()
    print("Verifying correctness...")
    result = verify_equivalence()
    print(f"  Max forward diff: {result['max_forward_diff']:.2e}")
    print(f"  Max roundtrip error (opt): {result['max_roundtrip_opt']:.2e}")
    print(f"  Equivalent: {result['equivalent']}")
