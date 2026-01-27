# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
RFT Basis Construction with Caching
=====================================

Complexity:
- Exact basis construction: O(N^3) via eigendecomposition
- Cached lookup: O(1) load + O(N^2) matrix-vector multiply for transform
- Lanczos (truncated): O(N^2 k) where k = number of eigenvectors retained

Key insight: The O(N³) eigendecomposition only needs to be done once
per (size, variant) pair. We cache results to disk and memory.

EXPERIMENTAL METHODS (clearly marked, not for production benchmarks):
- Circulant approximation: O(N log N) but KNOWN TO FAIL for RFT kernels
  (50-70% relative error, ~90° subspace angles vs exact basis)
- FFT + low-rank correction: Conjectural, needs validation

December 2025: Part of the "Next Level RFT" initiative.
"""

import numpy as np
from scipy.linalg import toeplitz, eigh
from scipy.fft import fft, ifft
from functools import lru_cache
from typing import Tuple, Optional, Dict
import os
from pathlib import Path

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2

# Cache directory for pre-computed bases
CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "rft_basis"


# =============================================================================
# SECTION 1: CACHED BASIS LOOKUP (O(1) for standard sizes)
# =============================================================================

# Standard sizes for pre-computation
STANDARD_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Variant generators
VARIANT_AUTOCORR = {
    'golden': lambda n, k: np.cos(2*np.pi*10*k/n) + np.cos(2*np.pi*10*PHI*k/n),
    'fibonacci': lambda n, k: _fib_autocorr(n, k),
    'harmonic': lambda n, k: sum((1.0/i)*np.cos(2*np.pi*i*4*k/n) for i in range(1, 6)),
    'geometric': lambda n, k: sum(np.cos(2*np.pi*(PHI**i)*k/n)/(i+1) for i in range(8) if PHI**i < n/2),
}


def _fib_autocorr(n: int, k: np.ndarray) -> np.ndarray:
    """Fibonacci-based autocorrelation."""
    fib = [1, 1]
    while fib[-1] < n:
        fib.append(fib[-1] + fib[-2])
    fib = fib[:min(8, len(fib))]
    
    r = np.zeros_like(k, dtype=float)
    for f in fib:
        r += np.cos(2*np.pi*f*k/n) / len(fib)
    return r


def _build_basis_exact(n: int, variant: str = 'golden', decay_rate: float = 0.01) -> np.ndarray:
    """
    Build exact eigenbasis (O(N³) but cached).
    """
    k = np.arange(n)
    
    # Get autocorrelation function
    if variant not in VARIANT_AUTOCORR:
        raise ValueError(f"Unknown variant: {variant}")
    
    r = VARIANT_AUTOCORR[variant](n, k)
    
    # Apply decay regularization
    decay = np.exp(-decay_rate * k)
    r_reg = r * decay
    r_reg[0] = 1.0
    
    # Build Toeplitz operator
    K = toeplitz(r_reg)
    
    # Eigendecomposition (O(N³))
    eigenvalues, eigenvectors = eigh(K)
    idx = np.argsort(eigenvalues)[::-1]
    
    return eigenvectors[:, idx]


@lru_cache(maxsize=128)
def get_cached_basis(n: int, variant: str = 'golden') -> np.ndarray:
    """
    Get RFT basis with aggressive caching.
    
    1. Check memory cache (LRU)
    2. Check disk cache
    3. Compute and cache
    
    For standard sizes and variants, this is O(1).
    """
    # Try disk cache
    cache_file = CACHE_DIR / f"{variant}_n{n}.npy"
    
    if cache_file.exists():
        return np.load(cache_file)
    
    # Compute exact basis
    basis = _build_basis_exact(n, variant)
    
    # Cache to disk for standard sizes
    if n in STANDARD_SIZES:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, basis)
    
    return basis


def precompute_all_bases():
    """Pre-compute and cache all standard bases."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    for n in STANDARD_SIZES:
        for variant in VARIANT_AUTOCORR.keys():
            cache_file = CACHE_DIR / f"{variant}_n{n}.npy"
            if not cache_file.exists():
                print(f"Computing {variant} basis for N={n}...")
                basis = _build_basis_exact(n, variant)
                np.save(cache_file, basis)
    
    print("All bases pre-computed.")


# =============================================================================
# SECTION 2: FAST APPROXIMATE CONSTRUCTION (O(N log² N))
# =============================================================================

def _circulant_eigenbasis(r: np.ndarray) -> np.ndarray:
    """
    EXPERIMENTAL - KNOWN TO FAIL FOR RFT KERNELS
    ============================================
    
    Circulant approximation of Toeplitz eigenbasis.
    
    Circulant matrices are diagonalized by DFT: C = F^H diag(Fc) F
    
    This is O(N log N) but experiments show:
    - 50-70% relative error in eigenvalues vs true Toeplitz
    - ~90° subspace angles between circulant and exact RFT basis
    - NOT suitable for any production or published benchmarks
    
    Keep for research/negative-result documentation only.
    """
    n = len(r)
    
    # Circulant embedding: make Toeplitz into circulant
    # For symmetric Toeplitz: c = [r[0], r[1], ..., r[n-1], r[n-1], ..., r[1]]
    c = np.concatenate([r, r[-2:0:-1]])
    
    # Eigenvalues of circulant = FFT of first column
    eigenvalues = fft(c).real[:n]  # Take first n eigenvalues
    
    # Eigenvectors of circulant = DFT columns
    # But we want sorted by eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    
    # Return DFT matrix columns in sorted order
    # This is approximate (circulant ≠ Toeplitz) but fast
    F = np.fft.fft(np.eye(n), norm='ortho')
    return F[:, idx].real


def _lanczos_top_eigenvectors(K: np.ndarray, k: int = None, tol: float = 1e-6) -> np.ndarray:
    """
    Lanczos algorithm for top-k eigenvectors.
    
    O(N² k) instead of O(N³) for full eigendecomposition.
    For k << N, this is much faster.
    """
    n = K.shape[0]
    if k is None:
        k = min(n // 4, 64)  # Default: quarter of N or 64
    k = min(k, n)
    
    # Lanczos iteration
    V = np.zeros((n, k+1))
    T = np.zeros((k+1, k+1))
    
    # Random starting vector
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    V[:, 0] = v
    
    beta = 0
    for j in range(k):
        w = K @ V[:, j]
        if j > 0:
            w = w - beta * V[:, j-1]
        alpha = np.dot(w, V[:, j])
        w = w - alpha * V[:, j]
        
        # Reorthogonalization
        for i in range(j+1):
            w = w - np.dot(w, V[:, i]) * V[:, i]
        
        beta = np.linalg.norm(w)
        if beta < tol:
            break
        
        V[:, j+1] = w / beta
        T[j, j] = alpha
        if j < k:
            T[j, j+1] = beta
            T[j+1, j] = beta
    
    # Eigendecomposition of tridiagonal T
    T_k = T[:j+1, :j+1]
    eig_vals, eig_vecs = eigh(T_k)
    idx = np.argsort(eig_vals)[::-1]
    
    # Ritz vectors
    ritz_vectors = V[:, :j+1] @ eig_vecs[:, idx]
    
    return ritz_vectors


def get_fast_basis(n: int, variant: str = 'golden', 
                   method: str = 'auto', k: int = None) -> np.ndarray:
    """
    Get RFT basis matrix.
    
    Args:
        n: Transform size
        variant: Which RFT variant ('golden', 'fibonacci', 'harmonic', 'geometric')
        method: 
            'cached' - O(1) lookup for standard sizes (RECOMMENDED)
            'exact'  - O(N³) eigendecomposition (correct, slow for large N)
            'lanczos'- O(N²k) truncated eigenbasis (approximate, for research)
            'circulant' - EXPERIMENTAL, KNOWN TO FAIL (do not use for benchmarks)
            'auto'   - cached if available, else exact for N≤256, else lanczos
        k: Number of eigenvectors for Lanczos (None = quarter of N)
    
    Returns:
        Orthonormal basis matrix (N x N, or N x k for truncated Lanczos)
    
    WARNING: 'circulant' method produces ~50-70% error vs exact basis.
    Only 'cached' and 'exact' are validated for production use.
    """
    # Auto-select method based on size
    if method == 'auto':
        if n in STANDARD_SIZES:
            method = 'cached'  # O(1) lookup
        elif n <= 256:
            method = 'exact'   # O(N³) but small N
        else:
            method = 'lanczos'  # O(N² k) for large N
    
    if method == 'cached':
        return get_cached_basis(n, variant)
    
    elif method == 'exact':
        return _build_basis_exact(n, variant)
    
    elif method == 'circulant':
        k_arr = np.arange(n)
        r = VARIANT_AUTOCORR[variant](n, k_arr)
        decay = np.exp(-0.01 * k_arr)
        r_reg = r * decay
        r_reg[0] = 1.0
        return _circulant_eigenbasis(r_reg)
    
    elif method == 'lanczos':
        k_arr = np.arange(n)
        r = VARIANT_AUTOCORR[variant](n, k_arr)
        decay = np.exp(-0.01 * k_arr)
        r_reg = r * decay
        r_reg[0] = 1.0
        K = toeplitz(r_reg)
        return _lanczos_top_eigenvectors(K, k)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# SECTION 3: FAST TRANSFORM (avoiding full matrix multiply)
# =============================================================================

def rft_forward_fast(x: np.ndarray, variant: str = 'golden', 
                     k: int = None) -> np.ndarray:
    """
    Fast RFT forward transform.
    
    For cached sizes: O(N²) matrix multiply (but fast due to NumPy/BLAS)
    For large sizes: O(N k) using truncated basis
    """
    n = len(x)
    
    if n in STANDARD_SIZES:
        # Full basis, O(N²) but well-optimized
        U = get_cached_basis(n, variant)
        return U.T @ x
    else:
        # Truncated basis for large non-standard sizes
        if k is None:
            k = min(n // 4, 256)
        U_k = get_fast_basis(n, variant, method='lanczos', k=k)
        return U_k.T @ x


def rft_inverse_fast(coeffs: np.ndarray, variant: str = 'golden',
                     original_size: int = None) -> np.ndarray:
    """
    Fast RFT inverse transform.
    """
    if original_size is None:
        original_size = len(coeffs)
    
    n = original_size
    
    if n in STANDARD_SIZES:
        U = get_cached_basis(n, variant)
        return U @ coeffs
    else:
        k = len(coeffs)
        U_k = get_fast_basis(n, variant, method='lanczos', k=k)
        return U_k @ coeffs


# =============================================================================
# SECTION 4: HYBRID FFT + CORRECTION (Research)
# =============================================================================

def rft_via_fft_correction(x: np.ndarray, variant: str = 'golden',
                           correction_rank: int = 8) -> np.ndarray:
    """
    EXPERIMENTAL - UNVALIDATED HYPOTHESIS
    =====================================
    
    Attempt RFT as FFT + low-rank correction.
    
    Hypothesis: If RFT eigenbasis U is "close to" DFT basis F,
    then U ≈ F + low-rank correction, giving O(N log N + Nr).
    
    CURRENT STATUS:
    - Mean overlap between DFT and RFT vectors: ~0.465 (not close!)
    - Min overlap: 0.161, suggesting bases are genuinely different
    - Singular value decay of (U-F) has NOT been validated
    - DO NOT use in production until exp_fft_correction_rank.py confirms
    
    This is a research direction, not a working algorithm.
    """
    n = len(x)
    
    # Get RFT basis
    U = get_fast_basis(n, variant, method='cached' if n in STANDARD_SIZES else 'exact')
    
    # FFT basis (normalized)
    F = np.fft.fft(np.eye(n), norm='ortho')
    
    # Compute correction matrix
    correction = U - F.real
    
    # Low-rank approximation of correction via SVD
    U_c, s_c, Vh_c = np.linalg.svd(correction, full_matrices=False)
    
    # Keep top-r singular values
    r = min(correction_rank, len(s_c))
    U_r = U_c[:, :r]
    s_r = s_c[:r]
    Vh_r = Vh_c[:r, :]
    
    # Reconstruct low-rank correction
    correction_lowrank = U_r @ np.diag(s_r) @ Vh_r
    
    # Apply: FFT + correction
    fft_result = np.fft.fft(x, norm='ortho').real
    correction_result = correction_lowrank.T @ x
    
    return fft_result + correction_result


# =============================================================================
# BENCHMARKS
# =============================================================================

def benchmark_construction_time():
    """Compare construction times for different methods."""
    import time
    
    sizes = [64, 128, 256, 512, 1024]
    methods = ['exact', 'circulant', 'lanczos']
    
    print("=" * 60)
    print("BASIS CONSTRUCTION TIME BENCHMARK")
    print("=" * 60)
    print(f"{'Size':<10} {'Exact':<12} {'Circulant':<12} {'Lanczos':<12}")
    print("-" * 60)
    
    for n in sizes:
        times = {}
        for method in methods:
            start = time.time()
            _ = get_fast_basis(n, 'golden', method=method)
            times[method] = time.time() - start
        
        print(f"{n:<10} {times['exact']*1000:>8.2f} ms  {times['circulant']*1000:>8.2f} ms  {times['lanczos']*1000:>8.2f} ms")
    
    print("=" * 60)


def benchmark_transform_accuracy():
    """Compare reconstruction accuracy for different methods."""
    
    print("=" * 60)
    print("TRANSFORM ACCURACY BENCHMARK")
    print("=" * 60)
    
    n = 256
    np.random.seed(42)
    x = np.random.randn(n)
    
    # Exact basis
    U_exact = get_fast_basis(n, 'golden', method='exact')
    coeffs_exact = U_exact.T @ x
    x_rec_exact = U_exact @ coeffs_exact
    
    # Circulant approximation
    U_circ = get_fast_basis(n, 'golden', method='circulant')
    coeffs_circ = U_circ.T @ x
    x_rec_circ = U_circ @ coeffs_circ
    
    # Lanczos (truncated)
    U_lanc = get_fast_basis(n, 'golden', method='lanczos', k=64)
    coeffs_lanc = U_lanc.T @ x
    x_rec_lanc = U_lanc @ coeffs_lanc[:U_lanc.shape[1]]
    
    print(f"Method       | Reconstruction Error | Unitarity Error")
    print("-" * 60)
    print(f"Exact        | {np.linalg.norm(x - x_rec_exact):.2e}            | {np.linalg.norm(U_exact.T @ U_exact - np.eye(n)):.2e}")
    print(f"Circulant    | {np.linalg.norm(x - x_rec_circ):.2e}            | {np.linalg.norm(U_circ.T @ U_circ - np.eye(n)):.2e}")
    print(f"Lanczos (64) | {np.linalg.norm(x - x_rec_lanc):.2e}            | N/A (truncated)")
    print("=" * 60)


if __name__ == "__main__":
    print("Fast RFT Basis Construction Module")
    print("=" * 60)
    
    # Pre-compute all standard bases
    print("\n[1] Pre-computing standard bases...")
    precompute_all_bases()
    
    # Benchmark construction time
    print("\n[2] Benchmarking construction time...")
    benchmark_construction_time()
    
    # Benchmark accuracy
    print("\n[3] Benchmarking accuracy...")
    benchmark_transform_accuracy()
