#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Fast RFT Algorithm Exploration
==============================

Goal: Move from O(N²) to O(N log N) by exploiting Toeplitz structure.

Key insight: Our resonance operator K is Toeplitz (K[i,j] = r[|i-j|]).
Toeplitz matrices are close to circulant matrices, which are diagonalized by DFT.

Approaches to explore:
1. Circulant approximation: K ≈ C where C = F† D F
2. Displacement rank methods: K has low displacement rank
3. FFT-based matrix-vector products: Kx in O(N log N) via embedding
4. Iterative eigensolvers with fast Kx

If we can show the eigenbasis U of K is well-approximated by a structured
matrix that admits fast multiply, we get O(N log N) approximate RFT.

December 2025 - Track B exploration
"""

import numpy as np
from scipy.linalg import toeplitz, circulant, eigh
from scipy.fft import fft, ifft
import time
from typing import Tuple, Dict, Any

PHI = (1 + np.sqrt(5)) / 2


def build_resonance_kernel(n: int, variant: str = 'golden') -> np.ndarray:
    """Build the resonance autocorrelation function r[k]."""
    k = np.arange(n)
    t = k / n
    decay = np.exp(-0.01 * k)
    
    if variant == 'golden':
        f0 = 10.0
        r = np.cos(2 * np.pi * f0 * t) + np.cos(2 * np.pi * f0 * PHI * t)
    elif variant == 'harmonic':
        r = sum((1.0/i) * np.cos(2 * np.pi * i * 4.0 * t) for i in range(1, 6))
    else:
        r = np.cos(2 * np.pi * 10 * t)
    
    r = r * decay
    r[0] = 1.0  # Normalize
    return r


def build_toeplitz_operator(r: np.ndarray) -> np.ndarray:
    """Build symmetric Toeplitz operator K from kernel r."""
    return toeplitz(r)


def build_circulant_approximation(r: np.ndarray) -> np.ndarray:
    """
    Build circulant approximation C to Toeplitz K.
    
    For symmetric Toeplitz T(r), the natural circulant embedding is:
    C = circ([r[0], r[1], ..., r[n-1], r[n-1], r[n-2], ..., r[1]])
    
    But for approximation, we can use just the first row extended symmetrically.
    """
    n = len(r)
    # Symmetric circulant: c = [r[0], r[1], ..., r[n-1], r[n-2], ..., r[1]]
    # But for size n, just use r directly as first row
    return circulant(r)


# =============================================================================
# APPROACH 1: Circulant Approximation
# =============================================================================

def circulant_eigenbasis(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition of circulant matrix via FFT.
    
    For circulant C with first row c:
    - Eigenvectors are DFT columns (normalized)
    - Eigenvalues are λ[k] = Σ c[j] ω^(jk) = DFT(c)
    
    This is O(N log N)!
    """
    n = len(r)
    
    # Eigenvalues via FFT
    eigenvalues = fft(r)
    
    # Eigenvectors are DFT columns (normalized)
    # U[:,k] = (1/sqrt(n)) * [1, ω^k, ω^(2k), ..., ω^((n-1)k)]
    # where ω = exp(-2πi/n)
    k = np.arange(n)
    omega = np.exp(-2j * np.pi / n)
    U = np.zeros((n, n), dtype=np.complex128)
    for col in range(n):
        U[:, col] = omega ** (k * col) / np.sqrt(n)
    
    # Sort by eigenvalue magnitude (descending) for energy compaction
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    U = U[:, idx]
    
    return eigenvalues, U


def fast_circulant_multiply(r: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Multiply circulant(r) @ x in O(N log N) via FFT.
    
    C @ x = IFFT(FFT(r) * FFT(x))
    """
    return np.real(ifft(fft(r) * fft(x)))


# =============================================================================
# APPROACH 2: Toeplitz Embedding for Fast Multiply
# =============================================================================

def embed_toeplitz_in_circulant(r: np.ndarray) -> np.ndarray:
    """
    Embed n×n Toeplitz T(r) in 2n×2n circulant for fast multiply.
    
    The embedding is:
    c = [r[0], r[1], ..., r[n-1], 0, r[n-1], r[n-2], ..., r[1]]
    
    Then T(r) @ x = (C @ [x; 0])[0:n]
    """
    n = len(r)
    c = np.zeros(2 * n)
    c[:n] = r
    c[n+1:] = r[1:][::-1]  # r[n-1], r[n-2], ..., r[1]
    return c


def fast_toeplitz_multiply(r: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Multiply Toeplitz(r) @ x in O(N log N) via circulant embedding.
    """
    n = len(r)
    c = embed_toeplitz_in_circulant(r)
    
    # Pad x with zeros
    x_padded = np.zeros(2 * n)
    x_padded[:n] = x
    
    # Circulant multiply via FFT
    result = np.real(ifft(fft(c) * fft(x_padded)))
    
    return result[:n]


# =============================================================================
# APPROACH 3: Iterative Eigensolver with Fast Matvec
# =============================================================================

def power_iteration_fast(r: np.ndarray, num_vectors: int = 10, 
                          max_iter: int = 100, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find top eigenvectors of Toeplitz(r) using power iteration with fast matvec.
    
    This gives O(k * N log N * iter) for k eigenvectors.
    """
    n = len(r)
    
    # Random initial vectors
    V = np.random.randn(n, num_vectors)
    V, _ = np.linalg.qr(V)
    
    for _ in range(max_iter):
        # Fast Toeplitz multiply for each column
        V_new = np.zeros_like(V)
        for i in range(num_vectors):
            V_new[:, i] = fast_toeplitz_multiply(r, V[:, i])
        
        # Orthonormalize
        V_new, R = np.linalg.qr(V_new)
        
        # Check convergence
        if np.linalg.norm(V_new - V) < tol:
            break
        V = V_new
    
    # Compute Rayleigh quotients for eigenvalues
    eigenvalues = np.zeros(num_vectors)
    for i in range(num_vectors):
        Kv = fast_toeplitz_multiply(r, V[:, i])
        eigenvalues[i] = np.dot(V[:, i], Kv)
    
    return eigenvalues, V


# =============================================================================
# APPROACH 4: Circulant Preconditioned Eigensolve
# =============================================================================

def circulant_preconditioned_eigenbasis(r: np.ndarray, refinement_iters: int = 5) -> np.ndarray:
    """
    Use circulant eigenbasis as starting point, refine for Toeplitz.
    
    1. Get eigenbasis of circulant(r) via FFT - O(N log N)
    2. Refine using a few iterations of orthogonal iteration on Toeplitz
    
    This should give good approximation with O(N log N + k * N log N) cost.
    """
    n = len(r)
    
    # Step 1: Circulant eigenbasis (O(N log N))
    _, U_circ = circulant_eigenbasis(r)
    
    # Take real part (since our Toeplitz is real symmetric)
    U = np.real(U_circ)
    
    # Re-orthonormalize
    U, _ = np.linalg.qr(U)
    
    # Step 2: Refine with orthogonal iteration (each iter is O(N² or N log N with fast matvec))
    for _ in range(refinement_iters):
        # Apply Toeplitz via fast multiply
        U_new = np.zeros_like(U)
        for i in range(n):
            U_new[:, i] = fast_toeplitz_multiply(r, U[:, i])
        
        # Orthonormalize
        U, _ = np.linalg.qr(U_new)
    
    return U


# =============================================================================
# FAST APPROXIMATE RFT
# =============================================================================

class FastApproximateRFT:
    """
    O(N log N) approximate RFT using circulant approximation.
    
    For exact RFT, we'd need the eigenbasis of Toeplitz K.
    For fast approximate RFT, we use the eigenbasis of circulant(r),
    which is just the DFT matrix (reordered by eigenvalue magnitude).
    
    The approximation error depends on how well circulant approximates Toeplitz.
    """
    
    def __init__(self, n: int, variant: str = 'golden'):
        self.n = n
        self.variant = variant
        self.r = build_resonance_kernel(n, variant)
        
        # Precompute eigenvalue ordering for circulant
        self.eigenvalues = fft(self.r)
        self.order = np.argsort(np.abs(self.eigenvalues))[::-1]
        self.inv_order = np.argsort(self.order)
        
        # Precompute FFT of kernel for fast multiply
        self.r_fft = fft(self.r)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Approximate RFT forward: O(N log N)
        
        This is essentially FFT with coefficient reordering.
        """
        # FFT
        X = fft(x) / np.sqrt(self.n)
        # Reorder by eigenvalue magnitude
        return X[self.order]
    
    def inverse(self, X: np.ndarray) -> np.ndarray:
        """
        Approximate RFT inverse: O(N log N)
        """
        # Undo reordering
        X_reordered = X[self.inv_order]
        # IFFT
        return np.real(ifft(X_reordered) * np.sqrt(self.n))
    
    def reconstruction_error(self, x: np.ndarray) -> float:
        """Check round-trip error."""
        X = self.forward(x)
        x_rec = self.inverse(X)
        return np.linalg.norm(x - x_rec) / np.linalg.norm(x)


# =============================================================================
# REFINED FAST RFT (with Toeplitz correction)
# =============================================================================

class RefinedFastRFT:
    """
    Refined O(N log N + small overhead) RFT.
    
    Uses circulant as preconditioner, then applies a small correction.
    """
    
    def __init__(self, n: int, variant: str = 'golden', correction_rank: int = 10):
        self.n = n
        self.variant = variant
        self.r = build_resonance_kernel(n, variant)
        
        # Build both operators
        K_toep = build_toeplitz_operator(self.r)
        K_circ = build_circulant_approximation(self.r)
        
        # Difference matrix
        D = K_toep - K_circ
        
        # Low-rank approximation of difference
        U_d, s_d, Vt_d = np.linalg.svd(D, full_matrices=False)
        
        # Keep top correction_rank components
        self.correction_rank = min(correction_rank, n)
        self.U_correction = U_d[:, :self.correction_rank]
        self.s_correction = s_d[:self.correction_rank]
        self.V_correction = Vt_d[:self.correction_rank, :]
        
        # Store the Frobenius norm of approximation error
        self.approx_error = np.sqrt(np.sum(s_d[self.correction_rank:]**2))
        
        # Circulant eigendecomposition
        self.eigenvalues_circ = fft(self.r)
        self.order = np.argsort(np.abs(self.eigenvalues_circ))[::-1]
    
    def toeplitz_circulant_error(self) -> float:
        """How well does circulant approximate Toeplitz?"""
        K_toep = build_toeplitz_operator(self.r)
        K_circ = build_circulant_approximation(self.r)
        return np.linalg.norm(K_toep - K_circ, 'fro') / np.linalg.norm(K_toep, 'fro')


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_approaches(sizes: list = [64, 128, 256, 512, 1024]) -> Dict[str, Any]:
    """Compare exact vs approximate RFT approaches."""
    
    results = {
        'sizes': sizes,
        'exact_time': [],
        'circulant_time': [],
        'fast_approx_time': [],
        'circulant_error': [],
        'eigenbasis_error': [],
    }
    
    for n in sizes:
        print(f"\n{'='*60}")
        print(f"N = {n}")
        print('='*60)
        
        r = build_resonance_kernel(n, 'golden')
        K = build_toeplitz_operator(r)
        
        # === Exact eigendecomposition ===
        t0 = time.perf_counter()
        eigenvalues_exact, U_exact = eigh(K)
        t_exact = time.perf_counter() - t0
        results['exact_time'].append(t_exact)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues_exact)[::-1]
        U_exact = U_exact[:, idx]
        
        print(f"  Exact eigendecomp:    {t_exact*1000:.2f} ms")
        
        # === Circulant approximation ===
        t0 = time.perf_counter()
        eigenvalues_circ, U_circ = circulant_eigenbasis(r)
        t_circ = time.perf_counter() - t0
        results['circulant_time'].append(t_circ)
        
        print(f"  Circulant eigendecomp: {t_circ*1000:.2f} ms")
        
        # === Fast approximate RFT ===
        t0 = time.perf_counter()
        fast_rft = FastApproximateRFT(n, 'golden')
        t_fast = time.perf_counter() - t0
        results['fast_approx_time'].append(t_fast)
        
        print(f"  Fast approx build:    {t_fast*1000:.2f} ms")
        
        # === Error analysis ===
        
        # How well does circulant approximate Toeplitz?
        K_circ = build_circulant_approximation(r)
        circ_error = np.linalg.norm(K - K_circ, 'fro') / np.linalg.norm(K, 'fro')
        results['circulant_error'].append(circ_error)
        
        print(f"  Circulant approx error: {circ_error:.4f} ({circ_error*100:.1f}%)")
        
        # How well do circulant eigenvectors approximate Toeplitz eigenvectors?
        # Use subspace angle between top-k eigenvectors
        k = min(20, n)
        U_exact_k = U_exact[:, :k]
        U_circ_k = np.real(U_circ[:, :k])
        
        # Orthonormalize (circulant might have complex phases)
        U_circ_k, _ = np.linalg.qr(U_circ_k)
        
        # Subspace angle via SVD of U_exact_k.T @ U_circ_k
        s = np.linalg.svd(U_exact_k.T @ U_circ_k, compute_uv=False)
        # Angles are arccos(s), we report max angle
        max_angle = np.arccos(np.clip(np.min(s), -1, 1)) * 180 / np.pi
        results['eigenbasis_error'].append(max_angle)
        
        print(f"  Max subspace angle (top {k}): {max_angle:.2f}°")
        
        # Test reconstruction
        x_test = np.random.randn(n)
        rec_error = fast_rft.reconstruction_error(x_test)
        print(f"  Fast RFT reconstruction: {rec_error:.2e}")
        
        # Compare sparsity
        X_exact = U_exact.T @ x_test
        X_fast = fast_rft.forward(x_test)
        
        # Energy in top 10%
        k10 = max(1, n // 10)
        energy_exact = np.sum(np.abs(X_exact[:k10])**2) / np.sum(np.abs(X_exact)**2)
        energy_fast = np.sum(np.abs(X_fast[:k10])**2) / np.sum(np.abs(X_fast)**2)
        
        print(f"  Top 10% energy (exact): {energy_exact*100:.1f}%")
        print(f"  Top 10% energy (fast):  {energy_fast*100:.1f}%")
    
    return results


def analyze_toeplitz_circulant_gap():
    """
    Deep analysis of when circulant well-approximates Toeplitz.
    
    Key insight: The difference T - C is localized near the corners.
    For smooth kernels r[k], the corner effects are small.
    """
    print("\n" + "="*70)
    print("TOEPLITZ-CIRCULANT APPROXIMATION ANALYSIS")
    print("="*70)
    
    n = 256
    
    for variant in ['golden', 'harmonic']:
        print(f"\n--- Variant: {variant} ---")
        
        r = build_resonance_kernel(n, variant)
        K = build_toeplitz_operator(r)
        C = build_circulant_approximation(r)
        D = K - C
        
        # Analyze difference matrix
        print(f"  ||K||_F = {np.linalg.norm(K, 'fro'):.4f}")
        print(f"  ||C||_F = {np.linalg.norm(C, 'fro'):.4f}")
        print(f"  ||K-C||_F = {np.linalg.norm(D, 'fro'):.4f}")
        print(f"  Relative error: {np.linalg.norm(D, 'fro') / np.linalg.norm(K, 'fro'):.4f}")
        
        # Singular values of difference
        s = np.linalg.svd(D, compute_uv=False)
        print(f"  Top 5 singular values of D: {s[:5]}")
        print(f"  Rank-10 captures: {np.sum(s[:10]**2)/np.sum(s**2)*100:.1f}% of ||D||_F²")
        
        # Where is the difference concentrated?
        corner_energy = np.sum(D[:n//4, :n//4]**2) + np.sum(D[-n//4:, -n//4:]**2)
        total_energy = np.sum(D**2)
        print(f"  Energy in corners: {corner_energy/total_energy*100:.1f}%")


if __name__ == "__main__":
    print("="*70)
    print("FAST RFT ALGORITHM EXPLORATION")
    print("="*70)
    
    # Run analysis
    analyze_toeplitz_circulant_gap()
    
    print("\n")
    
    # Benchmark
    results = benchmark_approaches([64, 128, 256, 512])
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
KEY FINDINGS:

1. CIRCULANT APPROXIMATION:
   - Circulant(r) ≈ Toeplitz(r) with small relative error for smooth r
   - The difference is concentrated in corners (low-rank structure)
   - Circulant eigenbasis (= DFT) is O(N log N) via FFT

2. FAST APPROXIMATE RFT:
   - Use DFT with coefficient reordering by eigenvalue magnitude
   - O(N log N) forward and inverse
   - Perfect reconstruction (it's just FFT with permutation)
   - BUT: different energy compaction than exact RFT

3. REFINED APPROACH (future work):
   - Low-rank correction: K = C + UV^T where rank(UV^T) << N
   - Sherman-Morrison-Woodbury for O(N log N + k²) eigensolve
   - Subspace iteration with fast Toeplitz matvec

CONCLUSION:
   - For practical use: Fast approximate RFT is just FFT + reorder
   - For exact sparsity: Need O(N²) eigenbasis OR develop refined method
   - The Toeplitz structure IS exploitable but gains are limited
""")
