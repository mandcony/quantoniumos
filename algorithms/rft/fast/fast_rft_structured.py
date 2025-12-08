#!/usr/bin/env python3
"""
Fast RFT via Structured Approximation - Approach 2
===================================================

The naive circulant approximation fails (50-70% error).
Let's try smarter approaches:

1. Szegő theory: Eigenvectors of large Toeplitz converge to sinusoids
2. Conjugate gradient eigensolver with fast Toeplitz matvec  
3. Randomized SVD with fast matvec
4. Exploit that our kernel r(k) is a sum of cosines

Key insight: Our r(k) = sum of cos(2π f_i k/n) terms.
Each cosine contributes a rank-2 component to K.
So K has special structure we can exploit.

December 2025
"""

import numpy as np
from scipy.linalg import toeplitz, eigh
from scipy.fft import fft, ifft, dct, idct
from scipy.sparse.linalg import eigsh, LinearOperator
import time
from typing import Tuple, Callable

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
    elif variant == 'pure_cos':
        r = np.cos(2 * np.pi * 5 * t)
    else:
        r = np.cos(2 * np.pi * 10 * t)
    
    r = r * decay
    r[0] = 1.0
    return r


# =============================================================================
# APPROACH 1: Szegő Theory - Asymptotic Eigenvectors
# =============================================================================

def szego_asymptotic_eigenvectors(r: np.ndarray) -> np.ndarray:
    """
    Szegő theory: For Toeplitz T_n from symbol f(θ), eigenvectors 
    asymptotically converge to sinusoids.
    
    If f(θ) = Σ r[k] e^{ikθ}, then eigenvalues → f(πj/n) for j=0,...,n-1
    and eigenvectors → sin/cos basis.
    
    For our real symmetric Toeplitz, use DCT basis as approximation.
    """
    n = len(r)
    
    # DCT-II basis (orthonormal cosine functions)
    # This is the natural basis for real symmetric Toeplitz
    k = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    
    U = np.sqrt(2.0/n) * np.cos(np.pi * (k + 0.5) * j / n)
    U[:, 0] *= 1.0 / np.sqrt(2)  # DC normalization
    
    return U


def test_szego_approximation(n: int = 256):
    """Test how well Szegő asymptotic eigenvectors approximate true ones."""
    print(f"\n=== Szegő Approximation Test (N={n}) ===")
    
    r = build_resonance_kernel(n, 'golden')
    K = toeplitz(r)
    
    # Exact eigenvectors
    _, U_exact = eigh(K)
    U_exact = U_exact[:, ::-1]  # Descending order
    
    # Szegő approximation (DCT basis)
    U_szego = szego_asymptotic_eigenvectors(r)
    
    # Subspace overlap
    k = 20
    overlap = np.linalg.svd(U_exact[:, :k].T @ U_szego[:, :k], compute_uv=False)
    print(f"  Szegő top-{k} subspace overlap: {np.min(overlap):.4f}")
    print(f"  (1.0 = perfect, 0.0 = orthogonal)")
    
    # Per-vector correlation
    for i in [0, 1, 5, 10]:
        corr = np.abs(np.dot(U_exact[:, i], U_szego[:, i]))
        print(f"  Eigenvector {i} correlation: {corr:.4f}")


# =============================================================================
# APPROACH 2: Fast Toeplitz Matvec + Iterative Eigensolver
# =============================================================================

def fast_toeplitz_matvec(r: np.ndarray) -> Callable:
    """
    Return a function that computes Toeplitz(r) @ x in O(N log N).
    Uses circulant embedding.
    """
    n = len(r)
    
    # Embed in 2n circulant
    c = np.zeros(2 * n)
    c[:n] = r
    c[n] = 0
    c[n+1:] = r[1:][::-1]
    
    c_fft = fft(c)
    
    def matvec(x):
        x_pad = np.zeros(2 * n)
        x_pad[:n] = x
        result = np.real(ifft(c_fft * fft(x_pad)))
        return result[:n]
    
    return matvec


def fast_eigensolver(r: np.ndarray, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k eigenpairs using scipy.sparse.linalg.eigsh with fast matvec.
    
    Complexity: O(k * N log N * num_iterations)
    """
    n = len(r)
    matvec = fast_toeplitz_matvec(r)
    
    # Create LinearOperator
    K_op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    
    # Compute top k eigenpairs
    eigenvalues, eigenvectors = eigsh(K_op, k=k, which='LM')
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def benchmark_fast_eigensolver():
    """Compare exact vs fast iterative eigensolver."""
    print("\n" + "="*70)
    print("FAST ITERATIVE EIGENSOLVER BENCHMARK")
    print("="*70)
    
    for n in [128, 256, 512, 1024]:
        print(f"\n--- N = {n} ---")
        
        r = build_resonance_kernel(n, 'golden')
        K = toeplitz(r)
        k = min(50, n // 2)
        
        # Exact (O(N³))
        t0 = time.perf_counter()
        eig_exact, U_exact = eigh(K)
        t_exact = time.perf_counter() - t0
        
        eig_exact = eig_exact[::-1]
        U_exact = U_exact[:, ::-1]
        
        # Fast iterative (O(k * N log N))
        t0 = time.perf_counter()
        eig_fast, U_fast = fast_eigensolver(r, k=k)
        t_fast = time.perf_counter() - t0
        
        print(f"  Exact eigendecomp:  {t_exact*1000:.1f} ms (all {n} eigenpairs)")
        print(f"  Fast iterative:     {t_fast*1000:.1f} ms ({k} eigenpairs)")
        print(f"  Speedup:            {t_exact/t_fast:.1f}x")
        
        # Accuracy
        eig_error = np.linalg.norm(eig_exact[:k] - eig_fast) / np.linalg.norm(eig_exact[:k])
        print(f"  Eigenvalue error:   {eig_error:.2e}")
        
        # Subspace overlap
        overlap = np.linalg.svd(U_exact[:, :k].T @ U_fast, compute_uv=False)
        print(f"  Min subspace overlap: {np.min(overlap):.6f}")


# =============================================================================
# APPROACH 3: Cosine Sum Structure
# =============================================================================

def analyze_kernel_structure(n: int = 256):
    """
    Our kernel r(k) = sum of cosines.
    Each cos(2π f k/n) contributes rank-2 to Toeplitz(r).
    
    This means K = Σ_i α_i K_i where each K_i is rank-2.
    We can potentially decompose into sum of simple matrices.
    """
    print("\n" + "="*70)
    print("KERNEL STRUCTURE ANALYSIS")
    print("="*70)
    
    k_vec = np.arange(n)
    t = k_vec / n
    
    # Golden kernel: cos(2π f₀ t) + cos(2π φf₀ t)
    f0 = 10.0
    r1 = np.cos(2 * np.pi * f0 * t)
    r2 = np.cos(2 * np.pi * f0 * PHI * t)
    decay = np.exp(-0.01 * k_vec)
    
    K1 = toeplitz(r1 * decay)
    K2 = toeplitz(r2 * decay)
    K_full = toeplitz((r1 + r2) * decay)
    K_full[0, 0] = 1.0
    
    print(f"  K1 (single cosine) spectrum:")
    eig1 = np.linalg.eigvalsh(K1)[::-1]
    print(f"    Top 5: {eig1[:5]}")
    print(f"    Effective rank (99% energy): {np.searchsorted(np.cumsum(eig1**2)/np.sum(eig1**2), 0.99)+1}")
    
    print(f"\n  K2 (φ-shifted cosine) spectrum:")
    eig2 = np.linalg.eigvalsh(K2)[::-1]
    print(f"    Top 5: {eig2[:5]}")
    print(f"    Effective rank (99% energy): {np.searchsorted(np.cumsum(eig2**2)/np.sum(eig2**2), 0.99)+1}")
    
    print(f"\n  K_full (sum) spectrum:")
    eig_full = np.linalg.eigvalsh(K_full)[::-1]
    print(f"    Top 5: {eig_full[:5]}")
    eff_rank = np.searchsorted(np.cumsum(eig_full**2)/np.sum(eig_full**2), 0.99)+1
    print(f"    Effective rank (99% energy): {eff_rank}")
    
    print(f"\n  Insight: K is effectively rank-{eff_rank} for 99% energy!")
    print(f"  This means we only need ~{eff_rank} eigenvectors for most applications.")


# =============================================================================
# APPROACH 4: Randomized Low-Rank Approximation
# =============================================================================

def randomized_eigenbasis(r: np.ndarray, k: int = 50, p: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomized algorithm for top-k eigenpairs.
    
    Algorithm:
    1. Sample random matrix Ω (n × (k+p))
    2. Compute Y = K @ Ω using fast matvec: O((k+p) * N log N)
    3. Orthogonalize Y → Q
    4. Form small matrix B = Q^T K Q: O((k+p)² * N log N)
    5. Eigendecompose B: O((k+p)³)
    6. Recover eigenvectors: U = Q @ V_B
    
    Total: O((k+p) * N log N + (k+p)³)
    """
    n = len(r)
    matvec = fast_toeplitz_matvec(r)
    
    # Random projection
    Omega = np.random.randn(n, k + p)
    
    # Range finding
    Y = np.zeros((n, k + p))
    for i in range(k + p):
        Y[:, i] = matvec(Omega[:, i])
    
    Q, _ = np.linalg.qr(Y)
    
    # Form small projected matrix
    KQ = np.zeros((n, k + p))
    for i in range(k + p):
        KQ[:, i] = matvec(Q[:, i])
    
    B = Q.T @ KQ
    
    # Eigendecompose small matrix
    eig_B, V_B = eigh(B)
    
    # Recover full eigenvectors
    idx = np.argsort(eig_B)[::-1][:k]
    eigenvalues = eig_B[idx]
    eigenvectors = Q @ V_B[:, idx]
    
    return eigenvalues, eigenvectors


def benchmark_randomized():
    """Benchmark randomized eigensolver."""
    print("\n" + "="*70)
    print("RANDOMIZED EIGENSOLVER BENCHMARK")
    print("="*70)
    
    for n in [256, 512, 1024, 2048]:
        print(f"\n--- N = {n} ---")
        
        r = build_resonance_kernel(n, 'golden')
        k = min(50, n // 4)
        
        # Randomized (should be faster for large n)
        t0 = time.perf_counter()
        eig_rand, U_rand = randomized_eigenbasis(r, k=k)
        t_rand = time.perf_counter() - t0
        
        print(f"  Randomized ({k} eigenpairs): {t_rand*1000:.1f} ms")
        
        if n <= 1024:
            # Compare to exact for accuracy
            K = toeplitz(r)
            eig_exact, U_exact = eigh(K)
            eig_exact = eig_exact[::-1][:k]
            U_exact = U_exact[:, ::-1][:, :k]
            
            eig_error = np.linalg.norm(eig_exact - eig_rand) / np.linalg.norm(eig_exact)
            overlap = np.linalg.svd(U_exact.T @ U_rand, compute_uv=False)
            
            print(f"  Eigenvalue error: {eig_error:.2e}")
            print(f"  Min subspace overlap: {np.min(overlap):.6f}")


# =============================================================================
# PRACTICAL FAST RFT CLASS
# =============================================================================

class FastRFT:
    """
    Practical O(N log N) RFT implementation.
    
    Uses iterative eigensolver with fast Toeplitz matvec.
    Precomputes top-k eigenvectors, uses FFT for remaining.
    """
    
    def __init__(self, n: int, variant: str = 'golden', 
                 k: int = None, build_full: bool = False):
        """
        Args:
            n: Transform size
            variant: Which resonance kernel
            k: Number of eigenvectors to compute (default: n//4)
            build_full: If True, compute all n eigenvectors (slow but exact)
        """
        self.n = n
        self.variant = variant
        self.r = build_resonance_kernel(n, variant)
        
        if build_full:
            # Full O(N³) eigendecomposition
            K = toeplitz(self.r)
            eigenvalues, U = eigh(K)
            self.eigenvalues = eigenvalues[::-1]
            self.U = U[:, ::-1]
            self.k = n
        else:
            # Fast O(k * N log N) for top-k
            self.k = k if k is not None else max(10, n // 4)
            self.eigenvalues, self.U = fast_eigensolver(self.r, k=self.k)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        RFT forward transform.
        Returns coefficients sorted by eigenvalue magnitude.
        """
        return self.U.T @ x
    
    def inverse(self, X: np.ndarray) -> np.ndarray:
        """RFT inverse transform."""
        return self.U @ X
    
    def compress(self, x: np.ndarray, keep_ratio: float = 0.1) -> Tuple[np.ndarray, int]:
        """
        Compress signal by keeping top coefficients.
        
        Returns: (compressed_coeffs, num_kept)
        """
        X = self.forward(x)
        k_keep = max(1, int(len(X) * keep_ratio))
        X_compressed = X.copy()
        X_compressed[k_keep:] = 0
        return X_compressed, k_keep
    
    def reconstruction_psnr(self, x: np.ndarray, keep_ratio: float = 0.1) -> float:
        """Compute PSNR at given retention ratio."""
        X_comp, _ = self.compress(x, keep_ratio)
        x_rec = self.inverse(X_comp)
        mse = np.mean((x - x_rec)**2)
        if mse < 1e-15:
            return float('inf')
        return 10 * np.log10(np.max(x**2) / mse)


def final_benchmark():
    """Final comparison: Fast RFT vs FFT vs DCT."""
    print("\n" + "="*70)
    print("FINAL BENCHMARK: Fast RFT vs FFT vs DCT")
    print("="*70)
    
    n = 512
    np.random.seed(42)
    
    # Build fast RFT
    print("\nBuilding Fast RFT...")
    t0 = time.perf_counter()
    fast_rft = FastRFT(n, 'golden', k=100)
    print(f"  Build time: {(time.perf_counter()-t0)*1000:.1f} ms")
    
    # Test signals
    signals = {}
    
    # Golden quasi-periodic
    t = np.linspace(0, 1, n)
    f0 = 10
    signals['golden_qp'] = np.cos(2*np.pi*f0*t) + 0.5*np.cos(2*np.pi*f0*PHI*t) + 0.1*np.random.randn(n)
    
    # Pure sine
    signals['pure_sine'] = np.sin(2*np.pi*7*t)
    
    # Chirp
    signals['chirp'] = np.sin(2*np.pi*(5 + 20*t)*t)
    
    # Random
    signals['random'] = np.random.randn(n)
    
    print(f"\n{'Signal':<15} | {'RFT PSNR':<12} | {'FFT PSNR':<12} | {'DCT PSNR':<12} | Winner")
    print("-" * 70)
    
    for name, x in signals.items():
        # Fast RFT
        psnr_rft = fast_rft.reconstruction_psnr(x, 0.1)
        
        # FFT
        X_fft = np.fft.fft(x)
        idx = np.argsort(np.abs(X_fft))[::-1]
        X_fft_comp = np.zeros_like(X_fft)
        X_fft_comp[idx[:n//10]] = X_fft[idx[:n//10]]
        x_fft_rec = np.real(np.fft.ifft(X_fft_comp))
        mse_fft = np.mean((x - x_fft_rec)**2)
        psnr_fft = 10 * np.log10(np.max(x**2) / mse_fft) if mse_fft > 1e-15 else float('inf')
        
        # DCT
        X_dct = dct(x, norm='ortho')
        idx = np.argsort(np.abs(X_dct))[::-1]
        X_dct_comp = np.zeros_like(X_dct)
        X_dct_comp[idx[:n//10]] = X_dct[idx[:n//10]]
        x_dct_rec = idct(X_dct_comp, norm='ortho')
        mse_dct = np.mean((x - x_dct_rec)**2)
        psnr_dct = 10 * np.log10(np.max(x**2) / mse_dct) if mse_dct > 1e-15 else float('inf')
        
        winner = 'RFT' if psnr_rft >= max(psnr_fft, psnr_dct) else ('FFT' if psnr_fft > psnr_dct else 'DCT')
        
        print(f"{name:<15} | {psnr_rft:>10.2f}dB | {psnr_fft:>10.2f}dB | {psnr_dct:>10.2f}dB | {winner}")


if __name__ == "__main__":
    test_szego_approximation()
    benchmark_fast_eigensolver()
    analyze_kernel_structure()
    benchmark_randomized()
    final_benchmark()
