# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Fast RFT Projection Benchmark
=============================

Goal
----
1) Project onto top-k manifold eigenvectors using Toeplitz–FFT matvec inside
   Lanczos (scipy.sparse.linalg.eigsh with LinearOperator).
2) Measure wall-time scaling vs. naïve dense N^2 multiply/eigsh.
3) Probe blockwise low-rank structure of Φ (manifold_projection) as a first step
   toward butterfly / low-rank factorizations.

This script is intentionally lightweight (no PSNR tables). It focuses on runtime.

December 2025.
"""

import time
import numpy as np
from scipy.linalg import toeplitz
from scipy.fft import fft, ifft
from scipy.sparse.linalg import eigsh, LinearOperator

import sys
sys.path.insert(0, "/workspaces/quantoniumos")

from algorithms.rft.variants.patent_variants import generate_rft_manifold_projection

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

PHI = (1 + np.sqrt(5)) / 2


def build_r_manifold(n: int) -> np.ndarray:
    """Recreate the manifold_projection autocorrelation first column r."""
    k = np.arange(n)
    t = k / n
    u = 2 * np.pi * t
    v = 2 * np.pi * PHI * t
    twist = PHI * u
    x = (2 + np.cos(v + twist)) * np.cos(u)
    y = (2 + np.cos(v + twist)) * np.sin(u)
    z = np.sin(v + twist)
    r = x + 0.3 * y + 0.1 * z
    r = r / (np.max(np.abs(r)) + 1e-10)
    return r


def make_toeplitz_fft_operator(r: np.ndarray) -> LinearOperator:
    """Create LinearOperator for symmetric Toeplitz matvec via FFT (O(N log N))."""
    n = len(r)
    # Circulant embedding length 2n
    emb = np.concatenate([r, [0.0], r[:0:-1]])
    fft_emb = fft(emb)
    fft_len = emb.size

    def matvec(x: np.ndarray) -> np.ndarray:
        vec = np.zeros(fft_len, dtype=float)
        vec[:n] = x
        y = ifft(fft_emb * fft(vec)).real[:n]
        return y

    return LinearOperator((n, n), matvec=matvec, rmatvec=matvec, dtype=float)


def project_topk(operator, x: np.ndarray, k: int, tol: float = 1e-6):
    """Compute top-k eigenvectors (largest magnitude) and project x onto them."""
    w, V = eigsh(operator, k=k, which="LM", tol=tol)
    coeffs = V.T @ x
    x_rec = V @ coeffs
    return w, V, coeffs, x_rec


def time_matvec(op, n_runs: int = 3):
    """Average matvec time for random vectors."""
    rng = np.random.default_rng(0)
    times = []
    for _ in range(n_runs):
        x = rng.standard_normal(op.shape[0])
        t0 = time.perf_counter()
        _ = op @ x
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------

def benchmark_projection(ns=(512, 1024, 2048, 4096), k=16):
    print("=" * 90)
    print("PROJECTION WITH TOEPLITZ–FFT LANCZOS VS DENSE N^2")
    print("=" * 90)
    print(f"k={k} leading eigenvectors (which='LM')")
    print()
    header = f"{'N':>6}  {'build_dense(s)':>14} {'eig_dense(s)':>12} {'eig_fft(s)':>10} {'mv_dense(ms)':>12} {'mv_fft(ms)':>11} {'rel_err λ':>10}"
    print(header)
    print("-" * len(header))

    rng = np.random.default_rng(123)

    for n in ns:
        r = build_r_manifold(n)

        # Dense Toeplitz matrix
        build_dense = np.nan
        eig_dense = np.nan
        mv_dense_ms = np.nan
        rel_err = np.nan

        if n <= 2048:  # avoid huge dense costs
            t0 = time.perf_counter()
            K = toeplitz(r)
            build_dense = time.perf_counter() - t0

            # Dense matvec timing
            mv_dense_ms = time_matvec(K) * 1e3

            t0 = time.perf_counter()
            w_dense, V_dense, _, _ = project_topk(K, rng.standard_normal(n), k)
            eig_dense = time.perf_counter() - t0
        else:
            K = None

        # FFT-based LinearOperator
        op_fft = make_toeplitz_fft_operator(r)
        mv_fft_ms = time_matvec(op_fft) * 1e3
        t0 = time.perf_counter()
        w_fft, _, _, _ = project_topk(op_fft, rng.standard_normal(n), k)
        eig_fft = time.perf_counter() - t0

        # Compare eigenvalues when dense available
        if n <= 2048:
            rel_err = np.linalg.norm(np.sort(w_dense) - np.sort(w_fft)) / (np.linalg.norm(w_dense) + 1e-12)

        print(f"{n:6d}  {build_dense:14.3f} {eig_dense:12.3f} {eig_fft:10.3f} {mv_dense_ms:12.2f} {mv_fft_ms:11.2f} {rel_err:10.3e}")

    print()


def investigate_block_low_rank(n=512, blocks=8, rank=8):
    """Probe blockwise low-rank structure of Φ for manifold_projection."""
    print("=" * 90)
    print("BLOCKWISE LOW-RANK PROBE OF Φ (manifold_projection)")
    print("=" * 90)
    print(f"N={n}, blocks={blocks}x{blocks}, target rank={rank}")
    Phi = generate_rft_manifold_projection(n)
    bsz = n // blocks
    errs = []
    ranks_eff = []

    for i in range(blocks):
        for j in range(blocks):
            block = Phi[i*bsz:(i+1)*bsz, j*bsz:(j+1)*bsz]
            # SVD-based rank-r approximation
            u, s, vt = np.linalg.svd(block, full_matrices=False)
            s_trunc = s[:rank]
            approx = (u[:, :rank] * s_trunc) @ vt[:rank, :]
            err = np.linalg.norm(block - approx, ord='fro') / (np.linalg.norm(block, ord='fro') + 1e-12)
            errs.append(err)
            # effective rank at 1e-3 relative tolerance
            thr = 1e-3 * s[0]
            ranks_eff.append(np.sum(s > thr))

    errs = np.array(errs)
    ranks_eff = np.array(ranks_eff)
    print(f"Mean rel error (rank-{rank} blocks): {errs.mean():.3f}")
    print(f"Max  rel error (rank-{rank} blocks): {errs.max():.3f}")
    print(f"Median rel error: {np.median(errs):.3f}")
    print(f"Effective rank @1e-3 thresh (mean): {ranks_eff.mean():.1f}; max: {ranks_eff.max()}")
    print("Observation: if errors/ranks stay low, Φ is blockwise compressible → candidate for butterfly/low-rank factorization.")
    print()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    benchmark_projection()
    investigate_block_low_rank()
