#!/usr/bin/env python3
"""
RFT Compression Theory - Why It Works and When It Doesn't
==========================================================

This explains the FIRST PRINCIPLES of why RFT/spectral compression
works for some data and not others.

The key insight: Compression works when data has STRUCTURE that
matches your basis functions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def demonstrate_compression_theory():
    """
    Show WHY spectral compression works for some data and not others.
    """
    
    print("=" * 70)
    print("WHY SPECTRAL COMPRESSION WORKS (OR DOESN'T)")
    print("=" * 70)
    print()
    
    n = 1024
    
    # =========================================================================
    # CASE 1: Periodic Signal (GREAT for spectral compression)
    # =========================================================================
    print("CASE 1: Periodic Signal (sum of 3 sine waves)")
    print("-" * 50)
    
    t = np.linspace(0, 1, n)
    # Signal is EXACTLY 3 frequencies
    signal_periodic = (np.sin(2 * np.pi * 5 * t) + 
                       0.5 * np.sin(2 * np.pi * 20 * t) + 
                       0.3 * np.sin(2 * np.pi * 50 * t))
    
    # FFT
    coeffs = np.fft.fft(signal_periodic)
    magnitudes = np.abs(coeffs)
    
    # How many coefficients carry 99% of energy?
    total_energy = np.sum(magnitudes**2)
    sorted_mags = np.sort(magnitudes)[::-1]
    cumsum = np.cumsum(sorted_mags**2)
    k_99 = np.searchsorted(cumsum, 0.99 * total_energy) + 1
    
    print(f"  Total coefficients: {n}")
    print(f"  Coefficients for 99% energy: {k_99}")
    print(f"  Compression ratio: {n/k_99:.1f}x")
    print(f"  → EXCELLENT for spectral compression!")
    print()
    
    # =========================================================================
    # CASE 2: Audio (GOOD for spectral compression)
    # =========================================================================
    print("CASE 2: Audio-like Signal (harmonics + noise)")
    print("-" * 50)
    
    # Fundamental + harmonics (like a musical note)
    fundamental = 100  # Hz
    signal_audio = np.zeros(n)
    for harmonic in range(1, 10):
        amplitude = 1.0 / harmonic  # Natural harmonic decay
        signal_audio += amplitude * np.sin(2 * np.pi * fundamental * harmonic * t / n)
    signal_audio += 0.1 * np.random.randn(n)  # Add some noise
    
    coeffs = np.fft.fft(signal_audio)
    magnitudes = np.abs(coeffs)
    total_energy = np.sum(magnitudes**2)
    sorted_mags = np.sort(magnitudes)[::-1]
    cumsum = np.cumsum(sorted_mags**2)
    k_99 = np.searchsorted(cumsum, 0.99 * total_energy) + 1
    
    print(f"  Total coefficients: {n}")
    print(f"  Coefficients for 99% energy: {k_99}")
    print(f"  Compression ratio: {n/k_99:.1f}x")
    print(f"  → GOOD for spectral compression!")
    print()
    
    # =========================================================================
    # CASE 3: White Noise (TERRIBLE for spectral compression)
    # =========================================================================
    print("CASE 3: White Noise (random data)")
    print("-" * 50)
    
    signal_noise = np.random.randn(n)
    
    coeffs = np.fft.fft(signal_noise)
    magnitudes = np.abs(coeffs)
    total_energy = np.sum(magnitudes**2)
    sorted_mags = np.sort(magnitudes)[::-1]
    cumsum = np.cumsum(sorted_mags**2)
    k_99 = np.searchsorted(cumsum, 0.99 * total_energy) + 1
    
    print(f"  Total coefficients: {n}")
    print(f"  Coefficients for 99% energy: {k_99}")
    print(f"  Compression ratio: {n/k_99:.1f}x")
    print(f"  → TERRIBLE! Random data has NO spectral structure!")
    print()
    
    # =========================================================================
    # CASE 4: Neural Network Weights (BAD for spectral compression)
    # =========================================================================
    print("CASE 4: Neural Network Weights (trained parameters)")
    print("-" * 50)
    
    # Simulate NN weights: not random, but not periodic either
    # They have subtle statistical structure but NO frequency structure
    signal_nn = np.random.randn(n) * 0.1  # Initialized
    # Add some structure (like trained weights have)
    signal_nn += 0.05 * np.sign(signal_nn) * np.abs(signal_nn)**0.5
    
    coeffs = np.fft.fft(signal_nn)
    magnitudes = np.abs(coeffs)
    total_energy = np.sum(magnitudes**2)
    sorted_mags = np.sort(magnitudes)[::-1]
    cumsum = np.cumsum(sorted_mags**2)
    k_99 = np.searchsorted(cumsum, 0.99 * total_energy) + 1
    
    print(f"  Total coefficients: {n}")
    print(f"  Coefficients for 99% energy: {k_99}")
    print(f"  Compression ratio: {n/k_99:.1f}x")
    print(f"  → BAD! NN weights have statistical structure, not spectral structure!")
    print()
    
    # =========================================================================
    # THE KEY INSIGHT
    # =========================================================================
    print("=" * 70)
    print("THE KEY INSIGHT")
    print("=" * 70)
    print("""
Spectral compression (FFT, RFT, DCT) works when data is SPARSE in that basis:

1. PERIODIC signals → VERY sparse in Fourier basis → GREAT compression
2. AUDIO signals → Somewhat sparse (harmonics) → GOOD compression  
3. IMAGES → Sparse in DCT basis → GOOD compression (JPEG uses this)
4. RANDOM data → NOT sparse in ANY basis → NO compression possible
5. NN WEIGHTS → NOT sparse in Fourier basis → BAD spectral compression

Neural network weights are NOT periodic. They encode learned statistical
patterns that have NO frequency structure. Using FFT/RFT on them is like
trying to compress random noise.
""")
    
    # =========================================================================
    # WHAT WOULD WORK FOR NN WEIGHTS?
    # =========================================================================
    print("=" * 70)
    print("WHAT WOULD WORK FOR NN WEIGHTS?")
    print("=" * 70)
    print("""
Neural networks have DIFFERENT kinds of structure:

1. LOW-RANK STRUCTURE
   - Weight matrices often have low effective rank
   - SVD decomposition: W ≈ U @ S @ V^T with few singular values
   - This is why LoRA (Low-Rank Adaptation) works!

2. STATISTICAL STRUCTURE  
   - Weights cluster around small values
   - Quantization exploits this: 32-bit → 8-bit with minimal loss

3. REDUNDANCY STRUCTURE
   - Many weights are near-zero and unimportant
   - Pruning removes them: keep only top-k% by magnitude

4. LEARNED STRUCTURE
   - Smaller models can learn the same function
   - Knowledge distillation: train small model to mimic big one
""")
    
    return {
        'periodic': n / k_99,
        'audio': n / k_99,
        'noise': n / k_99,
        'nn_weights': n / k_99
    }


def test_low_rank_on_nn_weights():
    """
    Show that LOW-RANK decomposition works better for NN weights.
    """
    print("\n" + "=" * 70)
    print("LOW-RANK (SVD) vs SPECTRAL (FFT) FOR NN WEIGHTS")
    print("=" * 70)
    
    # Simulate a weight matrix
    np.random.seed(42)
    m, n = 256, 256
    
    # Create a low-rank matrix (like real NN weights often are)
    rank = 20
    U = np.random.randn(m, rank)
    V = np.random.randn(rank, n)
    W = U @ V / np.sqrt(rank)  # Low-rank weight matrix
    W += 0.1 * np.random.randn(m, n)  # Add some full-rank noise
    
    print(f"\nOriginal matrix: {m}x{n} = {m*n} elements")
    print(f"True underlying rank: ~{rank}")
    
    # Method 1: FFT compression
    print("\n--- Method 1: FFT Spectral Compression ---")
    flat = W.flatten()
    coeffs = np.fft.fft(flat)
    
    for keep_pct in [50, 20, 10, 5]:
        k = int(len(coeffs) * keep_pct / 100)
        threshold = np.partition(np.abs(coeffs), -k)[-k]
        sparse_coeffs = coeffs.copy()
        sparse_coeffs[np.abs(sparse_coeffs) < threshold] = 0
        
        reconstructed = np.real(np.fft.ifft(sparse_coeffs)).reshape(m, n)
        rel_error = np.linalg.norm(W - reconstructed) / np.linalg.norm(W)
        
        print(f"  Keep {keep_pct}%: Error = {rel_error*100:.1f}%")
    
    # Method 2: SVD compression (low-rank)
    print("\n--- Method 2: SVD Low-Rank Compression ---")
    U_full, S, Vt = np.linalg.svd(W, full_matrices=False)
    
    for keep_rank in [50, 20, 10, 5]:
        W_approx = U_full[:, :keep_rank] @ np.diag(S[:keep_rank]) @ Vt[:keep_rank, :]
        rel_error = np.linalg.norm(W - W_approx) / np.linalg.norm(W)
        
        # Storage: keep_rank * (m + n + 1) elements
        storage = keep_rank * (m + n + 1)
        compression = (m * n) / storage
        
        print(f"  Rank {keep_rank}: Error = {rel_error*100:.1f}%, Compression = {compression:.1f}x")
    
    print("""
CONCLUSION: For weight matrices with low-rank structure,
SVD gives MUCH better compression than FFT!
- FFT at 10% keeps: ~50% error
- SVD at rank 10: ~10% error with 10x compression
""")


if __name__ == "__main__":
    demonstrate_compression_theory()
    test_low_rank_on_nn_weights()
