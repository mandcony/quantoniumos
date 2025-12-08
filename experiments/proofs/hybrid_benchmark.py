#!/usr/bin/env python3
"""
Reproducible Benchmark: DCT+RFT Hybrid vs Standard Transforms

Goal: Find a measurable application win for the hybrid decomposition.

Candidate applications:
1. Signal compression (PSNR at given bit budget)
2. Denoising (SNR improvement)
3. Sparse representation (coefficient count at given fidelity)

Key insight from sparsity_theorem.py:
- RFT has IDENTICAL sparsity to DFT (they differ only in phase)
- DCT is often better than both for smooth signals
- The hybrid might help for signals with BOTH smooth and oscillatory components

Test signals:
- Synthetic: piecewise smooth + quasi-periodic texture
- Real: audio samples with harmonic content
"""

import numpy as np
from scipy.fft import fft, ifft, dct, idct
from scipy.io import wavfile
import os
import json
from datetime import datetime

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


def build_closed_form_rft(n, beta=1.0, sigma=1.0):
    """Build the closed-form RFT matrix Ψ = D_φ C_σ F"""
    k = np.arange(n)
    F = np.exp(-2j * np.pi * np.outer(k, k) / n) / np.sqrt(n)
    C_sigma = np.diag(np.exp(1j * np.pi * sigma * k**2 / n))
    frac_parts = (k / PHI) % 1.0
    D_phi = np.diag(np.exp(2j * np.pi * beta * frac_parts))
    return D_phi @ C_sigma @ F


# =============================================================================
# Sparse Approximation Functions
# =============================================================================

def sparse_approx_dct(x, k):
    """Keep top-k DCT coefficients."""
    c = dct(x, norm='ortho')
    idx = np.argsort(np.abs(c))[::-1]
    c_sparse = np.zeros_like(c)
    c_sparse[idx[:k]] = c[idx[:k]]
    return idct(c_sparse, norm='ortho')


def sparse_approx_dft(x, k):
    """Keep top-k DFT coefficients."""
    c = fft(x) / np.sqrt(len(x))
    idx = np.argsort(np.abs(c))[::-1]
    c_sparse = np.zeros_like(c)
    c_sparse[idx[:k]] = c[idx[:k]]
    return np.real(ifft(c_sparse * np.sqrt(len(x))))


def sparse_approx_hybrid(x, k1, k2):
    """
    Hybrid DCT + RFT approximation.
    
    1. Keep top-k1 DCT coefficients for structure
    2. Compute residual
    3. Keep top-k2 DFT/RFT coefficients of residual for texture
    
    Note: Since |RFT| = |DFT|, we use DFT for efficiency.
    """
    n = len(x)
    
    # Step 1: DCT approximation
    c_dct = dct(x, norm='ortho')
    idx_dct = np.argsort(np.abs(c_dct))[::-1]
    c_dct_sparse = np.zeros_like(c_dct)
    c_dct_sparse[idx_dct[:k1]] = c_dct[idx_dct[:k1]]
    x_struct = idct(c_dct_sparse, norm='ortho')
    
    # Step 2: Residual
    r = x - x_struct
    
    # Step 3: DFT approximation of residual
    c_dft = fft(r) / np.sqrt(n)
    idx_dft = np.argsort(np.abs(c_dft))[::-1]
    c_dft_sparse = np.zeros_like(c_dft)
    c_dft_sparse[idx_dft[:k2]] = c_dft[idx_dft[:k2]]
    x_texture = np.real(ifft(c_dft_sparse * np.sqrt(n)))
    
    return x_struct + x_texture


# =============================================================================
# Test Signals
# =============================================================================

def generate_piecewise_smooth_plus_texture(n, num_pieces=4, texture_freqs=5, noise_std=0.0):
    """
    Generate a signal with:
    - Piecewise polynomial smooth structure (DCT-friendly)
    - Quasi-periodic texture at irrational frequencies (not DCT-friendly)
    """
    t = np.linspace(0, 1, n)
    
    # Piecewise smooth structure
    x_struct = np.zeros(n)
    piece_len = n // num_pieces
    for i in range(num_pieces):
        start = i * piece_len
        end = (i + 1) * piece_len if i < num_pieces - 1 else n
        # Random polynomial of degree 2
        coeffs = np.random.randn(3)
        t_local = np.linspace(0, 1, end - start)
        x_struct[start:end] = coeffs[0] + coeffs[1] * t_local + coeffs[2] * t_local**2
    
    # Quasi-periodic texture at golden-ratio frequencies
    x_texture = np.zeros(n)
    for j in range(1, texture_freqs + 1):
        freq = (j * PHI) % 1.0
        amp = 0.3 / j  # Decaying amplitude
        phase = np.random.uniform(0, 2 * np.pi)
        x_texture += amp * np.sin(2 * np.pi * freq * n * t + phase)
    
    # Combine
    x = x_struct + x_texture
    
    if noise_std > 0:
        x += noise_std * np.random.randn(n)
    
    return x, x_struct, x_texture


def generate_audio_like_signal(n, fundamental=440, num_harmonics=8):
    """
    Generate an audio-like signal with:
    - Harmonic series (DFT-friendly)
    - Amplitude envelope (DCT-friendly)
    """
    t = np.linspace(0, 1, n)
    
    # Harmonic content
    x = np.zeros(n)
    for h in range(1, num_harmonics + 1):
        freq = fundamental * h / n
        amp = 1.0 / h  # Natural harmonic decay
        x += amp * np.sin(2 * np.pi * freq * np.arange(n))
    
    # Amplitude envelope (smooth)
    envelope = np.exp(-2 * t) * np.sin(np.pi * t)**2
    x *= envelope
    
    return x


# =============================================================================
# Metrics
# =============================================================================

def psnr(original, reconstructed):
    """Peak Signal-to-Noise Ratio in dB."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse < 1e-15:
        return 100.0  # Perfect reconstruction
    max_val = np.max(np.abs(original))
    return 20 * np.log10(max_val / np.sqrt(mse))


def compression_ratio(n, k):
    """Compression ratio = original size / compressed size."""
    return n / k


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_signal(x, name, k_values):
    """Run benchmark on a single signal."""
    n = len(x)
    results = {'signal': name, 'n': n, 'methods': {}}
    
    for method in ['dct', 'dft', 'hybrid']:
        results['methods'][method] = []
        
        for k in k_values:
            if method == 'dct':
                x_approx = sparse_approx_dct(x, k)
            elif method == 'dft':
                x_approx = sparse_approx_dft(x, k)
            else:  # hybrid
                # Split budget: more to DCT since it's typically better for structure
                k1 = int(0.6 * k)
                k2 = k - k1
                x_approx = sparse_approx_hybrid(x, k1, k2)
            
            p = psnr(x, x_approx)
            cr = compression_ratio(n, k)
            
            results['methods'][method].append({
                'k': k,
                'psnr': float(p),
                'compression_ratio': float(cr)
            })
    
    return results


def run_full_benchmark():
    """Run complete benchmark suite."""
    print("=" * 70)
    print("REPRODUCIBLE BENCHMARK: DCT+RFT HYBRID vs STANDARD TRANSFORMS")
    print("=" * 70)
    print()
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'phi': float(PHI),
        'benchmarks': []
    }
    
    # Test configurations
    n = 512
    k_values = [8, 16, 32, 64, 128]
    
    # Signal 1: Piecewise smooth + golden texture
    print("Signal 1: Piecewise smooth + quasi-periodic texture")
    np.random.seed(42)
    x1, x1_struct, x1_texture = generate_piecewise_smooth_plus_texture(
        n, num_pieces=4, texture_freqs=5
    )
    results1 = benchmark_signal(x1, 'piecewise_smooth_plus_texture', k_values)
    all_results['benchmarks'].append(results1)
    
    print(f"  n={n}, structure energy: {np.sum(x1_struct**2):.2f}, texture energy: {np.sum(x1_texture**2):.2f}")
    print()
    print("  k   | DCT PSNR | DFT PSNR | Hybrid PSNR | Winner")
    print("  " + "-" * 55)
    for i, k in enumerate(k_values):
        dct_p = results1['methods']['dct'][i]['psnr']
        dft_p = results1['methods']['dft'][i]['psnr']
        hyb_p = results1['methods']['hybrid'][i]['psnr']
        winner = 'DCT' if dct_p >= max(dft_p, hyb_p) else ('DFT' if dft_p >= hyb_p else 'HYBRID')
        print(f"  {k:3d} | {dct_p:8.2f} | {dft_p:8.2f} | {hyb_p:11.2f} | {winner}")
    print()
    
    # Signal 2: Audio-like harmonic signal
    print("Signal 2: Audio-like harmonic signal with envelope")
    x2 = generate_audio_like_signal(n, fundamental=440, num_harmonics=8)
    results2 = benchmark_signal(x2, 'audio_harmonic', k_values)
    all_results['benchmarks'].append(results2)
    
    print(f"  n={n}")
    print()
    print("  k   | DCT PSNR | DFT PSNR | Hybrid PSNR | Winner")
    print("  " + "-" * 55)
    for i, k in enumerate(k_values):
        dct_p = results2['methods']['dct'][i]['psnr']
        dft_p = results2['methods']['dft'][i]['psnr']
        hyb_p = results2['methods']['hybrid'][i]['psnr']
        winner = 'DCT' if dct_p >= max(dft_p, hyb_p) else ('DFT' if dft_p >= hyb_p else 'HYBRID')
        print(f"  {k:3d} | {dct_p:8.2f} | {dft_p:8.2f} | {hyb_p:11.2f} | {winner}")
    print()
    
    # Signal 3: Pure random signal (baseline - no method should dominate)
    print("Signal 3: Random signal (baseline)")
    np.random.seed(123)
    x3 = np.random.randn(n)
    results3 = benchmark_signal(x3, 'random', k_values)
    all_results['benchmarks'].append(results3)
    
    print(f"  n={n}")
    print()
    print("  k   | DCT PSNR | DFT PSNR | Hybrid PSNR | Winner")
    print("  " + "-" * 55)
    for i, k in enumerate(k_values):
        dct_p = results3['methods']['dct'][i]['psnr']
        dft_p = results3['methods']['dft'][i]['psnr']
        hyb_p = results3['methods']['hybrid'][i]['psnr']
        winner = 'DCT' if dct_p >= max(dft_p, hyb_p) else ('DFT' if dft_p >= hyb_p else 'HYBRID')
        print(f"  {k:3d} | {dct_p:8.2f} | {dft_p:8.2f} | {hyb_p:11.2f} | {winner}")
    print()
    
    # Signal 4: Mixed signal designed to favor hybrid
    print("Signal 4: Designed mixed signal (50% smooth polynomial, 50% pure tone)")
    t = np.linspace(0, 1, n)
    x4_smooth = 3 * t**2 - 2 * t**3  # Smooth cubic
    x4_tone = 0.5 * np.sin(2 * np.pi * 50 * t)  # Pure sinusoid
    x4 = x4_smooth + x4_tone
    results4 = benchmark_signal(x4, 'smooth_plus_tone', k_values)
    all_results['benchmarks'].append(results4)
    
    print(f"  n={n}")
    print()
    print("  k   | DCT PSNR | DFT PSNR | Hybrid PSNR | Winner")
    print("  " + "-" * 55)
    for i, k in enumerate(k_values):
        dct_p = results4['methods']['dct'][i]['psnr']
        dft_p = results4['methods']['dft'][i]['psnr']
        hyb_p = results4['methods']['hybrid'][i]['psnr']
        winner = 'DCT' if dct_p >= max(dft_p, hyb_p) else ('DFT' if dft_p >= hyb_p else 'HYBRID')
        print(f"  {k:3d} | {dct_p:8.2f} | {dft_p:8.2f} | {hyb_p:11.2f} | {winner}")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    # Count wins
    wins = {'dct': 0, 'dft': 0, 'hybrid': 0}
    for bench in all_results['benchmarks']:
        for i in range(len(k_values)):
            scores = {m: bench['methods'][m][i]['psnr'] for m in ['dct', 'dft', 'hybrid']}
            winner = max(scores, key=scores.get)
            wins[winner] += 1
    
    total = sum(wins.values())
    print(f"Win counts across all signals and k values:")
    print(f"  DCT:    {wins['dct']:3d} / {total} ({100*wins['dct']/total:.1f}%)")
    print(f"  DFT:    {wins['dft']:3d} / {total} ({100*wins['dft']/total:.1f}%)")
    print(f"  Hybrid: {wins['hybrid']:3d} / {total} ({100*wins['hybrid']/total:.1f}%)")
    print()
    
    # Save results
    output_path = '/workspaces/quantoniumos/experiments/proofs/benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {output_path}")
    
    return all_results


if __name__ == "__main__":
    results = run_full_benchmark()
