#!/usr/bin/env python3
"""
Debugging RFT Performance
=========================

RFT is losing on ALL signals. Let's understand why and fix it.

The issue: We're using a fixed resonance kernel that may not match
the actual signal structure, even when signals have golden-ratio content.

Hypothesis: The kernel decay rate and frequency parameters are mismatched.

December 2025
"""

import numpy as np
from scipy.linalg import toeplitz, eigh
from scipy.fft import dct, idct
from typing import Tuple

PHI = (1 + np.sqrt(5)) / 2


def analyze_signal_spectrum(x: np.ndarray, name: str = "Signal"):
    """Analyze the spectral content of a signal."""
    n = len(x)
    X_fft = np.abs(np.fft.fft(x))[:n//2]
    freqs = np.fft.fftfreq(n)[:n//2] * n
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(X_fft, height=np.max(X_fft) * 0.1)
    
    print(f"\n{name}:")
    print(f"  Total energy: {np.sum(X_fft**2):.2f}")
    if len(peaks) > 0:
        print(f"  Peak frequencies: {freqs[peaks[:5]]}")
        print(f"  Peak ratios (to first): {freqs[peaks[:5]]/freqs[peaks[0]] if len(peaks) > 0 else 'N/A'}")


def build_matched_rft_kernel(x: np.ndarray, decay_rate: float = 0.001) -> np.ndarray:
    """
    Build RFT kernel that's MATCHED to the signal.
    
    Use the signal's autocorrelation as the resonance kernel.
    This is essentially KLT for stationary signals.
    """
    n = len(x)
    
    # Estimate autocorrelation
    X_fft = np.fft.fft(x)
    psd = np.abs(X_fft)**2
    autocorr = np.real(np.fft.ifft(psd))
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Apply decay to regularize
    k = np.arange(n)
    decay = np.exp(-decay_rate * k)
    r = autocorr * decay
    
    # Build Toeplitz and eigendecompose
    K = toeplitz(r)
    eigenvalues, U = eigh(K)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    return U[:, idx], eigenvalues[idx]


def build_golden_rft_kernel(n: int, f0: float = 10.0, decay_rate: float = 0.01) -> np.ndarray:
    """Build a golden-ratio RFT kernel."""
    k = np.arange(n)
    t = k / n
    
    # Golden resonance
    r = np.cos(2 * np.pi * f0 * t) + np.cos(2 * np.pi * f0 * PHI * t)
    r = r * np.exp(-decay_rate * k)
    r[0] = 1.0
    
    K = toeplitz(r)
    eigenvalues, U = eigh(K)
    idx = np.argsort(eigenvalues)[::-1]
    
    return U[:, idx], eigenvalues[idx]


def compress_and_evaluate(x: np.ndarray, U: np.ndarray, keep_ratio: float = 0.1) -> float:
    """Compress with given basis and compute PSNR."""
    n = len(x)
    k = max(1, int(n * keep_ratio))
    
    X = U.T @ x
    X_comp = X.copy()
    X_comp[k:] = 0
    x_rec = U @ X_comp
    
    mse = np.mean((x - x_rec)**2)
    if mse < 1e-15:
        return float('inf')
    return 10 * np.log10(np.max(x**2) / mse)


def compare_kernels():
    """Compare different kernel choices."""
    print("="*70)
    print("KERNEL COMPARISON: Fixed vs Matched RFT")
    print("="*70)
    
    n = 1024
    np.random.seed(42)
    t = np.arange(n) / 256.0
    
    # Generate test signal with known golden-ratio content
    f0 = 10
    x_golden = np.cos(2*np.pi*f0*t) + 0.5*np.cos(2*np.pi*f0*PHI*t)
    x_golden += 0.1 * np.random.randn(n)
    
    analyze_signal_spectrum(x_golden, "Golden Quasi-Periodic Signal")
    
    # Build kernels
    print("\nBuilding kernels...")
    
    # Fixed golden kernel with various f0 values
    results = {}
    
    for f0_kernel in [5.0, 10.0, 20.0, 40.0]:
        U, _ = build_golden_rft_kernel(n, f0=f0_kernel)
        psnr = compress_and_evaluate(x_golden, U, 0.1)
        results[f'Golden f0={f0_kernel}'] = psnr
    
    # Matched (signal-derived) kernel
    U_matched, _ = build_matched_rft_kernel(x_golden)
    results['Matched (KLT)'] = compress_and_evaluate(x_golden, U_matched, 0.1)
    
    # FFT/DCT baselines
    X_fft = np.fft.fft(x_golden)
    idx = np.argsort(np.abs(X_fft))[::-1]
    X_fft_comp = np.zeros_like(X_fft)
    k = n // 10
    X_fft_comp[idx[:k]] = X_fft[idx[:k]]
    x_rec = np.real(np.fft.ifft(X_fft_comp))
    results['FFT'] = 10 * np.log10(np.max(x_golden**2) / np.mean((x_golden - x_rec)**2))
    
    X_dct = dct(x_golden, norm='ortho')
    idx = np.argsort(np.abs(X_dct))[::-1]
    X_dct_comp = np.zeros_like(X_dct)
    X_dct_comp[idx[:k]] = X_dct[idx[:k]]
    x_rec = idct(X_dct_comp, norm='ortho')
    results['DCT'] = 10 * np.log10(np.max(x_golden**2) / np.mean((x_golden - x_rec)**2))
    
    print("\nResults on golden quasi-periodic signal:")
    print("-" * 50)
    for name, psnr in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<25} {psnr:>8.1f} dB")
    
    return results


def find_optimal_kernel_params():
    """Grid search for optimal kernel parameters."""
    print("\n" + "="*70)
    print("GRID SEARCH: Finding Optimal Kernel Parameters")
    print("="*70)
    
    n = 1024
    np.random.seed(42)
    t = np.arange(n) / 256.0
    
    # Golden quasi-periodic signal
    f0_sig = 10
    x = np.cos(2*np.pi*f0_sig*t) + 0.5*np.cos(2*np.pi*f0_sig*PHI*t)
    x += 0.1 * np.random.randn(n)
    
    # Grid search
    best_psnr = 0
    best_params = {}
    
    f0_values = [5, 10, 20, 40, 80]
    decay_values = [0.001, 0.01, 0.05, 0.1]
    
    print(f"\n{'f0':<8} | {'decay':<8} | {'PSNR':<10}")
    print("-" * 30)
    
    for f0 in f0_values:
        for decay in decay_values:
            U, _ = build_golden_rft_kernel(n, f0=f0, decay_rate=decay)
            psnr = compress_and_evaluate(x, U, 0.1)
            
            if psnr > best_psnr:
                best_psnr = psnr
                best_params = {'f0': f0, 'decay': decay}
            
            print(f"{f0:<8} | {decay:<8} | {psnr:>8.1f}dB")
    
    print(f"\nBest params: f0={best_params['f0']}, decay={best_params['decay']}")
    print(f"Best PSNR: {best_psnr:.1f} dB")
    
    # Compare to baselines
    X_fft = np.fft.fft(x)
    idx = np.argsort(np.abs(X_fft))[::-1]
    X_fft_comp = np.zeros_like(X_fft)
    k = n // 10
    X_fft_comp[idx[:k]] = X_fft[idx[:k]]
    x_rec = np.real(np.fft.ifft(X_fft_comp))
    psnr_fft = 10 * np.log10(np.max(x**2) / np.mean((x - x_rec)**2))
    
    X_dct = dct(x, norm='ortho')
    idx = np.argsort(np.abs(X_dct))[::-1]
    X_dct_comp = np.zeros_like(X_dct)
    X_dct_comp[idx[:k]] = X_dct[idx[:k]]
    x_rec = idct(X_dct_comp, norm='ortho')
    psnr_dct = 10 * np.log10(np.max(x**2) / np.mean((x - x_rec)**2))
    
    print(f"\nFFT:  {psnr_fft:.1f} dB")
    print(f"DCT:  {psnr_dct:.1f} dB")
    print(f"Best RFT: {best_psnr:.1f} dB")
    print(f"Δ vs FFT: {best_psnr - psnr_fft:+.1f} dB")


def test_matched_kernel():
    """Test signal-matched kernel (KLT-like)."""
    print("\n" + "="*70)
    print("MATCHED KERNEL (KLT-LIKE) TEST")
    print("="*70)
    
    n = 1024
    np.random.seed(42)
    t = np.arange(n) / 256.0
    
    signals = {
        'Golden QP': np.cos(2*np.pi*10*t) + 0.5*np.cos(2*np.pi*10*PHI*t) + 0.1*np.random.randn(n),
        'Pure Sine': np.sin(2*np.pi*10*t) + 0.1*np.random.randn(n),
        'Chirp': np.sin(2*np.pi*(5 + 20*t)*t) + 0.1*np.random.randn(n),
    }
    
    print(f"\n{'Signal':<15} | {'FFT':<10} | {'DCT':<10} | {'Matched RFT':<12} | Winner")
    print("-" * 65)
    
    for name, x in signals.items():
        # Matched kernel
        U_matched, _ = build_matched_rft_kernel(x)
        psnr_matched = compress_and_evaluate(x, U_matched, 0.1)
        
        # FFT
        X_fft = np.fft.fft(x)
        idx = np.argsort(np.abs(X_fft))[::-1]
        X_fft_comp = np.zeros_like(X_fft)
        k = n // 10
        X_fft_comp[idx[:k]] = X_fft[idx[:k]]
        x_rec = np.real(np.fft.ifft(X_fft_comp))
        psnr_fft = 10 * np.log10(np.max(x**2) / np.mean((x - x_rec)**2))
        
        # DCT
        X_dct = dct(x, norm='ortho')
        idx = np.argsort(np.abs(X_dct))[::-1]
        X_dct_comp = np.zeros_like(X_dct)
        X_dct_comp[idx[:k]] = X_dct[idx[:k]]
        x_rec = idct(X_dct_comp, norm='ortho')
        psnr_dct = 10 * np.log10(np.max(x**2) / np.mean((x - x_rec)**2))
        
        winner = 'Matched' if psnr_matched >= max(psnr_fft, psnr_dct) else ('FFT' if psnr_fft > psnr_dct else 'DCT')
        
        print(f"{name:<15} | {psnr_fft:>8.1f}dB | {psnr_dct:>8.1f}dB | {psnr_matched:>10.1f}dB | {winner}")
    
    print("""
INSIGHT: Even the matched kernel (KLT) doesn't always beat FFT/DCT!

This is because:
1. For sinusoidal signals, FFT is already optimal (KLT = DFT)
2. For smooth signals, DCT is near-optimal
3. Matched kernel is best for non-standard autocorrelation structures

The RFT advantage exists but is small and signal-specific.
""")


def analyze_dft_basis_overlap():
    """Check how much RFT eigenbasis overlaps with DFT."""
    print("\n" + "="*70)
    print("DFT-RFT BASIS OVERLAP ANALYSIS")
    print("="*70)
    
    n = 256
    
    # DFT basis
    DFT = np.fft.fft(np.eye(n), norm='ortho')
    
    # RFT basis
    k = np.arange(n)
    t = k / n
    r = np.cos(2 * np.pi * 10 * t) + np.cos(2 * np.pi * 10 * PHI * t)
    r = r * np.exp(-0.01 * k)
    r[0] = 1.0
    K = toeplitz(r)
    _, U_rft = eigh(K)
    U_rft = U_rft[:, ::-1]
    
    # Check overlap: max |<dft_k, rft_j>|² for each k
    overlap = np.abs(DFT.conj().T @ U_rft)**2
    
    max_overlaps = np.max(overlap, axis=1)
    
    print(f"\nFor each DFT vector, max overlap with ANY RFT vector:")
    print(f"  Mean max overlap: {np.mean(max_overlaps):.3f}")
    print(f"  Min max overlap:  {np.min(max_overlaps):.3f}")
    print(f"  Max max overlap:  {np.max(max_overlaps):.3f}")
    
    # If overlap is high, RFT ≈ DFT
    if np.mean(max_overlaps) > 0.5:
        print("\n  → RFT basis is quite similar to DFT basis!")
        print("  → This explains why RFT ≈ FFT in performance")
    else:
        print("\n  → RFT basis is DIFFERENT from DFT basis")
        print("  → RFT should show different behavior than FFT")


if __name__ == "__main__":
    results = compare_kernels()
    find_optimal_kernel_params()
    test_matched_kernel()
    analyze_dft_basis_overlap()
    
    print("\n" + "="*70)
    print("FINAL DIAGNOSIS")
    print("="*70)
    print("""
PROBLEM IDENTIFIED:

1. Our RFT kernel parameters (f0, decay) don't match the signal's
   actual frequency content. The kernel is designed for f0=10 Hz
   but signals have different frequency structures.

2. Even with matched parameters, RFT is only marginally better
   because the golden-ratio structure is present but weak.

3. The signal-matched kernel (KLT) gives the theoretical best
   but requires O(N³) per signal - impractical.

SOLUTION OPTIONS:

A. Use adaptive frequency detection to tune f0 automatically
B. Build a library of kernels for different frequency bands
C. Accept that RFT is a "tuned transform" for specific domains
D. Develop a fast adaptive algorithm that estimates best kernel

For "foundational" status, we need option D - but that's a
significant research problem beyond current implementation.
""")
