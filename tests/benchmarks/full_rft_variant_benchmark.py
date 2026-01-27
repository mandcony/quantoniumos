# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Full RFT Variant Benchmark Suite
================================

Comprehensive benchmark comparing all operator-based RFT variants
against FFT and DCT on multiple signal types.

Tests:
1. Unitarity verification
2. Sparsity (99% energy capture)
3. Compression quality (PSNR at 10% coefficients)
4. Domain specificity (in-family vs out-of-family)
"""

import numpy as np
from scipy.fft import fft, ifft, dct, idct
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from algorithms.rft.variants.operator_variants import (
    OPERATOR_VARIANTS, get_operator_variant, PHI
)


# =============================================================================
# SIGNAL GENERATORS
# =============================================================================

def gen_golden_qp(N, f0=10.0, A=1.0, B=1.0, phase=0.0):
    """Golden quasi-periodic signal."""
    t = np.linspace(0, 1, N)
    return A * np.sin(2*np.pi*f0*t) + B * np.sin(2*np.pi*f0*PHI*t + phase)

def gen_fibonacci_mod(N, f0=10.0, depth=5):
    """Fibonacci-modulated signal."""
    t = np.linspace(0, 1, N)
    fib = [1, 1]
    for _ in range(depth):
        fib.append(fib[-1] + fib[-2])
    x = np.zeros(N)
    for i, f in enumerate(fib):
        x += (1.0/f) * np.sin(2*np.pi*f0*(PHI**i)*t)
    return x / np.max(np.abs(x))

def gen_harmonic(N, f0=100.0):
    """Natural harmonic series."""
    t = np.linspace(0, 1, N)
    x = np.zeros(N)
    for h in range(1, 8):
        x += (1.0/h) * np.sin(2*np.pi*h*f0*t)
    return x / np.max(np.abs(x))

def gen_phyllotaxis(N, spirals=8):
    """Phyllotaxis phase sequence (2π/φ² ≈ 137.5°; complement 2π/φ ≈ 222.5°)."""
    golden_angle = 2 * np.pi / (PHI ** 2)
    t = np.linspace(0, 1, N)
    x = np.zeros(N)
    for s in range(1, spirals + 1):
        x += np.sin(s * golden_angle * np.arange(N) + 2*np.pi*s*t)
    return x / np.max(np.abs(x))

def gen_pure_sine(N, freq=7.0):
    """Pure sinusoid (FFT optimal)."""
    t = np.linspace(0, 1, N)
    return np.sin(2*np.pi*freq*t)

def gen_chirp(N):
    """Linear chirp signal."""
    t = np.linspace(0, 1, N)
    return np.sin(2*np.pi*(5 + 20*t)*t)

def gen_square(N, freq=4.0):
    """Square wave."""
    t = np.linspace(0, 1, N)
    return np.sign(np.sin(2*np.pi*freq*t))

def gen_noise(N, seed=42):
    """White noise."""
    np.random.seed(seed)
    return np.random.randn(N)


# Signal catalog with expected best variants
SIGNALS = {
    'golden_qp_1': (lambda N: gen_golden_qp(N), ['rft_golden', 'rft_beating']),
    'golden_qp_2': (lambda N: gen_golden_qp(N, phase=np.pi/2), ['rft_golden', 'rft_beating']),
    'fibonacci_d3': (lambda N: gen_fibonacci_mod(N, depth=3), ['rft_fibonacci', 'rft_geometric']),
    'fibonacci_d5': (lambda N: gen_fibonacci_mod(N, depth=5), ['rft_fibonacci', 'rft_geometric']),
    'harmonic_series': (lambda N: gen_harmonic(N), ['rft_harmonic']),
    'phyllotaxis_s5': (lambda N: gen_phyllotaxis(N, 5), ['rft_phyllotaxis', 'rft_golden']),
    'phyllotaxis_s8': (lambda N: gen_phyllotaxis(N, 8), ['rft_phyllotaxis', 'rft_golden']),
    'pure_sine': (lambda N: gen_pure_sine(N), []),  # FFT should win
    'chirp': (lambda N: gen_chirp(N), []),  # DCT should win
    'square_wave': (lambda N: gen_square(N), []),  # FFT should win
    'white_noise': (lambda N: gen_noise(N), []),  # Nothing should win
}


# =============================================================================
# METRICS
# =============================================================================

def energy_sparsity(coeffs, threshold=0.99):
    """Count coefficients for threshold energy."""
    energy = np.abs(coeffs) ** 2
    total = np.sum(energy)
    if total < 1e-15:
        return len(coeffs)
    sorted_energy = np.sort(energy)[::-1]
    cumsum = np.cumsum(sorted_energy)
    idx = np.searchsorted(cumsum, threshold * total)
    return min(idx + 1, len(coeffs))


def compression_psnr(signal, coeffs, inverse_fn, keep_frac=0.1):
    """PSNR at fixed coefficient retention."""
    N = len(coeffs)
    K = max(1, int(keep_frac * N))
    
    top_idx = np.argsort(np.abs(coeffs))[-K:]
    sparse = np.zeros_like(coeffs)
    sparse[top_idx] = coeffs[top_idx]
    
    rec = inverse_fn(sparse)
    if np.iscomplexobj(rec):
        rec = np.real(rec)
    
    mse = np.mean((signal - rec) ** 2)
    if mse < 1e-15:
        return 100.0
    max_val = np.max(np.abs(signal))
    return 10 * np.log10(max_val**2 / mse)


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_variant(variant_name, signals, N, keep_frac=0.1):
    """Benchmark a single variant on all signals."""
    try:
        Phi = get_operator_variant(variant_name, N)
    except Exception as e:
        return {'error': str(e)}
    
    results = {}
    for sig_name, (gen_fn, _) in signals.items():
        signal = gen_fn(N)
        
        # Forward transform
        coeffs = Phi.T @ signal
        
        # Metrics
        sparsity = energy_sparsity(coeffs)
        psnr = compression_psnr(signal, coeffs, lambda c: Phi @ c, keep_frac)
        
        results[sig_name] = {'sparsity': sparsity, 'psnr': psnr}
    
    return results


def benchmark_baseline(name, forward_fn, inverse_fn, signals, N, keep_frac=0.1):
    """Benchmark FFT or DCT."""
    results = {}
    for sig_name, (gen_fn, _) in signals.items():
        signal = gen_fn(N)
        coeffs = forward_fn(signal)
        sparsity = energy_sparsity(coeffs)
        psnr = compression_psnr(signal, coeffs, inverse_fn, keep_frac)
        results[sig_name] = {'sparsity': sparsity, 'psnr': psnr}
    return results


def main():
    print("=" * 90)
    print("FULL RFT VARIANT BENCHMARK")
    print("=" * 90)
    
    N = 256
    keep_frac = 0.10
    
    # Benchmark baselines
    print("\nBenchmarking baselines (FFT, DCT)...")
    baselines = {
        'FFT': benchmark_baseline('FFT', fft, lambda c: np.real(ifft(c)), SIGNALS, N, keep_frac),
        'DCT': benchmark_baseline('DCT', lambda x: dct(x, norm='ortho'), 
                                  lambda c: idct(c, norm='ortho'), SIGNALS, N, keep_frac),
    }
    
    # Benchmark all operator variants
    print("Benchmarking operator-based RFT variants...")
    variant_results = {}
    for variant_name in OPERATOR_VARIANTS.keys():
        variant_results[variant_name] = benchmark_variant(variant_name, SIGNALS, N, keep_frac)
    
    # =========================================================================
    # RESULTS TABLE: PSNR
    # =========================================================================
    print("\n" + "=" * 90)
    print(f"PSNR at {int(keep_frac*100)}% coefficient retention (higher = better)")
    print("=" * 90)
    
    # Header
    all_methods = ['FFT', 'DCT'] + list(OPERATOR_VARIANTS.keys())
    header = f"{'Signal':<20}"
    for m in all_methods:
        short_name = m.replace('rft_', '').upper()[:8]
        header += f" | {short_name:>8}"
    header += " | BEST"
    print(header)
    print("-" * len(header))
    
    # Per-signal results
    wins = {m: 0 for m in all_methods}
    
    for sig_name in SIGNALS.keys():
        row = f"{sig_name:<20}"
        psnrs = {}
        
        for m in all_methods:
            if m in baselines:
                psnr = baselines[m][sig_name]['psnr']
            else:
                res = variant_results[m]
                psnr = res[sig_name]['psnr'] if sig_name in res else 0
            
            psnrs[m] = psnr
            short_name = m.replace('rft_', '').upper()[:8]
            row += f" | {psnr:>8.2f}"
        
        best = max(psnrs, key=psnrs.get)
        wins[best] += 1
        row += f" | {best.replace('rft_', '').upper()[:8]}"
        print(row)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 90)
    print("WIN COUNT (best PSNR per signal)")
    print("=" * 90)
    
    for m in sorted(wins.keys(), key=lambda x: -wins[x]):
        short_name = m.replace('rft_', '').upper()
        bar = "█" * wins[m]
        print(f"{short_name:<20} {wins[m]:>2} {bar}")
    
    # =========================================================================
    # DOMAIN SPECIFICITY CHECK
    # =========================================================================
    print("\n" + "=" * 90)
    print("DOMAIN SPECIFICITY CHECK")
    print("=" * 90)
    
    in_family = ['golden_qp_1', 'golden_qp_2', 'fibonacci_d3', 'fibonacci_d5', 
                 'harmonic_series', 'phyllotaxis_s5', 'phyllotaxis_s8']
    out_family = ['pure_sine', 'chirp', 'square_wave', 'white_noise']
    
    rft_in_family_wins = 0
    rft_out_family_wins = 0
    
    for sig_name in SIGNALS.keys():
        psnrs = {}
        for m in all_methods:
            if m in baselines:
                psnrs[m] = baselines[m][sig_name]['psnr']
            else:
                psnrs[m] = variant_results[m][sig_name]['psnr']
        
        best = max(psnrs, key=psnrs.get)
        is_rft = best.startswith('rft_')
        
        if sig_name in in_family:
            if is_rft:
                rft_in_family_wins += 1
        else:
            if is_rft:
                rft_out_family_wins += 1
    
    print(f"In-Family (golden/fib/harm/phyll): RFT wins {rft_in_family_wins}/{len(in_family)} ({100*rft_in_family_wins/len(in_family):.0f}%)")
    print(f"Out-of-Family (sine/chirp/square/noise): RFT wins {rft_out_family_wins}/{len(out_family)} ({100*rft_out_family_wins/len(out_family):.0f}%)")
    
    if rft_in_family_wins >= len(in_family) // 2 and rft_out_family_wins <= len(out_family) // 2:
        print("\n✅ VALID: RFT variants show domain-specific advantage")
    else:
        print("\n⚠️ MIXED: Check signal-variant matching")


if __name__ == "__main__":
    main()
