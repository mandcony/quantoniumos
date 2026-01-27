# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Experiment 4: Router vs Oracle on PSNR
======================================

Goal: Measure how close the AdaptiveRouter comes to oracle transform selection.

For each signal:
1. Compute PSNR for ALL transforms at fixed compression ratio
2. Oracle = best PSNR across all transforms
3. Router = PSNR for classifier's pick
4. Measure Δ = oracle - router

If Δ is small (<0.5 dB average), the router is doing something real.
If Δ is large, the classifier needs calibration.

This directly answers: "Does the router pick something close to optimal?"
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scipy.fft import fft, ifft, dct, idct
from algorithms.rft.variants.operator_variants import get_operator_variant, PHI
from algorithms.rft.routing.signal_classifier import classify_signal, AdaptiveRouter


# =============================================================================
# SIGNAL GENERATORS (same as full_rft_variant_benchmark.py)
# =============================================================================

def gen_golden_qp(N, f0=10.0, phase=0.0):
    """Golden quasi-periodic signal."""
    t = np.linspace(0, 1, N)
    return np.sin(2*np.pi*f0*t) + np.sin(2*np.pi*f0*PHI*t + phase)


def gen_fibonacci_mod(N, f0=10.0, depth=5):
    """Fibonacci-modulated signal."""
    t = np.linspace(0, 1, N)
    fib = [1, 1]
    for _ in range(depth):
        fib.append(fib[-1] + fib[-2])
    x = np.zeros(N)
    for i, f in enumerate(fib):
        x += (1.0/f) * np.sin(2*np.pi*f0*(PHI**i)*t)
    return x / (np.max(np.abs(x)) + 1e-10)


def gen_harmonic(N, f0=100.0):
    """Natural harmonic series."""
    t = np.linspace(0, 1, N)
    x = np.zeros(N)
    for h in range(1, 8):
        x += (1.0/h) * np.sin(2*np.pi*h*f0*t)
    return x / (np.max(np.abs(x)) + 1e-10)


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


# =============================================================================
# TRANSFORM FUNCTIONS
# =============================================================================

def apply_transform_with_inverse(x: np.ndarray, transform: str) -> Tuple[np.ndarray, callable]:
    """
    Apply transform and return (coefficients, inverse_function).
    """
    n = len(x)
    
    if transform == 'fft':
        coeffs = fft(x, norm='ortho')
        def inv(c): return ifft(c, norm='ortho').real
        return coeffs, inv
    
    elif transform == 'dct':
        coeffs = dct(x, norm='ortho')
        def inv(c): return idct(c, norm='ortho')
        return coeffs, inv
    
    elif transform.startswith('rft_'):
        Phi = get_operator_variant(transform, n)
        coeffs = Phi.T @ x
        def inv(c): return Phi @ c
        return coeffs, inv
    
    else:
        raise ValueError(f"Unknown transform: {transform}")


def compress_and_psnr(x: np.ndarray, coeffs: np.ndarray, inv_fn: callable, 
                      keep_frac: float = 0.1) -> float:
    """
    Keep top fraction of coefficients, reconstruct, compute PSNR.
    """
    n = len(coeffs)
    k = max(1, int(n * keep_frac))
    
    # Keep top-k by magnitude
    magnitudes = np.abs(coeffs)
    threshold = np.sort(magnitudes)[::-1][k-1] if k < n else 0
    mask = magnitudes >= threshold
    
    # Sparse coefficients
    sparse = np.zeros_like(coeffs)
    sparse[mask] = coeffs[mask]
    
    # Limit to exactly k coefficients
    if np.sum(mask) > k:
        indices = np.argsort(magnitudes)[::-1][:k]
        sparse = np.zeros_like(coeffs)
        sparse[indices] = coeffs[indices]
    
    # Reconstruct
    x_rec = inv_fn(sparse)
    if np.iscomplexobj(x_rec):
        x_rec = x_rec.real
    
    # PSNR
    mse = np.mean((x - x_rec)**2)
    if mse < 1e-15:
        return 100.0
    max_val = np.max(np.abs(x))
    return 10 * np.log10(max_val**2 / mse)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run router vs oracle comparison."""
    
    print("=" * 80)
    print("EXPERIMENT 4: ROUTER VS ORACLE")
    print("=" * 80)
    print()
    
    N = 256
    keep_frac = 0.1  # Keep top 10% coefficients
    
    # All transforms to test
    transforms = ['fft', 'dct', 'rft_golden', 'rft_fibonacci', 
                  'rft_harmonic', 'rft_geometric', 'rft_beating']
    
    # Test signals
    test_signals = {
        'golden_qp_0': gen_golden_qp(N, phase=0),
        'golden_qp_90': gen_golden_qp(N, phase=np.pi/2),
        'fibonacci_d3': gen_fibonacci_mod(N, depth=3),
        'fibonacci_d5': gen_fibonacci_mod(N, depth=5),
        'harmonic': gen_harmonic(N),
        'pure_sine_7': gen_pure_sine(N, freq=7),
        'pure_sine_20': gen_pure_sine(N, freq=20),
        'chirp': gen_chirp(N),
        'square': gen_square(N),
        'noise_1': gen_noise(N, seed=1),
        'noise_2': gen_noise(N, seed=2),
    }
    
    # Collect results
    results = []
    router = AdaptiveRouter(enable_rft=True, enable_adaptive=True)
    
    print(f"{'Signal':<20} {'Oracle Best':<15} {'Router Pick':<15} {'Oracle PSNR':<12} {'Router PSNR':<12} {'Δ (dB)':<10}")
    print("-" * 90)
    
    total_delta = 0
    n_signals = 0
    confusion = {}  # confusion[oracle][router] = count
    
    for sig_name, x in test_signals.items():
        # Compute PSNR for all transforms
        psnrs = {}
        inv_fns = {}
        
        for t in transforms:
            try:
                coeffs, inv_fn = apply_transform_with_inverse(x, t)
                psnrs[t] = compress_and_psnr(x, coeffs, inv_fn, keep_frac)
                inv_fns[t] = inv_fn
            except Exception as e:
                psnrs[t] = -np.inf
        
        # Oracle: best transform
        oracle_best = max(psnrs, key=psnrs.get)
        oracle_psnr = psnrs[oracle_best]
        
        # Router: classifier's pick
        router_pick = router.route(x)
        if router_pick not in psnrs:
            router_pick = 'dct'  # fallback
        router_psnr = psnrs.get(router_pick, psnrs['dct'])
        
        # Delta
        delta = oracle_psnr - router_psnr
        
        print(f"{sig_name:<20} {oracle_best:<15} {router_pick:<15} {oracle_psnr:<12.2f} {router_psnr:<12.2f} {delta:<10.2f}")
        
        results.append({
            'signal': sig_name,
            'oracle_best': oracle_best,
            'oracle_psnr': oracle_psnr,
            'router_pick': router_pick,
            'router_psnr': router_psnr,
            'delta': delta,
        })
        
        total_delta += delta
        n_signals += 1
        
        # Confusion matrix
        if oracle_best not in confusion:
            confusion[oracle_best] = {}
        if router_pick not in confusion[oracle_best]:
            confusion[oracle_best][router_pick] = 0
        confusion[oracle_best][router_pick] += 1
    
    # Summary
    avg_delta = total_delta / n_signals
    
    print("-" * 90)
    print(f"\n{'SUMMARY':^90}")
    print("=" * 90)
    print(f"  Average PSNR gap (oracle - router): {avg_delta:.2f} dB")
    
    correct_picks = sum(1 for r in results if r['oracle_best'] == r['router_pick'])
    print(f"  Exact match rate: {correct_picks}/{n_signals} ({100*correct_picks/n_signals:.1f}%)")
    
    # "Close enough" = within same transform family
    def same_family(t1, t2):
        if t1 == t2:
            return True
        if t1.startswith('rft_') and t2.startswith('rft_'):
            return True  # Any RFT variant is "close"
        return False
    
    close_picks = sum(1 for r in results if same_family(r['oracle_best'], r['router_pick']))
    print(f"  Same-family rate: {close_picks}/{n_signals} ({100*close_picks/n_signals:.1f}%)")
    
    # Verdict
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    
    if avg_delta < 0.5:
        print("✓ Router is performing well (<0.5 dB average gap from oracle)")
    elif avg_delta < 2.0:
        print("⚠ Router needs calibration (0.5-2.0 dB gap)")
    else:
        print("✗ Router is NOT working (>2.0 dB gap from oracle)")
    
    print()
    print("Confusion matrix (rows=oracle, cols=router):")
    all_transforms = sorted(set(list(confusion.keys()) + 
                               [t for c in confusion.values() for t in c.keys()]))
    
    header_label = "Oracle / Router"
    print(f"{header_label:<15}", end='')
    for t in all_transforms:
        print(f"{t[:8]:<10}", end='')
    print()
    
    for oracle_t in all_transforms:
        print(f"{oracle_t[:15]:<15}", end='')
        for router_t in all_transforms:
            count = confusion.get(oracle_t, {}).get(router_t, 0)
            print(f"{count:<10}", end='')
        print()
    
    print("=" * 90)
    
    # Routing stats
    print("\nRouter distribution:")
    for t, count in sorted(router.get_routing_stats().items()):
        print(f"  {t}: {count}")


if __name__ == "__main__":
    run_experiment()
