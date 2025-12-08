"""
Patent Variant Discovery Benchmark
==================================

Tests all 20 new patent-aligned RFT variants to find which ones
excel on different signal types.

Goal: Find the RIGHT RFT variants that match the patent claims.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scipy.fft import fft, ifft, dct, idct
from algorithms.rft.variants.patent_variants import (
    PATENT_VARIANTS, get_patent_variant, PHI, GOLDEN_ANGLE
)
from algorithms.rft.variants.operator_variants import (
    OPERATOR_VARIANTS, get_operator_variant
)


# =============================================================================
# SIGNAL GENERATORS - Expanded for patent claims
# =============================================================================

def gen_golden_qp(N, f0=10.0, phase=0.0):
    """Golden quasi-periodic signal."""
    t = np.linspace(0, 1, N)
    return np.sin(2*np.pi*f0*t) + np.sin(2*np.pi*f0*PHI*t + phase)


def gen_fibonacci_mod(N, depth=5):
    """Fibonacci-modulated signal."""
    t = np.linspace(0, 1, N)
    fib = [1, 1]
    for _ in range(depth):
        fib.append(fib[-1] + fib[-2])
    x = np.zeros(N)
    for i, f in enumerate(fib):
        x += (1.0/f) * np.sin(2*np.pi*10*(PHI**i)*t)
    return x / (np.max(np.abs(x)) + 1e-10)


def gen_harmonic(N, f0=100.0):
    """Natural harmonic series."""
    t = np.linspace(0, 1, N)
    x = np.zeros(N)
    for h in range(1, 8):
        x += (1.0/h) * np.sin(2*np.pi*h*f0*t)
    return x / (np.max(np.abs(x)) + 1e-10)


def gen_phyllotaxis(N, spirals=8):
    """Phyllotaxis (golden angle) pattern."""
    t = np.linspace(0, 1, N)
    x = np.zeros(N)
    for s in range(1, spirals + 1):
        x += np.sin(s * GOLDEN_ANGLE * np.arange(N) + 2*np.pi*s*t)
    return x / (np.max(np.abs(x)) + 1e-10)


def gen_pure_sine(N, freq=7.0):
    """Pure sinusoid (FFT optimal)."""
    t = np.linspace(0, 1, N)
    return np.sin(2*np.pi*freq*t)


def gen_chirp(N):
    """Linear chirp."""
    t = np.linspace(0, 1, N)
    return np.sin(2*np.pi*(5 + 20*t)*t)


def gen_noise(N, seed=42):
    """White noise."""
    np.random.seed(seed)
    return np.random.randn(N)


def gen_damped_oscillation(N, freq=10.0, decay=3.0):
    """Damped sinusoid."""
    t = np.linspace(0, 1, N)
    return np.sin(2*np.pi*freq*t) * np.exp(-decay*t)


def gen_am_modulated(N, carrier=50.0, mod=5.0):
    """AM modulated signal."""
    t = np.linspace(0, 1, N)
    return (1 + 0.5*np.sin(2*np.pi*mod*t)) * np.sin(2*np.pi*carrier*t)


def gen_fm_modulated(N, carrier=50.0, mod=5.0, depth=2.0):
    """FM modulated signal."""
    t = np.linspace(0, 1, N)
    return np.sin(2*np.pi*carrier*t + depth*np.sin(2*np.pi*mod*t))


def gen_spiral_signal(N):
    """Signal based on golden spiral."""
    t = np.linspace(0, 1, N)
    theta = 4 * np.pi * t
    b = np.log(PHI) / (np.pi / 2)
    r = np.exp(b * theta)
    return r * np.cos(theta) / np.max(np.abs(r))


def gen_beating(N, f1=10.0):
    """Beating pattern with golden ratio frequencies."""
    t = np.linspace(0, 1, N)
    f2 = f1 * PHI
    return np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)


def gen_chaotic_logistic(N, r_param=3.9):
    """Chaotic logistic map signal."""
    x = np.zeros(N)
    x[0] = 0.1
    for i in range(1, N):
        x[i] = r_param * x[i-1] * (1 - x[i-1])
    return x - np.mean(x)


def gen_trefoil_projection(N):
    """Trefoil knot x-projection."""
    t = 2 * np.pi * np.arange(N) / N
    x = np.sin(t) + 2 * np.sin(2*t)
    return x / (np.max(np.abs(x)) + 1e-10)


def gen_torus_curve(N):
    """Curve on torus with golden winding."""
    t = 2 * np.pi * np.arange(N) / N
    R, r_minor = PHI, 1.0
    u, v = t, PHI * t
    x = (R + r_minor * np.cos(v)) * np.cos(u)
    return x / (np.max(np.abs(x)) + 1e-10)


# All test signals
SIGNALS = {
    # RFT-golden family
    'golden_qp_0': gen_golden_qp,
    'golden_qp_90': lambda N: gen_golden_qp(N, phase=np.pi/2),
    'fibonacci_d5': gen_fibonacci_mod,
    'phyllotaxis': gen_phyllotaxis,
    'spiral': gen_spiral_signal,
    'beating': gen_beating,
    
    # Classical transform optimal
    'pure_sine': gen_pure_sine,
    'chirp': gen_chirp,
    'noise': gen_noise,
    'harmonic': gen_harmonic,
    
    # Modulated signals
    'damped_osc': gen_damped_oscillation,
    'am_mod': gen_am_modulated,
    'fm_mod': gen_fm_modulated,
    
    # Topological structures
    'trefoil': gen_trefoil_projection,
    'torus': gen_torus_curve,
    
    # Chaotic
    'chaotic': gen_chaotic_logistic,
}


# =============================================================================
# METRICS
# =============================================================================

def compute_psnr(x, transform_fn, inverse_fn, keep_frac=0.1):
    """Compute PSNR at fixed compression ratio."""
    n = len(x)
    coeffs = transform_fn(x)
    
    k = max(1, int(n * keep_frac))
    magnitudes = np.abs(coeffs)
    indices = np.argsort(magnitudes)[::-1][:k]
    
    sparse = np.zeros_like(coeffs)
    sparse[indices] = coeffs[indices]
    
    x_rec = inverse_fn(sparse)
    if np.iscomplexobj(x_rec):
        x_rec = x_rec.real
    
    mse = np.mean((x - x_rec)**2)
    if mse < 1e-15:
        return 100.0
    
    max_val = np.max(np.abs(x))
    return 10 * np.log10(max_val**2 / mse)


def compute_sparsity(x, transform_fn, threshold=0.99):
    """Count coefficients for threshold energy."""
    coeffs = transform_fn(x)
    energy = np.abs(coeffs) ** 2
    total = np.sum(energy)
    if total < 1e-15:
        return len(coeffs)
    sorted_energy = np.sort(energy)[::-1]
    cumsum = np.cumsum(sorted_energy)
    idx = np.searchsorted(cumsum, threshold * total)
    return min(idx + 1, len(coeffs))


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_all_variants():
    """Benchmark all patent variants + baselines on all signals."""
    
    N = 256
    keep_frac = 0.1
    
    # Collect all variants
    all_variants = {}
    
    # Baselines
    all_variants['fft'] = {
        'name': 'FFT',
        'forward': lambda x: fft(x, norm='ortho'),
        'inverse': lambda c: ifft(c, norm='ortho').real,
    }
    all_variants['dct'] = {
        'name': 'DCT',
        'forward': lambda x: dct(x, norm='ortho'),
        'inverse': lambda c: idct(c, norm='ortho'),
    }
    
    # Original operator variants
    for vname, vinfo in OPERATOR_VARIANTS.items():
        Phi = vinfo['generator'](N)
        all_variants[vname] = {
            'name': vinfo['name'],
            'forward': lambda x, P=Phi: P.T @ x,
            'inverse': lambda c, P=Phi: P @ c,
        }
    
    # NEW: Patent variants
    for vname, vinfo in PATENT_VARIANTS.items():
        try:
            Phi = vinfo['generator'](N)
            all_variants[vname] = {
                'name': vinfo['name'],
                'forward': lambda x, P=Phi: P.T @ x,
                'inverse': lambda c, P=Phi: P @ c,
                'claim': vinfo.get('claim', ''),
            }
        except Exception as e:
            print(f"Warning: Could not load {vname}: {e}")
    
    # Results matrix
    results = {}
    
    print("=" * 100)
    print("PATENT VARIANT DISCOVERY BENCHMARK")
    print("=" * 100)
    print(f"\nTesting {len(all_variants)} transforms on {len(SIGNALS)} signals (N={N}, keep={keep_frac*100:.0f}%)")
    print()
    
    # Header
    print(f"{'Signal':<15}", end='')
    for vname in ['fft', 'dct', 'rft_golden']:
        print(f"{vname[:10]:<12}", end='')
    print("| BEST PATENT VARIANT")
    print("-" * 100)
    
    variant_wins = {v: 0 for v in all_variants}
    
    for sig_name, gen_fn in SIGNALS.items():
        x = gen_fn(N)
        
        sig_results = {}
        for vname, vinfo in all_variants.items():
            try:
                psnr = compute_psnr(x, vinfo['forward'], vinfo['inverse'], keep_frac)
                sig_results[vname] = psnr
            except:
                sig_results[vname] = -np.inf
        
        results[sig_name] = sig_results
        
        # Find best overall and best patent variant
        best_overall = max(sig_results, key=sig_results.get)
        variant_wins[best_overall] += 1
        
        # Best among patent variants only
        patent_results = {k: v for k, v in sig_results.items() if k in PATENT_VARIANTS}
        if patent_results:
            best_patent = max(patent_results, key=patent_results.get)
            best_patent_psnr = patent_results[best_patent]
        else:
            best_patent = "N/A"
            best_patent_psnr = 0
        
        # Print row
        print(f"{sig_name:<15}", end='')
        for vname in ['fft', 'dct', 'rft_golden']:
            psnr = sig_results.get(vname, 0)
            print(f"{psnr:<12.2f}", end='')
        
        # Highlight best patent variant
        marker = "★" if best_overall in PATENT_VARIANTS else " "
        print(f"| {marker} {best_patent:<25} ({best_patent_psnr:.2f} dB)")
    
    print("-" * 100)
    
    # Summary: Which variants won?
    print("\n" + "=" * 100)
    print("VARIANT WIN COUNTS (best PSNR on each signal)")
    print("=" * 100)
    
    sorted_wins = sorted(variant_wins.items(), key=lambda x: -x[1])
    
    print("\nTop 10 Winners:")
    for i, (vname, wins) in enumerate(sorted_wins[:10]):
        if wins > 0:
            claim = ""
            if vname in PATENT_VARIANTS:
                claim = f" ({PATENT_VARIANTS[vname].get('claim', '')[:40]})"
            print(f"  {i+1}. {vname:<30} {wins} wins{claim}")
    
    # Patent variant summary
    print("\n" + "=" * 100)
    print("PATENT VARIANTS PERFORMANCE SUMMARY")
    print("=" * 100)
    
    for vname, vinfo in PATENT_VARIANTS.items():
        wins = variant_wins.get(vname, 0)
        avg_psnr = np.mean([results[s].get(vname, 0) for s in SIGNALS])
        print(f"  {vinfo['name']:<25} | wins={wins:<2} | avg PSNR={avg_psnr:.2f} dB | {vinfo.get('claim', '')[:35]}")
    
    # Best signal for each patent variant
    print("\n" + "=" * 100)
    print("BEST SIGNAL FOR EACH PATENT VARIANT")
    print("=" * 100)
    
    for vname, vinfo in PATENT_VARIANTS.items():
        best_sig = max(SIGNALS.keys(), key=lambda s: results[s].get(vname, -np.inf))
        best_psnr = results[best_sig].get(vname, 0)
        
        # Compare to baselines
        fft_psnr = results[best_sig].get('fft', 0)
        dct_psnr = results[best_sig].get('dct', 0)
        golden_psnr = results[best_sig].get('rft_golden', 0)
        
        improvement = best_psnr - max(fft_psnr, dct_psnr)
        vs_golden = best_psnr - golden_psnr
        
        symbol = "★" if improvement > 0 else " "
        print(f"  {symbol} {vinfo['name']:<25} → {best_sig:<15} ({best_psnr:.1f} dB, +{improvement:.1f} vs FFT/DCT, {vs_golden:+.1f} vs golden)")
    
    print("=" * 100)
    
    return results


if __name__ == "__main__":
    benchmark_all_variants()
