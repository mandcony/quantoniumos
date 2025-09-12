#!/usr/bin/env python3
"""
RFT Convolution Invariance Test

Goal: Prove DFT convolution identity fails under RFT, demonstrating fundamental difference.

DFT Property: DFT(x ⊛ h) = DFT(x) ⊙ DFT(h) (convolution becomes pointwise multiplication)
RFT Test: Show ‖RFT(x⊛h) − RFT(x)⊙RFT(h)‖/‖RFT(x⊛h)‖ ≥ 10% over 100 random draws

Pass Criteria:
- Median relative error ≥ 10% over 100 random signal pairs
- At least 90% of samples show error ≥ 5%

This demonstrates RFT does not preserve the convolution property that defines DFT,
proving they are fundamentally different transforms.
"""

import numpy as np
import time
from typing import Tuple, List, Dict
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from src.core.canonical_true_rft import CanonicalTrueRFT
except ImportError:
    print("Warning: Could not import CanonicalTrueRFT, using standalone implementation")
    
    class CanonicalTrueRFT:
        """Standalone RFT implementation for testing"""
        
        def __init__(self, n: int):
            self.n = n
            self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            
        def forward_transform(self, x: np.ndarray) -> np.ndarray:
            """Compute RFT using golden ratio parameterization"""
            n = len(x)
            result = np.zeros(n, dtype=complex)
            
            for k in range(n):
                for j in range(n):
                    # RFT kernel with golden ratio parameterization
                    angle = 2 * np.pi * k * j * self.phi / n
                    kernel = np.exp(-1j * angle)
                    result[k] += x[j] * kernel
                    
            return result


def circular_convolution(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Compute circular convolution of two signals.
    
    x ⊛ h = Σ_{m=0}^{N-1} x[m] * h[(n-m) mod N]
    """
    n = len(x)
    assert len(h) == n, "Signals must have same length"
    
    result = np.zeros(n, dtype=complex)
    
    for i in range(n):
        for m in range(n):
            result[i] += x[m] * h[(i - m) % n]
    
    return result


def test_dft_convolution_property(x: np.ndarray, h: np.ndarray) -> float:
    """
    Verify DFT convolution property: DFT(x⊛h) = DFT(x)⊙DFT(h)
    
    Returns relative error (should be ~0 for DFT)
    """
    # Compute convolution
    conv = circular_convolution(x, h)
    
    # DFT of convolution
    dft_conv = np.fft.fft(conv)
    
    # DFT of individual signals
    dft_x = np.fft.fft(x)
    dft_h = np.fft.fft(h)
    
    # Pointwise multiplication
    dft_mult = dft_x * dft_h
    
    # Compute relative error
    error_norm = np.linalg.norm(dft_conv - dft_mult)
    signal_norm = np.linalg.norm(dft_conv)
    
    if signal_norm > 1e-12:
        relative_error = error_norm / signal_norm
    else:
        relative_error = 0.0
    
    return relative_error


def test_rft_convolution_property(x: np.ndarray, h: np.ndarray, rft: CanonicalTrueRFT) -> float:
    """
    Test if RFT satisfies convolution property: RFT(x⊛h) ?= RFT(x)⊙RFT(h)
    
    Returns relative error (should be large for RFT, proving distinctness)
    """
    # Compute convolution
    conv = circular_convolution(x, h)
    
    # RFT of convolution
    rft_conv = rft.forward_transform(conv)
    
    # RFT of individual signals
    rft_x = rft.forward_transform(x)
    rft_h = rft.forward_transform(h)
    
    # Pointwise multiplication
    rft_mult = rft_x * rft_h
    
    # Compute relative error
    error_norm = np.linalg.norm(rft_conv - rft_mult)
    signal_norm = np.linalg.norm(rft_conv)
    
    if signal_norm > 1e-12:
        relative_error = error_norm / signal_norm
    else:
        relative_error = 0.0
    
    return relative_error


def generate_test_signals(n: int, num_pairs: int = 100) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate diverse test signal pairs for robust testing.
    
    Includes:
    - Random complex signals
    - Real sinusoids
    - Impulse responses
    - Smooth signals
    - Noisy signals
    """
    np.random.seed(42)  # Reproducible results
    signal_pairs = []
    
    for i in range(num_pairs):
        if i < 20:
            # Random complex signals
            x = np.random.randn(n) + 1j * np.random.randn(n)
            h = np.random.randn(n) + 1j * np.random.randn(n)
        elif i < 40:
            # Real sinusoids
            freqs_x = np.random.randint(1, n//4, size=3)
            freqs_h = np.random.randint(1, n//4, size=3)
            t = np.arange(n)
            
            x = sum(np.sin(2 * np.pi * f * t / n) for f in freqs_x)
            h = sum(np.cos(2 * np.pi * f * t / n) for f in freqs_h)
        elif i < 60:
            # Impulse-like signals
            x = np.zeros(n, dtype=complex)
            h = np.zeros(n, dtype=complex)
            
            # Random impulses
            impulse_locs_x = np.random.choice(n, size=min(3, n//4), replace=False)
            impulse_locs_h = np.random.choice(n, size=min(3, n//4), replace=False)
            
            x[impulse_locs_x] = np.random.randn(len(impulse_locs_x)) + 1j * np.random.randn(len(impulse_locs_x))
            h[impulse_locs_h] = np.random.randn(len(impulse_locs_h)) + 1j * np.random.randn(len(impulse_locs_h))
        elif i < 80:
            # Smooth signals (low-pass)
            x_freq = np.random.randn(n) + 1j * np.random.randn(n)
            h_freq = np.random.randn(n) + 1j * np.random.randn(n)
            
            # Low-pass filter (keep only low frequencies)
            cutoff = n // 4
            x_freq[cutoff:-cutoff] = 0
            h_freq[cutoff:-cutoff] = 0
            
            x = np.fft.ifft(x_freq)
            h = np.fft.ifft(h_freq)
        else:
            # Mixed: real + complex components
            x_real = np.random.randn(n)
            x_imag = np.random.randn(n) * 0.5
            h_real = np.random.randn(n)
            h_imag = np.random.randn(n) * 0.5
            
            x = x_real + 1j * x_imag
            h = h_real + 1j * h_imag
        
        # Normalize to prevent numerical issues
        x = x / (np.linalg.norm(x) + 1e-12)
        h = h / (np.linalg.norm(h) + 1e-12)
        
        signal_pairs.append((x, h))
    
    return signal_pairs


def test_convolution_invariance(n: int = 32, num_pairs: int = 100) -> Dict:
    """
    Test RFT convolution invariance failure across multiple signal pairs.
    
    Pass criteria:
    - Median RFT error ≥ 10%
    - At least 90% of samples show RFT error ≥ 5%
    - DFT error should be < 1% (control test)
    """
    print("=" * 80)
    print("RFT CONVOLUTION INVARIANCE TEST")
    print("=" * 80)
    print(f"Testing n={n}, {num_pairs} signal pairs")
    print(f"Pass criteria: Median RFT error ≥ 10%, ≥90% samples with error ≥ 5%")
    print("-" * 80)
    
    # Generate test signals
    print("Generating test signals...")
    signal_pairs = generate_test_signals(n, num_pairs)
    
    # Initialize RFT
    rft = CanonicalTrueRFT(n)
    
    # Test results
    dft_errors = []
    rft_errors = []
    
    print("Running convolution tests...")
    
    for i, (x, h) in enumerate(signal_pairs):
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{num_pairs}")
        
        # Test DFT (control - should satisfy convolution property)
        dft_error = test_dft_convolution_property(x, h)
        dft_errors.append(dft_error * 100)  # Convert to percentage
        
        # Test RFT (should violate convolution property)
        rft_error = test_rft_convolution_property(x, h, rft)
        rft_errors.append(rft_error * 100)  # Convert to percentage
    
    # Analyze results
    dft_errors = np.array(dft_errors)
    rft_errors = np.array(rft_errors)
    
    # Statistics
    dft_median = np.median(dft_errors)
    dft_mean = np.mean(dft_errors)
    dft_max = np.max(dft_errors)
    
    rft_median = np.median(rft_errors)
    rft_mean = np.mean(rft_errors)
    rft_min = np.min(rft_errors)
    rft_max = np.max(rft_errors)
    
    # Pass criteria
    rft_samples_above_5pct = np.sum(rft_errors >= 5.0)
    rft_fraction_above_5pct = rft_samples_above_5pct / len(rft_errors)
    
    pass_median = rft_median >= 10.0
    pass_fraction = rft_fraction_above_5pct >= 0.90
    pass_dft_control = dft_median < 1.0  # DFT should work correctly
    
    overall_pass = pass_median and pass_fraction and pass_dft_control
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\nDFT (Control Test):")
    print(f"  Median error: {dft_median:.4f}% {'✓' if pass_dft_control else '✗'}")
    print(f"  Mean error: {dft_mean:.4f}%")
    print(f"  Max error: {dft_max:.4f}%")
    
    print(f"\nRFT (Distinctness Test):")
    print(f"  Median error: {rft_median:.2f}% {'✓' if pass_median else '✗'}")
    print(f"  Mean error: {rft_mean:.2f}%")
    print(f"  Min error: {rft_min:.2f}%")
    print(f"  Max error: {rft_max:.2f}%")
    print(f"  Samples ≥5%: {rft_samples_above_5pct}/{len(rft_errors)} ({rft_fraction_above_5pct:.1%}) {'✓' if pass_fraction else '✗'}")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"Median error criterion: {'PASS' if pass_median else 'FAIL'} ({rft_median:.1f}% ≥ 10%)")
    print(f"Sample fraction criterion: {'PASS' if pass_fraction else 'FAIL'} ({rft_fraction_above_5pct:.1%} ≥ 90%)")
    print(f"DFT control test: {'PASS' if pass_dft_control else 'FAIL'} ({dft_median:.3f}% < 1%)")
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
    
    if overall_pass:
        print("\n🎉 CONVOLUTION INVARIANCE VIOLATION CONFIRMED!")
        print("   - RFT does NOT satisfy DFT's convolution property")
        print("   - This proves RFT is fundamentally different from DFT")
        print("   - RFT cannot be a re-parameterized DFT")
    else:
        print("\n❌ Test failed - RFT may be too similar to DFT")
        if not pass_dft_control:
            print("   WARNING: DFT control test failed - check implementation")
    
    return {
        'overall_pass': overall_pass,
        'pass_median': pass_median,
        'pass_fraction': pass_fraction,
        'pass_dft_control': pass_dft_control,
        'dft_errors': dft_errors,
        'rft_errors': rft_errors,
        'dft_median': dft_median,
        'rft_median': rft_median,
        'rft_fraction_above_5pct': rft_fraction_above_5pct
    }


def analyze_error_distribution(rft_errors: np.ndarray, dft_errors: np.ndarray) -> None:
    """Analyze and display error distribution statistics"""
    print(f"\n" + "=" * 80)
    print("ERROR DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Percentiles
    rft_percentiles = [5, 25, 50, 75, 95]
    dft_percentiles = [5, 25, 50, 75, 95]
    
    print(f"\nRFT Error Percentiles:")
    for p in rft_percentiles:
        val = np.percentile(rft_errors, p)
        print(f"  {p:2d}th percentile: {val:6.2f}%")
    
    print(f"\nDFT Error Percentiles:")
    for p in dft_percentiles:
        val = np.percentile(dft_errors, p)
        print(f"  {p:2d}th percentile: {val:8.4f}%")
    
    # Error ranges
    print(f"\nError Range Analysis:")
    
    rft_ranges = [
        (0, 5, "0-5%"),
        (5, 10, "5-10%"),
        (10, 20, "10-20%"),
        (20, 50, "20-50%"),
        (50, 100, "50-100%"),
        (100, float('inf'), ">100%")
    ]
    
    for min_err, max_err, label in rft_ranges:
        if max_err == float('inf'):
            count = np.sum(rft_errors >= min_err)
        else:
            count = np.sum((rft_errors >= min_err) & (rft_errors < max_err))
        
        fraction = count / len(rft_errors)
        print(f"  RFT errors {label:8s}: {count:3d} samples ({fraction:5.1%})")


if __name__ == "__main__":
    start_time = time.time()
    
    # Run main test
    results = test_convolution_invariance(n=32, num_pairs=100)
    
    # Extended analysis
    analyze_error_distribution(results['rft_errors'], results['dft_errors'])
    
    # Test multiple sizes for robustness
    print(f"\n" + "=" * 80)
    print("MULTI-SIZE VALIDATION")
    print("=" * 80)
    
    all_pass = True
    for n in [16, 32, 64]:
        print(f"\nTesting n={n}...")
        result = test_convolution_invariance(n=n, num_pairs=50)
        status = "PASS" if result['overall_pass'] else "FAIL"
        print(f"  Result: {status} (median error: {result['rft_median']:.1f}%)")
        
        if not result['overall_pass']:
            all_pass = False
    
    # Final verdict
    print(f"\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if results['overall_pass'] and all_pass:
        print("✅ TEST PASSED: RFT violates DFT convolution property")
        print("   RFT is mathematically distinct from DFT")
        print("   Convolution theorem does not hold for RFT")
        print("   This proves RFT is not a re-parameterized DFT")
    else:
        print("❌ TEST FAILED: RFT may be too similar to DFT")
        print("   Further investigation needed")
    
    elapsed = time.time() - start_time
    print(f"\nTest completed in {elapsed:.2f} seconds")
    
    # Exit with appropriate code
    sys.exit(0 if (results['overall_pass'] and all_pass) else 1)
