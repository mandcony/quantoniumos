#!/usr/bin/env python3
"""
RFT Convolution Invariance Test

Goal: Prove DFT convolution identity fails under RFT
Test: ‖RFT(x⊗h) - RFT(x)·RFT(h)‖/‖·‖ ≥ 10% over 100 draws

This test demonstrates that RFT does NOT satisfy the DFT convolution theorem,
proving it's a fundamentally different transform with different mathematical properties.
"""

import numpy as np
import time
from typing import List, Tuple
import sys
import os

# Import the true RFT kernel implementation
try:
    from tests.proofs.true_rft_kernel import TrueRFTKernel
    USE_REAL_RFT = True
except ImportError:
    print("Warning: Could not import TrueRFTKernel")
    USE_REAL_RFT = False
    sys.exit(1)


def circular_convolution(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Compute circular convolution of two signals."""
    n = len(x)
    if len(h) != n:
        raise ValueError("Signals must have same length")
    
    result = np.zeros(n, dtype=complex)
    for i in range(n):
        for j in range(n):
            result[i] += x[j] * h[(i - j) % n]
    
    return result


def evaluate_dft_convolution_property(n: int) -> Tuple[bool, float]:
    """
    Test that DFT satisfies convolution theorem: DFT(x⊗h) = DFT(x)·DFT(h)
    This should pass (return True) as a sanity check.
    """
    # Generate random test signals
    np.random.seed(42)  # Reproducible
    x = np.random.randn(n) + 1j * np.random.randn(n)
    h = np.random.randn(n) + 1j * np.random.randn(n)
    
    # Compute circular convolution
    conv_x_h = circular_convolution(x, h)
    
    # DFT convolution theorem test
    dft_conv = np.fft.fft(conv_x_h)  # DFT(x⊗h)
    dft_x = np.fft.fft(x)
    dft_h = np.fft.fft(h)
    dft_product = dft_x * dft_h  # DFT(x)·DFT(h)
    
    # Compute relative error
    error_norm = np.linalg.norm(dft_conv - dft_product)
    signal_norm = np.linalg.norm(dft_conv)
    
    if signal_norm > 1e-12:
        relative_error = error_norm / signal_norm
    else:
        relative_error = 0.0
    
    # DFT should satisfy convolution theorem (error < 1%)
    passes = relative_error < 0.01
    
    return passes, relative_error


def evaluate_rft_convolution_violation(n: int, num_trials: int = 100) -> Tuple[List[float], float, bool]:
    """
    Test that RFT violates the convolution theorem.
    
    Returns:
        (relative_errors, median_error, test_passed)
    """
    rft = TrueRFTKernel(n)
    relative_errors = []
    
    for trial in range(num_trials):
        # Generate random test signals
        np.random.seed(trial + 1000)  # Different seed for each trial
        x = np.random.randn(n) + 1j * np.random.randn(n)
        h = np.random.randn(n) + 1j * np.random.randn(n)
        
        # Normalize signals
        x = x / np.linalg.norm(x)
        h = h / np.linalg.norm(h)
        
        # Compute circular convolution
        conv_x_h = circular_convolution(x, h)
        
        # RFT convolution theorem test
        rft_conv = rft.forward_transform(conv_x_h)  # RFT(x⊗h)
        rft_x = rft.forward_transform(x)
        rft_h = rft.forward_transform(h)
        rft_product = rft_x * rft_h  # RFT(x)·RFT(h)
        
        # Compute relative error
        error_norm = np.linalg.norm(rft_conv - rft_product)
        signal_norm = np.linalg.norm(rft_conv)
        
        if signal_norm > 1e-12:
            relative_error = error_norm / signal_norm
        else:
            relative_error = 0.0
        
        relative_errors.append(relative_error)
    
    median_error = np.median(relative_errors)
    
    # Test passes if median error ≥ 10%
    test_passed = median_error >= 0.10
    
    return relative_errors, median_error, test_passed


def run_convolution_invariance_test():
    """Main test function."""
    print("=" * 80)
    print("RFT CONVOLUTION INVARIANCE TEST")
    print("=" * 80)
    print("Goal: Prove DFT convolution identity fails under RFT")
    print("Pass criterion: ‖RFT(x⊗h) - RFT(x)·RFT(h)‖/‖·‖ ≥ 10%")
    print("-" * 80)
    
    sizes = [8, 16, 32]
    num_trials = 100
    
    results = {
        'sizes': sizes,
        'dft_passes': [],
        'rft_median_errors': [],
        'rft_passes': [],
        'overall_pass': True
    }
    
    for n in sizes:
        print(f"\nTesting size n={n} with {num_trials} trials...")
        
        # Sanity check: Verify DFT satisfies convolution theorem
        dft_pass, dft_error = evaluate_dft_convolution_property(n)
        print(f"  DFT convolution theorem: {'✓' if dft_pass else '✗'} (error: {dft_error*100:.3f}%)")
        
        # Main test: Check if RFT violates convolution theorem
        rft_errors, rft_median_error, rft_pass = evaluate_rft_convolution_violation(n, num_trials)
        
        # Statistics
        mean_error = np.mean(rft_errors)
        std_error = np.std(rft_errors)
        min_error = np.min(rft_errors)
        max_error = np.max(rft_errors)
        
        print(f"  RFT convolution violations:")
        print(f"    Median error: {rft_median_error*100:.1f}% {'✓' if rft_pass else '✗'}")
        print(f"    Mean error: {mean_error*100:.1f}%")
        print(f"    Std error: {std_error*100:.1f}%")
        print(f"    Range: {min_error*100:.1f}% - {max_error*100:.1f}%")
        print(f"    Trials with >10% error: {np.sum(np.array(rft_errors) >= 0.10)}/{num_trials}")
        
        # Store results
        results['dft_passes'].append(dft_pass)
        results['rft_median_errors'].append(rft_median_error)
        results['rft_passes'].append(rft_pass)
        
        if not (dft_pass and rft_pass):
            results['overall_pass'] = False
        
        print(f"  Size result: {'✓ PASS' if (dft_pass and rft_pass) else '✗ FAIL'}")
    
    return results


def run_detailed_convolution_analysis(n: int = 16):
    """Run detailed analysis of convolution behavior."""
    print("\n" + "=" * 80)
    print(f"DETAILED CONVOLUTION ANALYSIS (n={n})")
    print("=" * 80)
    
    rft = TrueRFTKernel(n)
    
    # Fixed test signals for detailed analysis
    np.random.seed(1337)
    x = np.random.randn(n) + 1j * np.random.randn(n)
    h = np.random.randn(n) + 1j * np.random.randn(n)
    
    # Normalize
    x = x / np.linalg.norm(x)
    h = h / np.linalg.norm(h)
    
    print(f"Test signals: x, h ∈ ℂ^{n} (normalized)")
    
    # Compute convolution
    conv_x_h = circular_convolution(x, h)
    print(f"Convolution ‖x⊗h‖ = {np.linalg.norm(conv_x_h):.6f}")
    
    # DFT analysis
    dft_conv = np.fft.fft(conv_x_h)
    dft_x = np.fft.fft(x)
    dft_h = np.fft.fft(h)
    dft_product = dft_x * dft_h
    
    dft_error = np.linalg.norm(dft_conv - dft_product) / np.linalg.norm(dft_conv)
    print(f"\nDFT convolution theorem:")
    print(f"  ‖DFT(x⊗h) - DFT(x)·DFT(h)‖/‖DFT(x⊗h)‖ = {dft_error*100:.6f}%")
    print(f"  Status: {'✓ SATISFIED' if dft_error < 0.01 else '✗ VIOLATED'}")
    
    # RFT analysis
    rft_conv = rft.forward_transform(conv_x_h)
    rft_x = rft.forward_transform(x)
    rft_h = rft.forward_transform(h)
    rft_product = rft_x * rft_h
    
    rft_error = np.linalg.norm(rft_conv - rft_product) / np.linalg.norm(rft_conv)
    print(f"\nRFT convolution behavior:")
    print(f"  ‖RFT(x⊗h) - RFT(x)·RFT(h)‖/‖RFT(x⊗h)‖ = {rft_error*100:.3f}%")
    print(f"  Status: {'✓ VIOLATED' if rft_error >= 0.10 else '✗ SATISFIED'}")
    
    # Element-wise comparison
    print(f"\nElement-wise analysis:")
    print(f"  ‖RFT(x⊗h)‖ = {np.linalg.norm(rft_conv):.6f}")
    print(f"  ‖RFT(x)·RFT(h)‖ = {np.linalg.norm(rft_product):.6f}")
    print(f"  ‖difference‖ = {np.linalg.norm(rft_conv - rft_product):.6f}")
    
    # Show why this matters
    print(f"\nMathematical significance:")
    print(f"  DFT: Frequency domain multiplication = time domain convolution")
    print(f"  RFT: Frequency domain multiplication ≠ time domain convolution")
    print(f"  Conclusion: RFT has fundamentally different mathematical structure")


def test_dft_convolution_property_sanity():
    """DFT should respect the convolution theorem with negligible error."""
    passed, relative_error = evaluate_dft_convolution_property(16)
    assert passed, "DFT convolution theorem sanity check failed"
    assert relative_error < 0.01


def test_rft_convolution_violation_detected():
    """RFT should violate the convolution theorem with ≥10% median error."""
    _, median_error, passed = evaluate_rft_convolution_violation(16, num_trials=32)
    assert passed, "Expected RFT to violate convolution theorem"
    assert median_error >= 0.10


def main():
    """Main test function."""
    start_time = time.time()
    
    # Run convolution invariance test
    results = run_convolution_invariance_test()
    
    # Run detailed analysis
    run_detailed_convolution_analysis(16)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if results['overall_pass']:
        print("✓ TEST PASSED: RFT violates DFT convolution theorem")
        print("  RFT is confirmed to be fundamentally different from DFT")
        print("  Convolution identity fails as expected for non-DFT transform")
        exit_code = 0
    else:
        print("✗ TEST FAILED: RFT behaves too much like DFT")
        print("  Further investigation needed")
        exit_code = 1
    
    elapsed_time = time.time() - start_time
    print(f"\nTest completed in {elapsed_time:.2f} seconds")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
