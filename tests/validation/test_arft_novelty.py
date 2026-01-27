# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Validation Script for Adaptive Resonant Fourier Transform (ARFT)
Tests:
1. Unitarity (Energy Preservation)
2. Magnitude Difference vs FFT (The "Novelty Threshold")
3. Sparsity Comparison on Quasi-Periodic Signals
"""

import numpy as np
import sys
import os

# Add workspace root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from algorithms.rft.kernels.arft_kernel import build_resonant_kernel, arft_forward, arft_inverse

def test_arft_properties():
    print("="*60)
    print("ARFT VALIDATION SUITE")
    print("="*60)
    
    N = 256
    coupling = 0.15
    
    # 1. Generate Kernel
    print(f"\n[1] Generating ARFT Kernel (N={N}, coupling={coupling})...")
    # Create a synthetic "resonance map" simulating a structured signal
    t = np.linspace(0, 1, N)
    phi = (1 + np.sqrt(5)) / 2
    resonance_map = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 5 * phi * t)
    
    kernel = build_resonant_kernel(N, resonance_map=resonance_map, coupling=coupling)
    
    # 2. Test Unitarity
    print("\n[2] Testing Unitarity...")
    I = np.eye(N, dtype=complex)
    K_dag_K = kernel.conj().T @ kernel
    error = np.linalg.norm(K_dag_K - I)
    print(f"    ||K^H K - I||_F = {error:.2e}")
    if error < 1e-12:
        print("    ✅ Unitarity CONFIRMED")
    else:
        print("    ❌ Unitarity FAILED")
        
    # 3. Test Magnitude Difference vs FFT
    print("\n[3] Testing Magnitude Difference vs FFT...")
    # Test signal: Random complex
    np.random.seed(42)
    x = np.random.randn(N) + 1j * np.random.randn(N)
    
    y_fft = np.fft.fft(x, norm='ortho')
    y_arft = arft_forward(x, kernel)
    
    mag_fft = np.abs(y_fft)
    mag_arft = np.abs(y_arft)
    
    mag_diff = np.linalg.norm(mag_fft - mag_arft)
    print(f"    || |FFT(x)| - |ARFT(x)| || = {mag_diff:.4f}")
    
    if mag_diff > 1e-6:
        print("    ✅ Magnitude Difference CONFIRMED (Novelty Threshold Passed)")
    else:
        print("    ❌ Magnitudes Identical (Failed Novelty Check)")
        
    # 4. Sparsity Test on Quasi-Periodic Signal
    print("\n[4] Testing Sparsity on Golden Quasi-Periodic Signal...")
    # Signal: sum of golden ratio harmonics
    x_qp = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 10 * phi * t)
    
    y_fft_qp = np.fft.fft(x_qp, norm='ortho')
    y_arft_qp = arft_forward(x_qp, kernel)
    
    # Gini Index (Sparsity Metric: higher is sparser)
    def gini(array):
        array = np.abs(array)
        array += 1e-9 # avoid div by zero
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((2 * np.sum(index * array)) / (n * np.sum(array))) - ((n + 1) / n)

    gini_fft = gini(y_fft_qp)
    gini_arft = gini(y_arft_qp)
    
    print(f"    FFT Gini Sparsity:  {gini_fft:.4f}")
    print(f"    ARFT Gini Sparsity: {gini_arft:.4f}")
    
    if gini_arft > gini_fft:
        print(f"    ✅ ARFT is SPARSER by {(gini_arft/gini_fft - 1)*100:.2f}%")
    else:
        print(f"    ❌ ARFT is LESS SPARSE by {(1 - gini_arft/gini_fft)*100:.2f}%")

if __name__ == "__main__":
    test_arft_properties()
