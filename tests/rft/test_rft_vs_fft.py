#!/usr/bin/env python3
"""
Test and benchmark closed-form RFT against standard FFT
Validates:
- Transform correctness
- Unitarity
- Performance comparison
- Frequency domain behavior
"""
import numpy as np
import time
from algorithms.rft.core.closed_form_rft import rft_forward, rft_inverse, rft_unitary_error, rft_matrix

def test_unitarity():
    """Test that RFT is a unitary transform"""
    print("=" * 70)
    print("TEST 1: Unitarity Verification")
    print("=" * 70)
    
    sizes = [8, 16, 32, 64, 128, 256]
    for n in sizes:
        error = rft_unitary_error(n, trials=10)
        status = "✓ PASS" if error < 1e-10 else "✗ FAIL"
        print(f"N={n:4d}: Round-trip error = {error:.2e} {status}")
    print()

def test_parseval():
    """Test Parseval's theorem (energy preservation)"""
    print("=" * 70)
    print("TEST 2: Energy Preservation (Parseval's Theorem)")
    print("=" * 70)
    
    sizes = [8, 16, 32, 64, 128]
    for n in sizes:
        # Generate random signal
        rng = np.random.default_rng(42)
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        
        # Time domain energy
        energy_time = np.sum(np.abs(x) ** 2)
        
        # RFT frequency domain energy
        X_rft = rft_forward(x)
        energy_rft = np.sum(np.abs(X_rft) ** 2)
        
        # FFT frequency domain energy
        X_fft = np.fft.fft(x, norm="ortho")
        energy_fft = np.sum(np.abs(X_fft) ** 2)
        
        # Compute relative errors
        err_rft = abs(energy_rft - energy_time) / max(1e-16, energy_time)
        err_fft = abs(energy_fft - energy_time) / max(1e-16, energy_time)
        
        status = "✓ PASS" if err_rft < 1e-10 else "✗ FAIL"
        print(f"N={n:4d}: Time={energy_time:.6f}, RFT={energy_rft:.6f}, FFT={energy_fft:.6f}")
        print(f"       RFT error={err_rft:.2e}, FFT error={err_fft:.2e} {status}")
    print()

def test_orthogonality():
    """Test matrix orthogonality: Ψ† Ψ = I"""
    print("=" * 70)
    print("TEST 3: Matrix Orthogonality (Ψ† Ψ = I)")
    print("=" * 70)
    
    sizes = [8, 16, 32, 64]
    for n in sizes:
        Psi = rft_matrix(n)
        PsiH = np.conj(Psi.T)
        
        # Should be identity
        I_test = PsiH @ Psi
        I_true = np.eye(n, dtype=np.complex128)
        
        error = np.linalg.norm(I_test - I_true, 'fro') / n
        status = "✓ PASS" if error < 1e-10 else "✗ FAIL"
        print(f"N={n:4d}: ||Ψ†Ψ - I||_F / n = {error:.2e} {status}")
    print()

def test_signal_reconstruction():
    """Test reconstruction of known signals"""
    print("=" * 70)
    print("TEST 4: Signal Reconstruction")
    print("=" * 70)
    
    n = 64
    
    # Test signals
    signals = {
        "Impulse": np.array([1.0] + [0.0] * (n-1)),
        "Constant": np.ones(n),
        "Sine wave": np.sin(2 * np.pi * 5 * np.arange(n) / n),
        "Complex exp": np.exp(2j * np.pi * 3 * np.arange(n) / n),
        "Random": np.random.default_rng(123).normal(size=n) + 1j * np.random.default_rng(456).normal(size=n)
    }
    
    for name, x in signals.items():
        # RFT round trip
        X_rft = rft_forward(x)
        x_rec_rft = rft_inverse(X_rft)
        err_rft = np.linalg.norm(x_rec_rft - x) / max(1e-16, np.linalg.norm(x))
        
        # FFT round trip
        X_fft = np.fft.fft(x)
        x_rec_fft = np.fft.ifft(X_fft)
        err_fft = np.linalg.norm(x_rec_fft - x) / max(1e-16, np.linalg.norm(x))
        
        status = "✓ PASS" if err_rft < 1e-10 else "✗ FAIL"
        print(f"{name:15s}: RFT error={err_rft:.2e}, FFT error={err_fft:.2e} {status}")
    print()

def test_spectrum_comparison():
    """Compare RFT and FFT spectral characteristics"""
    print("=" * 70)
    print("TEST 5: Spectrum Comparison (RFT vs FFT)")
    print("=" * 70)
    
    n = 64
    # Create a test signal with multiple frequencies
    t = np.arange(n)
    x = (np.sin(2 * np.pi * 5 * t / n) + 
         0.5 * np.sin(2 * np.pi * 10 * t / n) + 
         0.25 * np.cos(2 * np.pi * 3 * t / n))
    
    X_rft = rft_forward(x)
    X_fft = np.fft.fft(x, norm="ortho")
    
    print(f"Signal: Multi-frequency sinusoid (N={n})")
    print(f"Peak RFT magnitude: {np.max(np.abs(X_rft)):.6f}")
    print(f"Peak FFT magnitude: {np.max(np.abs(X_fft)):.6f}")
    print(f"RFT spectrum L2 norm: {np.linalg.norm(X_rft):.6f}")
    print(f"FFT spectrum L2 norm: {np.linalg.norm(X_fft):.6f}")
    
    # The spectra will differ due to phase modulation, but norms should match
    norm_diff = abs(np.linalg.norm(X_rft) - np.linalg.norm(X_fft))
    print(f"Spectrum norm difference: {norm_diff:.2e}")
    print()

def benchmark_performance():
    """Benchmark RFT vs FFT performance"""
    print("=" * 70)
    print("TEST 6: Performance Benchmark")
    print("=" * 70)
    
    sizes = [64, 128, 256, 512, 1024, 2048]
    n_iterations = 100
    
    print(f"Running {n_iterations} iterations per size...\n")
    print(f"{'Size':>6} | {'RFT (ms)':>10} | {'FFT (ms)':>10} | {'Ratio':>8}")
    print("-" * 50)
    
    for n in sizes:
        rng = np.random.default_rng(789)
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        
        # Warm up
        _ = rft_forward(x)
        _ = np.fft.fft(x, norm="ortho")
        
        # Benchmark RFT
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = rft_forward(x)
        time_rft = (time.perf_counter() - start) * 1000 / n_iterations
        
        # Benchmark FFT
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = np.fft.fft(x, norm="ortho")
        time_fft = (time.perf_counter() - start) * 1000 / n_iterations
        
        ratio = time_rft / time_fft
        print(f"{n:6d} | {time_rft:10.4f} | {time_fft:10.4f} | {ratio:8.2f}x")
    
    print("\nNote: RFT uses FFT internally with additional phase modulation")
    print()

def test_linearity():
    """Test that RFT is linear: RFT(ax + by) = a*RFT(x) + b*RFT(y)"""
    print("=" * 70)
    print("TEST 7: Linearity Property")
    print("=" * 70)
    
    n = 64
    rng = np.random.default_rng(999)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    y = rng.normal(size=n) + 1j * rng.normal(size=n)
    a, b = 2.5 + 1.5j, -1.2 + 0.8j
    
    # Direct computation
    lhs = rft_forward(a * x + b * y)
    
    # Linear combination
    rhs = a * rft_forward(x) + b * rft_forward(y)
    
    error = np.linalg.norm(lhs - rhs) / max(1e-16, np.linalg.norm(lhs))
    status = "✓ PASS" if error < 1e-10 else "✗ FAIL"
    
    print(f"Testing: RFT(ax + by) = a·RFT(x) + b·RFT(y)")
    print(f"Coefficients: a = {a}, b = {b}")
    print(f"Relative error: {error:.2e} {status}")
    print()

def main():
    print("\n" + "=" * 70)
    print(" RFT vs FFT Comprehensive Test Suite")
    print("=" * 70)
    print()
    
    # Run all tests
    test_unitarity()
    test_parseval()
    test_orthogonality()
    test_signal_reconstruction()
    test_spectrum_comparison()
    test_linearity()
    benchmark_performance()
    
    print("=" * 70)
    print(" Test Suite Complete")
    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
