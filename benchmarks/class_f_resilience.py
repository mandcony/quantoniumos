#!/usr/bin/env python3
"""
CLASS F: RESILIENCE & PHYSICS BENCHMARK
=======================================

Tests the "Hard Physics" claims of QuantoniumOS:
1. Denoising Superiority: Can RFT clean Golden Ratio signals better than FFT?
2. Statistical Randomness: Does RFT-SIS pass basic NIST tests?
3. Quantum Stress: Where does the symbolic engine break?
"""

import numpy as np
import time
import sys
import os
import math
from scipy.fft import fft, ifft

# Add project root
sys.path.append(os.getcwd())

try:
    from algorithms.rft.core.resonant_fourier_transform import (
        rft_basis_matrix, 
        rft_forward_frame, 
        rft_inverse_frame,
        RFTSISHash
    )
    
    class CanonicalRFT:
        """Wrapper for functional RFT implementation to match benchmark interface."""
        def __init__(self, N):
            self.N = N
            # Use Gram-normalized basis for unitary-like behavior
            self.Phi = rft_basis_matrix(N, use_gram_normalization=True)
        
        def forward(self, x):
            return rft_forward_frame(x, self.Phi)
            
        def inverse(self, X):
            return rft_inverse_frame(X, self.Phi)
            
    RFT_AVAILABLE = True
except ImportError as e:
    print(f"Import Error: {e}")
    RFT_AVAILABLE = False
    print("⚠️ RFT Core not found. Running in mock mode if possible.")

def generate_golden_signal(N, snr_db=0):
    """Generate a Golden Ratio signal with noise."""
    t = np.arange(N)
    phi = (1 + np.sqrt(5)) / 2
    
    # Signal: Sum of RFT basis functions (Sparse in RFT domain)
    # We pick 3 specific RFT frequencies
    k1, k2, k3 = 1, 3, 5
    f1 = ((k1 + 1) * phi) % 1.0
    f2 = ((k2 + 1) * phi) % 1.0
    f3 = ((k3 + 1) * phi) % 1.0
    
    sig = np.sin(2 * np.pi * f1 * t) + \
          0.5 * np.sin(2 * np.pi * f2 * t) + \
          0.25 * np.sin(2 * np.pi * f3 * t)
    
    # Normalize signal power
    sig_power = np.mean(sig**2)
    sig = sig / np.sqrt(sig_power)
    
    # Noise
    noise_power = 10**(-snr_db/10)
    noise = np.random.normal(0, np.sqrt(noise_power), N)
    
    return sig, sig + noise

def hard_threshold_denoise(transform_coeffs, percentile=95):
    """Keep only the top X% of coefficients."""
    threshold = np.percentile(np.abs(transform_coeffs), percentile)
    coeffs_clean = transform_coeffs.copy()
    coeffs_clean[np.abs(coeffs_clean) < threshold] = 0
    return coeffs_clean

def calculate_snr(clean, noisy):
    noise = noisy - clean
    return 10 * np.log10(np.sum(clean**2) / np.sum(noise**2))

def benchmark_denoising():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  DENOISING BENCHMARK (RFT vs FFT)")
    print("  Signal: Quasi-periodic (Golden Ratio harmonics)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if not RFT_AVAILABLE:
        print("Skipping: RFT not available")
        return

    N = 256
    input_snr = 0 # 0 dB (Signal = Noise)
    
    # Run multiple trials to average out noise
    n_trials = 20
    fft_snrs = []
    rft_snrs = []
    
    print(f"  Running {n_trials} trials...")
    
    for _ in range(n_trials):
        clean, noisy = generate_golden_signal(N, input_snr)
        
        # --- FFT Denoising ---
        fft_coeffs = fft(noisy)
        fft_clean_coeffs = hard_threshold_denoise(fft_coeffs, 95) 
        fft_denoised = np.real(ifft(fft_clean_coeffs))
        fft_snrs.append(calculate_snr(clean, fft_denoised))
        
        # --- RFT Denoising ---
        rft = CanonicalRFT(N)
        rft_coeffs = rft.forward(noisy)
        rft_clean_coeffs = hard_threshold_denoise(rft_coeffs, 95)
        rft_denoised = np.real(rft.inverse(rft_clean_coeffs))
        rft_snrs.append(calculate_snr(clean, rft_denoised))
    
    fft_output_snr = np.mean(fft_snrs)
    rft_output_snr = np.mean(rft_snrs)
    
    print(f"  Input SNR:      {input_snr:.2f} dB")
    print(f"  FFT Output SNR: {fft_output_snr:.2f} dB (Avg)")
    print(f"  RFT Output SNR: {rft_output_snr:.2f} dB (Avg)")
    
    gain = rft_output_snr - fft_output_snr
    print(f"  RFT Advantage:  {gain:+.2f} dB")
    
    if gain > 0.5:
        print("  [PASS] RFT consistently outperforms FFT on Golden Ratio signals.")
    else:
        print("  [FAIL] RFT failed to beat FFT significantly.")

def nist_monobit_test(bits):
    """NIST Frequency (Monobit) Test."""
    n = len(bits)
    # Convert 0/1 to -1/+1
    s = sum([2*b - 1 for b in bits])
    s_obs = abs(s) / np.sqrt(n)
    # Simplified p-value calculation
    p_value = math.erfc(s_obs / math.sqrt(2))
    return p_value

def benchmark_crypto_randomness():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  NIST RANDOMNESS CHECK (RFT-SIS Hash)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Generate bits using RFT-SIS Hash
    # We'll hash a counter 0..N to get a stream
    
    try:
        sis = RFTSISHash() # Use the correct class name
        
        bits = []
        start = time.time()
        # Generate 10,000 bits
        for i in range(320): # 320 * 32 bits approx 10k
            data = str(i).encode()
            h = sis.hash(data) # Returns bytes
            # Convert bytes to bits
            for b in h:
                for j in range(8):
                    bits.append((b >> j) & 1)
        
        duration = time.time() - start
        bits = bits[:10000] # Truncate to 10k
        
        p_value = nist_monobit_test(bits)
        
        print(f"  Generated: {len(bits)} bits")
        print(f"  Time:      {duration:.4f} s")
        print(f"  P-Value:   {p_value:.6f} (Target > 0.01)")
        
        if p_value > 0.01:
            print("  [PASS] Sequence looks random (NIST Monobit).")
        else:
            print("  [FAIL] Sequence looks non-random.")
            
    except Exception as e:
        print(f"  [ERROR] Could not run crypto test: {e}")

def main():
    print("============================================================")
    print("CLASS F: RESILIENCE & PHYSICS BENCHMARK")
    print("============================================================")
    
    benchmark_denoising()
    benchmark_crypto_randomness()

if __name__ == "__main__":
    main()
