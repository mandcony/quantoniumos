#!/usr/bin/env python3
"""
Sparsity Theorem Analysis for Golden-Quasi-Periodic Signals

Goal: Prove (or disprove) that RFT gives better sparsity than DFT for 
signals with golden-ratio frequency components.

Strategy:
1. Define the signal class precisely
2. Compute RFT and DFT coefficients analytically
3. Derive coefficient decay bounds
4. Compare sparsity metrics
"""

import numpy as np
from scipy.fft import fft, dct
from scipy.linalg import qr
import matplotlib.pyplot as plt

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2

def build_closed_form_rft(n, beta=1.0, sigma=1.0):
    """Build the closed-form RFT matrix Ψ = D_φ C_σ F"""
    # DFT matrix (normalized)
    k = np.arange(n)
    F = np.exp(-2j * np.pi * np.outer(k, k) / n) / np.sqrt(n)
    
    # Chirp matrix C_σ
    C_sigma = np.diag(np.exp(1j * np.pi * sigma * k**2 / n))
    
    # Golden phase matrix D_φ
    frac_parts = (k / PHI) % 1.0
    D_phi = np.diag(np.exp(2j * np.pi * beta * frac_parts))
    
    # Ψ = D_φ C_σ F
    Psi = D_phi @ C_sigma @ F
    return Psi


def golden_quasi_periodic_signal(n, K, amplitudes=None, seed=None):
    """
    Generate a K-golden-quasi-periodic signal:
    x_m = Σ_{j=1}^K a_j exp(2πi · {jφ} · m / n)
    
    where {·} denotes fractional part.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if amplitudes is None:
        amplitudes = np.random.randn(K) + 1j * np.random.randn(K)
    
    m = np.arange(n)
    x = np.zeros(n, dtype=complex)
    
    for j in range(1, K + 1):
        freq = (j * PHI) % 1.0  # fractional part of jφ
        x += amplitudes[j-1] * np.exp(2j * np.pi * freq * m / n)
    
    return x, amplitudes


def compute_sparsity_metrics(coeffs, thresholds=[0.01, 0.001, 0.0001]):
    """Compute sparsity at various relative thresholds."""
    max_coeff = np.max(np.abs(coeffs))
    results = {}
    for tau in thresholds:
        abs_threshold = tau * max_coeff
        n_small = np.sum(np.abs(coeffs) < abs_threshold)
        results[tau] = n_small / len(coeffs)
    return results


def energy_concentration(coeffs, k):
    """Fraction of energy in top-k coefficients."""
    sorted_energy = np.sort(np.abs(coeffs)**2)[::-1]
    total_energy = np.sum(sorted_energy)
    top_k_energy = np.sum(sorted_energy[:k])
    return top_k_energy / total_energy


def analyze_signal(n, K, num_trials=100):
    """
    Compare RFT vs DFT sparsity for K-golden-quasi-periodic signals.
    """
    Psi = build_closed_form_rft(n)
    
    rft_concentration = []
    dft_concentration = []
    dct_concentration = []
    
    for trial in range(num_trials):
        x, _ = golden_quasi_periodic_signal(n, K, seed=trial)
        
        # Compute coefficients
        rft_coeffs = Psi @ x
        dft_coeffs = fft(x) / np.sqrt(n)
        dct_coeffs = dct(np.real(x), norm='ortho')  # DCT only for real part
        
        # Energy concentration in top-K coefficients
        rft_concentration.append(energy_concentration(rft_coeffs, K))
        dft_concentration.append(energy_concentration(dft_coeffs, K))
        dct_concentration.append(energy_concentration(dct_coeffs, K))
    
    return {
        'rft': np.mean(rft_concentration),
        'dft': np.mean(dft_concentration),
        'dct': np.mean(dct_concentration),
        'rft_std': np.std(rft_concentration),
        'dft_std': np.std(dft_concentration),
        'dct_std': np.std(dct_concentration),
    }


def theoretical_dft_analysis(n, K):
    """
    Analyze DFT coefficients of golden-quasi-periodic signals theoretically.
    
    For x_m = Σ_{j=1}^K a_j exp(2πi · {jφ} · m / n)
    
    The DFT coefficient at frequency k is:
    X_k = (1/√n) Σ_m x_m exp(-2πi k m / n)
        = (1/√n) Σ_m Σ_j a_j exp(2πi m ({jφ} - k/n))
        = Σ_j a_j · (1/√n) Σ_m exp(2πi m ({jφ} - k/n))
    
    The inner sum is a geometric series:
    = Σ_j a_j · (1/√n) · [1 - exp(2πi n ({jφ} - k/n))] / [1 - exp(2πi ({jφ} - k/n))]
    
    Key insight: The DFT spreads energy across ALL bins because {jφ} is 
    generically irrational and doesn't align with any k/n.
    """
    print("=== Theoretical DFT Analysis ===")
    print(f"n={n}, K={K}")
    print()
    
    # For each j, compute the frequency mismatch
    for j in range(1, K + 1):
        freq_j = (j * PHI) % 1.0
        
        # Find closest DFT bin
        closest_k = int(round(freq_j * n))
        mismatch = freq_j - closest_k / n
        
        # The DFT coefficient at bin k has magnitude proportional to
        # |sin(πn·mismatch) / sin(π·mismatch)|
        # which is maximized when mismatch = 0 (exact alignment)
        
        print(f"j={j}: freq={freq_j:.6f}, closest_k={closest_k}, mismatch={mismatch:.6f}")
        
        # Compute leakage factor
        if abs(mismatch) < 1e-10:
            leakage = 1.0
        else:
            leakage = abs(np.sin(np.pi * n * mismatch) / (n * np.sin(np.pi * mismatch)))
        
        print(f"       DFT leakage factor: {leakage:.4f}")
    
    print()


def theoretical_rft_analysis(n, K, beta=1.0, sigma=1.0):
    """
    Analyze RFT coefficients theoretically.
    
    Ψ = D_φ C_σ F, so Ψx = D_φ C_σ (Fx)
    
    The RFT applies:
    1. DFT: spreads energy due to frequency mismatch
    2. Chirp: rotates phases by exp(iπσk²/n) 
    3. Golden phase: rotates by exp(2πiβ{k/φ})
    
    Key question: Does the golden phase "undo" the spreading from (1)?
    
    Answer: For the closed-form RFT, NO. The phases are applied AFTER the DFT,
    so they just rotate the already-spread coefficients. They don't concentrate.
    
    The closed-form RFT has the same sparsity as DFT (they're equivalent up to phases).
    """
    print("=== Theoretical RFT Analysis ===")
    print(f"n={n}, K={K}, β={beta}, σ={sigma}")
    print()
    
    print("Key insight: Ψ = D_φ C_σ F")
    print("Since D_φ and C_σ are diagonal, they only rotate phases.")
    print("They do NOT change coefficient magnitudes.")
    print()
    print("Therefore: |Ψx|_k = |Fx|_k for all k")
    print("The RFT has IDENTICAL sparsity to the DFT.")
    print()
    print("This is the trivial equivalence from Remark 1.5 in the paper.")
    print()


def verify_magnitude_equivalence(n, K, num_trials=10):
    """Verify that |Ψx| = |Fx| (up to normalization)."""
    Psi = build_closed_form_rft(n)
    
    print("=== Magnitude Equivalence Verification ===")
    
    max_diff = 0
    for trial in range(num_trials):
        x, _ = golden_quasi_periodic_signal(n, K, seed=trial)
        
        rft_coeffs = Psi @ x
        dft_coeffs = fft(x) / np.sqrt(n)
        
        # Compare magnitudes
        diff = np.abs(np.abs(rft_coeffs) - np.abs(dft_coeffs))
        max_diff = max(max_diff, np.max(diff))
    
    print(f"Max |Ψx| - |Fx| difference: {max_diff:.2e}")
    if max_diff < 1e-10:
        print("CONFIRMED: RFT and DFT have identical coefficient magnitudes.")
        print("Therefore, they have IDENTICAL sparsity properties.")
    else:
        print("WARNING: Magnitudes differ. This contradicts theory.")
    
    return max_diff


def main():
    print("=" * 70)
    print("SPARSITY ANALYSIS FOR RFT vs DFT")
    print("=" * 70)
    print()
    
    n = 256
    K = 5
    
    # Theoretical analysis
    theoretical_dft_analysis(n, K)
    theoretical_rft_analysis(n, K)
    
    # Verify magnitude equivalence
    diff = verify_magnitude_equivalence(n, K)
    print()
    
    # Empirical comparison anyway
    print("=== Empirical Energy Concentration (Top-K) ===")
    for n in [64, 128, 256, 512]:
        for K in [3, 5, 10]:
            results = analyze_signal(n, K, num_trials=50)
            print(f"n={n:4d}, K={K:2d}: "
                  f"RFT={results['rft']:.4f}±{results['rft_std']:.4f}, "
                  f"DFT={results['dft']:.4f}±{results['dft_std']:.4f}, "
                  f"DCT={results['dct']:.4f}±{results['dct_std']:.4f}")
    
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The closed-form RFT Ψ = D_φ C_σ F has:")
    print("  |Ψx|_k = |Fx|_k  for all k")
    print()
    print("Therefore, the closed-form RFT has NO sparsity advantage over DFT.")
    print("Any claimed sparsity advantage would require a DIFFERENT construction.")
    print()
    print("The Sparsity Conjecture (Conjecture 8.3) is FALSE for the closed-form RFT.")


if __name__ == "__main__":
    main()
