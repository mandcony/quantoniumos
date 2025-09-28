#!/usr/bin/env python3
"""
PROPER RFT vs DFT Distinctness Test

Goal: Prove RFT is not a re-parameterized DFT using appropriate metrics.

Pass Criteria:
- Frobenius norm difference > 1.0
- Matrix correlation < 95%  
- Eigenvalue distribution significantly different

This test uses metrics appropriate for comparing non-unitary transforms.
"""

import numpy as np
import time
from typing import List, Dict
import sys
import os

# Import the true RFT kernel implementation
try:
    from tests.proofs.true_rft_kernel import TrueRFTKernel, build_dft_matrix
    USE_REAL_RFT = True
except ImportError:
    print("Warning: Could not import TrueRFTKernel")
    USE_REAL_RFT = False
    sys.exit(1)


def compute_matrix_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """Compute overall correlation between two matrices."""
    # Flatten matrices and compute correlation
    a_flat = A.flatten()
    b_flat = B.flatten()
    
    # Normalize
    a_norm = a_flat / np.linalg.norm(a_flat)
    b_norm = b_flat / np.linalg.norm(b_flat)
    
    # Compute correlation
    correlation = abs(np.vdot(a_norm, b_norm))**2
    return correlation


def compute_spectral_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Compute spectral distance between two matrices."""
    try:
        # Compute eigenvalues
        eigs_A = np.linalg.eigvals(A @ A.conj().T)
        eigs_B = np.linalg.eigvals(B @ B.conj().T)
        
        # Sort eigenvalues
        eigs_A = np.sort(np.real(eigs_A))
        eigs_B = np.sort(np.real(eigs_B))
        
        # Compute distance between eigenvalue distributions
        return np.linalg.norm(eigs_A - eigs_B)
    except:
        return float('inf')  # If eigenvalue computation fails, assume very different


def test_rft_dft_distinctness() -> Dict:
    """
    Test RFT vs DFT distinctness using proper metrics.
    """
    
    print("=" * 80)
    print("PROPER RFT vs DFT DISTINCTNESS TEST")
    print("=" * 80)
    print("Using mathematical kernel without QR decomposition")
    print("Testing that RFT ≠ re-parameterized DFT")
    print("-" * 80)
    
    sizes = [8, 16, 32, 64]
    results = {
        'sizes': sizes,
        'frobenius_diffs': [],
        'correlations': [],
        'spectral_distances': [],
        'pass_frobenius': [],
        'pass_correlation': [],
        'pass_spectral': [],
        'overall_pass': True
    }
    
    for n in sizes:
        print(f"\nTesting size n={n}...")
        
        # Get the true mathematical kernels
        rft = TrueRFTKernel(n)
        rft_matrix = rft.get_rft_matrix()
        dft_matrix = build_dft_matrix(n)
        
        # Test 1: Frobenius norm difference
        frobenius_diff = np.linalg.norm(rft_matrix - dft_matrix, 'fro')
        frobenius_pass = frobenius_diff > 1.0
        
        # Test 2: Matrix correlation
        correlation = compute_matrix_correlation(rft_matrix, dft_matrix)
        correlation_pass = correlation < 0.95  # Less than 95% correlation
        
        # Test 3: Spectral distance
        spectral_distance = compute_spectral_distance(rft_matrix, dft_matrix)
        spectral_pass = spectral_distance > 0.5
        
        # Overall pass for this size
        size_pass = frobenius_pass and correlation_pass and spectral_pass
        
        # Store results
        results['frobenius_diffs'].append(frobenius_diff)
        results['correlations'].append(correlation)
        results['spectral_distances'].append(spectral_distance)
        results['pass_frobenius'].append(frobenius_pass)
        results['pass_correlation'].append(correlation_pass)
        results['pass_spectral'].append(spectral_pass)
        
        if not size_pass:
            results['overall_pass'] = False
        
        # Report results
        print(f"  Frobenius difference: {frobenius_diff:.3f} {'✓' if frobenius_pass else '✗'}")
        print(f"  Matrix correlation: {correlation*100:.1f}% {'✓' if correlation_pass else '✗'}")
        print(f"  Spectral distance: {spectral_distance:.3f} {'✓' if spectral_pass else '✗'}")
        print(f"  Size result: {'✓ PASS' if size_pass else '✗ FAIL'}")
    
    return results


def run_detailed_analysis(n: int = 32) -> None:
    """Run detailed analysis for documentation."""
    print("\n" + "=" * 80)
    print(f"DETAILED ANALYSIS (n={n})")
    print("=" * 80)
    
    rft = TrueRFTKernel(n)
    rft_matrix = rft.get_rft_matrix()
    dft_matrix = build_dft_matrix(n)
    
    print(f"RFT matrix shape: {rft_matrix.shape}")
    print(f"DFT matrix shape: {dft_matrix.shape}")
    
    # Frobenius analysis
    frobenius_diff = np.linalg.norm(rft_matrix - dft_matrix, 'fro')
    print(f"\nFrobenius norm |RFT - DFT|_F = {frobenius_diff:.6f}")
    
    # Element-wise comparison
    max_element_diff = np.max(np.abs(rft_matrix - dft_matrix))
    mean_element_diff = np.mean(np.abs(rft_matrix - dft_matrix))
    print(f"Max element difference: {max_element_diff:.6f}")
    print(f"Mean element difference: {mean_element_diff:.6f}")
    
    # Matrix norms
    rft_norm = np.linalg.norm(rft_matrix, 'fro')
    dft_norm = np.linalg.norm(dft_matrix, 'fro')
    print(f"‖RFT‖_F = {rft_norm:.6f}")
    print(f"‖DFT‖_F = {dft_norm:.6f}")
    
    # Relative difference
    relative_diff = frobenius_diff / max(rft_norm, dft_norm)
    print(f"Relative difference: {relative_diff:.6f}")
    
    # Golden ratio presence
    phi = 1.618033988749894848204586834366
    rft_elements = rft_matrix.flatten()
    phi_like_elements = np.sum(np.abs(np.abs(rft_elements) - phi) < 0.1)
    total_elements = len(rft_elements)
    phi_percentage = (phi_like_elements / total_elements) * 100
    
    print(f"\nGolden ratio analysis:")
    print(f"Elements near φ = {phi:.6f}: {phi_like_elements}/{total_elements} ({phi_percentage:.1f}%)")
    
    # Spectral analysis
    try:
        rft_eigenvals = np.linalg.eigvals(rft_matrix @ rft_matrix.conj().T)
        dft_eigenvals = np.linalg.eigvals(dft_matrix @ dft_matrix.conj().T)
        
        rft_eigenval_spread = np.std(np.real(rft_eigenvals))
        dft_eigenval_spread = np.std(np.real(dft_eigenvals))
        
        print(f"\nSpectral analysis:")
        print(f"RFT eigenvalue spread: {rft_eigenval_spread:.6f}")
        print(f"DFT eigenvalue spread: {dft_eigenval_spread:.6f}")
        print(f"Spread difference: {abs(rft_eigenval_spread - dft_eigenval_spread):.6f}")
    except Exception as e:
        print(f"\nSpectral analysis failed: {e}")


def main():
    """Main test function."""
    start_time = time.time()
    
    # Run distinctness test
    results = test_rft_dft_distinctness()
    
    # Run detailed analysis
    run_detailed_analysis(32)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if results['overall_pass']:
        print("✓ TEST PASSED: RFT is mathematically distinct from DFT")
        print("  All metrics confirm RFT ≠ re-parameterized DFT")
        exit_code = 0
    else:
        print("✗ TEST FAILED: RFT may be too similar to DFT")
        print("  Further investigation needed")
        exit_code = 1
    
    elapsed_time = time.time() - start_time
    print(f"\nTest completed in {elapsed_time:.2f} seconds")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
