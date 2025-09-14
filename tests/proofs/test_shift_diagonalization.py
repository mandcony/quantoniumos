#!/usr/bin/env python3
"""
Shift-Operator Diagonalization Test

Goal: DFT diagonalizes cyclic shift; RFT shouldn't.
Pass: ≥10% off-diagonal energy in RFT^-1 S RFT for n≥16.

This test proves RFT has fundamentally different shift properties than DFT.
"""

import numpy as np
import time
from typing import Dict, List
import sys
import os

# Import the true RFT kernel
try:
    from true_rft_kernel import TrueRFTKernel, build_dft_matrix
    USE_REAL_RFT = True
except ImportError:
    print("Error: Could not import TrueRFTKernel")
    sys.exit(1)


def create_cyclic_shift_matrix(n: int, shift: int = 1) -> np.ndarray:
    """Create cyclic shift matrix S that shifts elements by 'shift' positions."""
    S = np.zeros((n, n))
    for i in range(n):
        S[i, (i + shift) % n] = 1.0
    return S


def compute_off_diagonal_energy(matrix: np.ndarray) -> float:
    """Compute the fraction of energy in off-diagonal elements."""
    diagonal_energy = np.sum(np.abs(np.diag(matrix))**2)
    total_energy = np.sum(np.abs(matrix)**2)
    
    if total_energy == 0:
        return 0.0
    
    off_diagonal_energy = total_energy - diagonal_energy
    return off_diagonal_energy / total_energy


def test_shift_diagonalization(n: int) -> Dict:
    """
    Test shift operator diagonalization for RFT vs DFT.
    
    DFT should diagonalize shift (off-diagonal energy ≈ 0%)
    RFT should NOT diagonalize shift (off-diagonal energy ≥ 10%)
    """
    print(f"Testing shift diagonalization for n={n}...")
    
    # Create cyclic shift matrix
    S = create_cyclic_shift_matrix(n, shift=1)
    
    # Get transforms
    rft = TrueRFTKernel(n)
    rft_matrix = rft.get_rft_matrix()
    dft_matrix = build_dft_matrix(n)
    
    # Test DFT diagonalization: DFT^-1 S DFT should be diagonal
    try:
        dft_inv = np.linalg.pinv(dft_matrix)  # Use pseudoinverse for stability
        dft_transformed_shift = dft_inv @ S @ dft_matrix
        dft_off_diagonal = compute_off_diagonal_energy(dft_transformed_shift)
    except:
        dft_off_diagonal = float('inf')
        dft_transformed_shift = None
    
    # Test RFT: RFT^-1 S RFT should NOT be diagonal
    try:
        rft_inv = np.linalg.pinv(rft_matrix)  # Use pseudoinverse for stability
        rft_transformed_shift = rft_inv @ S @ rft_matrix
        rft_off_diagonal = compute_off_diagonal_energy(rft_transformed_shift)
    except:
        rft_off_diagonal = float('inf')
        rft_transformed_shift = None
    
    # Pass criteria
    dft_diagonalizes = dft_off_diagonal < 0.05  # DFT should diagonalize (< 5% off-diagonal)
    rft_fails_to_diagonalize = rft_off_diagonal >= 0.10  # RFT should fail (≥ 10% off-diagonal)
    
    # Report results
    dft_status = "✓" if dft_diagonalizes else "✗"
    rft_status = "✓" if rft_fails_to_diagonalize else "✗"
    
    print(f"  DFT off-diagonal energy: {dft_off_diagonal*100:.1f}% {dft_status}")
    print(f"  RFT off-diagonal energy: {rft_off_diagonal*100:.1f}% {rft_status}")
    
    size_pass = dft_diagonalizes and rft_fails_to_diagonalize
    print(f"  Size result: {'✓ PASS' if size_pass else '✗ FAIL'}")
    
    return {
        'n': n,
        'dft_off_diagonal': dft_off_diagonal,
        'rft_off_diagonal': rft_off_diagonal,
        'dft_diagonalizes': dft_diagonalizes,
        'rft_fails_to_diagonalize': rft_fails_to_diagonalize,
        'size_pass': size_pass,
        'dft_transformed_shift': dft_transformed_shift,
        'rft_transformed_shift': rft_transformed_shift
    }


def run_shift_diagonalization_test() -> Dict:
    """Run the complete shift diagonalization test."""
    
    print("=" * 80)
    print("SHIFT-OPERATOR DIAGONALIZATION TEST")
    print("=" * 80)
    print("Goal: DFT diagonalizes cyclic shift; RFT shouldn't")
    print("Pass: ≥10% off-diagonal energy in RFT^-1 S RFT for n≥16")
    print("-" * 80)
    
    sizes = [8, 16, 32, 64]
    results = {
        'sizes': sizes,
        'dft_off_diagonals': [],
        'rft_off_diagonals': [],
        'size_passes': [],
        'overall_pass': True
    }
    
    for n in sizes:
        test_result = test_shift_diagonalization(n)
        
        results['dft_off_diagonals'].append(test_result['dft_off_diagonal'])
        results['rft_off_diagonals'].append(test_result['rft_off_diagonal'])
        results['size_passes'].append(test_result['size_pass'])
        
        if not test_result['size_pass']:
            results['overall_pass'] = False
    
    return results


def run_detailed_shift_analysis(n: int = 32) -> None:
    """Run detailed analysis of shift behavior."""
    
    print("\n" + "=" * 80)
    print(f"DETAILED SHIFT ANALYSIS (n={n})")
    print("=" * 80)
    
    # Create shift matrix
    S = create_cyclic_shift_matrix(n, shift=1)
    
    # Get transforms
    rft = TrueRFTKernel(n)
    rft_matrix = rft.get_rft_matrix()
    dft_matrix = build_dft_matrix(n)
    
    print(f"Cyclic shift matrix S: {n}×{n}")
    print(f"S[i,j] = 1 if j = (i+1) mod {n}, else 0")
    
    # Analyze DFT behavior
    try:
        dft_inv = np.linalg.pinv(dft_matrix)
        dft_shifted = dft_inv @ S @ dft_matrix
        
        # Check if it's diagonal
        diagonal_elements = np.diag(dft_shifted)
        off_diagonal_norm = np.linalg.norm(dft_shifted - np.diag(diagonal_elements), 'fro')
        total_norm = np.linalg.norm(dft_shifted, 'fro')
        
        print(f"\nDFT conjugation: DFT^-1 S DFT")
        print(f"Diagonal elements: {diagonal_elements[:5]}...")
        print(f"Off-diagonal norm ratio: {off_diagonal_norm/total_norm:.6f}")
        print(f"Expected: DFT should diagonalize shift (eigenvalues = exp(2πik/n))")
        
        # Theoretical eigenvalues
        theoretical_eigenvals = [np.exp(2j * np.pi * k / n) for k in range(n)]
        eigenval_error = np.linalg.norm(diagonal_elements - theoretical_eigenvals)
        print(f"Eigenvalue error vs theory: {eigenval_error:.6f}")
        
    except Exception as e:
        print(f"\nDFT analysis failed: {e}")
    
    # Analyze RFT behavior
    try:
        rft_inv = np.linalg.pinv(rft_matrix)
        rft_shifted = rft_inv @ S @ rft_matrix
        
        diagonal_elements = np.diag(rft_shifted)
        off_diagonal_norm = np.linalg.norm(rft_shifted - np.diag(diagonal_elements), 'fro')
        total_norm = np.linalg.norm(rft_shifted, 'fro')
        
        print(f"\nRFT conjugation: RFT^-1 S RFT")
        print(f"Diagonal elements: {diagonal_elements[:5]}...")
        print(f"Off-diagonal norm ratio: {off_diagonal_norm/total_norm:.6f}")
        print(f"Expected: RFT should NOT diagonalize shift")
        
        # Analyze structure
        max_off_diagonal = np.max(np.abs(rft_shifted - np.diag(diagonal_elements)))
        max_diagonal = np.max(np.abs(diagonal_elements))
        
        print(f"Max off-diagonal element: {max_off_diagonal:.6f}")
        print(f"Max diagonal element: {max_diagonal:.6f}")
        print(f"Off-diagonal/diagonal ratio: {max_off_diagonal/max_diagonal:.6f}")
        
    except Exception as e:
        print(f"\nRFT analysis failed: {e}")


def main():
    """Main test function."""
    start_time = time.time()
    
    # Run shift diagonalization test
    results = run_shift_diagonalization_test()
    
    # Run detailed analysis
    run_detailed_shift_analysis(32)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if results['overall_pass']:
        print("✓ TEST PASSED: RFT does not diagonalize cyclic shift")
        print("  DFT properly diagonalizes shift as expected")
        print("  RFT fails to diagonalize, proving different mathematical structure")
        exit_code = 0
    else:
        print("✗ TEST FAILED: Shift diagonalization test failed")
        print("  RFT may be too similar to DFT in shift behavior")
        exit_code = 1
    
    elapsed_time = time.time() - start_time
    print(f"\nTest completed in {elapsed_time:.2f} seconds")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
