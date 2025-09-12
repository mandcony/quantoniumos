#!/usr/bin/env python3
"""
RFT vs DFT Distinctness Test

Goal: Prove RFT is not a re-parameterized DFT by showing fundamental basis separation.

Pass Criteria:
- Median residual ≥ 5% when projecting RFT basis onto DFT basis
- Subspace angle > 5° between RFT and DFT basis vectors

This test demonstrates that RFT and DFT span fundamentally different subspaces
and cannot be transformed into each other through simple re-parameterization.
"""

import numpy as np
import time
from typing import Tuple, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from true_rft_kernel import TrueRFTKernel, build_dft_matrix
    USE_REAL_RFT = True
except ImportError:
    print("Warning: Could not import TrueRFTKernel, using standalone implementation")
    USE_REAL_RFT = False
    
def compute_rft_basis(n: int) -> np.ndarray:
    """Compute RFT basis using the true kernel implementation"""
    if USE_REAL_RFT:
        rft = TrueRFTKernel(n)
        return rft.get_rft_matrix()
    else:
        # Fallback implementation
        basis = np.zeros((n, n), dtype=complex)
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        for k in range(n):
            for j in range(n):
                angle = 2 * np.pi * k * j * phi / n
                basis[k, j] = np.exp(-1j * angle)
                
        return basis / np.sqrt(n)  # Normalize


def compute_dft_basis(n: int) -> np.ndarray:
    """Compute standard DFT basis matrix"""
    if USE_REAL_RFT:
        return build_dft_matrix(n)
    else:
        # Fallback implementation
        basis = np.zeros((n, n), dtype=complex)
        
        for k in range(n):
            for j in range(n):
                angle = 2 * np.pi * k * j / n
                basis[k, j] = np.exp(-1j * angle)
                
        return basis / np.sqrt(n)  # Normalize


def compute_subspace_angle(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the principal angle between two subspaces.
    
    Uses SVD of A^H * B to find canonical angles between subspaces.
    Returns the principal (largest) angle in degrees.
    """
    # Ensure matrices are orthonormal
    U_A, _, _ = np.linalg.svd(A, full_matrices=False)
    U_B, _, _ = np.linalg.svd(B, full_matrices=False)
    
    # Compute cross-correlation matrix
    C = np.conj(U_A.T) @ U_B
    
    # SVD to get canonical correlations
    _, sigma, _ = np.linalg.svd(C, full_matrices=False)
    
    # Principal angle (largest angle)
    # sigma contains cosines of canonical angles
    min_cosine = np.min(sigma.real)  # Minimum cosine = maximum angle
    principal_angle_rad = np.arccos(np.clip(min_cosine, -1, 1))
    
    return np.degrees(principal_angle_rad)


def compute_projection_residual(rft_basis: np.ndarray, dft_basis: np.ndarray) -> Tuple[float, List[float]]:
    """
    Project RFT basis vectors onto DFT subspace and compute residuals.
    
    For each RFT basis vector, project onto DFT subspace and measure
    the residual (unprojectable component).
    
    Returns:
        median_residual: Median residual as percentage
        all_residuals: List of all residual percentages
    """
    residuals = []
    
    for i in range(rft_basis.shape[0]):
        rft_vector = rft_basis[i, :]
        
        # Project RFT vector onto DFT subspace
        # projection = DFT @ DFT^H @ rft_vector
        projection = dft_basis.conj().T @ (dft_basis @ rft_vector)
        
        # Compute residual
        residual = rft_vector - projection
        residual_norm = np.linalg.norm(residual)
        original_norm = np.linalg.norm(rft_vector)
        
        # Residual as percentage
        residual_percent = (residual_norm / original_norm) * 100
        residuals.append(residual_percent)
    
    return np.median(residuals), residuals


def test_rft_dft_distinctness(sizes: List[int] = [8, 16, 32, 64]) -> dict:
    """
    Test RFT vs DFT distinctness across multiple sizes.
    
    Pass criteria:
    - Median residual ≥ 5% for projection test
    - Subspace angle > 5° for angle test
    """
    results = {
        'sizes': sizes,
        'median_residuals': [],
        'subspace_angles': [],
        'all_residuals': [],
        'pass_projection': [],
        'pass_angle': [],
        'overall_pass': True
    }
    
    print("=" * 80)
    print("RFT vs DFT DISTINCTNESS TEST")
    print("=" * 80)
    print(f"Testing sizes: {sizes}")
    print(f"Pass criteria: Median residual ≥ 5%, Subspace angle > 5°")
    print("-" * 80)
    
    for n in sizes:
        print(f"\nTesting size n={n}...")
        
        # Get basis matrices
        rft_basis = compute_rft_basis(n)
        dft_basis = compute_dft_basis(n)
        
        # Test 1: Projection residual
        median_residual, all_residuals = compute_projection_residual(rft_basis, dft_basis)
        pass_projection = median_residual >= 5.0
        
        # Test 2: Subspace angle
        subspace_angle = compute_subspace_angle(rft_basis, dft_basis)
        pass_angle = subspace_angle > 5.0
        
        # Store results
        results['median_residuals'].append(median_residual)
        results['subspace_angles'].append(subspace_angle)
        results['all_residuals'].append(all_residuals)
        results['pass_projection'].append(pass_projection)
        results['pass_angle'].append(pass_angle)
        
        # Print results
        print(f"  Median projection residual: {median_residual:.2f}% {'✓' if pass_projection else '✗'}")
        print(f"  Subspace angle: {subspace_angle:.2f}° {'✓' if pass_angle else '✗'}")
        print(f"  Individual residuals: {[f'{r:.1f}%' for r in all_residuals[:5]]}...")
        
        if not (pass_projection and pass_angle):
            results['overall_pass'] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_projection_pass = all(results['pass_projection'])
    all_angle_pass = all(results['pass_angle'])
    overall_pass = all_projection_pass and all_angle_pass
    
    print(f"Projection test: {'PASS' if all_projection_pass else 'FAIL'}")
    print(f"Angle test: {'PASS' if all_angle_pass else 'FAIL'}")
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
    
    if overall_pass:
        print("\n🎉 RFT IS PROVEN DISTINCT FROM DFT!")
        print("   - RFT basis vectors cannot be represented in DFT subspace")
        print("   - Subspaces have significant angular separation")
        print("   - RFT is NOT a re-parameterized DFT")
    else:
        print("\n❌ Test failed - RFT may be too similar to DFT")
    
    results['overall_pass'] = overall_pass
    return results


def run_extended_analysis(n: int = 32) -> None:
    """Run extended analysis for detailed understanding"""
    print(f"\n" + "=" * 80)
    print(f"EXTENDED ANALYSIS (n={n})")
    print("=" * 80)
    
    if USE_REAL_RFT:
        rft_basis = compute_rft_basis(n)
    else:
        rft_basis = compute_rft_basis(n)
    dft_basis = compute_dft_basis(n)
    
    # Analyze frequency content differences
    print("\nFrequency content analysis:")
    
    # Compare a few basis vectors directly
    for k in [0, 1, n//4, n//2]:
        if k < n:
            rft_vec = rft_basis[k, :]
            dft_vec = dft_basis[k, :]
            
            # Compute correlation
            correlation = np.abs(np.vdot(rft_vec, dft_vec))**2
            print(f"  |⟨RFT[{k}], DFT[{k}]⟩|² = {correlation:.4f}")
    
    # Analyze golden ratio effect
    phi = (1 + np.sqrt(5)) / 2
    print(f"\nGolden ratio φ = {phi:.6f}")
    print(f"φ/1 ratio = {phi:.6f} (should be ≠ 1 for distinctness)")
    
    # Spectral analysis
    rft_eigenvals = np.linalg.eigvals(rft_basis @ rft_basis.conj().T)
    dft_eigenvals = np.linalg.eigvals(dft_basis @ dft_basis.conj().T)
    
    print(f"\nSpectral properties:")
    print(f"  RFT eigenvalue spread: {np.max(rft_eigenvals.real) - np.min(rft_eigenvals.real):.6f}")
    print(f"  DFT eigenvalue spread: {np.max(dft_eigenvals.real) - np.min(dft_eigenvals.real):.6f}")


if __name__ == "__main__":
    start_time = time.time()
    
    # Run main test
    results = test_rft_dft_distinctness([8, 16, 32, 64])
    
    # Run extended analysis
    run_extended_analysis(32)
    
    # Final verdict
    print(f"\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if results['overall_pass']:
        print("✅ TEST PASSED: RFT is mathematically distinct from DFT")
        print("   RFT cannot be expressed as a re-parameterized DFT")
        print("   This proves genuine algorithmic novelty")
    else:
        print("❌ TEST FAILED: RFT may be too similar to DFT")
        print("   Further investigation needed")
    
    elapsed = time.time() - start_time
    print(f"\nTest completed in {elapsed:.2f} seconds")
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_pass'] else 1)
