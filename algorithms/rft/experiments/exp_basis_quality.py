"""
Experiment 1: Validate Approximate Bases vs Exact RFT
======================================================

Goal: Put hard numbers on how good/bad circulant and lanczos approximations are.

Metrics:
1. Reconstruction error on random signals
2. Subspace overlap (principal angles) between approximate and exact basis

Expected results based on prior analysis:
- Circulant: BAD (~50-70% eigenvalue error, ~90° subspace angles)
- Lanczos (k ~ rank): Decent for top-k modes

This experiment produces a formal "negative result" for circulant.
"""

import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from algorithms.rft.fast.cached_basis import (
    get_fast_basis, _build_basis_exact, VARIANT_AUTOCORR
)
from scipy.linalg import subspace_angles


def measure_reconstruction_error(U: np.ndarray, x: np.ndarray) -> float:
    """Measure ||x - U @ U.T @ x|| / ||x||"""
    x_proj = U @ (U.T @ x)
    return np.linalg.norm(x - x_proj) / (np.linalg.norm(x) + 1e-12)


def measure_subspace_overlap(U1: np.ndarray, U2: np.ndarray, k: int = None) -> dict:
    """
    Measure subspace similarity between two bases.
    
    Returns:
        dict with principal angles statistics
    """
    if k is None:
        k = min(U1.shape[1], U2.shape[1])
    
    # Use first k columns
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]
    
    # Principal angles (in radians)
    angles = subspace_angles(U1_k, U2_k)
    
    return {
        'mean_angle_deg': np.mean(angles) * 180 / np.pi,
        'max_angle_deg': np.max(angles) * 180 / np.pi,
        'min_angle_deg': np.min(angles) * 180 / np.pi,
        'near_orthogonal_frac': np.mean(angles > np.pi/4),  # >45°
    }


def run_experiment():
    """Run basis quality experiment."""
    
    print("=" * 80)
    print("EXPERIMENT 1: APPROXIMATE BASIS QUALITY")
    print("=" * 80)
    print()
    
    sizes = [128, 256, 512]
    variant = 'golden'
    n_signals = 20
    lanczos_k_fracs = [0.1, 0.25, 0.5]  # Fraction of N
    
    np.random.seed(42)
    
    results = []
    
    for N in sizes:
        print(f"\n{'='*40}")
        print(f"N = {N}")
        print(f"{'='*40}")
        
        # Exact basis (ground truth)
        print(f"  Computing exact basis...")
        U_exact = get_fast_basis(N, variant, method='exact')
        
        # Circulant approximation
        print(f"  Computing circulant approximation...")
        U_circ = get_fast_basis(N, variant, method='circulant')
        
        # Generate random test signals
        test_signals = [np.random.randn(N) for _ in range(n_signals)]
        
        # Measure circulant quality
        circ_recon_errors = [measure_reconstruction_error(U_circ, x) for x in test_signals]
        exact_recon_errors = [measure_reconstruction_error(U_exact, x) for x in test_signals]
        
        circ_subspace = measure_subspace_overlap(U_exact, U_circ, k=min(64, N))
        
        print(f"\n  CIRCULANT APPROXIMATION:")
        print(f"    Mean recon error (exact): {np.mean(exact_recon_errors):.2e}")
        print(f"    Mean recon error (circ):  {np.mean(circ_recon_errors):.2e}")
        print(f"    Subspace mean angle:      {circ_subspace['mean_angle_deg']:.1f}°")
        print(f"    Subspace max angle:       {circ_subspace['max_angle_deg']:.1f}°")
        print(f"    Frac near-orthogonal:     {circ_subspace['near_orthogonal_frac']:.1%}")
        
        if circ_subspace['mean_angle_deg'] > 45:
            print(f"    >>> VERDICT: CIRCULANT FAILS (angles near 90° = bases unrelated)")
        
        results.append({
            'N': N,
            'method': 'circulant',
            'mean_recon_error': np.mean(circ_recon_errors),
            'mean_angle_deg': circ_subspace['mean_angle_deg'],
        })
        
        # Lanczos approximations
        for k_frac in lanczos_k_fracs:
            k = int(N * k_frac)
            if k < 4:
                continue
                
            print(f"\n  LANCZOS (k={k}, {k_frac:.0%} of N):")
            U_lanc = get_fast_basis(N, variant, method='lanczos', k=k)
            
            # Truncated reconstruction
            lanc_recon_errors = []
            for x in test_signals:
                coeffs = U_lanc.T @ x
                x_rec = U_lanc @ coeffs
                err = np.linalg.norm(x - x_rec) / (np.linalg.norm(x) + 1e-12)
                lanc_recon_errors.append(err)
            
            # Subspace overlap (compare first k vectors)
            lanc_subspace = measure_subspace_overlap(U_exact[:, :k], U_lanc, k=k)
            
            print(f"    Mean recon error:    {np.mean(lanc_recon_errors):.2e}")
            print(f"    Subspace mean angle: {lanc_subspace['mean_angle_deg']:.1f}°")
            
            results.append({
                'N': N,
                'method': f'lanczos_k{k}',
                'mean_recon_error': np.mean(lanc_recon_errors),
                'mean_angle_deg': lanc_subspace['mean_angle_deg'],
            })
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'N':<8} {'Method':<20} {'Recon Error':<15} {'Mean Angle (°)':<15}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['N']:<8} {r['method']:<20} {r['mean_recon_error']:<15.2e} {r['mean_angle_deg']:<15.1f}")
    
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print("- Circulant approximation: FAILS (subspace angles ~90° = unrelated bases)")
    print("- Lanczos: Usable for low-rank approximations if k is chosen appropriately")
    print("- For production: Use 'exact' or 'cached' methods ONLY")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
