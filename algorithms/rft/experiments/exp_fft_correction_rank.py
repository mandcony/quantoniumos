# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Experiment 2: Is (U - F) Low-Rank?
==================================

Goal: Determine if the difference between RFT basis U and DFT basis F
has fast-decaying singular values, enabling O(N log N + Nr) transform.

If singular values decay fast → FFT + low-rank correction is viable
If singular values decay slowly → FFT + correction is NOT a shortcut

This directly validates/invalidates the hypothesis in rft_via_fft_correction().
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from algorithms.rft.fast.cached_basis import get_fast_basis


def run_experiment():
    """Analyze singular value structure of U - F."""
    
    print("=" * 80)
    print("EXPERIMENT 2: IS (U - F) LOW-RANK?")
    print("=" * 80)
    print()
    
    sizes = [128, 256, 512]
    variant = 'golden'
    test_ranks = [2, 4, 8, 16, 32, 64]
    n_test_signals = 20
    
    np.random.seed(42)
    
    all_results = {}
    
    for N in sizes:
        print(f"\n{'='*60}")
        print(f"N = {N}")
        print(f"{'='*60}")
        
        # Get exact RFT basis
        U = get_fast_basis(N, variant, method='exact')
        
        # Get DFT basis (real part for comparison)
        F = np.fft.fft(np.eye(N), norm='ortho')
        F_real = F.real
        
        # Compute difference
        C = U - F_real
        
        # SVD of difference
        U_c, s_c, Vh_c = np.linalg.svd(C, full_matrices=False)
        
        # Singular value analysis
        print(f"\n  SINGULAR VALUE SPECTRUM OF (U - F):")
        print(f"    Max singular value:     {s_c[0]:.4f}")
        print(f"    Min singular value:     {s_c[-1]:.4f}")
        print(f"    Condition number:       {s_c[0] / (s_c[-1] + 1e-12):.1f}")
        
        # Energy concentration
        total_energy = np.sum(s_c**2)
        cumulative_energy = np.cumsum(s_c**2) / total_energy
        
        print(f"\n  ENERGY CONCENTRATION:")
        for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
            k_needed = np.searchsorted(cumulative_energy, threshold) + 1
            print(f"    {threshold:.0%} energy captured by top {k_needed} singular values")
        
        # Test approximation quality for various ranks
        print(f"\n  APPROXIMATION ERROR AT VARIOUS RANKS:")
        print(f"    {'Rank':<8} {'Rel Error':<15} {'Transform Error':<15}")
        print(f"    {'-'*40}")
        
        rank_results = []
        
        for r in test_ranks:
            if r > N:
                continue
            
            # Low-rank approximation of C
            C_r = U_c[:, :r] @ np.diag(s_c[:r]) @ Vh_c[:r, :]
            
            # Matrix approximation error
            matrix_error = np.linalg.norm(C - C_r, 'fro') / np.linalg.norm(C, 'fro')
            
            # Transform approximation error on test signals
            transform_errors = []
            for _ in range(n_test_signals):
                x = np.random.randn(N)
                
                # Exact RFT
                y_exact = U.T @ x
                
                # Approximate: FFT + low-rank correction
                fft_result = np.fft.fft(x, norm='ortho').real
                correction = C_r.T @ x
                y_approx = fft_result + correction
                
                rel_err = np.linalg.norm(y_exact - y_approx) / (np.linalg.norm(y_exact) + 1e-12)
                transform_errors.append(rel_err)
            
            mean_transform_error = np.mean(transform_errors)
            
            print(f"    {r:<8} {matrix_error:<15.4f} {mean_transform_error:<15.4f}")
            
            rank_results.append({
                'rank': r,
                'matrix_error': matrix_error,
                'transform_error': mean_transform_error,
            })
        
        all_results[N] = {
            'singular_values': s_c,
            'rank_results': rank_results,
        }
        
        # Verdict for this N
        # Check if small rank gives acceptable error
        best_small_rank = None
        for rr in rank_results:
            if rr['rank'] <= 16 and rr['transform_error'] < 0.1:
                best_small_rank = rr['rank']
                break
        
        if best_small_rank:
            print(f"\n  >>> VERDICT: Low-rank correction MAY work (rank {best_small_rank} gives <10% error)")
        else:
            print(f"\n  >>> VERDICT: Low-rank correction FAILS (no small rank gives <10% error)")
    
    # Save singular value plots
    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(sizes), figsize=(4*len(sizes), 4))
    if len(sizes) == 1:
        axes = [axes]
    
    for ax, N in zip(axes, sizes):
        s = all_results[N]['singular_values']
        ax.semilogy(s, 'b-', linewidth=1.5)
        ax.set_xlabel('Index')
        ax.set_ylabel('Singular Value')
        ax.set_title(f'N={N}: Singular Values of (U - F)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp2_singular_values.png', dpi=150)
    print(f"\n  Plot saved to: {output_dir / 'exp2_singular_values.png'}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    low_rank_viable = False
    for N, res in all_results.items():
        for rr in res['rank_results']:
            if rr['rank'] <= 16 and rr['transform_error'] < 0.1:
                low_rank_viable = True
    
    if low_rank_viable:
        print("Some sizes show viable low-rank correction.")
        print("Further investigation warranted for specific applications.")
    else:
        print("LOW-RANK CORRECTION IS NOT VIABLE.")
        print("The difference (U - F) does not have fast-decaying singular values.")
        print("FFT + correction is NOT a valid shortcut to O(N log N) RFT.")
    
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
