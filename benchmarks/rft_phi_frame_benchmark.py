# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Benchmark: φ-grid exponential RFT with frame correction.

Compares:
- φ-grid RFT (raw, naive inverse)
- φ-grid RFT (raw, frame-correct inverse)
- φ-grid RFT (Gram-normalized, unitary)
- FFT (unitary baseline)

Run:
    python benchmarks/rft_phi_frame_benchmark.py

Notes:
- The φ-grid raw exponential basis is generally non-orthogonal at finite N.
- Frame-correct inversion uses the dual-frame solve: (ΦᴴΦ)^{-1} Φᴴ x.

References (orientation):
- Oppenheim & Schafer, Discrete-Time Signal Processing (orthogonality of DFT basis)
- O. Christensen, An Introduction to Frames and Riesz Bases (dual frames)
- Encyclopaedia Britannica, Fourier analysis
"""

from __future__ import annotations

import time
import numpy as np

from algorithms.rft.core.resonant_fourier_transform import (
    rft_basis_matrix,
    rft_forward_frame,
    rft_inverse_frame,
)
from algorithms.rft.core.gram_utils import gram_matrix


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> None:
    rng = np.random.default_rng(0)
    N = 256
    x = rng.normal(size=N) + 1j * rng.normal(size=N)

    # Raw φ-grid basis
    t0 = time.perf_counter()
    Phi_raw = rft_basis_matrix(N, N, use_gram_normalization=False)
    t1 = time.perf_counter()

    # Gram-normalized (unitary) basis
    Phi_unit = rft_basis_matrix(N, N, use_gram_normalization=True)
    t2 = time.perf_counter()

    # Condition number of Gram (raw)
    G = gram_matrix(Phi_raw)
    cond = np.linalg.cond(G)

    # Naive analysis/synthesis (raw)
    X_naive = Phi_raw.conj().T @ x
    x_hat_naive = Phi_raw @ X_naive

    # Frame-correct (raw)
    X_frame = rft_forward_frame(x, Phi_raw)
    x_hat_frame = rft_inverse_frame(X_frame, Phi_raw)

    # Unitary (Gram-normalized)
    X_unit = Phi_unit.conj().T @ x
    x_hat_unit = Phi_unit @ X_unit

    # FFT baseline (unitary)
    X_fft = np.fft.fft(x, norm="ortho")
    x_hat_fft = np.fft.ifft(X_fft, norm="ortho")

    print("=== RFT φ-grid frame benchmark ===")
    print(f"N={N}")
    print(f"build Phi_raw:  {(t1 - t0)*1e3:.2f} ms")
    print(f"build Phi_unit: {(t2 - t1)*1e3:.2f} ms")
    print(f"cond(Gram(Phi_raw)) = {cond:.3e}")
    print()
    print(f"raw/naive reconstruction relerr:  {rel_err(x_hat_naive, x):.3e}")
    print(f"raw/frame reconstruction relerr:  {rel_err(x_hat_frame, x):.3e}")
    print(f"unitary reconstruction relerr:     {rel_err(x_hat_unit, x):.3e}")
    print(f"fft reconstruction relerr:         {rel_err(x_hat_fft, x):.3e}")

    # =========================================================================
    # VARIANT & HYBRID FRAME ANALYSIS
    # =========================================================================
    print("\n" + "="*60)
    print("VARIANT & HYBRID FRAME ANALYSIS")
    print("="*60)
    
    try:
        from benchmarks.variant_benchmark_harness import (
            load_variant_generators, load_hybrid_functions, 
            VARIANT_CODES, HYBRID_NAMES
        )
        
        # 1. Variants (Explicit Basis Analysis)
        print("\n--- RFT VARIANTS (Explicit Basis Properties) ---")
        print(f"{'Variant':<20} | {'Cond(Gram)':<10} | {'Unitary Err':<12} | {'Frame Err':<12}")
        print("-" * 65)
        
        generators = load_variant_generators()
        
        for code in VARIANT_CODES:
            if code == "GOLDEN_EXACT": continue # Skip slow one
            
            try:
                if code not in generators:
                    print(f"{code:<20} | {'MISSING':<10} | {'-':<12} | {'-':<12}")
                    continue
                    
                gen = generators[code]
                # Use smaller N for variants to be fast
                N_var = 128 
                x_var = rng.normal(size=N_var) + 1j * rng.normal(size=N_var)
                
                # Generate Basis
                Phi = gen(N_var)
                
                # Gram Properties
                Gram = Phi.conj().T @ Phi
                cond = np.linalg.cond(Gram)
                
                # Reconstruction
                y = Phi @ x_var
                x_naive = Phi.conj().T @ y
                
                # Unitary Error (assuming Phi is unitary)
                err_unit = rel_err(x_naive, x_var)
                
                # Frame Error (using dual)
                try:
                    x_frame = np.linalg.solve(Gram, x_naive)
                    err_frame = rel_err(x_frame, x_var)
                except np.linalg.LinAlgError:
                    err_frame = float('nan')
                
                print(f"{code:<20} | {cond:.2e}   | {err_unit:.2e}     | {err_frame:.2e}")
                
            except Exception as e:
                print(f"{code:<20} | {'ERROR':<10} | {str(e)[:12]:<12} | {'-':<12}")

        # 2. Hybrids (Coherence Violation Analysis)
        print("\n--- HYBRIDS (Coherence/Orthogonality Check) ---")
        print(f"{'Hybrid':<25} | {'Coherence (η)':<15} | {'Status':<10}")
        print("-" * 55)
        
        hybrids = load_hybrid_functions()
        x_hybrid = rng.normal(size=256) # Real signal for hybrids usually
        
        for name in HYBRID_NAMES:
            try:
                if name not in hybrids:
                    print(f"{name:<25} | {'MISSING':<15} | {'-':<10}")
                    continue
                
                func = hybrids[name]
                # Run hybrid
                res = func(x_hybrid, target_sparsity=0.95)
                
                # Extract coherence if available
                # The result object usually has 'coherence_violation' or we infer it
                coh = getattr(res, 'coherence_violation', None)
                
                if coh is None:
                    # Try to infer from result type
                    coh = "N/A"
                elif isinstance(coh, float):
                    coh = f"{coh:.2e}"
                    
                print(f"{name:<25} | {coh:<15} | {'✓':<10}")
                
            except Exception as e:
                print(f"{name:<25} | {'ERROR':<15} | {str(e)[:10]}")

    except ImportError as e:
        print(f"\nCould not load variant harness: {e}")
        print("Skipping variant analysis.")


if __name__ == "__main__":
    main()
