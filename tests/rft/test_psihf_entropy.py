# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Ψ†F Structure Test via Column Entropy
Tests that Ψ†F has high-entropy columns (not monomial/permutation-like)
"""
import numpy as np
from algorithms.rft.core.phi_phase_fft import rft_forward

def _psi_mat(n: int, beta: float, sigma: float) -> np.ndarray:
    # Build Ψ by applying rft_forward to canonical basis (cost O(n^2 log n) but small n in test)
    E = np.eye(n, dtype=np.complex128)
    cols = [rft_forward(E[:,j], beta=beta, sigma=sigma) for j in range(n)]
    return np.column_stack(cols)

def _shannon_entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-16, 1.0)
    return float(-(p * np.log(p)).sum())

def test_psihf_column_entropy_high():
    """
    Proves Ψ†F is NOT a simple permutation/monomial matrix by showing
    high Shannon entropy in column distributions. Monomial → entropy ≈ 0;
    random-ish → entropy ≈ log(n).
    """
    n = 128
    beta, sigma = 0.83, 1.25
    Psi = _psi_mat(n, beta, sigma)
    F = np.fft.fft(np.eye(n), norm="ortho", axis=0)
    S = Psi.conj().T @ F  # Ψᴴ F
    # Columnwise entropy of |S|^2: monomial (one-hot) ⇒ entropy ~ 0; random-ish ⇒ high.
    ent = []
    for j in range(n):
        pj = np.abs(S[:, j])**2
        pj /= max(1e-16, pj.sum())
        ent.append(_shannon_entropy(pj))
    mean_ent = float(np.mean(ent))
    # Upper bound for uniform over n is log(n); require reasonably high fraction.
    assert mean_ent > 0.6 * np.log(n), f"Entropy too low: mean={mean_ent}, n={n}"
    print(f"✓ High Ψ†F entropy confirmed: mean = {mean_ent:.4f} (> 0.6·log({n}) = {0.6*np.log(n):.4f})")
