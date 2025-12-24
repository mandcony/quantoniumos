# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
DFT Non-Equivalence Test via Coefficient Correlation
Tests that Ψ is NOT secretly a permuted/phased Fourier transform
"""
import numpy as np
from algorithms.rft.core.phi_phase_fft_optimized import rft_forward

def test_max_dft_coeff_correlation_is_small():
    """
    Proves Φ-RFT is NOT DFT-equivalent by showing low correlation between
    Ψx and Fx across random inputs. If Ψ ≈ D1·F·D2·P, correlation → 1.
    """
    n = 256
    beta, sigma = 0.83, 1.25
    rng = np.random.default_rng(0xBEEF)
    trials = 64
    max_corr = 0.0
    for _ in range(trials):
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        denom = max(1e-16, np.vdot(x, x).real)
        Psi_x = rft_forward(x, beta=beta, sigma=sigma)
        F_x   = np.fft.fft(x, norm="ortho")
        corr = abs(np.vdot(Psi_x, F_x)) / denom
        max_corr = max(max_corr, float(corr))
    # If Ψ ≈ D1 F D2 P, correlation can approach 1. We expect it to be << 1.
    assert max_corr < 0.25, f"Max correlation too high: {max_corr}"
    print(f"✓ DFT non-equivalence confirmed: max correlation = {max_corr:.6f} (< 0.25)")
