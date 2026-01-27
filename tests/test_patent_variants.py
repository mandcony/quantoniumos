# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Tests for Patent-Aligned RFT Variants.
Verifies structure, orthonormality, and determinism.
"""
import pytest
import numpy as np
from algorithms.rft.variants import patent_variants

# List of all variant generator functions
VARIANTS = [
    patent_variants.generate_rft_polar_golden,
    patent_variants.generate_rft_spiral_golden,
    patent_variants.generate_rft_loxodrome,
    patent_variants.generate_rft_complex_exp,
    patent_variants.generate_rft_exp_decay_golden,
    patent_variants.generate_rft_mobius,
    patent_variants.generate_rft_winding,
    patent_variants.generate_rft_euler_torus,
    patent_variants.generate_rft_euler_sphere,
    patent_variants.generate_rft_klein,
    patent_variants.generate_rft_phase_coherent,
    patent_variants.generate_rft_kuramoto,
    patent_variants.generate_rft_entropy_modulated,
    patent_variants.generate_rft_bloom_hash,
    patent_variants.generate_rft_manifold_projection,
    patent_variants.generate_rft_hopf_fibration,
    patent_variants.generate_rft_hybrid_phase,
    patent_variants.generate_rft_recursive_modulation,
    patent_variants.generate_rft_trefoil_knot,
]

@pytest.mark.parametrize("variant_func", VARIANTS)
def test_variant_properties(variant_func):
    """Verify basic properties of the RFT variant basis."""
    N = 32 # Keep small for speed
    
    # 1. Generate basis
    Psi = variant_func(N)
    
    # 2. Check shape
    assert Psi.shape == (N, N), f"{variant_func.__name__} returned wrong shape"
    
    # 3. Check orthonormality: Psi.H @ Psi = I
    # Note: These are numerical eigenbases, so we expect machine precision errors
    # but they should be close to unitary.
    I_approx = Psi.conj().T @ Psi
    I_ideal = np.eye(N)
    
    # Check max deviation from Identity
    max_error = np.max(np.abs(I_approx - I_ideal))
    assert max_error < 1e-10, f"{variant_func.__name__} not unitary (max err: {max_error})"
    
    # 4. Determinism check
    Psi2 = variant_func(N)
    assert np.allclose(Psi, Psi2), f"{variant_func.__name__} is not deterministic"

def test_variant_energy_conservation():
    """Verify Parseval's theorem (energy conservation) for a random signal."""
    N = 32
    np.random.seed(42)
    x = np.random.randn(N) + 1j * np.random.randn(N)
    x /= np.linalg.norm(x) # Normalize input energy to 1.0
    
    variant_func = patent_variants.generate_rft_polar_golden
    Psi = variant_func(N)
    
    # Forward transform: y = Psi.H @ x (projection onto basis)
    y = Psi.conj().T @ x
    
    energy_in = np.linalg.norm(x)**2
    energy_out = np.linalg.norm(y)**2
    
    assert np.isclose(energy_in, energy_out, atol=1e-10), "Energy not conserved"
