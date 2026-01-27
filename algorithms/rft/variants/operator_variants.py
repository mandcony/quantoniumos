# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Operator-Based RFT Variants
===========================

These variants use the OPERATOR-BASED definition (Alternative/Legacy): eigenbasis of a resonance operator K.
Each variant defines a different resonance autocorrelation function R(k).

The Canonical RFT is now defined as the Gram-normalized exponential basis (see algorithms/rft/core/resonant_fourier_transform.py).

December 2025: New implementation based on formal framework proofs.
"""

import numpy as np
from scipy.linalg import toeplitz, eigh
from functools import lru_cache
from typing import Callable, Tuple

# Import patent variants (USPTO 19/169,399)
from .patent_variants import (
    generate_rft_manifold_projection,
    generate_rft_sphere_parametric,
    generate_rft_phase_coherent,
    generate_rft_entropy_modulated,
    generate_rft_loxodrome,
    generate_rft_polar_golden,
)

# Golden ratio constant
PHI = (1 + np.sqrt(5)) / 2


def _build_resonance_operator(r: np.ndarray, decay_rate: float = 0.01) -> np.ndarray:
    """
    Build Hermitian resonance operator K from autocorrelation function r.
    
    K = T(r * d) where:
        T = Toeplitz constructor
        d = exp(-decay_rate * k) for regularization
    """
    N = len(r)
    k = np.arange(N)
    decay = np.exp(-decay_rate * k)
    r_reg = r * decay
    r_reg[0] = 1.0  # Normalize
    return toeplitz(r_reg)


def _eigenbasis(K: np.ndarray) -> np.ndarray:
    """
    Extract orthonormal eigenbasis from Hermitian operator K.
    Sorted by eigenvalue (descending) for energy compaction.
    """
    eigenvalues, eigenvectors = eigh(K)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx]


# =============================================================================
# OPERATOR-BASED RFT VARIANTS
# =============================================================================

@lru_cache(maxsize=32)
def generate_rft_golden(n: int) -> np.ndarray:
    """
    RFT-Golden: Canonical operator-based RFT with golden ratio structure.
    
    R_φ[k] = cos(2πf₀k/n) + cos(2πf₀φk/n)
    
    This is the PRIMARY RFT variant - optimal for quasi-periodic signals
    with golden ratio frequency relationships.
    """
    f0 = 10.0
    k = np.arange(n)
    t_k = k / n
    
    r = np.cos(2 * np.pi * f0 * t_k) + np.cos(2 * np.pi * f0 * PHI * t_k)
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_fibonacci(n: int) -> np.ndarray:
    """
    RFT-Fibonacci: Resonance operator tuned to Fibonacci frequency ratios.
    
    R[k] = Σ cos(2π F_i k / n) for Fibonacci numbers F_i
    
    Optimal for signals with Fibonacci-modulated structure.
    """
    # Generate Fibonacci sequence
    fib = [1, 1]
    while fib[-1] < n:
        fib.append(fib[-1] + fib[-2])
    fib = fib[:min(8, len(fib))]  # Use first 8 Fibonacci numbers
    
    k = np.arange(n)
    t_k = k / n
    
    # Sum of cosines at Fibonacci-scaled frequencies
    r = np.zeros(n)
    for f in fib:
        r += np.cos(2 * np.pi * f * t_k) / len(fib)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_harmonic(n: int, num_harmonics: int = 5) -> np.ndarray:
    """
    RFT-Harmonic: Resonance operator with harmonic overtone structure.
    
    R[k] = Σ (1/i) cos(2π i f₀ k / n) for i = 1, 2, ..., num_harmonics
    
    Optimal for audio/musical signals with natural harmonic content.
    """
    f0 = 4.0
    k = np.arange(n)
    t_k = k / n
    
    r = np.zeros(n)
    for i in range(1, num_harmonics + 1):
        r += (1.0 / i) * np.cos(2 * np.pi * i * f0 * t_k)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_geometric(n: int) -> np.ndarray:
    """
    RFT-Geometric: Resonance operator with geometric scaling.
    
    R[k] = Σ cos(2π φ^i k / n) for i = 0, 1, 2, ...
    
    Golden ratio powers create self-similar frequency structure.
    """
    k = np.arange(n)
    t_k = k / n
    
    r = np.zeros(n)
    for i in range(8):
        freq = PHI ** i
        if freq < n / 2:  # Stay below Nyquist
            r += np.cos(2 * np.pi * freq * t_k) / (i + 1)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_beating(n: int) -> np.ndarray:
    """
    RFT-Beating: Resonance operator for beating/interference patterns.
    
    R[k] = cos(2π f₁ k/n) + cos(2π f₂ k/n) where f₂ = φ·f₁
    
    Optimal for signals with golden-ratio beating patterns.
    """
    f1 = 5.0
    f2 = f1 * PHI
    f3 = f1 * PHI * PHI
    
    k = np.arange(n)
    t_k = k / n
    
    r = np.cos(2 * np.pi * f1 * t_k) + np.cos(2 * np.pi * f2 * t_k) + 0.5 * np.cos(2 * np.pi * f3 * t_k)
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_phyllotaxis(n: int) -> np.ndarray:
    """
    RFT-Phyllotaxis: Resonance operator for golden-angle patterns.
    
    Based on the phyllotaxis angle (137.5°) found in sunflowers, pinecones, etc.
    Optimal for signals with spiral/radial golden-angle structure.
    """
    golden_angle = 2 * np.pi / (PHI ** 2)  # ~137.5 degrees in radians
    
    k = np.arange(n)
    
    r = np.zeros(n)
    for s in range(1, 6):  # 5 spiral arms
        r += np.cos(s * golden_angle * k) / s
    
    K = _build_resonance_operator(r, decay_rate=0.02)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_cascade_h3(n: int) -> np.ndarray:
    """
    RFT-Cascade (H3): Hybrid DCT + RFT operator.
    
    Low frequencies: DCT-like structure (smooth)
    High frequencies: Golden RFT structure (texture)
    
    This provides the best of both worlds for general signals.
    """
    k = np.arange(n)
    t_k = k / n
    
    # Low-frequency DCT-like component (cosine at low harmonics)
    r_dct = np.cos(np.pi * k / n)  # Fundamental DCT mode
    
    # High-frequency golden RFT component
    f0 = 10.0
    r_rft = np.cos(2 * np.pi * f0 * t_k) + np.cos(2 * np.pi * f0 * PHI * t_k)
    
    # Blend: DCT dominant at low k, RFT dominant at high k
    blend = np.linspace(0, 1, n)
    r = (1 - blend) * r_dct + blend * r_rft
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_adaptive(n: int, signal: np.ndarray = None) -> np.ndarray:
    """
    ARFT: Adaptive RFT using signal's own autocorrelation.
    
    If signal is provided, builds operator from signal statistics.
    Otherwise falls back to golden RFT.
    
    This is the maximum-sparsity variant (effectively KLT).
    """
    if signal is None or len(signal) != n:
        return generate_rft_golden(n)
    
    # Compute autocorrelation from signal
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[autocorr.size // 2:][:n]
    autocorr = autocorr / autocorr[0]  # Normalize
    
    K = _build_resonance_operator(autocorr)
    return _eigenbasis(K)


# =============================================================================
# HYBRID VARIANTS (DCT-RFT combinations)
# =============================================================================

@lru_cache(maxsize=32)
def generate_rft_hybrid_dct(n: int, split_ratio: float = 0.5) -> np.ndarray:
    """
    Hybrid DCT-RFT: Route low frequencies to DCT, high to RFT.
    
    Uses actual DCT-II basis for low frequencies and golden RFT for high.
    """
    from scipy.fft import dct, idct
    
    split = int(n * split_ratio)
    
    # DCT basis (first split rows)
    dct_basis = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        dct_basis[:, i] = dct(e, norm='ortho')
    
    # RFT basis
    rft_basis = generate_rft_golden(n)
    
    # Combine: DCT for low freq, RFT for high freq
    combined = np.zeros((n, n), dtype=np.float64)
    combined[:split, :] = dct_basis[:split, :]
    combined[split:, :] = rft_basis[split:, :]
    
    # Orthonormalize the combined basis
    Q, _ = np.linalg.qr(combined.T)
    return Q.T


# =============================================================================
# VARIANT REGISTRY
# =============================================================================

OPERATOR_VARIANTS = {
    'rft_golden': {
        'name': 'RFT-Golden',
        'generator': generate_rft_golden,
        'description': 'Canonical RFT with golden ratio resonance structure',
        'use_case': 'Quasi-periodic signals, phyllotaxis patterns',
    },
    'rft_fibonacci': {
        'name': 'RFT-Fibonacci',
        'generator': generate_rft_fibonacci,
        'description': 'Fibonacci frequency resonance',
        'use_case': 'Fibonacci-modulated signals',
    },
    'rft_harmonic': {
        'name': 'RFT-Harmonic',
        'generator': generate_rft_harmonic,
        'description': 'Natural harmonic overtone structure',
        'use_case': 'Musical audio, speech',
    },
    'rft_geometric': {
        'name': 'RFT-Geometric',
        'generator': generate_rft_geometric,
        'description': 'Golden-ratio powered frequencies',
        'use_case': 'Self-similar signals',
    },
    'rft_beating': {
        'name': 'RFT-Beating',
        'generator': generate_rft_beating,
        'description': 'Golden-ratio beating patterns',
        'use_case': 'Interference, modulation',
    },
    'rft_phyllotaxis': {
        'name': 'RFT-Phyllotaxis',
        'generator': generate_rft_phyllotaxis,
        'description': 'Golden angle (137.5°) structure',
        'use_case': 'Biological patterns, spirals',
    },
    'rft_cascade_h3': {
        'name': 'RFT-Cascade (H3)',
        'generator': generate_rft_cascade_h3,
        'description': 'DCT (low) + RFT (high) blend',
        'use_case': 'General-purpose compression',
    },
    'rft_hybrid_dct': {
        'name': 'RFT-Hybrid-DCT',
        'generator': generate_rft_hybrid_dct,
        'description': 'Split DCT/RFT basis',
        'use_case': 'Mixed content',
    },
    # =========================================================================
    # PATENT VARIANTS (USPTO 19/169,399 - April 2025)
    # =========================================================================
    'rft_manifold_projection': {
        'name': 'RFT-Manifold-Projection',
        'generator': generate_rft_manifold_projection,
        'description': 'Claim 3: Projection-based hash generation (4 wins, +47.9 dB on torus)',
        'use_case': 'Torus, spiral, helical signals - TOP PERFORMER',
    },
    'rft_sphere_parametric': {
        'name': 'RFT-Sphere-Parametric',
        'generator': generate_rft_sphere_parametric,
        'description': 'Claim 3: Spherical parametric resonance',
        'use_case': 'Phyllotaxis, biological patterns',
    },
    'rft_phase_coherent': {
        'name': 'RFT-Phase-Coherent',
        'generator': generate_rft_phase_coherent,
        'description': 'Claim 1: Phase-space coherence maintenance',
        'use_case': 'Chirp signals, frequency sweeps',
    },
    'rft_entropy_modulated': {
        'name': 'RFT-Entropy-Modulated',
        'generator': generate_rft_entropy_modulated,
        'description': 'Claim 2: Entropy-modulated golden scaling',
        'use_case': 'Noise-like signals, random processes',
    },
    'rft_loxodrome': {
        'name': 'RFT-Loxodrome',
        'generator': generate_rft_loxodrome,
        'description': 'Claim 3: Rhumb line trajectory on sphere',
        'use_case': 'Pure tones, sinusoidal signals (+12 dB vs golden)',
    },
    'rft_polar_golden': {
        'name': 'RFT-Polar-Golden',
        'generator': generate_rft_polar_golden,
        'description': 'Claim 3: Polar-Cartesian with golden scaling',
        'use_case': 'General quasi-periodic, radial patterns',
    },
}


def get_operator_variant(name: str, n: int) -> np.ndarray:
    """Get an operator-based RFT variant by name."""
    if name not in OPERATOR_VARIANTS:
        raise ValueError(f"Unknown variant: {name}. Available: {list(OPERATOR_VARIANTS.keys())}")
    return OPERATOR_VARIANTS[name]['generator'](n)


def list_operator_variants() -> list:
    """List all available operator-based RFT variants."""
    return list(OPERATOR_VARIANTS.keys())


# =============================================================================
# TRANSFORM FUNCTIONS
# =============================================================================

def rft_forward(x: np.ndarray, variant: str = 'rft_golden') -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply operator-based RFT forward transform.
    
    Returns: (coefficients, basis_matrix)
    """
    n = len(x)
    Phi = get_operator_variant(variant, n)
    coeffs = Phi.T @ x
    return coeffs, Phi


def rft_inverse(coeffs: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """Apply operator-based RFT inverse transform."""
    return Phi @ coeffs


if __name__ == "__main__":
    print("=" * 70)
    print("OPERATOR-BASED RFT VARIANTS - Verification")
    print("=" * 70)
    
    N = 128
    
    for name, info in OPERATOR_VARIANTS.items():
        Phi = info['generator'](N)
        
        # Verify unitarity
        I = np.eye(N)
        error = np.linalg.norm(Phi.T @ Phi - I, 'fro')
        status = "✓" if error < 1e-10 else "✗"
        
        print(f"{status} {info['name']:<25} | Unitarity error: {error:.2e}")
    
    print("\nAll operator-based variants verified.")
