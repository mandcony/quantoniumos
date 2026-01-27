# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Patent-Aligned RFT Variants for US Patent Application 19/169,399
================================================================

These 20 new RFT variants are designed to match the claims in:
"Hybrid Computational Framework for Quantum and Resonance Simulation"

Claim 1: Symbolic Resonance Fourier Transform Engine
Claim 2: Resonance-Based Cryptographic Subsystem  
Claim 3: Geometric Structures for RFT-Based Cryptographic Waveform Hashing
Claim 4: Hybrid Mode Integration

Each variant uses a different resonance autocorrelation function R(k)
derived from the geometric/topological structures described in the patent.

December 2025: Part of the RFT Discovery Initiative.
"""

import numpy as np
from scipy.linalg import toeplitz, eigh
from functools import lru_cache
from typing import Tuple

# Golden ratio constant
PHI = (1 + np.sqrt(5)) / 2

# Golden angle (phyllotaxis) - ~137.5 degrees
GOLDEN_ANGLE = 2 * np.pi / (PHI ** 2)

# Euler's number for complex exponential transforms
E = np.e


def _build_resonance_operator(r: np.ndarray, decay_rate: float = 0.01) -> np.ndarray:
    """Build Hermitian resonance operator K from autocorrelation function r."""
    N = len(r)
    k = np.arange(N)
    decay = np.exp(-decay_rate * k)
    r_reg = r * decay
    r_reg[0] = 1.0
    return toeplitz(r_reg)


def _eigenbasis(K: np.ndarray) -> np.ndarray:
    """Extract orthonormal eigenbasis sorted by eigenvalue (descending)."""
    eigenvalues, eigenvectors = eigh(K)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx]


# =============================================================================
# PATENT CLAIM 3: Polar-Cartesian with Golden Ratio Scaling
# =============================================================================

@lru_cache(maxsize=32)
def generate_rft_polar_golden(n: int) -> np.ndarray:
    """
    RFT-Polar-Golden: Polar coordinate resonance with golden ratio radial scaling.
    
    Patent Claim 3: "polar-to-Cartesian coordinate systems with golden ratio 
    scaling applied to harmonic relationships"
    
    R[k] = cos(θ_k) * r_k where r_k = φ^(k/n) and θ_k = 2πk/n
    """
    k = np.arange(n)
    theta = 2 * np.pi * k / n
    radius = PHI ** (k / n)  # Golden ratio radial scaling
    
    # Polar to Cartesian: x = r*cos(θ)
    r = radius * np.cos(theta)
    r = r / (np.max(np.abs(r)) + 1e-10)  # Normalize
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_spiral_golden(n: int) -> np.ndarray:
    """
    RFT-Spiral-Golden: Golden spiral resonance pattern.
    
    r(θ) = a * e^(bθ) where b = ln(φ)/(π/2) for golden spiral
    
    Used in phyllotaxis, nautilus shells, galaxies.
    """
    k = np.arange(n)
    theta = 2 * np.pi * k / n
    
    # Golden spiral: b chosen so r doubles each quarter turn by φ
    b = np.log(PHI) / (np.pi / 2)
    radius = np.exp(b * theta)
    radius = radius / radius[-1]  # Normalize to [0, 1]
    
    # Resonance from spiral x-projection
    r = radius * np.cos(theta)
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r, decay_rate=0.02)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_loxodrome(n: int) -> np.ndarray:
    """
    RFT-Loxodrome: Rhumb line on a sphere with golden ratio pitch.
    
    Models constant-angle paths on curved manifolds.
    Related to navigation and geodesic structures.
    """
    k = np.arange(n)
    t = k / n
    
    # Loxodrome parameters
    pitch = np.arctan(1 / PHI)  # Golden angle pitch
    
    # Parametric loxodrome on sphere
    theta = 2 * np.pi * t * 3  # Multiple wraps
    phi = 2 * pitch * theta  # Latitude increases with longitude
    
    # Project to autocorrelation
    r = np.cos(theta) * np.cos(phi)
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


# =============================================================================
# PATENT CLAIM 3: Complex Exponential Transforms
# =============================================================================

@lru_cache(maxsize=32)
def generate_rft_complex_exp(n: int) -> np.ndarray:
    """
    RFT-Complex-Exp: Complex geometric coordinate generation via exponential transforms.
    
    Patent Claim 3: "complex geometric coordinate generation via exponential transforms"
    
    R[k] = Re(e^(i*φ*k) + e^(i*φ^2*k))
    """
    k = np.arange(n)
    t = k / n
    
    # Complex exponentials at golden ratio frequencies
    z1 = np.exp(1j * 2 * np.pi * PHI * k / n)
    z2 = np.exp(1j * 2 * np.pi * (PHI ** 2) * k / n)
    
    r = np.real(z1 + z2)
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_exp_decay_golden(n: int) -> np.ndarray:
    """
    RFT-Exp-Decay-Golden: Exponentially decaying golden oscillation.
    
    R[k] = e^(-k/τ) * cos(2π φ k / n) where τ = n/φ
    
    Models damped resonance with golden decay constant.
    """
    k = np.arange(n)
    tau = n / PHI  # Golden decay time constant
    
    decay = np.exp(-k / tau)
    oscillation = np.cos(2 * np.pi * PHI * k / n)
    
    r = decay * oscillation
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r, decay_rate=0.005)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_mobius(n: int) -> np.ndarray:
    """
    RFT-Mobius: Möbius transformation resonance.
    
    Möbius maps preserve circles and angles - key to conformal mappings.
    Uses z -> (az+b)/(cz+d) structure with golden ratio coefficients.
    """
    k = np.arange(n)
    z = np.exp(2j * np.pi * k / n)  # Unit circle
    
    # Möbius transformation with golden coefficients
    a, b = PHI, 1.0
    c, d = 1.0, PHI
    
    w = (a * z + b) / (c * z + d)
    
    r = np.real(w)
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


# =============================================================================
# PATENT CLAIM 3: Topological Winding Numbers and Euler Characteristics
# =============================================================================

@lru_cache(maxsize=32)
def generate_rft_winding(n: int, num_winds: int = 3) -> np.ndarray:
    """
    RFT-Winding: Topological winding number resonance.
    
    Patent Claim 3: "topological winding number computation"
    
    R[k] represents the phase accumulated over multiple complete rotations,
    with golden-ratio frequency modulation.
    """
    k = np.arange(n)
    t = k / n
    
    # Multiple winding contributions
    r = np.zeros(n)
    for w in range(1, num_winds + 1):
        # Each winding at golden-scaled frequency
        freq = w * PHI
        r += np.cos(2 * np.pi * freq * t) / w
    
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_euler_torus(n: int) -> np.ndarray:
    """
    RFT-Euler-Torus: Torus surface resonance (Euler characteristic = 0).
    
    Patent Claim 3: "Euler characteristic approximation for cryptographic signatures"
    
    R[k] from torus parametrization with golden ratio major/minor radii.
    """
    k = np.arange(n)
    t = 2 * np.pi * k / n
    
    # Torus parameters: major radius R, minor radius r
    R = PHI
    minor_r = 1.0
    
    # Parametric torus (u, v) -> (x, y, z)
    # We sample along a curve with golden ratio winding
    u = t
    v = PHI * t  # Golden ratio winding on torus
    
    x = (R + minor_r * np.cos(v)) * np.cos(u)
    y = (R + minor_r * np.cos(v)) * np.sin(u)
    z = minor_r * np.sin(v)
    
    # Use x-projection as resonance
    r = x / (np.max(np.abs(x)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_euler_sphere(n: int) -> np.ndarray:
    """
    RFT-Euler-Sphere: Sphere surface resonance (Euler characteristic = 2).
    
    Spherical coordinates with golden ratio latitude-longitude coupling.
    """
    k = np.arange(n)
    t = k / n
    
    # Spherical coordinates with golden coupling
    theta = np.pi * t  # Latitude (0 to π)
    phi = 2 * np.pi * PHI * t  # Longitude (golden winding)
    
    # Spherical to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Combined projection
    r = x + 0.5 * z
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)  
def generate_rft_klein(n: int) -> np.ndarray:
    """
    RFT-Klein: Klein bottle-inspired non-orientable resonance.
    
    Non-orientable topology (Euler characteristic = 0, like torus but different).
    """
    k = np.arange(n)
    t = 2 * np.pi * k / n
    
    # Klein bottle parametrization (immersion in R^3)
    u = t
    v = PHI * t
    
    # Simplified Klein bottle coordinates
    r_val = 4 * (1 - np.cos(u) / 2)
    x = r_val * np.cos(u) * np.cos(v)
    y = r_val * np.cos(u) * np.sin(v)
    
    r = x / (np.max(np.abs(x)) + 1e-10)
    
    K = _build_resonance_operator(r, decay_rate=0.02)
    return _eigenbasis(K)


# =============================================================================
# PATENT CLAIM 1: Phase-Space Coherence
# =============================================================================

@lru_cache(maxsize=32)
def generate_rft_phase_coherent(n: int) -> np.ndarray:
    """
    RFT-Phase-Coherent: Phase-space coherence retention mechanism.
    
    Patent Claim 1: "phase-space coherence retention mechanism for maintaining 
    structural dependencies between symbolic amplitudes and phase interactions"
    
    Uses coupled phase oscillators with golden ratio coupling.
    """
    k = np.arange(n)
    t = k / n
    
    # Two coupled oscillators
    omega1 = 2 * np.pi * 5
    omega2 = 2 * np.pi * 5 * PHI
    coupling = 0.3
    
    # Phase coherent superposition
    phase1 = omega1 * t
    phase2 = omega2 * t + coupling * np.sin(omega1 * t)
    
    r = np.cos(phase1) + np.cos(phase2)
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_kuramoto(n: int, num_oscillators: int = 5) -> np.ndarray:
    """
    RFT-Kuramoto: Kuramoto-model coupled oscillator resonance.
    
    Models synchronization phenomena in phase-coupled systems.
    Natural frequencies distributed around golden ratio.
    """
    k = np.arange(n)
    t = k / n
    
    # Natural frequencies centered on PHI with spread
    omegas = [PHI + 0.2 * (i - num_oscillators // 2) for i in range(num_oscillators)]
    
    r = np.zeros(n)
    for omega in omegas:
        r += np.cos(2 * np.pi * omega * t)
    
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


# =============================================================================
# PATENT CLAIM 2: Cryptographic Waveform Structures
# =============================================================================

@lru_cache(maxsize=32)
def generate_rft_entropy_modulated(n: int) -> np.ndarray:
    """
    RFT-Entropy-Modulated: Dynamic entropy mapping for cryptographic use.
    
    Patent Claim 2: "dynamic entropy mapping engine for continuous modulation 
    of key material based on symbolic resonance states"
    
    Uses deterministic chaos (logistic map) at golden ratio parameter.
    """
    k = np.arange(n)
    
    # Logistic map at r = PHI + 2 (chaotic regime)
    # This is deterministic but highly sensitive to initial conditions
    r_param = PHI + 2  # ~3.618 (chaotic)
    x = np.zeros(n)
    x[0] = 0.1  # Initial condition
    
    for i in range(1, n):
        x[i] = r_param * x[i-1] * (1 - x[i-1])
    
    # Modulate with golden frequency
    r = x * np.cos(2 * np.pi * PHI * k / n)
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r, decay_rate=0.02)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_bloom_hash(n: int, num_hashes: int = 5) -> np.ndarray:
    """
    RFT-Bloom-Hash: Bloom filter-inspired resonance structure.
    
    Patent Claim 2: "topological hashing module for extracting waveform features 
    into Bloom-like filters"
    
    Multiple hash functions at golden-ratio-scaled frequencies.
    """
    k = np.arange(n)
    t = k / n
    
    r = np.zeros(n)
    for h in range(num_hashes):
        # Each "hash function" is a different golden-scaled frequency
        freq = PHI ** h
        phase = 2 * np.pi * h / num_hashes  # Distributed phases
        r += np.cos(2 * np.pi * freq * t + phase) / (h + 1)
    
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


# =============================================================================
# PATENT CLAIM 3: Manifold-Based Hash Generation
# =============================================================================

@lru_cache(maxsize=32)
def generate_rft_manifold_projection(n: int) -> np.ndarray:
    """
    RFT-Manifold-Projection: Manifold mapping that preserves geometric relationships.
    
    Patent Claim 3: "manifold-based hash generation that preserves geometric 
    relationships in the cryptographic output space"
    
    Projects from higher-dimensional manifold to 1D resonance.
    """
    k = np.arange(n)
    t = k / n
    
    # 3D manifold coordinates (twisted torus)
    u = 2 * np.pi * t
    v = 2 * np.pi * PHI * t
    
    # Twisted torus with golden twist
    twist = PHI * u
    x = (2 + np.cos(v + twist)) * np.cos(u)
    y = (2 + np.cos(v + twist)) * np.sin(u)
    z = np.sin(v + twist)
    
    # Project to 1D preserving some structure
    r = x + 0.3 * y + 0.1 * z
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_hopf_fibration(n: int) -> np.ndarray:
    """
    RFT-Hopf: Hopf fibration-based resonance.
    
    S³ → S² fibration with golden ratio fiber twist.
    Deep topological structure used in physics.
    """
    k = np.arange(n)
    t = k / n
    
    # Parameters for Hopf fibration
    eta = 2 * np.pi * t
    xi = 2 * np.pi * PHI * t
    
    # S³ to S² projection (stereographic)
    x = np.sin(eta) * np.cos(xi)
    y = np.sin(eta) * np.sin(xi)
    z = np.cos(eta)
    
    r = x + PHI * y
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


# =============================================================================
# PATENT CLAIM 4: Hybrid Mode Integration
# =============================================================================

@lru_cache(maxsize=32)
def generate_rft_hybrid_phase(n: int) -> np.ndarray:
    """
    RFT-Hybrid-Phase: Phase-aware modular architecture.
    
    Patent Claim 4: "symbolic amplitude and phase-state transformations 
    propagate coherently across encryption and storage layers"
    
    Combines amplitude modulation with phase modulation.
    """
    k = np.arange(n)
    t = k / n
    
    # Amplitude envelope (smooth)
    envelope = 0.5 + 0.5 * np.cos(np.pi * t)
    
    # Phase modulation (golden)
    phase = 2 * np.pi * 10 * t + 2 * np.sin(2 * np.pi * PHI * t)
    
    r = envelope * np.cos(phase)
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_recursive_modulation(n: int, depth: int = 4) -> np.ndarray:
    """
    RFT-Recursive-Modulation: Recursive modulation controller.
    
    Patent Claim 2: "recursive modulation controller adapted to modify 
    waveform structure in real time"
    
    Self-similar modulation at multiple scales.
    """
    k = np.arange(n)
    t = k / n
    
    r = np.zeros(n)
    freq = 1.0
    for d in range(depth):
        r += np.cos(2 * np.pi * freq * t) / (d + 1)
        freq *= PHI  # Golden ratio scaling between levels
    
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


@lru_cache(maxsize=32)
def generate_rft_trefoil_knot(n: int) -> np.ndarray:
    """
    RFT-Trefoil: Trefoil knot resonance.
    
    Simplest non-trivial knot - has deep topological invariants.
    """
    k = np.arange(n)
    t = 2 * np.pi * k / n
    
    # Trefoil knot parametrization
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    
    r = x / (np.max(np.abs(x)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


# =============================================================================
# VARIANT REGISTRY
# =============================================================================

PATENT_VARIANTS = {
    # Claim 3: Polar-Cartesian with Golden Ratio
    'rft_polar_golden': {
        'name': 'RFT-Polar-Golden',
        'generator': generate_rft_polar_golden,
        'description': 'Polar coordinates with golden ratio radial scaling',
        'claim': 'Claim 3: polar-to-Cartesian coordinate systems',
    },
    'rft_spiral_golden': {
        'name': 'RFT-Spiral-Golden',
        'generator': generate_rft_spiral_golden,
        'description': 'Golden logarithmic spiral resonance',
        'claim': 'Claim 3: golden ratio scaling',
    },
    'rft_loxodrome': {
        'name': 'RFT-Loxodrome',
        'generator': generate_rft_loxodrome,
        'description': 'Rhumb line on sphere with golden pitch',
        'claim': 'Claim 3: manifold mappings',
    },
    
    # Claim 3: Complex Exponential Transforms
    'rft_complex_exp': {
        'name': 'RFT-Complex-Exp',
        'generator': generate_rft_complex_exp,
        'description': 'Complex exponential at golden frequencies',
        'claim': 'Claim 3: complex geometric coordinate generation',
    },
    'rft_exp_decay_golden': {
        'name': 'RFT-Exp-Decay-Golden',
        'generator': generate_rft_exp_decay_golden,
        'description': 'Exponentially damped golden oscillation',
        'claim': 'Claim 3: exponential transforms',
    },
    'rft_mobius': {
        'name': 'RFT-Mobius',
        'generator': generate_rft_mobius,
        'description': 'Möbius transformation with golden coefficients',
        'claim': 'Claim 3: geometric coordinate transformations',
    },
    
    # Claim 3: Topological Structures
    'rft_winding': {
        'name': 'RFT-Winding',
        'generator': generate_rft_winding,
        'description': 'Topological winding number resonance',
        'claim': 'Claim 3: topological winding number computation',
    },
    'rft_euler_torus': {
        'name': 'RFT-Euler-Torus',
        'generator': generate_rft_euler_torus,
        'description': 'Torus surface (χ=0) resonance',
        'claim': 'Claim 3: Euler characteristic approximation',
    },
    'rft_euler_sphere': {
        'name': 'RFT-Euler-Sphere',
        'generator': generate_rft_euler_sphere,
        'description': 'Sphere surface (χ=2) resonance',
        'claim': 'Claim 3: Euler characteristic approximation',
    },
    'rft_klein': {
        'name': 'RFT-Klein',
        'generator': generate_rft_klein,
        'description': 'Klein bottle non-orientable resonance',
        'claim': 'Claim 3: topological invariants',
    },
    
    # Claim 1: Phase-Space Coherence
    'rft_phase_coherent': {
        'name': 'RFT-Phase-Coherent',
        'generator': generate_rft_phase_coherent,
        'description': 'Coupled phase oscillators',
        'claim': 'Claim 1: phase-space coherence retention',
    },
    'rft_kuramoto': {
        'name': 'RFT-Kuramoto',
        'generator': generate_rft_kuramoto,
        'description': 'Kuramoto synchronization model',
        'claim': 'Claim 1: structural dependencies between amplitudes',
    },
    
    # Claim 2: Cryptographic Structures
    'rft_entropy_modulated': {
        'name': 'RFT-Entropy-Modulated',
        'generator': generate_rft_entropy_modulated,
        'description': 'Chaotic entropy with golden modulation',
        'claim': 'Claim 2: dynamic entropy mapping engine',
    },
    'rft_bloom_hash': {
        'name': 'RFT-Bloom-Hash',
        'generator': generate_rft_bloom_hash,
        'description': 'Bloom filter-like multi-hash structure',
        'claim': 'Claim 2: Bloom-like filters',
    },
    
    # Claim 3: Manifold Mappings
    'rft_manifold_projection': {
        'name': 'RFT-Manifold-Projection',
        'generator': generate_rft_manifold_projection,
        'description': 'Twisted torus manifold projection',
        'claim': 'Claim 3: manifold-based hash generation',
    },
    'rft_hopf_fibration': {
        'name': 'RFT-Hopf-Fibration',
        'generator': generate_rft_hopf_fibration,
        'description': 'S³→S² Hopf fibration resonance',
        'claim': 'Claim 3: manifold mappings',
    },
    
    # Claim 4: Hybrid Integration
    'rft_hybrid_phase': {
        'name': 'RFT-Hybrid-Phase',
        'generator': generate_rft_hybrid_phase,
        'description': 'AM-PM combined modulation',
        'claim': 'Claim 4: phase-aware architecture',
    },
    'rft_recursive_modulation': {
        'name': 'RFT-Recursive-Modulation',
        'generator': generate_rft_recursive_modulation,
        'description': 'Self-similar multi-scale modulation',
        'claim': 'Claim 2: recursive modulation controller',
    },
    'rft_trefoil_knot': {
        'name': 'RFT-Trefoil-Knot',
        'generator': generate_rft_trefoil_knot,
        'description': 'Trefoil knot topological resonance',
        'claim': 'Claim 3: topological invariants',
    },
    
    # Additional topology variant
    'rft_figure8_knot': {
        'name': 'RFT-Figure8-Knot',
        'generator': lambda n: _figure8_knot(n),
        'description': 'Figure-8 knot resonance',
        'claim': 'Claim 3: node linkage invariants',
    },
}


@lru_cache(maxsize=32)
def _figure8_knot(n: int) -> np.ndarray:
    """Figure-8 knot parametrization."""
    k = np.arange(n)
    t = 2 * np.pi * k / n
    
    x = (2 + np.cos(2*t)) * np.cos(3*t)
    y = (2 + np.cos(2*t)) * np.sin(3*t)
    z = np.sin(4*t)
    
    r = x / (np.max(np.abs(x)) + 1e-10)
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


def get_patent_variant(name: str, n: int) -> np.ndarray:
    """Get a patent-aligned RFT variant by name."""
    if name not in PATENT_VARIANTS:
        raise ValueError(f"Unknown variant: {name}. Available: {list(PATENT_VARIANTS.keys())}")
    return PATENT_VARIANTS[name]['generator'](n)


def list_patent_variants() -> list:
    """List all patent-aligned RFT variants."""
    return list(PATENT_VARIANTS.keys())


# =============================================================================
# VERIFICATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PATENT-ALIGNED RFT VARIANTS - US Patent Application 19/169,399")
    print("=" * 80)
    
    N = 128
    
    for name, info in PATENT_VARIANTS.items():
        try:
            Phi = info['generator'](N)
            
            # Verify unitarity
            I = np.eye(N)
            error = np.linalg.norm(Phi.T @ Phi - I, 'fro')
            status = "✓" if error < 1e-10 else "✗"
            
            print(f"{status} {info['name']:<25} | {info['claim'][:45]:<45} | err={error:.1e}")
        except Exception as e:
            print(f"✗ {info['name']:<25} | ERROR: {e}")
    
    print("\n" + "=" * 80)
    print(f"Total variants: {len(PATENT_VARIANTS)}")
    print("=" * 80)
