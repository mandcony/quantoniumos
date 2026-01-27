#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Tetrahedral Φ-RFT Variant Hypotheses Test Suite
================================================

Tests 10 hypotheses about tetrahedral generalizations of Φ-RFT.
These extend the phase operators Dφ and Cσ into 3D/4D simplex geometry.

Mathematical Foundation:
- Tetrahedron vertices in 3D: 4 points with maximal angular separation
- A4 symmetry group: 12 rotational symmetries of tetrahedron
- Simplex embedding: Maps 1D signal to 4D barycentric coordinates
- Multi-axis phase cycling: Rotates through tetrahedral directions

Hypotheses:
T1: Tetrahedral Phase Modulation (Δ-RFT) - higher decorrelation
T2: Tetrahedral Chirp Operator (CΔσ) - uniform energy spread
T3: 3-Simplex Harmonic Embedding (SH3-RFT) - cluster sparsity
T4: Tetrahedral Rotational Kernel (KRFT-Δ) - reduced coherence
T5: Tetrahedral Log-Frequency Warp (logΔ-RFT) - mixed signal sparsity
T6: 3-Vertex Cascade (H3Δ) - lower BPP
T7: Tetrahedral Attention Routing (H5Δ) - better dictionary selection
T8: Tetrahedral Sensing Matrix (ΦΔ-sense) - CS recovery
T9: Tetrahedral Quasicrystal Decoder (QCΔ-RFT) - sharper peaks
T10: Tetrahedral Quantization Minimizer (QΔ-RFT) - lower MSE

Author: QuantoniumOS Team
Date: 2024
"""

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.fft import fft, ifft, dct, idct
from scipy.stats import kurtosis
from scipy.linalg import norm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Constants
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618

# Tetrahedron vertices (unit vectors from center)
# Regular tetrahedron inscribed in unit sphere
TETRA_VERTICES = np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1]
], dtype=np.float64) / np.sqrt(3)

# A4 group generators (tetrahedral rotation group, order 12)
# 3-fold rotations around each vertex axis
def _rotation_matrix_axis_angle(axis, angle):
    """Rodrigues' rotation formula."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

# Generate A4 group elements (12 rotations)
def generate_A4_group():
    """Generate the 12 rotational symmetries of a tetrahedron."""
    rotations = [np.eye(3)]  # Identity
    
    # 3-fold rotations around each vertex (8 elements: 4 vertices × 2 angles)
    for v in TETRA_VERTICES:
        for angle in [2*np.pi/3, 4*np.pi/3]:
            rotations.append(_rotation_matrix_axis_angle(v, angle))
    
    # 2-fold rotations around edge midpoints (3 elements)
    edge_axes = [
        (TETRA_VERTICES[0] + TETRA_VERTICES[1]) / 2,
        (TETRA_VERTICES[0] + TETRA_VERTICES[2]) / 2,
        (TETRA_VERTICES[0] + TETRA_VERTICES[3]) / 2,
    ]
    for axis in edge_axes:
        rotations.append(_rotation_matrix_axis_angle(axis, np.pi))
    
    return rotations[:12]  # Exactly 12 elements

A4_GROUP = generate_A4_group()


# =============================================================================
# Tetrahedral Transforms
# =============================================================================

def tetrahedral_phase(k: np.ndarray, n: int, beta: float = 0.83) -> np.ndarray:
    """
    T1: Tetrahedral Phase Modulation (Δ-RFT)
    
    θ(k) = 2π * (k/N) * (v · t)
    where t cycles through tetrahedron vertices
    """
    phases = np.zeros(n, dtype=np.complex128)
    for i, ki in enumerate(k):
        vertex_idx = int(ki) % 4
        v = TETRA_VERTICES[vertex_idx]
        # Direction vector cycles through vertices
        t = TETRA_VERTICES[(vertex_idx + 1) % 4]
        theta = 2 * np.pi * beta * (ki / n) * np.dot(v, t)
        phases[i] = np.exp(1j * theta)
    return phases


def tetrahedral_chirp(k: np.ndarray, n: int, sigma: float = 1.25) -> np.ndarray:
    """
    T2: Tetrahedral Chirp Operator (CΔσ)
    
    Uses tetrahedral edge lengths to construct 3D chirp.
    """
    # Edge lengths from vertex 0 to others (all equal for regular tetrahedron)
    edge_lengths = np.array([
        np.linalg.norm(TETRA_VERTICES[0] - TETRA_VERTICES[j])
        for j in range(1, 4)
    ])
    
    phases = np.zeros(n, dtype=np.complex128)
    for i, ki in enumerate(k):
        # Cycle through 3 edge directions
        edge_idx = int(ki) % 3
        edge_weight = edge_lengths[edge_idx]
        theta = np.pi * sigma * (ki * ki / n) * edge_weight
        phases[i] = np.exp(1j * theta)
    return phases


def simplex_embed(signal: np.ndarray) -> np.ndarray:
    """
    T3: Embed 1D signal into 4D simplex barycentric coordinates.
    
    Maps each sample to a point in the 3-simplex (tetrahedron).
    """
    n = len(signal)
    # Normalize signal to [0, 1]
    s_min, s_max = signal.min(), signal.max()
    if s_max - s_min > 1e-10:
        s_norm = (signal - s_min) / (s_max - s_min)
    else:
        s_norm = np.zeros(n)
    
    # Map to barycentric coordinates (sum to 1)
    # Use phase-based distribution across 4 vertices
    bary = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        phase = 2 * np.pi * s_norm[i]
        # Distribute weight based on angular distance to each vertex
        weights = np.array([
            np.cos(phase) + 1,
            np.cos(phase + np.pi/2) + 1,
            np.cos(phase + np.pi) + 1,
            np.cos(phase + 3*np.pi/2) + 1
        ])
        bary[i] = weights / weights.sum()
    
    return bary


def simplex_project(bary: np.ndarray) -> np.ndarray:
    """Project barycentric coordinates to 3D tetrahedron points."""
    return bary @ TETRA_VERTICES[:, :3] if bary.shape[1] == 4 else bary @ TETRA_VERTICES


def delta_rft_forward(x: np.ndarray, variant: str = "T1") -> np.ndarray:
    """
    Forward Δ-RFT transform with specified variant.
    
    All variants: D_Δ * C_Δ * F (unitary structure preserved)
    """
    n = len(x)
    k = np.arange(n, dtype=np.float64)
    
    # FFT base
    X = fft(x, norm="ortho")
    
    if variant == "T1":
        # Tetrahedral phase modulation
        D = tetrahedral_phase(k, n)
        C = np.exp(1j * np.pi * 1.25 * k * k / n)  # Standard chirp
    elif variant == "T2":
        # Tetrahedral chirp
        D = np.exp(1j * 2 * np.pi * 0.83 * (k / PHI - np.floor(k / PHI)))
        C = tetrahedral_chirp(k, n)
    elif variant == "T5":
        # Tetrahedral log-frequency warp
        D = tetrahedral_phase(k, n)
        # 3D log warp aligned with tetrahedral axes
        log_k = np.log1p(k) / np.log1p(n)
        C = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            axis_idx = int(k[i]) % 3
            axis = TETRA_VERTICES[axis_idx + 1] - TETRA_VERTICES[0]
            warp = log_k[i] * np.linalg.norm(axis)
            C[i] = np.exp(1j * 2 * np.pi * warp)
    else:
        # Default: combined tetrahedral phase and chirp
        D = tetrahedral_phase(k, n)
        C = tetrahedral_chirp(k, n)
    
    return D * C * X


def delta_rft_inverse(Y: np.ndarray, variant: str = "T1") -> np.ndarray:
    """Inverse Δ-RFT transform."""
    n = len(Y)
    k = np.arange(n, dtype=np.float64)
    
    if variant == "T1":
        D = tetrahedral_phase(k, n)
        C = np.exp(1j * np.pi * 1.25 * k * k / n)
    elif variant == "T2":
        D = np.exp(1j * 2 * np.pi * 0.83 * (k / PHI - np.floor(k / PHI)))
        C = tetrahedral_chirp(k, n)
    elif variant == "T5":
        D = tetrahedral_phase(k, n)
        log_k = np.log1p(k) / np.log1p(n)
        C = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            axis_idx = int(k[i]) % 3
            axis = TETRA_VERTICES[axis_idx + 1] - TETRA_VERTICES[0]
            warp = log_k[i] * np.linalg.norm(axis)
            C[i] = np.exp(1j * 2 * np.pi * warp)
    else:
        D = tetrahedral_phase(k, n)
        C = tetrahedral_chirp(k, n)
    
    # Inverse: F^H * C^H * D^H
    return ifft(np.conj(C) * np.conj(D) * Y, norm="ortho")


def rotational_kernel_transform(x: np.ndarray) -> np.ndarray:
    """
    T4: Apply Φ-RFT with A4 rotational kernel.
    
    Cycles through tetrahedral symmetry group orientations.
    """
    n = len(x)
    k = np.arange(n, dtype=np.float64)
    
    # FFT base
    X = fft(x, norm="ortho")
    
    # Apply A4 group rotation to phase
    phases = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        rot_idx = int(k[i]) % 12
        R = A4_GROUP[rot_idx]
        # Use rotation to modify phase direction
        direction = R @ TETRA_VERTICES[0]
        theta = 2 * np.pi * 0.83 * (k[i] / n) * direction[0]
        phases[i] = np.exp(1j * theta)
    
    return phases * X


def three_vertex_cascade(x: np.ndarray, threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    T6: 3-Vertex Cascade (H3Δ)
    
    Three-branch residual cascade using tetrahedral projections.
    Branch 1: DCT (structural)
    Branch 2: Δ-RFT with vertex 0-1 axis
    Branch 3: Δ-RFT with vertex 0-2 axis
    """
    n = len(x)
    residual = x.copy().astype(np.complex128)
    
    # Branch 1: DCT
    dct_coeffs = dct(residual.real, norm="ortho")
    thresh_dct = threshold * np.max(np.abs(dct_coeffs))
    mask_dct = np.abs(dct_coeffs) > thresh_dct
    dct_sparse = dct_coeffs * mask_dct
    branch1 = idct(dct_sparse, norm="ortho")
    residual -= branch1
    
    # Branch 2: Δ-RFT axis 0-1
    rft1_coeffs = delta_rft_forward(residual, variant="T1")
    thresh_rft = threshold * np.max(np.abs(rft1_coeffs))
    mask_rft1 = np.abs(rft1_coeffs) > thresh_rft
    rft1_sparse = rft1_coeffs * mask_rft1
    branch2 = delta_rft_inverse(rft1_sparse, variant="T1")
    residual -= branch2
    
    # Branch 3: Δ-RFT axis 0-2 (different tetrahedral direction)
    rft2_coeffs = delta_rft_forward(residual, variant="T2")
    mask_rft2 = np.abs(rft2_coeffs) > thresh_rft
    rft2_sparse = rft2_coeffs * mask_rft2
    branch3 = delta_rft_inverse(rft2_sparse, variant="T2")
    
    return branch1, branch2.real, branch3.real


def tetrahedral_sensing_matrix(m: int, n: int) -> np.ndarray:
    """
    T8: Build sensing matrix from tetrahedral RFT atoms.
    
    Returns m×n sensing matrix for compressed sensing.
    """
    # Full Δ-RFT matrix
    full_matrix = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        full_matrix[:, i] = delta_rft_forward(e_i, variant="T1")
    
    # Select m rows (deterministic, tetrahedral-guided selection)
    row_indices = []
    for j in range(m):
        # Use golden ratio + tetrahedral offset for row selection
        idx = int((j * PHI * n + j % 4 * n / 4)) % n
        while idx in row_indices:
            idx = (idx + 1) % n
        row_indices.append(idx)
    
    return full_matrix[row_indices, :]


# =============================================================================
# Test Signals
# =============================================================================

def generate_test_signals(n: int = 512) -> Dict[str, np.ndarray]:
    """Generate test signals for benchmarking."""
    t = np.arange(n, dtype=np.float64)
    
    signals = {}
    
    # Mixed wave + text (ASCII bytes normalized)
    text = "The quick brown fox jumps over the lazy dog. " * 20
    text_bytes = np.frombuffer(text.encode()[:n], dtype=np.uint8).astype(np.float64)
    text_norm = (text_bytes / 128.0) - 1.0
    wave = 0.5 * np.sin(2 * np.pi * 5 * t / n)
    signals['mixed_wave_text'] = wave + text_norm
    
    # 3-component mixture (for T8, T10)
    comp1 = np.sin(2 * np.pi * 3 * t / n)
    comp2 = 0.5 * np.sin(2 * np.pi * 7 * t / n + np.pi/4)
    comp3 = 0.3 * np.sin(2 * np.pi * 11 * t / n + np.pi/3)
    signals['three_component'] = comp1 + comp2 + comp3
    
    # Quasicrystal-like (for T9)
    qc = np.zeros(n)
    for m in range(5):
        freq = PHI ** m
        qc += np.cos(2 * np.pi * freq * t / n) / (m + 1)
    signals['quasicrystal'] = qc
    
    # Structured discrete (ASCII)
    code = "def f(x): return x**2 + phi*x + 1\n" * 20
    code_bytes = np.frombuffer(code.encode()[:n], dtype=np.uint8).astype(np.float64)
    signals['source_code'] = (code_bytes / 128.0) - 1.0
    
    # Random baseline
    np.random.seed(42)
    signals['random'] = np.random.randn(n)
    
    # Fibonacci-structured (builds on H9)
    fib = [1, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    fib = np.array(fib[:n], dtype=np.float64)
    signals['fibonacci'] = fib / fib.max()
    
    return signals


# =============================================================================
# Metrics
# =============================================================================

def energy_concentration(coeffs: np.ndarray, percentile: float = 0.99) -> float:
    """Fraction of coefficients needed for percentile of energy."""
    energy = np.abs(coeffs) ** 2
    total = energy.sum()
    if total < 1e-10:
        return 1.0
    sorted_energy = np.sort(energy)[::-1]
    cumsum = np.cumsum(sorted_energy)
    idx = np.searchsorted(cumsum, percentile * total)
    return (idx + 1) / len(coeffs)


def coherence_violation(A: np.ndarray, B: np.ndarray) -> float:
    """Measure coherence between two basis representations."""
    # Normalize
    A_norm = A / (np.linalg.norm(A) + 1e-10)
    B_norm = B / (np.linalg.norm(B) + 1e-10)
    return np.abs(np.vdot(A_norm, B_norm))


def mutual_coherence(matrix: np.ndarray) -> float:
    """Maximum absolute inner product between distinct columns."""
    n_cols = matrix.shape[1]
    max_coh = 0.0
    for i in range(n_cols):
        for j in range(i + 1, min(i + 50, n_cols)):  # Sample for speed
            col_i = matrix[:, i] / (np.linalg.norm(matrix[:, i]) + 1e-10)
            col_j = matrix[:, j] / (np.linalg.norm(matrix[:, j]) + 1e-10)
            coh = np.abs(np.vdot(col_i, col_j))
            max_coh = max(max_coh, coh)
    return max_coh


def reconstruction_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean squared error between original and reconstructed."""
    return np.mean(np.abs(original - reconstructed) ** 2)


# =============================================================================
# Hypothesis Tests
# =============================================================================

@dataclass
class TestResult:
    hypothesis_id: str
    hypothesis_text: str
    test_performed: str
    metric_name: str
    delta_rft_value: float
    baseline_value: float  # DCT or standard RFT
    dft_value: float
    verdict: str
    confidence: float
    details: str


def test_T1_tetrahedral_phase(signals: Dict[str, np.ndarray]) -> TestResult:
    """T1: Tetrahedral Phase Modulation - higher decorrelation."""
    print("[T1] Testing tetrahedral phase decorrelation...")
    
    signal = signals['mixed_wave_text']
    n = len(signal)
    
    # Compute transforms
    delta_coeffs = delta_rft_forward(signal, variant="T1")
    dct_coeffs = dct(signal, norm="ortho")
    dft_coeffs = fft(signal, norm="ortho")
    
    # Measure decorrelation via sparsity (lower = more decorrelated)
    delta_sparse = energy_concentration(delta_coeffs)
    dct_sparse = energy_concentration(dct_coeffs)
    dft_sparse = energy_concentration(dft_coeffs)
    
    # Also measure cross-coherence
    delta_dct_coh = coherence_violation(delta_coeffs, dct_coeffs)
    
    verdict = "CONFIRMED" if delta_sparse < dct_sparse else "REJECTED"
    confidence = max(0, 1 - delta_sparse / dct_sparse) if delta_sparse < dct_sparse else 0
    
    return TestResult(
        hypothesis_id="T1",
        hypothesis_text="Tetrahedral phase yields higher decorrelation on mixed signals",
        test_performed="Measured sparsity (99% energy concentration) on wave+text mixture",
        metric_name="Energy concentration (lower = better decorrelation)",
        delta_rft_value=delta_sparse,
        baseline_value=dct_sparse,
        dft_value=dft_sparse,
        verdict=verdict,
        confidence=confidence,
        details=f"Δ-RFT: {delta_sparse:.3f}, DCT: {dct_sparse:.3f}, Δ-DCT coherence: {delta_dct_coh:.3f}"
    )


def test_T2_tetrahedral_chirp(signals: Dict[str, np.ndarray]) -> TestResult:
    """T2: Tetrahedral Chirp - uniform energy spread."""
    print("[T2] Testing tetrahedral chirp energy distribution...")
    
    signal = signals['mixed_wave_text']
    n = len(signal)
    
    # Compute transforms
    delta_coeffs = delta_rft_forward(signal, variant="T2")
    dct_coeffs = dct(signal, norm="ortho")
    dft_coeffs = fft(signal, norm="ortho")
    
    # Measure energy uniformity via Gini coefficient (lower = more uniform)
    def gini(x):
        x = np.abs(x)
        if x.sum() < 1e-10:
            return 0
        x = np.sort(x)
        n = len(x)
        idx = np.arange(1, n + 1)
        return (2 * np.sum(idx * x) / (n * np.sum(x))) - (n + 1) / n
    
    delta_gini = gini(delta_coeffs)
    dct_gini = gini(dct_coeffs)
    dft_gini = gini(dft_coeffs)
    
    # Lower Gini = more uniform spread = better for chirp hypothesis
    verdict = "CONFIRMED" if delta_gini < dct_gini else "REJECTED"
    confidence = max(0, 1 - delta_gini / dct_gini) if delta_gini < dct_gini else 0
    
    return TestResult(
        hypothesis_id="T2",
        hypothesis_text="Tetrahedral chirp spreads energy more uniformly",
        test_performed="Measured Gini coefficient of coefficient magnitudes",
        metric_name="Gini coefficient (lower = more uniform)",
        delta_rft_value=delta_gini,
        baseline_value=dct_gini,
        dft_value=dft_gini,
        verdict=verdict,
        confidence=confidence,
        details=f"Δ-chirp Gini: {delta_gini:.4f}, DCT Gini: {dct_gini:.4f}"
    )


def test_T3_simplex_embedding(signals: Dict[str, np.ndarray]) -> TestResult:
    """T3: 3-Simplex Harmonic Embedding - cluster sparsity."""
    print("[T3] Testing simplex embedding for structured discrete data...")
    
    signal = signals['source_code']
    n = len(signal)
    
    # Standard approach
    dct_coeffs = dct(signal, norm="ortho")
    dct_sparse = energy_concentration(dct_coeffs)
    
    # Simplex embedding approach
    bary = simplex_embed(signal)
    # Project to 3D, then apply RFT to each dimension
    proj = simplex_project(bary)
    simplex_coeffs = np.zeros((n, 3), dtype=np.complex128)
    for d in range(3):
        simplex_coeffs[:, d] = delta_rft_forward(proj[:, d], variant="T1")
    
    # Measure sparsity across all dimensions
    all_coeffs = simplex_coeffs.flatten()
    simplex_sparse = energy_concentration(all_coeffs)
    
    dft_coeffs = fft(signal, norm="ortho")
    dft_sparse = energy_concentration(dft_coeffs)
    
    verdict = "CONFIRMED" if simplex_sparse < dct_sparse else "REJECTED"
    confidence = max(0, 1 - simplex_sparse / dct_sparse) if simplex_sparse < dct_sparse else 0
    
    return TestResult(
        hypothesis_id="T3",
        hypothesis_text="Simplex embedding improves sparsity for discrete data",
        test_performed="Embedded ASCII signal into 4D simplex, applied Δ-RFT",
        metric_name="Energy concentration after simplex+Δ-RFT",
        delta_rft_value=simplex_sparse,
        baseline_value=dct_sparse,
        dft_value=dft_sparse,
        verdict=verdict,
        confidence=confidence,
        details=f"Simplex+Δ: {simplex_sparse:.3f}, DCT: {dct_sparse:.3f}"
    )


def test_T4_rotational_kernel(signals: Dict[str, np.ndarray]) -> TestResult:
    """T4: Tetrahedral Rotational Kernel - reduced cross-basis coherence."""
    print("[T4] Testing A4 rotational kernel coherence...")
    
    signal = signals['mixed_wave_text']
    n = len(signal)
    
    # Build coherence matrices
    # Standard RFT vs DCT
    rft_coeffs = delta_rft_forward(signal, variant="T1")
    rot_coeffs = rotational_kernel_transform(signal)
    dct_coeffs = dct(signal, norm="ortho")
    
    # Cross-coherence: |<A, B>| / (||A|| ||B||)
    rft_dct_coh = coherence_violation(rft_coeffs, dct_coeffs)
    rot_dct_coh = coherence_violation(rot_coeffs, dct_coeffs)
    
    verdict = "CONFIRMED" if rot_dct_coh < rft_dct_coh else "REJECTED"
    confidence = max(0, 1 - rot_dct_coh / rft_dct_coh) if rot_dct_coh < rft_dct_coh else 0
    
    return TestResult(
        hypothesis_id="T4",
        hypothesis_text="A4 rotational kernel reduces cross-basis coherence",
        test_performed="Measured coherence between Δ-RFT and DCT representations",
        metric_name="Cross-basis coherence (lower = better)",
        delta_rft_value=rot_dct_coh,
        baseline_value=rft_dct_coh,
        dft_value=coherence_violation(fft(signal, norm="ortho"), dct_coeffs),
        verdict=verdict,
        confidence=confidence,
        details=f"Rotational: {rot_dct_coh:.4f}, Standard Δ: {rft_dct_coh:.4f}"
    )


def test_T5_log_frequency_warp(signals: Dict[str, np.ndarray]) -> TestResult:
    """T5: Tetrahedral Log-Frequency Warp - mixed signal sparsity."""
    print("[T5] Testing 3D log-frequency warp sparsity...")
    
    signal = signals['mixed_wave_text']
    n = len(signal)
    
    # Compare variants
    log_delta_coeffs = delta_rft_forward(signal, variant="T5")
    std_delta_coeffs = delta_rft_forward(signal, variant="T1")
    dct_coeffs = dct(signal, norm="ortho")
    
    log_sparse = energy_concentration(log_delta_coeffs)
    std_sparse = energy_concentration(std_delta_coeffs)
    dct_sparse = energy_concentration(dct_coeffs)
    
    verdict = "CONFIRMED" if log_sparse < dct_sparse else "REJECTED"
    confidence = max(0, 1 - log_sparse / dct_sparse) if log_sparse < dct_sparse else 0
    
    return TestResult(
        hypothesis_id="T5",
        hypothesis_text="Tetrahedral log-warp produces better sparsity on mixed signals",
        test_performed="Compared 3D log-warp Δ-RFT vs standard approaches",
        metric_name="Energy concentration (lower = sparser)",
        delta_rft_value=log_sparse,
        baseline_value=dct_sparse,
        dft_value=std_sparse,
        verdict=verdict,
        confidence=confidence,
        details=f"LogΔ: {log_sparse:.3f}, DCT: {dct_sparse:.3f}, StdΔ: {std_sparse:.3f}"
    )


def test_T6_three_vertex_cascade(signals: Dict[str, np.ndarray]) -> TestResult:
    """T6: 3-Vertex Cascade - lower BPP."""
    print("[T6] Testing 3-branch tetrahedral cascade...")
    
    signal = signals['source_code']
    n = len(signal)
    
    # 3-vertex cascade
    b1, b2, b3 = three_vertex_cascade(signal, threshold=0.1)
    cascade_recon = b1 + b2 + b3
    cascade_mse = reconstruction_mse(signal, cascade_recon)
    
    # Standard 2-way cascade (DCT + RFT)
    dct_coeffs = dct(signal, norm="ortho")
    thresh = 0.1 * np.max(np.abs(dct_coeffs))
    dct_sparse = dct_coeffs * (np.abs(dct_coeffs) > thresh)
    dct_recon = idct(dct_sparse, norm="ortho")
    residual = signal - dct_recon
    
    rft_coeffs = delta_rft_forward(residual, variant="T1")
    rft_sparse = rft_coeffs * (np.abs(rft_coeffs) > thresh)
    rft_recon = delta_rft_inverse(rft_sparse, variant="T1").real
    two_way_recon = dct_recon + rft_recon
    two_way_mse = reconstruction_mse(signal, two_way_recon)
    
    # Pure DCT
    pure_dct_recon = idct(dct_sparse, norm="ortho")
    pure_dct_mse = reconstruction_mse(signal, pure_dct_recon)
    
    verdict = "CONFIRMED" if cascade_mse < two_way_mse else "REJECTED"
    confidence = max(0, 1 - cascade_mse / two_way_mse) if cascade_mse < two_way_mse else 0
    
    return TestResult(
        hypothesis_id="T6",
        hypothesis_text="3-vertex cascade reduces reconstruction error",
        test_performed="Compared 3-way vs 2-way residual cascade at same threshold",
        metric_name="Reconstruction MSE (lower = better)",
        delta_rft_value=cascade_mse,
        baseline_value=two_way_mse,
        dft_value=pure_dct_mse,
        verdict=verdict,
        confidence=confidence,
        details=f"3-way: {cascade_mse:.6f}, 2-way: {two_way_mse:.6f}, DCT-only: {pure_dct_mse:.6f}"
    )


def test_T7_attention_routing(signals: Dict[str, np.ndarray]) -> TestResult:
    """T7: Tetrahedral Attention Routing - dictionary selection."""
    print("[T7] Testing tetrahedral attention for dictionary selection...")
    
    signal = signals['mixed_wave_text']
    n = len(signal)
    
    # Compute coefficients in multiple bases
    bases = {
        'dct': dct(signal, norm="ortho"),
        'delta_t1': delta_rft_forward(signal, variant="T1"),
        'delta_t2': delta_rft_forward(signal, variant="T2"),
        'dft': fft(signal, norm="ortho")
    }
    
    # Tetrahedral attention: weight each basis by energy in tetrahedral directions
    attention_weights = np.zeros(4)
    for i, (name, coeffs) in enumerate(bases.items()):
        energy = np.abs(coeffs) ** 2
        # Top-k energy concentration as attention weight
        attention_weights[i] = 1.0 / (energy_concentration(coeffs) + 0.01)
    attention_weights /= attention_weights.sum()
    
    # Weighted combination
    combined = np.zeros(n, dtype=np.complex128)
    for i, (name, coeffs) in enumerate(bases.items()):
        combined += attention_weights[i] * coeffs
    
    attention_sparse = energy_concentration(combined)
    dct_sparse = energy_concentration(bases['dct'])
    dft_sparse = energy_concentration(bases['dft'])
    
    verdict = "CONFIRMED" if attention_sparse < dct_sparse else "REJECTED"
    confidence = max(0, 1 - attention_sparse / dct_sparse) if attention_sparse < dct_sparse else 0
    
    return TestResult(
        hypothesis_id="T7",
        hypothesis_text="Tetrahedral attention improves dictionary selection",
        test_performed="Weighted basis combination using energy-based attention",
        metric_name="Combined representation sparsity",
        delta_rft_value=attention_sparse,
        baseline_value=dct_sparse,
        dft_value=dft_sparse,
        verdict=verdict,
        confidence=confidence,
        details=f"Attention weights: DCT={attention_weights[0]:.2f}, Δ1={attention_weights[1]:.2f}, Δ2={attention_weights[2]:.2f}"
    )


def test_T8_sensing_matrix(signals: Dict[str, np.ndarray]) -> TestResult:
    """T8: Tetrahedral Sensing Matrix - CS recovery."""
    print("[T8] Testing tetrahedral sensing matrix for compressed sensing...")
    
    signal = signals['three_component']
    n = len(signal)
    m = n // 4  # 25% measurements
    
    # Build sensing matrices
    delta_sense = tetrahedral_sensing_matrix(m, n)
    
    # Random Gaussian sensing matrix
    np.random.seed(42)
    gauss_sense = np.random.randn(m, n) / np.sqrt(m)
    
    # DCT sensing matrix (random rows of DCT matrix)
    dct_matrix = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        dct_matrix[:, i] = dct(e_i, norm="ortho")
    row_idx = np.random.choice(n, m, replace=False)
    dct_sense = dct_matrix[row_idx, :]
    
    # Measure mutual coherence (lower = better for CS)
    delta_coh = mutual_coherence(delta_sense)
    gauss_coh = mutual_coherence(gauss_sense)
    dct_coh = mutual_coherence(dct_sense)
    
    verdict = "CONFIRMED" if delta_coh < gauss_coh else "REJECTED"
    confidence = max(0, 1 - delta_coh / gauss_coh) if delta_coh < gauss_coh else 0
    
    return TestResult(
        hypothesis_id="T8",
        hypothesis_text="Tetrahedral sensing matrices have lower coherence",
        test_performed=f"Built {m}×{n} sensing matrices, measured mutual coherence",
        metric_name="Mutual coherence (lower = better for ℓ₁ recovery)",
        delta_rft_value=delta_coh,
        baseline_value=gauss_coh,
        dft_value=dct_coh,
        verdict=verdict,
        confidence=confidence,
        details=f"Δ-sense: {delta_coh:.4f}, Gaussian: {gauss_coh:.4f}, DCT: {dct_coh:.4f}"
    )


def test_T9_quasicrystal_decoder(signals: Dict[str, np.ndarray]) -> TestResult:
    """T9: Tetrahedral Quasicrystal Decoder - sharper peaks."""
    print("[T9] Testing tetrahedral decoder on quasicrystal signals...")
    
    signal = signals['quasicrystal']
    n = len(signal)
    
    # Apply transforms
    delta_coeffs = delta_rft_forward(signal, variant="T1")
    dct_coeffs = dct(signal, norm="ortho")
    dft_coeffs = fft(signal, norm="ortho")
    
    # Measure peak sharpness via kurtosis (higher = sharper)
    delta_kurt = kurtosis(np.abs(delta_coeffs))
    dct_kurt = kurtosis(np.abs(dct_coeffs))
    dft_kurt = kurtosis(np.abs(dft_coeffs))
    
    verdict = "CONFIRMED" if delta_kurt > dct_kurt else "REJECTED"
    confidence = max(0, 1 - dct_kurt / delta_kurt) if delta_kurt > dct_kurt else 0
    
    return TestResult(
        hypothesis_id="T9",
        hypothesis_text="Tetrahedral decoder produces sharper quasicrystal peaks",
        test_performed="Measured kurtosis of coefficient magnitudes",
        metric_name="Kurtosis (higher = sharper peaks)",
        delta_rft_value=delta_kurt,
        baseline_value=dct_kurt,
        dft_value=dft_kurt,
        verdict=verdict,
        confidence=confidence,
        details=f"Δ-RFT kurtosis: {delta_kurt:.2f}, DCT: {dct_kurt:.2f}, DFT: {dft_kurt:.2f}"
    )


def test_T10_quantization_minimizer(signals: Dict[str, np.ndarray]) -> TestResult:
    """T10: Tetrahedral Quantization Minimizer - builds on H9."""
    print("[T10] Testing tetrahedral quantization robustness on 3D mixtures...")
    
    signal = signals['three_component']
    n = len(signal)
    bits = 8
    
    def quantize_roundtrip(coeffs, bits=8):
        """Quantize and dequantize complex coefficients."""
        # Separate real/imag
        real = coeffs.real
        imag = coeffs.imag
        
        # Quantize to bits
        scale_r = (2**(bits-1) - 1) / (np.abs(real).max() + 1e-10)
        scale_i = (2**(bits-1) - 1) / (np.abs(imag).max() + 1e-10)
        
        q_real = np.round(real * scale_r) / scale_r
        q_imag = np.round(imag * scale_i) / scale_i
        
        return q_real + 1j * q_imag
    
    # Transform, quantize, inverse
    delta_coeffs = delta_rft_forward(signal, variant="T1")
    delta_q = quantize_roundtrip(delta_coeffs, bits)
    delta_recon = delta_rft_inverse(delta_q, variant="T1").real
    delta_mse = reconstruction_mse(signal, delta_recon)
    
    dct_coeffs = dct(signal, norm="ortho")
    dct_q = quantize_roundtrip(dct_coeffs.astype(np.complex128), bits).real
    dct_recon = idct(dct_q, norm="ortho")
    dct_mse = reconstruction_mse(signal, dct_recon)
    
    dft_coeffs = fft(signal, norm="ortho")
    dft_q = quantize_roundtrip(dft_coeffs, bits)
    dft_recon = ifft(dft_q, norm="ortho").real
    dft_mse = reconstruction_mse(signal, dft_recon)
    
    verdict = "CONFIRMED" if delta_mse < dct_mse else "REJECTED"
    confidence = max(0, 1 - delta_mse / dct_mse) if delta_mse < dct_mse else 0
    
    return TestResult(
        hypothesis_id="T10",
        hypothesis_text="Tetrahedral RFT reduces quantization MSE on 3D mixtures",
        test_performed=f"8-bit quantization roundtrip on 3-component signal",
        metric_name="Reconstruction MSE after quantization (lower = better)",
        delta_rft_value=delta_mse,
        baseline_value=dct_mse,
        dft_value=dft_mse,
        verdict=verdict,
        confidence=confidence,
        details=f"Δ-RFT MSE: {delta_mse:.6f}, DCT: {dct_mse:.6f}, DFT: {dft_mse:.6f}"
    )


# =============================================================================
# Main Runner
# =============================================================================

def run_all_tests():
    """Run all 10 tetrahedral hypothesis tests."""
    print("=" * 70)
    print("TETRAHEDRAL Φ-RFT HYPOTHESIS TEST SUITE")
    print("=" * 70)
    print()
    print("Testing 10 hypotheses about tetrahedral generalizations of Φ-RFT.")
    print("These extend phase operators Dφ and Cσ into 3D/4D simplex geometry.")
    print()
    
    # Generate test signals
    print("Generating test signals (N=512)...")
    signals = generate_test_signals(512)
    print(f"  Signals: {list(signals.keys())}")
    print()
    
    # Run all tests
    tests = [
        test_T1_tetrahedral_phase,
        test_T2_tetrahedral_chirp,
        test_T3_simplex_embedding,
        test_T4_rotational_kernel,
        test_T5_log_frequency_warp,
        test_T6_three_vertex_cascade,
        test_T7_attention_routing,
        test_T8_sensing_matrix,
        test_T9_quasicrystal_decoder,
        test_T10_quantization_minimizer,
    ]
    
    results: List[TestResult] = []
    for test_func in tests:
        try:
            result = test_func(signals)
            results.append(result)
            status = "✅" if result.verdict == "CONFIRMED" else "❌"
            print(f"  {status} {result.hypothesis_id}: {result.verdict} (conf: {result.confidence:.2f})")
        except Exception as e:
            print(f"  ⚠️ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    confirmed = [r for r in results if r.verdict == "CONFIRMED"]
    rejected = [r for r in results if r.verdict == "REJECTED"]
    
    print(f"\nConfirmed: {len(confirmed)}/10")
    for r in confirmed:
        print(f"  ✅ {r.hypothesis_id}: {r.hypothesis_text[:50]}...")
    
    print(f"\nRejected: {len(rejected)}/10")
    for r in rejected:
        print(f"  ❌ {r.hypothesis_id}: {r.hypothesis_text[:50]}...")
    
    # Save results
    output_dir = Path(__file__).parent
    
    # JSON
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj
    
    json_data = [
        _convert({
            'hypothesis_id': r.hypothesis_id,
            'hypothesis_text': r.hypothesis_text,
            'test_performed': r.test_performed,
            'metric_name': r.metric_name,
            'delta_rft_value': r.delta_rft_value,
            'baseline_value': r.baseline_value,
            'dft_value': r.dft_value,
            'verdict': r.verdict,
            'confidence': r.confidence,
            'details': r.details
        })
        for r in results
    ]
    
    json_path = output_dir / 'tetrahedral_rft_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    return results


if __name__ == "__main__":
    run_all_tests()
