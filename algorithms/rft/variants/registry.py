"""Φ-RFT variant generators and metadata registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from .golden_ratio_unitary import GoldenRatioUnitary

PHI = (1.0 + np.sqrt(5.0)) / 2.0


def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """Return the Q factor from the QR decomposition of *matrix*."""
    q, _ = np.linalg.qr(matrix)
    return q


def generate_original_phi_rft(n: int) -> np.ndarray:
    k = np.arange(n).reshape(1, -1)
    samples = np.arange(n).reshape(-1, 1)
    phi_k = PHI ** (-k)
    theta = 2 * np.pi * phi_k * samples / n + np.pi * phi_k * (samples**2) / (2 * n)
    raw = (1.0 / np.sqrt(n)) * np.exp(1j * theta)
    return _orthonormalize(raw)


def generate_harmonic_phase(n: int, alpha: float = 0.5) -> np.ndarray:
    samples = np.arange(n).reshape(-1, 1)
    k = np.arange(n).reshape(1, -1)
    phase = (2 * np.pi * k * samples / n) + (alpha * np.pi * (k * samples) ** 3 / (n**2))
    raw = (1.0 / np.sqrt(n)) * np.exp(1j * phase)
    return _orthonormalize(raw)


def generate_fibonacci_tilt(n: int) -> np.ndarray:
    fib = [1, 1]
    while len(fib) <= n + 5:
        fib.append(fib[-1] + fib[-2])
    f_k = np.array(fib[:n], dtype=np.float64).reshape(1, -1)
    f_n = float(fib[n])
    samples = np.arange(n).reshape(-1, 1)
    phase = 2 * np.pi * f_k * samples / f_n
    raw = (1.0 / np.sqrt(n)) * np.exp(1j * phase)
    return _orthonormalize(raw)


def generate_chaotic_mix(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(matrix)
    phases = np.diagonal(r) / np.abs(np.diagonal(r))
    return q @ np.diag(phases)


def generate_geometric_lattice(n: int) -> np.ndarray:
    samples = np.arange(n).reshape(-1, 1)
    k = np.arange(n).reshape(1, -1)
    phase = (2 * np.pi * k * samples / n) + (2 * np.pi * (samples**2 * k + samples * k**2) / (n**2))
    raw = (1.0 / np.sqrt(n)) * np.exp(1j * phase)
    return _orthonormalize(raw)


def generate_phi_chaotic_hybrid(n: int) -> np.ndarray:
    fib = generate_fibonacci_tilt(n)
    chaos = generate_chaotic_mix(n)
    combined = (fib + chaos) / np.sqrt(2)
    return _orthonormalize(combined)


def generate_adaptive_phi(n: int) -> np.ndarray:
    # Adaptive variant defers to hybrid generator for deterministic basis.
    return generate_phi_chaotic_hybrid(n)


def _fft_factorized_basis(
    n: int,
    *,
    golden_phase: np.ndarray,
    sigma: float = 1.25,
) -> np.ndarray:
    """Helper to build Φ-RFT-style unitary matrices via FFT factorization."""
    k = np.arange(n, dtype=np.float64)
    phase_quadratic = np.exp(1j * np.pi * sigma * (k * k) / float(n))
    fft_matrix = np.fft.fft(np.eye(n), norm="ortho")
    raw = golden_phase.reshape(-1, 1) * (phase_quadratic.reshape(-1, 1) * fft_matrix)
    return _orthonormalize(raw)


def generate_log_periodic_phi_rft(n: int, beta: float = 0.83, sigma: float = 1.25) -> np.ndarray:
    """Log-periodic Φ-RFT variant (Theorem 10 hybrid)."""
    k = np.arange(n, dtype=np.float64)
    logk = np.log1p(k) / np.log1p(float(n))
    phase = np.exp(1j * 2.0 * np.pi * beta * logk)
    return _fft_factorized_basis(n, golden_phase=phase, sigma=sigma)


def generate_convex_mixed_phi_rft(
    n: int,
    *,
    beta: float = 0.83,
    sigma: float = 1.25,
    mix: float = 0.5,
) -> np.ndarray:
    """Convex blend between standard and log-periodic Φ phases."""
    k = np.arange(n, dtype=np.float64)
    frac_k = np.modf(k / PHI)[0]
    theta_std = 2.0 * np.pi * beta * frac_k
    logk = np.log1p(k) / np.log1p(float(n))
    theta_log = 2.0 * np.pi * beta * logk
    mix_clamped = float(np.clip(mix, 0.0, 1.0))
    theta = (1.0 - mix_clamped) * theta_std + mix_clamped * theta_log
    phase = np.exp(1j * theta)
    return _fft_factorized_basis(n, golden_phase=phase, sigma=sigma)


def generate_exact_golden_ratio_unitary(n: int) -> np.ndarray:
    """High-fidelity Golden Ratio kernel using exact construction."""
    builder = GoldenRatioUnitary()
    matrix = builder.construct_rft_matrix(n)
    return _orthonormalize(matrix)


@dataclass(frozen=True)
class VariantInfo:
    name: str
    generator: Callable[[int], np.ndarray]
    innovation: str
    use_case: str


VARIANTS: Dict[str, VariantInfo] = {
    "original": VariantInfo(
        name="Original Φ-RFT",
        generator=generate_original_phi_rft,
        innovation="Golden-resonant phase",
        use_case="Quantum simulation",
    ),
    "harmonic_phase": VariantInfo(
        name="Harmonic-Phase",
        generator=generate_harmonic_phase,
        innovation="Cubic time-base",
        use_case="Nonlinear filtering",
    ),
    "fibonacci_tilt": VariantInfo(
        name="Fibonacci Tilt",
        generator=generate_fibonacci_tilt,
        innovation="Integer lattice alignment",
        use_case="Lattice structures (experimental)",
    ),
    "chaotic_mix": VariantInfo(
        name="Chaotic Mix",
        generator=generate_chaotic_mix,
        innovation="Haar-like randomness",
        use_case="Mixing/diffusion (experimental)",
    ),
    "geometric_lattice": VariantInfo(
        name="Geometric Lattice",
        generator=generate_geometric_lattice,
        innovation="Phase-engineered lattice",
        use_case="Analog / optical computing",
    ),
    "phi_chaotic_hybrid": VariantInfo(
        name="Φ-Chaotic Hybrid",
        generator=generate_phi_chaotic_hybrid,
        innovation="Structure + disorder",
        use_case="Resilient codecs",
    ),
    "adaptive_phi": VariantInfo(
        name="Adaptive Φ",
        generator=generate_adaptive_phi,
        innovation="Meta selection",
        use_case="Universal compression",
    ),
    "log_periodic": VariantInfo(
        name="Log-Periodic Φ-RFT",
        generator=generate_log_periodic_phi_rft,
        innovation="Log-frequency phase warp",
        use_case="Symbol compression",
    ),
    "convex_mix": VariantInfo(
        name="Convex Mixed Φ-RFT",
        generator=generate_convex_mixed_phi_rft,
        innovation="Hybrid log/standard phase",
        use_case="Adaptive textures",
    ),
    "golden_ratio_exact": VariantInfo(
        name="Exact Golden Ratio Kernel",
        generator=generate_exact_golden_ratio_unitary,
        innovation="Full resonance lattice",
        use_case="Theorem validation",
    ),
}

__all__ = [
    "PHI",
    "VariantInfo",
    "VARIANTS",
    "generate_original_phi_rft",
    "generate_harmonic_phase",
    "generate_fibonacci_tilt",
    "generate_chaotic_mix",
    "generate_geometric_lattice",
    "generate_phi_chaotic_hybrid",
    "generate_adaptive_phi",
    "generate_log_periodic_phi_rft",
    "generate_convex_mixed_phi_rft",
    "generate_exact_golden_ratio_unitary",
]
