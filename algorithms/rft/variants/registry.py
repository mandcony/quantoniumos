"""Φ-RFT variant generators and metadata registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

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
        use_case="Post-quantum crypto",
    ),
    "chaotic_mix": VariantInfo(
        name="Chaotic Mix",
        generator=generate_chaotic_mix,
        innovation="Haar-like randomness",
        use_case="Secure scrambling",
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
]
