# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC

"""Legacy φ-spaced exponential RFT (non-orthogonal, correlation-based inverse).

This file is a *lock* of the original implementation that existed in
`algorithms.rft.core.resonant_fourier_transform` prior to adding Gram-matrix
normalization and frame-correct inversion utilities.

It is preserved for backward compatibility with existing benchmarks and demos
that depend on the historical behavior:
- Basis is not generally orthogonal for irrational spacing at finite N
- Inverse uses correlation (matched filtering) rather than a true dual-frame
  solve.

If you need the mathematically complete version (Gram-normalized / frame-correct),
use `algorithms.rft.core.resonant_fourier_transform`.

References (orientation):
- Oppenheim & Schafer, Discrete-Time Signal Processing (DFT orthogonality facts)
- Encyclopaedia Britannica, Fourier analysis (Fourier-type expansions)
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import Optional

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = PHI - 1


def rft_frequency(k: int) -> float:
    """Legacy resonant frequency: f_k = (k+1) * φ."""

    return (k + 1) * PHI


def rft_phase(k: int) -> float:
    """Legacy phase offset: θ_k = 2πk/φ."""

    return 2 * np.pi * k / PHI


def rft_basis_function(k: int, t: np.ndarray) -> np.ndarray:
    """Legacy basis Ψ_k(t) = exp(2πi f_k t + i θ_k)."""

    f_k = rft_frequency(k)
    theta_k = rft_phase(k)
    return np.exp(2j * np.pi * f_k * t + 1j * theta_k)


@lru_cache(maxsize=32)
def rft_basis_matrix(N: int, T: int) -> np.ndarray:
    """Legacy N×T basis matrix Ψ with Ψ[k,t] = exp(2πi f_k (t/T) + i θ_k)."""

    t = np.arange(T) / T
    Psi = np.zeros((N, T), dtype=np.complex128)

    for k in range(N):
        Psi[k, :] = rft_basis_function(k, t)

    return Psi


def rft_forward(x: np.ndarray, T: Optional[int] = None) -> np.ndarray:
    """Legacy forward transform (data → waveform): W[t] = Σ_k x[k] Ψ_k(t)."""

    x = np.asarray(x, dtype=np.complex128)
    N = len(x)

    if T is None:
        T = N * 16

    Psi = rft_basis_matrix(N, T)
    return x @ Psi


def rft_inverse(W: np.ndarray, N: int) -> np.ndarray:
    """Legacy inverse (waveform → data) via correlation: x[k] = (1/T) Σ_t W[t] Ψ_k*(t)."""

    W = np.asarray(W, dtype=np.complex128)
    T = len(W)

    Psi = rft_basis_matrix(N, T)
    return (Psi @ np.conj(W)) / T


def rft(x: np.ndarray) -> np.ndarray:
    return rft_forward(x)


def irft(W: np.ndarray, N: int) -> np.ndarray:
    return rft_inverse(W, N)
