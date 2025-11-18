# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Closed-Form Φ-RFT Implementation
Optimized unitary transform with golden-ratio phase modulation
"""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

PHI = (1.0 + 5.0 ** 0.5) / 2.0  # golden ratio

def _frac(arr: np.ndarray) -> np.ndarray:
    frac, _ = np.modf(arr)
    return np.where(frac < 0.0, frac + 1.0, frac)

def rft_phase_vectors(n: int, *, beta: float = 1.0, sigma: float = 1.0, phi: float = PHI) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(n, dtype=np.float64)
    theta = 2.0 * np.pi * beta * _frac(k / float(phi))               # non-quadratic
    D_phi = np.exp(1j * theta).astype(np.complex128, copy=False)
    ctheta = np.pi * sigma * (k * k / float(n))                      # quadratic chirp
    C_sig = np.exp(1j * ctheta).astype(np.complex128, copy=False)
    return D_phi, C_sig

def rft_forward(x: ArrayLike, *, beta: float = 1.0, sigma: float = 1.0, phi: float = PHI) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    D_phi, C_sig = rft_phase_vectors(x.shape[0], beta=beta, sigma=sigma, phi=phi)
    X = np.fft.fft(x, norm="ortho")
    return D_phi * (C_sig * X)

def rft_inverse(y: ArrayLike, *, beta: float = 1.0, sigma: float = 1.0, phi: float = PHI) -> np.ndarray:
    y = np.asarray(y, dtype=np.complex128)
    D_phi, C_sig = rft_phase_vectors(y.shape[0], beta=beta, sigma=sigma, phi=phi)
    return np.fft.ifft(np.conj(C_sig) * np.conj(D_phi) * y, norm="ortho")

def rft_unitary_error(n: int, *, beta: float = 1.0, sigma: float = 1.0, phi: float = PHI, trials: int = 4) -> float:
    errs, rng = [], np.random.default_rng(0x5151AA)
    for _ in range(trials):
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        rel = np.linalg.norm(rft_inverse(rft_forward(x, beta=beta, sigma=sigma, phi=phi),
                                         beta=beta, sigma=sigma, phi=phi) - x) / max(1e-16, np.linalg.norm(x))
        errs.append(rel)
    return float(np.mean(errs))

def rft_matrix(n: int, *, beta: float = 1.0, sigma: float = 1.0, phi: float = PHI) -> np.ndarray:
    """Generate the nxn unitary matrix Ψ"""
    I = np.eye(n, dtype=np.complex128)
    # Apply to each column (basis vector)
    cols = [rft_forward(I[:, i], beta=beta, sigma=sigma, phi=phi) for i in range(n)]
    return np.column_stack(cols)
