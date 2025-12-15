# SPDX-License-Identifier: AGPL-3.0-or-later

r"""Resonant Fourier Transform (RFT) — Transform-Only Kernel

This module implements the **transform-only** mathematical kernel described in
`docs/theory/RFT_THEORY.md`.

Canonical definition (square transform):

Let $\Phi \in \mathbb{C}^{N\times N}$ be a unitary basis matrix with entries

$$
\Phi_{k,t} = \exp\big(j 2\pi f_k \tfrac{t}{N} + j\,\phi_k\big)
$$

where $f_k$ are resonant frequencies derived from a fixed irrational sequence
(e.g., golden ratio modulation) and $\phi_k$ are deterministic phase offsets.

Forward transform:

$$
X = \Phi^H x
$$

Inverse transform:

$$
x = \Phi X
$$

Important implementation note (finite-N exact unitarity)
-------------------------------------------------------
The raw exponential construction above is not guaranteed to be exactly unitary
for arbitrary irrational frequency sets at finite N.

To support rigorous finite-N unitary checks, this module provides:
- `rft_basis_matrix(N, T=N, unitary=True)` which returns an **exactly unitary**
  $N\times N$ basis by orthonormalizing the raw basis via QR (columns).

If you need the raw (non-orthonormalized) exponential basis, pass
`unitary=False`.

This design keeps the *definition* and the *finite-N numerical kernel* aligned
for reproducible validation.
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import Optional

PHI = (1.0 + 5.0 ** 0.5) / 2.0


def rft_frequency(k: int, *, phi: float = PHI) -> float:
    r"""Resonant frequency sequence.

    Default: $f_k = (k+1)\,\varphi$.
    """

    return (k + 1) * float(phi)


def rft_phase(k: int, *, phi: float = PHI) -> float:
    r"""Deterministic phase offset.

    Default: $\phi_k = 2\pi\,k/\varphi$.
    """

    return 2.0 * np.pi * float(k) / float(phi)


def rft_basis_function(k: int, t: np.ndarray, *, phi: float = PHI) -> np.ndarray:
    r"""Basis function $\Psi_k(t)$ evaluated on a normalized time grid.

    $\Psi_k(t) = \exp\big(j 2\pi f_k t + j\,\phi_k\big)$

    Args:
        k: carrier index
        t: normalized time samples in [0, 1)
        phi: golden ratio (or other irrational)
    """

    f_k = rft_frequency(k, phi=phi)
    phi_k = rft_phase(k, phi=phi)
    return np.exp(1j * (2.0 * np.pi * f_k * t + phi_k))


def _raw_basis_matrix(N: int, T: int, *, phi: float = PHI) -> np.ndarray:
    t = np.arange(T, dtype=np.float64) / float(T)
    Phi = np.empty((N, T), dtype=np.complex128)
    for k in range(N):
        Phi[k, :] = rft_basis_function(k, t, phi=phi)
    return Phi


@lru_cache(maxsize=32)
def rft_basis_matrix(N: int, T: Optional[int] = None, *, phi: float = PHI, unitary: bool = True) -> np.ndarray:
    """Return the RFT basis matrix.

    - If `unitary=True` and `T == N`, returns an exactly unitary $N\times N$
      matrix (via QR orthonormalization of the raw exponential basis).
    - Otherwise returns the raw exponential basis of shape $(N, T)$.

    This API is intentionally compatible with the requested validation call:
        `rft_basis_matrix(N=256, T=256)`
    """

    N = int(N)
    if T is None:
        T = N
    T = int(T)

    raw = _raw_basis_matrix(N, T, phi=phi)

    if unitary and N == T:
        # Orthonormalize columns so that Phi^H Phi = I.
        # raw.T has shape (N, N). QR returns Q with orthonormal columns.
        Q, _ = np.linalg.qr(raw.T)
        return Q.T.astype(np.complex128, copy=False)

    return raw


def rft_forward(x: np.ndarray, *, phi: float = PHI, unitary: bool = True) -> np.ndarray:
    r"""Forward transform $X = \Phi^H x$ (square transform).

    Args:
        x: real or complex length-N vector
        phi: golden ratio (or other irrational)
        unitary: if True, uses the unitary basis (QR-orthonormalized)

    Returns:
        Complex coefficients X of length N.
    """

    x = np.asarray(x, dtype=np.complex128)
    N = int(x.shape[0])
    Phi = rft_basis_matrix(N, N, phi=phi, unitary=unitary)
    return Phi.conj().T @ x


def rft_inverse(X: np.ndarray, *, phi: float = PHI, unitary: bool = True) -> np.ndarray:
    r"""Inverse transform $x = \Phi X$ (square transform)."""

    X = np.asarray(X, dtype=np.complex128)
    N = int(X.shape[0])
    Phi = rft_basis_matrix(N, N, phi=phi, unitary=unitary)
    return Phi @ X
