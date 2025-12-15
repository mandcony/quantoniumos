# SPDX-License-Identifier: AGPL-3.0-or-later

"""Gram-matrix utilities for frame normalization.

These utilities implement the standard linear-algebra constructions used for
frames / Riesz bases:

- Gram matrix: G = Φᴴ Φ
- Gram inverse square-root: G^{-1/2}

This is used to convert a full-rank (generally non-orthogonal) basis into an
orthonormalized (unitary) basis via:

    Φ̃ = Φ G^{-1/2}

References (orientation):
- Oppenheim & Schafer, Discrete-Time Signal Processing (orthogonality of DFT exponentials)
- O. Christensen, An Introduction to Frames and Riesz Bases (frame operator / dual frames)
- Encyclopaedia Britannica, Fourier analysis (Fourier-type expansions)
"""

from __future__ import annotations

import numpy as np


def gram_matrix(Phi: np.ndarray) -> np.ndarray:
    """Compute Gram matrix G = Φᴴ Φ."""

    Phi = np.asarray(Phi)
    return Phi.conj().T @ Phi


def gram_inverse_sqrt(G: np.ndarray, *, eps: float = 1e-15) -> np.ndarray:
    """Return G^{-1/2} using Hermitian eigendecomposition.

    Args:
        G: Hermitian positive definite (or semidefinite) Gram matrix
        eps: floor for eigenvalues to avoid division by zero under numerical noise

    Returns:
        Hermitian matrix approximating G^{-1/2}
    """

    G = np.asarray(G)
    eigvals, eigvecs = np.linalg.eigh(G)

    eigvals = np.maximum(eigvals, eps)
    inv_sqrt = 1.0 / np.sqrt(eigvals)

    return eigvecs @ np.diag(inv_sqrt) @ eigvecs.conj().T
