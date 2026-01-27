# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
import numpy as np

from algorithms.rft.core.gram_utils import gram_matrix
from algorithms.rft.core.resonant_fourier_transform import (
    rft_basis_matrix,
    rft_forward_frame,
    rft_inverse_frame,
)


def test_phi_basis_not_orthogonal_without_normalization() -> None:
    N = 256
    Phi = rft_basis_matrix(N, N, use_gram_normalization=False)
    G = gram_matrix(Phi)

    I = np.eye(N, dtype=np.complex128)
    # It should *not* be (close to) identity in general.
    assert not np.allclose(G, I, atol=1e-10, rtol=0.0)


def test_phi_basis_becomes_unitary_with_gram_normalization() -> None:
    N = 256
    Phi = rft_basis_matrix(N, N, use_gram_normalization=True)
    G = gram_matrix(Phi)

    I = np.eye(N, dtype=np.complex128)
    assert np.allclose(G, I, atol=1e-12, rtol=0.0)


def test_frame_correct_inverse_reconstructs() -> None:
    rng = np.random.default_rng(0)
    N = 256
    x = rng.normal(size=N) + 1j * rng.normal(size=N)

    Phi_raw = rft_basis_matrix(N, N, use_gram_normalization=False)

    # Naive coefficients (assumes orthonormal columns; generally wrong here)
    X_naive = Phi_raw.conj().T @ x
    x_hat_naive = Phi_raw @ X_naive
    naive_err = np.linalg.norm(x_hat_naive - x) / np.linalg.norm(x)

    # Frame-correct coefficients using the dual-frame solve
    X_frame = rft_forward_frame(x, Phi_raw)
    x_hat_frame = rft_inverse_frame(X_frame, Phi_raw)
    frame_err = np.linalg.norm(x_hat_frame - x) / np.linalg.norm(x)

    # Gram-normalized basis behaves like a unitary transform
    Phi_unit = rft_basis_matrix(N, N, use_gram_normalization=True)
    X_unit = Phi_unit.conj().T @ x
    x_hat_unit = Phi_unit @ X_unit
    unit_err = np.linalg.norm(x_hat_unit - x) / np.linalg.norm(x)

    # Frame and unitary should be essentially perfect; naive should be measurably worse.
    assert frame_err < 1e-10
    assert unit_err < 1e-10
    assert naive_err > 1e-6
