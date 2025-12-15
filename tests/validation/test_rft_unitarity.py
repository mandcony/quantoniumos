import numpy as np

from algorithms.rft.transform_core.resonant_fourier_transform import (
    rft_basis_matrix,
    rft_forward,
    rft_inverse,
)


def test_rft_basis_is_unitary_N256() -> None:
    N = 256
    Phi = rft_basis_matrix(N, N)

    gram = Phi.conj().T @ Phi
    identity = np.eye(N, dtype=np.complex128)

    # Tight tolerance: Phi is constructed to be unitary at finite N.
    assert np.allclose(gram, identity, atol=1e-10, rtol=0.0)


def test_rft_roundtrip_unitary_basis() -> None:
    rng = np.random.default_rng(0)
    N = 256
    x = rng.normal(size=N) + 1j * rng.normal(size=N)

    X = rft_forward(x)
    x_hat = rft_inverse(X)

    err = np.linalg.norm(x_hat - x) / np.linalg.norm(x)
    assert err < 1e-10
