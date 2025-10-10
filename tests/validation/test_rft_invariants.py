"""Unit tests for the canonical Resonance Fourier Transform implementation."""
import numpy as np
import pytest

from core.canonical_true_rft import CanonicalTrueRFT


@pytest.mark.parametrize("size", [4, 8, 16])
def test_unitarity_error_below_tolerance(size: int) -> None:
    rft = CanonicalTrueRFT(size)
    assert rft.get_unitarity_error() < 1e-12


@pytest.mark.parametrize("size", [4, 8, 16])
def test_roundtrip_accuracy(size: int) -> None:
    rng = np.random.default_rng(1337)
    rft = CanonicalTrueRFT(size)
    vector = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    transformed = rft.forward_transform(vector)
    reconstructed = rft.inverse_transform(transformed)
    assert np.allclose(vector, reconstructed, atol=1e-10)


def test_distinct_from_dft() -> None:
    size = 8
    rft = CanonicalTrueRFT(size)
    dft_matrix = np.fft.fft(np.eye(size)) / np.sqrt(size)
    distance = np.linalg.norm(rft.get_rft_matrix() - dft_matrix, ord="fro")
    assert distance > 1.0
