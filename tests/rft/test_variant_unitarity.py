# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
"""Unit tests that ensure every Φ-RFT variant stays unitary."""

import numpy as np
import pytest

from algorithms.rft.variants.manifest import iter_variants

_VARIANT_ENTRIES = list(iter_variants(include_experimental=True))

# 2D variants return (n², n²) matrices instead of (n, n)
_2D_VARIANTS = {"ROBUST_MANIFOLD_2D"}


@pytest.mark.parametrize("variant_entry", _VARIANT_ENTRIES, ids=lambda entry: entry.code)
def test_variant_generator_is_unitary(variant_entry):
    size = 32
    generator = variant_entry.info.generator
    basis = generator(size)
    
    # 2D variants produce (n², n²) matrices
    if variant_entry.code in _2D_VARIANTS:
        expected_size = size * size
    else:
        expected_size = size
    
    identity = basis.conj().T @ basis
    error = np.linalg.norm(identity - np.eye(expected_size))
    assert error < 1e-10, f"Variant {variant_entry.code} lost unitarity (error={error:.2e})"
