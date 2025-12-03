# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
"""Unit tests that ensure every Φ-RFT variant stays unitary."""

import numpy as np
import pytest

from algorithms.rft.variants.manifest import iter_variants

_VARIANT_ENTRIES = list(iter_variants(include_experimental=True))


@pytest.mark.parametrize("variant_entry", _VARIANT_ENTRIES, ids=lambda entry: entry.code)
def test_variant_generator_is_unitary(variant_entry):
    size = 32
    generator = variant_entry.info.generator
    basis = generator(size)
    identity = basis.conj().T @ basis
    error = np.linalg.norm(identity - np.eye(size))
    assert error < 1e-10, f"Variant {variant_entry.code} lost unitarity (error={error:.2e})"
