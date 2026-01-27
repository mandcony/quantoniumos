#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
"""Tests for quantization behavior in entropy utilities."""

import numpy as np

from algorithms.rft.compression.entropy import uniform_quantizer


def test_uniform_quantizer_truncates_toward_zero():
    coeffs = np.array([-1.9, -1.1, -0.9, 0.0, 0.9, 1.1, 1.9], dtype=np.float64)
    q = uniform_quantizer(coeffs, 1.0)
    expected = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    assert np.allclose(q, expected)


def test_uniform_quantizer_complex_components():
    coeffs = np.array([1.9 + 2.2j, -1.1 - 0.1j], dtype=np.complex128)
    q = uniform_quantizer(coeffs, 1.0)
    expected = np.array([1.0 + 2.0j, -1.0 + 0.0j], dtype=np.complex128)
    assert np.allclose(q, expected)


def test_uniform_quantizer_step_size_nonpositive_passthrough():
    coeffs = np.array([0.25, -2.5, 3.75], dtype=np.float64)
    q = uniform_quantizer(coeffs, 0.0)
    assert np.array_equal(q, coeffs)
