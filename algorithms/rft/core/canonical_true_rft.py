#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Canonical True RFT API (Gram-Normalized Φ-Grid Basis)

This module preserves the CanonicalTrueRFT class API and implements
the canonical RFT definition via Gram-matrix normalization of the
φ-grid exponential basis. This is the mathematically exact reference
implementation (O(N^3) basis build, O(N^2) transforms).
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from .resonant_fourier_transform import (
    PHI,
    rft_basis_matrix,
    rft_forward_frame,
    rft_inverse_frame,
)


class CanonicalTrueRFT:
    """
    Compatibility wrapper retaining the CanonicalTrueRFT API while
    implementing the canonical Gram-normalized Φ-grid basis.
    """

    def __init__(self, size: int, beta: float = 1.0, *, sigma: float = 1.0, phi: float = PHI,
                 validate_unitarity: bool = False, validation_trials: int = 2):
        """
        Initialize RFT operator for given size.

        Args:
            size: Transform size
            beta: Frequency scaling parameter
            sigma: Chirp width parameter for C_sigma
            phi: Golden ratio parameter
            validate_unitarity: If True, runs a quick stochastic unitarity check
            validation_trials: Number of trials for the unitary check
        """
        self.size = int(size)
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.phi = float(phi)

        # Canonical Gram-normalized basis (unitary).
        self._basis = rft_basis_matrix(self.size, self.size, use_gram_normalization=True)

        if validate_unitarity:
            err = self.get_unitarity_error()
            tol = 1e-12
            if err > tol:
                raise ValueError(f"Φ-RFT unitarity error {err:.2e} exceeds tolerance {tol:.2e}")

    def forward_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply forward Φ-RFT using closed-form implementation."""
        x = np.asarray(x, dtype=np.complex128)
        if x.shape[0] != self.size:
            raise ValueError(f"Input size {x.shape[0]} != RFT size {self.size}")
        return rft_forward_frame(x, self._basis)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Apply inverse RFT: x = Ψ y
        
        Args:
            y: RFT coefficients (complex vector)
            
        Returns:
            Reconstructed signal
        """
        y = np.asarray(y, dtype=np.complex128)
        if y.shape[0] != self.size:
            raise ValueError(f"Input size {y.shape[0]} != RFT size {self.size}")
        return rft_inverse_frame(y, self._basis)
    
    def get_unitarity_error(self) -> float:
        """Return stochastic unitarity error estimate for current parameters."""
        identity = np.eye(self.size, dtype=np.complex128)
        return float(np.linalg.norm(self._basis.conj().T @ self._basis - identity))
    
    def get_rft_matrix(self) -> np.ndarray:
        """Construct and return the Φ-RFT basis matrix Ψ (nxn)."""
        return self._basis.copy()


def validate_rft_properties(size: int = 64) -> dict:
    """
    Comprehensive validation of RFT properties for the research paper.
    
    Args:
        size: Transform size for validation
        
    Returns:
        Dictionary of validation metrics
    """
    print(f"Validating RFT properties for size {size}...")
    
    rft = CanonicalTrueRFT(size)
    
    # Test round-trip accuracy
    np.random.seed(1337)  # Reproducible testing
    test_signals = [
        np.random.randn(size) + 1j * np.random.randn(size),
        np.ones(size, dtype=complex),
        np.exp(2j * np.pi * np.arange(size) / size),  # Pure frequency
    ]
    
    max_roundtrip_error = 0.0
    for i, x in enumerate(test_signals):
        x = x / np.linalg.norm(x)  # Normalize
        y = rft.forward_transform(x)
        x_reconstructed = rft.inverse_transform(y)
        error = np.linalg.norm(x - x_reconstructed)
        max_roundtrip_error = max(max_roundtrip_error, error)
        print(f"  Test signal {i+1}: roundtrip error = {error:.2e}")
    
    # Unitarity validation
    unitarity_error = rft.get_unitarity_error()
    
    # DFT distinction (for paper validation)
    dft_matrix = np.fft.fft(np.eye(size)) / np.sqrt(size)  # Unitary DFT
    _psi = rft.get_rft_matrix()
    dft_distance = np.linalg.norm(_psi - dft_matrix, 'fro')
    
    results = {
        'size': size,
        'unitarity_error': unitarity_error,
        'max_roundtrip_error': max_roundtrip_error,
        'dft_distance': dft_distance,
        'paper_validation': {
            'unitarity_meets_spec': unitarity_error < 1e-12,
            'roundtrip_acceptable': max_roundtrip_error < 1e-10,
            'mathematically_distinct_from_dft': dft_distance > 1.0
        }
    }
    
    print(f"✓ Validation complete:")
    print(f"  Unitarity error: {unitarity_error:.2e} (spec: <1e-12)")
    print(f"  Max roundtrip error: {max_roundtrip_error:.2e}")
    print(f"  DFT distance: {dft_distance:.3f}")
    
    return results


if __name__ == "__main__":
    # Run validation as described in the paper
    validation_results = validate_rft_properties(64)
    
    # Additional test for paper metrics
    print("\n" + "="*50)
    print("PAPER VALIDATION METRICS")
    print("="*50)
    
    sizes = [8, 16, 32, 64, 128]
    for size in sizes:
        rft = CanonicalTrueRFT(size)
        error = rft.get_unitarity_error()
        print(f"Size {size:3d}: Unitarity error = {error:.2e}")
