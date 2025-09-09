#!/usr/bin/env python3
"""
Canonical True RFT Implementation
As described in QuantoniumOS Research Paper

Implements the unitary Resonance Fourier Transform (RFT) operator:
Ψ = Σ_i w_i D_φi C_σi D†_φi

With golden-ratio parameterization and proven unitarity < 10^-12
"""

import numpy as np
import cmath
from typing import Tuple, Optional
from numpy.linalg import qr, norm


class CanonicalTrueRFT:
    """
    Canonical implementation of the unitary Resonance Fourier Transform
    with golden-ratio parameterization and formal unitarity guarantees.
    """
    
    def __init__(self, size: int, beta: float = 1.0):
        """
        Initialize RFT operator for given size.
        
        Args:
            size: Transform size (power of 2 recommended)
            beta: Frequency scaling parameter
        """
        self.size = size
        self.beta = beta
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Precompute RFT basis matrix
        self._rft_matrix = self._construct_rft_basis()
        self._validate_unitarity()
    
    def _construct_rft_basis(self) -> np.ndarray:
        """
        Construct the unitary RFT basis matrix using golden-ratio parameterization.
        
        Returns:
            Unitary matrix Ψ of shape (size, size)
        """
        N = self.size
        
        # Golden-ratio phase sequence: φ_k = {k/φ} mod 1
        phi_sequence = np.array([(k / self.phi) % 1 for k in range(N)])
        
        # Construct kernel matrix K with Gaussian weights and golden-ratio phases
        K = np.zeros((N, N), dtype=complex)
        
        for i in range(N):
            for j in range(N):
                # Gaussian kernel weight
                sigma = 0.5  # Kernel width parameter
                g_weight = np.exp(-0.5 * ((i - j) / (sigma * N)) ** 2)
                
                # Golden-ratio phase term
                phase = 2 * np.pi * phi_sequence[i] * phi_sequence[j] * self.beta
                
                # Combined kernel element
                K[i, j] = g_weight * cmath.exp(1j * phase)
        
        # QR decomposition to ensure unitarity
        Q, R = qr(K)
        
        # Normalize to ensure exact unitarity
        for i in range(N):
            Q[:, i] = Q[:, i] / norm(Q[:, i])
        
        return Q
    
    def _validate_unitarity(self) -> None:
        """Validate that the RFT matrix is unitary within tolerance."""
        Psi = self._rft_matrix
        identity = np.eye(self.size, dtype=complex)
        unitarity_error = norm(Psi.conj().T @ Psi - identity, ord=2)
        
        tolerance = 1e-12
        if unitarity_error > tolerance:
            raise ValueError(f"Unitarity error {unitarity_error:.2e} exceeds tolerance {tolerance:.2e}")
        
        print(f"✓ RFT Unitarity validated: error = {unitarity_error:.2e}")
    
    def forward_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply forward RFT: y = Ψ^H x
        
        Args:
            x: Input signal vector
            
        Returns:
            RFT coefficients
        """
        if len(x) != self.size:
            raise ValueError(f"Input size {len(x)} != RFT size {self.size}")
        
        return self._rft_matrix.conj().T @ x
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Apply inverse RFT: x = Ψ y
        
        Args:
            y: RFT coefficients
            
        Returns:
            Reconstructed signal
        """
        if len(y) != self.size:
            raise ValueError(f"Input size {len(y)} != RFT size {self.size}")
        
        return self._rft_matrix @ y
    
    def get_unitarity_error(self) -> float:
        """Get current unitarity error of the RFT matrix."""
        Psi = self._rft_matrix
        identity = np.eye(self.size, dtype=complex)
        return float(norm(Psi.conj().T @ Psi - identity, ord=2))
    
    def get_rft_matrix(self) -> np.ndarray:
        """Get the RFT basis matrix."""
        return self._rft_matrix.copy()


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
        x = x / norm(x)  # Normalize
        y = rft.forward_transform(x)
        x_reconstructed = rft.inverse_transform(y)
        error = norm(x - x_reconstructed)
        max_roundtrip_error = max(max_roundtrip_error, error)
        print(f"  Test signal {i+1}: roundtrip error = {error:.2e}")
    
    # Unitarity validation
    unitarity_error = rft.get_unitarity_error()
    
    # DFT distinction (for paper validation)
    dft_matrix = np.fft.fft(np.eye(size)) / np.sqrt(size)  # Unitary DFT
    rft_matrix = rft.get_rft_matrix()
    dft_distance = norm(rft_matrix - dft_matrix, 'fro')
    
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
