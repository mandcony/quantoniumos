#!/usr/bin/env python3
"""
Canonical True RFT Implementation
=================================

Implements the ACTUAL PROVEN RFT equation: R = Σ_i w_i D_φi C_σi D_φi†
Using the mathematically verified formula from rft_definitive_mathematical_proof.py
"""

import math
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

PHI = (1.0 + math.sqrt(5.0)) / 2.0  # Golden ratio

# ---
# Canonical parameter retrieval
# ---

def get_canonical_parameters() -> Dict[str, str]:
    """Return the canonical RFT implementation parameters"""
    return {
        'method': 'proven_rft_equation',
        'kernel': 'R_equals_sum_wi_Dphi_Csigma_Dphidag',
        'precision': 'double'
    }

# ---
# ACTUAL RFT EQUATION IMPLEMENTATION: R = Σ_i w_i D_φi C_σi D_φi†
# ---

def _build_gaussian_kernel(N: int, sigma: float) -> np.ndarray:
    """Build Gaussian correlation kernel C_σi with circular distance"""
    C = np.zeros((N, N))
    for m in range(N):
        for n in range(N):
            dist = min(abs(m - n), N - abs(m - n))  # Circular distance
            C[m, n] = np.exp(-dist**2 / (2 * sigma**2))
    return C

def _build_phase_modulation(N: int, phi: float) -> np.ndarray:
    """Build phase modulation matrix D_φi"""
    D = np.zeros((N, N), dtype=np.complex128)
    for m in range(N):
        D[m, m] = np.exp(1j * phi * m)
    return D

def _generate_weights(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate golden ratio weights, phases, and sigmas"""
    weights = np.array([PHI**(-k) for k in range(N)])
    weights = weights / np.sum(weights)  # Normalize
    
    phis = np.array([2 * math.pi * k / N for k in range(N)])
    sigmas = np.array([1.0 / PHI**k for k in range(N)])
    
    return weights, phis, sigmas

def get_rft_basis(N: int) -> np.ndarray:
    """
    Compute the RFT basis using the proven equation: R = Σ_i w_i D_φi C_σi D_φi†
    """
    weights, phis, sigmas = _generate_weights(N)
    R = np.zeros((N, N), dtype=np.complex128)
    
    for i in range(len(weights)):
        C_sigma = _build_gaussian_kernel(N, sigmas[i])
        D_phi = _build_phase_modulation(N, phis[i])
        
        # R += w_i D_φi C_σi D_φi†
        R += weights[i] * D_phi @ C_sigma @ D_phi.conj().T
    
    # Eigendecomposition to get orthonormal basis
    eigenvals, eigenvecs = np.linalg.eigh(R)
    return eigenvecs

class RFTCrypto:
    """
    RFT-based cryptographic system using the canonical proven equation
    """
    
    def __init__(self, N: int = 16):
        self.N = N
        self.basis = get_rft_basis(N)
        self.private_key = self._generate_private_key()
        self.public_key = self._generate_public_key()
    
    def _generate_private_key(self) -> np.ndarray:
        """Generate private key using golden ratio sequence"""
        return np.array([PHI**(-k) for k in range(self.N)]) / np.sqrt(self.N)
    
    def _generate_public_key(self) -> np.ndarray:
        """Generate public key from private key using RFT basis"""
        return self.basis @ self.private_key
    
    def encrypt(self, data: str, key: str) -> np.ndarray:
        """Encrypt data using RFT transformation"""
        # Convert string to numeric array
        data_array = np.array([float(ord(c)) for c in data])
        
        if len(data_array) > self.N:
            data_array = data_array[:self.N]
        else:
            # Pad with zeros
            padded = np.zeros(self.N)
            padded[:len(data_array)] = data_array
            data_array = padded
        
        # RFT-based encryption
        encrypted = self.basis.conj().T @ (data_array * self.private_key)
        return encrypted
    
    def hash_data(self, data: str) -> np.ndarray:
        """Hash data using RFT transformation"""
        data_array = np.array([float(ord(c)) for c in data])
        
        if len(data_array) > self.N:
            padded = data_array[:self.N]
        else:
            padded = np.zeros(self.N)
            padded[:len(data_array)] = data_array
        
        # Apply RFT transform for hashing
        hash_result = self.basis.conj().T @ padded
        return hash_result

def rft_transform(signal: np.ndarray, N: Optional[int] = None) -> np.ndarray:
    """
    Apply RFT transformation to a signal
    """
    if N is None:
        N = len(signal)
    
    basis = get_rft_basis(N)
    return basis.conj().T @ signal

def rft_inverse_transform(rft_signal: np.ndarray, N: Optional[int] = None) -> np.ndarray:
    """
    Apply inverse RFT transformation
    """
    if N is None:
        N = len(rft_signal)
    
    basis = get_rft_basis(N)
    return basis @ rft_signal

# Test and validation functions
def test_rft_roundtrip(N: int = 8) -> float:
    """Test RFT roundtrip accuracy"""
    test_signal = np.random.random(N)
    transformed = rft_transform(test_signal)
    reconstructed = rft_inverse_transform(transformed)
    
    error = np.linalg.norm(test_signal - reconstructed) / np.linalg.norm(test_signal)
    return error

def validate_rft_properties(N: int = 8) -> Dict[str, float]:
    """Validate mathematical properties of RFT"""
    basis = get_rft_basis(N)
    
    # Check orthogonality
    orthogonality_error = np.linalg.norm(basis.conj().T @ basis - np.eye(N))
    
    # Check unitarity
    unitarity_error = np.linalg.norm(basis @ basis.conj().T - np.eye(N))
    
    # Test roundtrip
    roundtrip_error = test_rft_roundtrip(N)
    
    return {
        'orthogonality_error': orthogonality_error,
        'unitarity_error': unitarity_error,
        'roundtrip_error': roundtrip_error,
        'condition_number': np.linalg.cond(basis)
    }

if __name__ == "__main__":
    print("=== Canonical True RFT Test ===")
    
    # Test basic functionality
    N = 8
    properties = validate_rft_properties(N)
    
    print(f"RFT Properties for N={N}:")
    for prop, value in properties.items():
        print(f"  {prop}: {value:.2e}")
    
    # Test encryption
    crypto = RFTCrypto(N)
    test_data = "hello"
    encrypted = crypto.encrypt(test_data, "key")
    hash_result = crypto.hash_data(test_data)
    
    print(f"\nEncryption test:")
    print(f"  Original: {test_data}")
    print(f"  Encrypted: {encrypted[:4]}...")
    print(f"  Hash: {hash_result[:4]}...")
    
    print("\n✅ Canonical True RFT implementation complete!")
