#!/usr/bin/env python3
"""
Canonical True RFT Implementation
=================================

Implements the ACTUAL PROVEN RFT equation: R = Σ_i w_i D_φi C_σi D_φi†
Using the mathematically verified formula from rft_definitive_mathematical_proof.py
"""

import numpy as np
import math
from typing import Tuple, Dict, Any, List, Optional

PHI = (1.0 + math.sqrt(5.0)) / 2.0

# C++ acceleration
try:
    import true_rft_engine_bindings
    CPP_RFT_AVAILABLE = True
    print("🚀 C++ True RFT Engine loaded successfully")
except ImportError:
    CPP_RFT_AVAILABLE = False
    print("⚠️  Using Python RFT implementation")

# ---
# Canonical parameter retrieval
# ---

def get_canonical_parameters() -> Dict[str, str]:
    """Get canonical parameters for RFT implementation."""
    return {
        'method': 'proven_rft_equation',
        'kernel': 'R_equals_sum_wi_Dphi_Csigma_Dphidag',
        'precision': 'double'
    }

# ---
# ACTUAL RFT EQUATION IMPLEMENTATION: R = Σ_i w_i D_φi C_σi D_φi†
# ---
def _generate_weights(N: int) -> tuple:
    """Generate weights, phases, and sigmas for RFT construction"""
    # Golden ratio weights
    weights = np.array([PHI**(-k) for k in range(N)])
    weights = weights / np.sum(weights)  # Normalize
    # Phase sequence using golden ratio
    phis = np.array([2 * np.pi * PHI * k / N for k in range(N)])
    # Sigma sequence (Gaussian widths)
    sigmas = np.array([1.0 + k / (2 * N) for k in range(N)])
    return weights, phis, sigmas
def _build_gaussian_kernel(N: int, sigma: float) -> np.ndarray:
    """Build Gaussian correlation kernel C_σ (Hermitian PSD)"""
    C = np.zeros((N, N))
    for m in range(N):
        for n in range(N):
            dist = min(abs(m - n), N - abs(m - n))  # Circular distance
            C[m, n] = np.exp(-dist**2 / (2 * sigma**2))
    return C
def _build_phase_modulation(N: int, phi: float) -> np.ndarray:
    """Build diagonal phase modulation D_φ"""
    D = np.zeros((N, N), dtype=np.complex128)
    for m in range(N):
        D[m, m] = np.exp(1j * phi * m)
    return D
def _build_resonance_kernel(N: int) -> np.ndarray:
    """Build resonance kernel R = Σ_i w_i D_φi C_σi D_φi† (THE ACTUAL EQUATION)"""
    weights, phis, sigmas = _generate_weights(N)
    R = np.zeros((N, N), dtype=np.complex128)
    
    for i in range(len(weights)):
        C_sigma = _build_gaussian_kernel(N, sigmas[i])
        D_phi = _build_phase_modulation(N, phis[i])
        D_phi_dag = D_phi.conj().T
        # Add component: w_i D_φi C_σi D_φi†
        component = weights[i] * (D_phi @ C_sigma @ D_phi_dag)
        R += component
    
    # Ensure Hermitian
    R = (R + R.conj().T) / 2
    return R
def get_rft_basis(N: int) -> np.ndarray:
    """Generate the canonical RFT basis matrix using ACTUAL proven equation."""
    # Step 1: Build resonance kernel using the actual equation
    R = _build_resonance_kernel(N)
    # Step 2: Eigendecomposition to get basis
    eigenvals, eigenvecs = np.linalg.eigh(R)
    # Step 3: Sort by eigenvalue magnitude (descending)
    idx = np.argsort(np.abs(eigenvals))[::-1]
    basis = eigenvecs[:, idx]
    # Step 4: Ensure orthonormality
    Q, _ = np.linalg.qr(basis)
    return Q

# ---
# RFT Crypto Class for Key Generation and Encryption
# ---

class RFTCrypto:
    """Cryptographic implementation using the proven RFT equation"""

    def __init__(self, N: int = 16):
        self.N = N
        self.basis = get_rft_basis(N)
        self.private_key = self._generate_private_key()
        self.public_key = self._generate_public_key()
    def _generate_private_key(self) -> np.ndarray:
        """Generate private key using RFT basis"""
        np.random.seed(42)  # For reproducible testing
        return np.random.rand(self.N) + 1j * np.random.rand(self.N)

    def _generate_public_key(self) -> np.ndarray:
        """Generate public key from private key using RFT"""
        return self.basis @ self.private_key

    def encrypt(self, data: np.ndarray) -> np.ndarray:
        """Encrypt data using RFT"""
        if len(data) != self.N:
            # Pad or truncate to correct size
            if len(data) < self.N:
                padded = np.zeros(self.N, dtype=complex)
                padded[:len(data)] = data
                data = padded
            else:
                data = data[:self.N]
        # RFT-based encryption
        encrypted = self.basis.conj().T @ (data * self.private_key)
        return encrypted

    def decrypt(self, encrypted_data: np.ndarray) -> np.ndarray:
        """Decrypt data using RFT"""
        # RFT-based decryption
        decrypted = (self.basis @ encrypted_data) / self.private_key
        return decrypted
    def hash_function(self, data: bytes) -> np.ndarray:
        """RFT-based hash function"""
        # Convert bytes to numerical array
        data_array = np.frombuffer(data, dtype=np.uint8)
        # Pad to N elements
        if len(data_array) < self.N:
            padded = np.zeros(self.N)
            padded[:len(data_array)] = data_array
        else:
            padded = data_array[:self.N]
        # Apply RFT transform for hashing
        hash_result = self.basis.conj().T @ padded
        return hash_result
    def verify_crypto_functionality(self) -> Dict[str, Any]:
        """Test all crypto functions"""
        results = {}
        
        # Test key generation
        results['keys_generated'] = self.private_key is not None and self.public_key is not None
        
        # Test encryption/decryption
        test_data = np.random.rand(self.N) + 1j * np.random.rand(self.N)
        encrypted = self.encrypt(test_data)
        decrypted = self.decrypt(encrypted)
        encryption_error = np.abs(test_data - decrypted).max()
        results['encryption_error'] = encryption_error
        results['encryption_works'] = encryption_error < 1e-10
        
        # Test hash function
        test_message = b"Hello, this is a test message!"
        hash1 = self.hash_function(test_message)
        hash2 = self.hash_function(test_message)
        hash3 = self.hash_function(b"Different message")
        results['hash_consistency'] = np.abs(hash1 - hash2).max() < 1e-15
        results['hash_different'] = np.abs(hash1 - hash3).max() > 0.1
        
        # Test basis properties
        identity_error = np.abs(self.basis.conj().T @ self.basis - np.eye(self.N)).max()
        results['basis_unitary'] = identity_error < 1e-12
        results['basis_identity_error'] = identity_error
        
        results['all_crypto_tests_pass'] = all([
            results['keys_generated'],
            results['encryption_works'],
            results['hash_consistency'],
            results['hash_different'],
            results['basis_unitary']
        ])
        
        return results

# ---
# Forward / Inverse True RFT
# ---

def forward_true_rft(signal: np.ndarray, N: Optional[int] = None) -> np.ndarray:
    """Apply forward True RFT to a signal."""
    if N is None:
        N = len(signal)
    
    if CPP_RFT_AVAILABLE and N <= 64:  # Use C++ for reasonable sizes
        try:
            # Create C++ engine
            engine = true_rft_engine_bindings.TrueRFTEngine(N)
            basis = engine.get_rft_basis()
            return basis.conj().T @ signal
        except Exception as e:
            print(f"⚠️ C++ RFT failed: {e}, using Python fallback")
    
    # Python fallback
    basis = get_rft_basis(N)
    return basis.conj().T @ signal

def inverse_true_rft(spectrum: np.ndarray, N: Optional[int] = None) -> np.ndarray:
    """Apply inverse True RFT to a spectrum."""
    if N is None:
        N = len(spectrum)
    
    if CPP_RFT_AVAILABLE and N <= 64:  # Use C++ for reasonable sizes
        try:
            # Create C++ engine
            engine = true_rft_engine_bindings.TrueRFTEngine(N)
            basis = engine.get_rft_basis()
            return basis @ spectrum
        except Exception as e:
            print(f"⚠️ C++ RFT failed: {e}, using Python fallback")
    
    # Python fallback
    basis = get_rft_basis(N)
    return basis @ spectrum

# ---
# Validation utilities
# ---
def validate_true_rft(N: int = 8) -> Dict[str, Any]:
    """Validate unitarity and inversion accuracy of the True RFT."""
    basis = get_rft_basis(N)
    identity_error = np.linalg.norm(basis.conj().T @ basis - np.eye(N))
    
    # Test round-trip accuracy
    test_signal = np.random.default_rng().normal(size=N) + 1j * np.random.default_rng().normal(size=N)
    spectrum = forward_true_rft(test_signal, N)
    reconstructed = inverse_true_rft(spectrum, N)
    reconstruction_error = np.linalg.norm(test_signal - reconstructed)
    
    return {
        'basis_shape': basis.shape,
        'identity_error': identity_error,
        'reconstruction_error': reconstruction_error,
        'passes_unitarity': identity_error < 1e-12,
        'passes_roundtrip': reconstruction_error < 1e-12
    }

# ---
# Self-test
# ---

if __name__ == "__main__":
    print("Canonical True RFT Self-Test")
    print("=" * 40)
    
    params = get_canonical_parameters()
    for k, v in params.items():
        print(f"{k}: {v}")
    
    result = validate_true_rft(8)
    print("\nValidation Results:")
    for k, v in result.items():
        print(f"{k}: {v}")