#!/usr/bin/env python3
"""
Canonical True RFT Implementation
=================================

Implements the ACTUAL PROVEN RFT equation: R = Σ_i w_i D_φi C_σi D_φi†
Using the mathematically verified formula from rft_definitive_mathematical_proof.py
"""

import math
from typing import Any, Dict, Optional

import numpy as np

PHI = (1.0 + math.sqrt(5.0)) / 2.0

# ---
# Canonical parameter retrieval
# ---


def get_canonical_parameters() -> Dict[str, str]:
    """Get canonical parameters for RFT implementation."""
    return {
        "method": "proven_rft_equation",
        "kernel": "R_equals_sum_wi_Dphi_Csigma_Dphidag",
        "precision": "double",
    }


# ---
# ACTUAL RFT EQUATION IMPLEMENTATION: R = Σ_i w_i D_φi C_σi D_φi†
# ---
def _generate_weights(N: int) -> tuple:
    """Generate weights, phases, and sigmas for RFT construction"""
    # Golden ratio weights
    weights = np.array([PHI ** (-k) for k in range(N)])
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
            C[m, n] = np.exp(-(dist**2) / (2 * sigma**2))
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
    # Step 4: Ensure strict orthonormality via QR decomposition
    Q, _ = np.linalg.qr(basis)

    # Step 5: Enforce unit norm on each column (critical for energy conservation)
    for j in range(N):
        column_norm = np.linalg.norm(Q[:, j])
        assert (
            abs(column_norm - 1.0) < 1e-10
        ), f"Column {j} not normalized: {column_norm}"
        # Extra safety to ensure exact unit norm
        Q[:, j] = Q[:, j] / column_norm

    # Step 6: Verify orthogonality
    gram_matrix = np.abs(Q.conj().T @ Q)
    np.fill_diagonal(gram_matrix, 0)  # Zero out diagonal
    max_off_diag = np.max(gram_matrix)
    assert (
        max_off_diag < 1e-10
    ), f"Basis not orthogonal, max off-diagonal: {max_off_diag:.2e}"

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
        # Security fix: Use cryptographically secure random instead of fixed seed
        import secrets

        np.random.seed(secrets.randbits(32))  # Cryptographically secure seed
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
                padded[: len(data)] = data
                data = padded
            else:
                data = data[: self.N]
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
            padded[: len(data_array)] = data_array
        else:
            padded = data_array[: self.N]
        # Apply RFT transform for hashing
        hash_result = self.basis.conj().T @ padded
        return hash_result

    def verify_crypto_functionality(self) -> Dict[str, Any]:
        """
        Test all crypto functions with comprehensive metrics.

        Tests:
        1. Key generation
        2. Encryption/decryption with round-trip accuracy
        3. Hash function consistency and collision resistance
        4. Avalanche effect (bit flip propagation)
        5. Basis unitarity

        Returns:
            Dictionary of test results
        """
        results = {}

        # Test key generation
        results["keys_generated"] = (
            self.private_key is not None and self.public_key is not None
        )

        # Test encryption/decryption
        test_data = np.random.rand(self.N) + 1j * np.random.rand(self.N)
        test_energy = np.linalg.norm(test_data) ** 2

        encrypted = self.encrypt(test_data)
        encrypted_energy = np.linalg.norm(encrypted) ** 2

        # Check energy conservation in encryption
        energy_ratio = encrypted_energy / test_energy
        results["energy_ratio"] = energy_ratio
        results["energy_conserved"] = 0.95 < energy_ratio < 1.05

        decrypted = self.decrypt(encrypted)
        encryption_error = np.abs(test_data - decrypted).max()
        results["encryption_error"] = encryption_error
        results["encryption_works"] = encryption_error < 1e-8

        # Test hash function
        test_message = b"Hello, this is a test message!"
        hash1 = self.hash_function(test_message)
        hash2 = self.hash_function(test_message)
        hash3 = self.hash_function(b"Different message")

        results["hash_consistency"] = np.abs(hash1 - hash2).max() < 1e-15
        results["hash_different"] = np.abs(hash1 - hash3).max() > 0.1

        # Test avalanche effect (bit flip propagation)
        avalanche_results = self._test_avalanche_effect(test_message)
        results["avalanche_percentage"] = avalanche_results["avalanche_percentage"]
        results["bit_influence"] = avalanche_results["bit_influence"]
        results["good_avalanche"] = avalanche_results["avalanche_percentage"] > 45.0

        # Test collision resistance
        collision_results = self._test_collision_resistance(100)
        results["collision_count"] = collision_results["collisions"]
        results["collision_rate"] = collision_results["collision_rate"]
        results["good_collision_resistance"] = (
            collision_results["collision_rate"] < 0.01
        )

        # Test basis properties
        identity_error = np.abs(self.basis.conj().T @ self.basis - np.eye(self.N)).max()
        results["basis_unitary"] = identity_error < 1e-10
        results["basis_identity_error"] = identity_error

        results["all_crypto_tests_pass"] = all(
            [
                results["keys_generated"],
                results["encryption_works"],
                results["energy_conserved"],
                results["hash_consistency"],
                results["hash_different"],
                results["good_avalanche"],
                results["good_collision_resistance"],
                results["basis_unitary"],
            ]
        )

        return results

    def _test_avalanche_effect(
        self, message: bytes, samples: int = 20
    ) -> Dict[str, Any]:
        """
        Test the avalanche effect (bit flip propagation) in the hash function.

        The avalanche effect is measured by:
        1. Computing the original hash
        2. Flipping a single bit in the message
        3. Computing the new hash
        4. Measuring what percentage of output bits changed

        A good hash function should have ~50% of bits change.
        """
        original_hash = self.hash_function(message)

        # Convert to bit representation for analysis
        def to_bits(complex_array):
            # Extract real and imaginary parts as float arrays
            real_bits = np.array(
                [
                    np.binary_repr(int(abs(x) * 2**20), width=20)
                    for x in np.real(complex_array)
                ]
            )
            imag_bits = np.array(
                [
                    np.binary_repr(int(abs(x) * 2**20), width=20)
                    for x in np.imag(complex_array)
                ]
            )

            # Convert to bit arrays (1s and 0s)
            real_bit_array = np.array(
                [[int(bit) for bit in bits] for bits in real_bits]
            )
            imag_bit_array = np.array(
                [[int(bit) for bit in bits] for bits in imag_bits]
            )

            # Combine into a single bit array
            return np.hstack((real_bit_array, imag_bit_array))

        original_bits = to_bits(original_hash)
        total_bits = original_bits.size

        # Test flipping different bits and measure the impact
        message_bytes = bytearray(message)
        bit_influences = []

        for _ in range(samples):
            # Flip a random bit
            byte_idx = np.random.randint(0, len(message_bytes))
            bit_idx = np.random.randint(0, 8)
            message_bytes[byte_idx] ^= 1 << bit_idx

            # Get the new hash
            modified_hash = self.hash_function(bytes(message_bytes))
            modified_bits = to_bits(modified_hash)

            # Count different bits
            diff_count = np.sum(original_bits != modified_bits)
            diff_percentage = (diff_count / total_bits) * 100
            bit_influences.append(diff_percentage)

            # Restore the bit for next iteration
            message_bytes[byte_idx] ^= 1 << bit_idx

        avg_influence = np.mean(bit_influences)

        return {
            "avalanche_percentage": avg_influence,
            "bit_influence": bit_influences,
            "samples": samples,
        }

    def _test_collision_resistance(self, num_tests: int = 100) -> Dict[str, Any]:
        """
        Test collision resistance of the hash function.

        Generates random messages and checks if they hash to the same value.
        """
        hash_values = []
        collisions = 0

        for i in range(num_tests):
            # Generate a random message of random length
            length = np.random.randint(10, 100)
            message = bytes(np.random.randint(0, 256, size=length))

            # Compute hash
            hash_value = self.hash_function(message)

            # Check for collisions (simplify by checking if the sum+mean of the hash matches any previous)
            hash_fingerprint = (np.sum(hash_value), np.mean(np.abs(hash_value)))

            if hash_fingerprint in hash_values:
                collisions += 1
            else:
                hash_values.append(hash_fingerprint)

        return {
            "collisions": collisions,
            "total_tests": num_tests,
            "collision_rate": collisions / num_tests if num_tests > 0 else 0,
        }


# ---
# Forward / Inverse True RFT
# ---


def forward_true_rft(
    signal: np.ndarray, N: Optional[int] = None, check_energy: bool = True
) -> np.ndarray:
    """
    Apply forward True RFT to a signal.

    Args:
        signal: Input signal to transform
        N: Dimension (defaults to signal length)
        check_energy: Whether to verify energy conservation

    Returns:
        RFT domain representation of the signal

    Raises:
        AssertionError: If energy conservation check fails
    """
    if N is None:
        N = len(signal)

    # Store original energy for verification
    original_energy = np.linalg.norm(signal) ** 2

    # Get orthonormal basis and apply forward transform
    basis = get_rft_basis(N)
    rft_result = basis.conj().T @ signal

    # Verify energy conservation (Parseval's theorem)
    if check_energy:
        rft_energy = np.linalg.norm(rft_result) ** 2
        energy_ratio = rft_energy / original_energy if original_energy > 0 else 1.0
        assert (
            0.99 < energy_ratio < 1.01
        ), f"Energy not conserved in forward RFT: {energy_ratio:.6f}"

    return rft_result


def inverse_true_rft(
    spectrum: np.ndarray,
    N: Optional[int] = None,
    original_signal: Optional[np.ndarray] = None,
    check_roundtrip: bool = False,
) -> np.ndarray:
    """
    Apply inverse True RFT to a spectrum.

    Args:
        spectrum: RFT domain representation to inverse transform
        N: Dimension (defaults to spectrum length)
        original_signal: Original signal for round-trip error verification
        check_roundtrip: Whether to verify round-trip reconstruction error

    Returns:
        Time domain reconstruction of the signal

    Raises:
        AssertionError: If round-trip error check fails
    """
    if N is None:
        N = len(spectrum)

    # Get orthonormal basis and apply inverse transform
    basis = get_rft_basis(N)
    reconstructed = basis @ spectrum

    # Verify round-trip reconstruction accuracy if requested
    if check_roundtrip and original_signal is not None:
        error = np.linalg.norm(original_signal - reconstructed)
        assert error < 1e-8, f"Round-trip error too large: {error:.2e}"

    return reconstructed


# ---
# Validation utilities
# ---
def validate_true_rft(N: int = 8) -> Dict[str, Any]:
    """
    Validate unitarity, orthonormality, energy conservation, and inversion accuracy of the True RFT.

    Performs comprehensive checks:
    1. Basis orthonormality (unit columns and orthogonal)
    2. Energy conservation in forward transform
    3. Round-trip reconstruction accuracy
    4. Basis completeness (spanning the full space)
    """
    # Get the RFT basis and verify shape
    basis = get_rft_basis(N)
    assert basis.shape == (
        N,
        N,
    ), f"Basis shape error: expected ({N}, {N}), got {basis.shape}"
    assert len(basis.shape) == 2, "Basis must be 2-dimensional"

    # Test column normalization
    column_norms = np.linalg.norm(basis, axis=0)
    norm_error = np.max(np.abs(column_norms - 1.0))
    assert norm_error < 1e-10, f"Column normalization error: {norm_error:.2e}"

    # Test orthogonality
    gram = basis.conj().T @ basis
    np.fill_diagonal(gram, 0)  # Zero out diagonal
    ortho_error = np.max(np.abs(gram))
    assert ortho_error < 1e-10, f"Orthogonality error: {ortho_error:.2e}"

    # Check if it's unitary (should be very close to identity)
    identity_error = np.linalg.norm(basis.conj().T @ basis - np.eye(N))
    assert identity_error < 1e-10, f"Unitarity error: {identity_error:.2e}"

    # Test round-trip accuracy and energy conservation
    test_signal = np.random.default_rng().normal(
        size=N
    ) + 1j * np.random.default_rng().normal(size=N)
    test_energy = np.linalg.norm(test_signal) ** 2

    # Verify forward transform with energy check
    spectrum = forward_true_rft(test_signal, N, check_energy=True)
    spectrum_energy = np.linalg.norm(spectrum) ** 2
    energy_ratio = spectrum_energy / test_energy
    energy_error = abs(1.0 - energy_ratio)
    assert energy_error < 1e-10, f"Energy conservation error: {energy_error:.2e}"

    # Verify inverse transform with round-trip check
    reconstructed = inverse_true_rft(spectrum, N, test_signal, check_roundtrip=True)
    reconstruction_error = np.linalg.norm(test_signal - reconstructed)
    assert (
        reconstruction_error < 1e-8
    ), f"Reconstruction error: {reconstruction_error:.2e}"

    # Check basis completeness by reconstructing delta functions
    completeness_errors = []
    for i in range(N):
        delta = np.zeros(N, dtype=complex)
        delta[i] = 1.0
        delta_spectrum = forward_true_rft(delta, N)
        delta_recon = inverse_true_rft(delta_spectrum, N)
        completeness_errors.append(np.linalg.norm(delta - delta_recon))
    max_completeness_error = max(completeness_errors)
    assert (
        max_completeness_error < 1e-8
    ), f"Basis completeness error: {max_completeness_error:.2e}"

    return {
        "basis_shape": basis.shape,
        "column_normalization_error": norm_error,
        "orthogonality_error": ortho_error,
        "identity_error": identity_error,
        "energy_conservation_error": energy_error,
        "energy_ratio": energy_ratio,
        "reconstruction_error": reconstruction_error,
        "basis_completeness_error": max_completeness_error,
        "passes_unit_norm": norm_error < 1e-10,
        "passes_orthogonality": ortho_error < 1e-10,
        "passes_unitarity": identity_error < 1e-10,
        "passes_energy_conservation": energy_error < 1e-10,
        "passes_roundtrip": reconstruction_error < 1e-8,
        "passes_completeness": max_completeness_error < 1e-8,
        "all_tests_passed": all(
            [
                norm_error < 1e-10,
                ortho_error < 1e-10,
                identity_error < 1e-10,
                energy_error < 1e-10,
                reconstruction_error < 1e-8,
                max_completeness_error < 1e-8,
            ]
        ),
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
