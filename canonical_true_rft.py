"""
Canonical True RFT Implementation - Single Source of Truth

This module provides the definitive implementation of the True Resonance Fourier Transform
as defined in the QuantoniumOS specification:

Mathematical Definition:
    X = Ψ† x    (forward transform)
    x = Ψ X      (inverse transform)
    
Where Ψ are eigenvectors of the resonance kernel:
    R = Σᵢ wᵢ D_φᵢ C_σᵢ D_φᵢ†

This is the ONLY implementation that should be used for production RFT operations.
All other "RFT" implementations in this codebase are legacy/compatibility layers.

Author: QuantoniumOS Team
Version: 1.0.0 (Canonical)
"""

import math
import numpy as np
from typing import List, Optional, Tuple
import warnings

# Golden ratio with maximum precision
PHI = (1.0 + math.sqrt(5.0)) / 2.0

# Production parameters (canonical)
CANONICAL_WEIGHTS = [0.7, 0.3]
CANONICAL_THETA0 = [0.0, math.pi / 4.0]  
CANONICAL_OMEGA = [1.0, PHI]
CANONICAL_SIGMA0 = 1.0
CANONICAL_GAMMA = 0.3


def generate_phi_sequence(N: int, theta0: float, omega: float) -> np.ndarray:
    """
    Generate phase sequence φᵢ(k) = e^{j(θ₀ + ωk)} for kernel construction.
    """
    k = np.arange(N, dtype=float)
    phases = theta0 + omega * k
    return np.exp(1j * phases)


def generate_gaussian_kernel(N: int, component_idx: int, sigma0: float, gamma: float) -> np.ndarray:
    """
    Generate Gaussian correlation kernel Cσᵢ for component i.
    """
    C = np.zeros((N, N), dtype=complex)
    sigma = sigma0 * math.exp(-gamma * component_idx)
    
    for k in range(N):
        for n in range(N):
            diff = (k - n) % N
            if diff > N // 2:
                diff = N - diff
            C[k, n] = math.exp(-0.5 * (diff / sigma) ** 2)
    
    return C


def generate_resonance_kernel(
    N: int,
    weights: Optional[List[float]] = None,
    theta0_values: Optional[List[float]] = None,
    omega_values: Optional[List[float]] = None,
    sigma0: float = CANONICAL_SIGMA0,
    gamma: float = CANONICAL_GAMMA
) -> np.ndarray:
    """
    Generate resonance kernel R = Σᵢ wᵢ D_φᵢ C_σᵢ D_φᵢ†
    
    This is the core mathematical construction of the True RFT.
    """
    # Use canonical parameters if not provided
    if weights is None:
        weights = CANONICAL_WEIGHTS
    if theta0_values is None:
        theta0_values = CANONICAL_THETA0
    if omega_values is None:
        omega_values = CANONICAL_OMEGA
    
    # Ensure consistent lengths
    M = min(len(weights), len(theta0_values), len(omega_values))
    if M == 0:
        raise ValueError("At least one resonance component must be specified")
    
    weights = weights[:M]
    theta0_values = theta0_values[:M]
    omega_values = omega_values[:M]
    
    # Initialize kernel
    R = np.zeros((N, N), dtype=complex)
    
    # Accumulate weighted components
    for i in range(M):
        w = weights[i]
        if abs(w) < 1e-15:
            continue
            
        theta0 = theta0_values[i]
        omega = omega_values[i]
        
        # Generate phase sequence φᵢ
        phi = generate_phi_sequence(N, theta0, omega)
        
        # Generate Gaussian kernel Cσᵢ
        C = generate_gaussian_kernel(N, i, sigma0, gamma)
        
        # Accumulate R += wᵢ D_φᵢ C_σᵢ D_φᵢ†
        for k in range(N):
            for n in range(N):
                phase_factor = phi[k] * np.conj(phi[n])
                R[k, n] += w * phase_factor * C[k, n]
    
    return R


def forward_true_rft(x):
    """
    Forward True RFT transform: X = Ψ† x
    
    Args:
        x: Input sequence (complex or real)
        
    Returns:
        X: RFT coefficients
    """
    # Suppress complex casting warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        x = np.asarray(x, dtype=np.complex128)
        n = len(x)
        
        psi_matrix = get_rft_basis(n)
        X = psi_matrix.T.conj() @ x
        
        return X


def inverse_true_rft(
    rft_data: List[complex],
    weights: Optional[List[float]] = None,
    theta0_values: Optional[List[float]] = None,
    omega_values: Optional[List[float]] = None,
    sigma0: float = CANONICAL_SIGMA0,
    gamma: float = CANONICAL_GAMMA
) -> List[float]:
    """
    Canonical True RFT Inverse Transform: x = Ψ X
    
    This function implements the exact mathematical definition from the specification.
    Do not modify without updating the canonical definition.
    """
    X = np.asarray(rft_data, dtype=complex)
    N = len(X)
    
    if N == 0:
        return []
    
    # Generate same resonance kernel as forward transform
    R = generate_resonance_kernel(N, weights, theta0_values, omega_values, sigma0, gamma)
    
    # Eigendecomposition: R = Ψ Λ Ψ†
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=np.ComplexWarning)
        eigenvalues, eigenvectors = np.linalg.eigh(R)
    
    # Inverse transform: x = Ψ X
    x = eigenvectors @ X
    
    return [float(v.real) for v in x]


def validate_true_rft(
    test_signal: Optional[List[float]] = None,
    tolerance: float = 1e-12
) -> Tuple[bool, float]:
    """
    Validate that the True RFT implementation is unitary (exact reconstruction).
    
    Returns:
        (is_valid, reconstruction_error)
    """
    if test_signal is None:
        # Default test signal
        test_signal = [1.0, 0.5, 0.2, 0.8, 0.3]
    
    # Forward transform
    X = forward_true_rft(test_signal)
    
    # Inverse transform
    reconstructed = inverse_true_rft(X)
    
    # Compute reconstruction error
    error = np.linalg.norm(np.array(test_signal) - np.array(reconstructed))
    
    return error < tolerance, error


def get_canonical_parameters() -> dict:
    """
    Return the canonical parameters used in the True RFT.
    These are the exact values used in publication results.
    """
    return {
        'weights': CANONICAL_WEIGHTS.copy(),
        'theta0_values': CANONICAL_THETA0.copy(),
        'omega_values': CANONICAL_OMEGA.copy(),
        'sigma0': CANONICAL_SIGMA0,
        'gamma': CANONICAL_GAMMA,
        'phi_exact': PHI
    }


def get_rft_basis(N: int) -> np.ndarray:
    """
    Get the RFT basis matrix Ψ for a given size N using canonical parameters.
    
    Returns:
        N×N complex matrix where columns are eigenvectors of resonance kernel
    """
    R = generate_resonance_kernel(N)
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    return eigenvectors


# Self-validation on import
if __name__ == "__main__":
    print("Canonical True RFT Implementation")
    print("=" * 40)
    
    # Validate unitary property
    is_valid, error = validate_true_rft()
    print(f"Unitary validation: {'✓ PASS' if is_valid else '❌ FAIL'}")
    print(f"Reconstruction error: {error:.2e}")
    
    # Show canonical parameters
    params = get_canonical_parameters()
    print(f"Canonical parameters:")
    print(f"  Weights: {params['weights']}")
    print(f"  Phases: {[f'{x:.4f}' for x in params['theta0_values']]}")
    print(f"  Steps: {[f'{x:.4f}' for x in params['omega_values']]}")
    print(f"  φ (golden ratio): {params['phi_exact']:.10f}")
    
    # Small non-equivalence test
    psi_4 = get_rft_basis(4)
    dft_4 = np.fft.fft(np.eye(4)) / 2.0
    diff = np.linalg.norm(psi_4 - dft_4)
    print(f"Non-equivalence (N=4): ||Ψ - F|| = {diff:.4f} {'✓ NON-EQUIVALENT' if diff > 1e-3 else '❌ EQUIVALENT'}")
