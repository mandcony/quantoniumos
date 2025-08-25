"""
Canonical True RFT Implementation
================================
This module provides the canonical implementation of the True Resonance Fourier Transform
by importing from true_rft_exact.py.
"""

import numpy as np
from pathlib import Path
import sys

# Add the proper path for importing
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import the true unitary transform implementation
try:
    from true_rft_exact import (
        TrueResonanceFourierTransform,
        forward_true_rft,
        inverse_true_rft,
        get_rft_basis,
        PHI
    )
    print("Successfully imported TrueResonanceFourierTransform from true_rft_exact.py")
except ImportError as e:
    print(f"Error importing from true_rft_exact.py: {e}")
    raise

# For backward compatibility
def generate_phi_sequence(N):
    """Generate the phi sequence for N points using the golden ratio"""
    return np.array([PHI**k % 1 for k in range(N)])

def generate_resonance_kernel(N):
    """Generate the resonance kernel of size N"""
    transformer = TrueResonanceFourierTransform(N=N)
    return transformer.R

    # Modulate with resonance factor
    for k in range(n):
        resonance_factor = np.exp(-1j * 2 * np.pi * k * phi / n)
        symbol_index = k % len(symbols)
        fft_result[k] *= (
            resonance_factor
            * symbols[symbol_index]
            * alpha
            * np.exp(1j * beta * k * theta)
        )

    return fft_result


def inverse_true_rft(transformed, alpha=1.0, beta=0.3, theta=None, symbols=None):
    """
    Apply the inverse True Resonance Fourier Transform.

    Parameters:
    -----------
    transformed : array-like
        The transformed signal
    alpha : float
        Bandwidth parameter
    beta : float
        Gamma coefficient
    theta : float or None
        Phase angle, uses defaults if None
    symbols : array-like or None
        Symbol sequence, uses defaults if None

    Returns:
    --------
    array-like
        The original signal
    """
    # Convert to numpy array
    transformed = np.array(transformed, dtype=np.complex128)

    # Default theta if not provided
    if theta is None:
        theta = np.pi / 4

    # Default symbols if not provided
    if symbols is None:
        symbols = np.array([1, 1j, -1, -1j])  # QPSK symbols

    # Undo resonance modulation
    n = len(transformed)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    # Demodulate with resonance factor
    for k in range(n):
        resonance_factor = np.exp(1j * 2 * np.pi * k * phi / n)
        symbol_index = k % len(symbols)
        transformed[k] /= (
            resonance_factor
            * symbols[symbol_index]
            * alpha
            * np.exp(1j * beta * k * theta)
        )

    # Apply inverse FFT
    return np.fft.ifft(transformed)


def get_rft_basis(size, num_basis=None):
    """
    Generate the RFT basis functions.

    Parameters:
    -----------
    size : int
        Size of the basis functions
    num_basis : int or None
        Number of basis functions, uses size if None

    Returns:
    --------
    array-like
        The basis functions
    """
    if num_basis is None:
        num_basis = size

    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    basis = np.zeros((num_basis, size), dtype=np.complex128)

    for k in range(num_basis):
        for n in range(size):
            basis[k, n] = np.exp(-1j * 2 * np.pi * k * n * phi / size) / np.sqrt(size)

    return basis
