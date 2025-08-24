"""
Canonical True RFT Implementation
================================
This module provides the canonical implementation of the True Resonance Fourier Transform.
"""

import numpy as np


def forward_true_rft(signal, alpha=1.0, beta=0.3, theta=None, symbols=None):
    """
    Apply the True Resonance Fourier Transform to a signal.

    Parameters:
    -----------
    signal : array-like
        The input signal
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
        The transformed signal
    """
    # Convert to numpy array
    signal = np.array(signal, dtype=np.complex128)

    # Default theta if not provided
    if theta is None:
        theta = np.pi / 4

    # Default symbols if not provided
    if symbols is None:
        symbols = np.array([1, 1j, -1, -1j])  # QPSK symbols

    # Apply FFT
    fft_result = np.fft.fft(signal)

    # Apply resonance modulation
    n = len(signal)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

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
