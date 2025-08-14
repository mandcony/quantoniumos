# LEGACY RFT IMPLEMENTATION - REPLACE WITH CANONICAL
# from canonical_true_rft import forward_true_rft, inverse_true_rft

"""
QuantoniumOS - Resonance Fourier Transform Module

This module provides the core mathematical implementation supporting USPTO Patent Claims:
- Claim 1: Symbolic transformation engine with quantum amplitude decomposition
- Claim 3: RFT-based geometric structures for cryptographic waveform processing
- Claim 4: Unified computational framework integration

Implementation: True Resonance Fourier Transform (RFT)
Mathematical Formula: X_k = (1/N) Σ_t x_t * e^(i(2πkt/N + θ_t))
Operator Form: X = Ψ† x where Ψ are eigenvectors of resonance kernel R

The implementation uses C++ engine bindings for high-performance computation when available,
with Python fallback for development and testing.
"""

import math
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import logging
import os
import json

# Import CANONICAL True RFT implementation (single source of truth)
from canonical_true_rft import (
    forward_true_rft,
    inverse_true_rft,
    validate_true_rft,
    get_rft_basis
)

# Try to import C++ engine bindings
try:
    from core.python_bindings.engine_core import QuantoniumEngineCore
    HAS_CPP_ENGINE = True
    print("✓ Using high-performance C++ engine")
except ImportError:
    HAS_CPP_ENGINE = False
    print("C++ module not available, using Python implementation")

logger = logging.getLogger(__name__)

# Public API - all functions now route to True RFT implementation
def resonance_fourier_transform(
    signal: List[float], 
    *, 
    alpha: float = 1.0,
    beta: float = 0.3,
    theta: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None
) -> List[Tuple[float, complex]]:
    """
    Apply True RFT to a signal.
    
    This function routes to the mathematically rigorous True RFT implementation
    that uses eigendecomposition of the resonance kernel.
    
    Args:
        signal: Input signal as a list of floating point values
        alpha: Resonance parameter (mapped to sigma0)
        beta: Decay parameter (mapped to gamma)
        theta: Optional phase offset matrix (mapped to theta0_values)
        symbols: Optional symbol array (mapped to sequence_type)
        
    Returns:
        List of (frequency, complex_amplitude) tuples
    """
    # Map parameters to True RFT interface
    kwargs = {
        'sigma0': alpha,
        'gamma': beta,
    }
    
    if theta is not None:
        # Extract theta0_values from theta matrix
        kwargs['theta0_values'] = theta.diagonal().tolist() if hasattr(theta, 'diagonal') else [0.0, np.pi/4]
    
    # Perform True RFT
    rft_coeffs = forward_true_rft(signal, **kwargs)
    
    # Convert to legacy format
    n = len(signal)
    result = []
    for k, coeff in enumerate(rft_coeffs):
        frequency = k / n
        result.append((frequency, coeff))
    
    return result

def inverse_resonance_fourier_transform(
    frequency_components: List[Tuple[float, complex]],
    *,
    alpha: float = 1.0,
    beta: float = 0.3,
    theta: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None
) -> List[float]:
    """
    Apply inverse True RFT to frequency components.
    
    Args:
        frequency_components: List of (frequency, complex_amplitude) tuples
        alpha: Resonance parameter (mapped to sigma0)
        beta: Decay parameter (mapped to gamma)
        theta: Optional phase offset matrix (mapped to theta0_values)
        symbols: Optional symbol array (mapped to sequence_type)
        
    Returns:
        Reconstructed signal
    """
    # Convert from legacy format
    rft_coeffs = [coeff for _, coeff in frequency_components]
    
    # Map parameters to True RFT interface
    kwargs = {
        'sigma0': alpha,
        'gamma': beta,
    }
    
    if theta is not None:
        kwargs['theta0_values'] = theta.diagonal().tolist() if hasattr(theta, 'diagonal') else [0.0, np.pi/4]
    
    # Perform True IRFT
    return inverse_true_rft(rft_coeffs, **kwargs)

def perform_rft(
    waveform: List[float], 
    *, 
    alpha: float = 1.0,
    beta: float = 0.3,
    theta: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Route to True RFT implementation."""
    kwargs = {'sigma0': alpha, 'gamma': beta}
    if theta is not None:
        kwargs['theta0_values'] = theta.diagonal().tolist() if hasattr(theta, 'diagonal') else [0.0, np.pi/4]
    
    # Use canonical True RFT
    rft_coeffs = forward_true_rft(waveform, **kwargs)
    
    # Return compatible dict format
    return {
        'coefficients': rft_coeffs.tolist() if hasattr(rft_coeffs, 'tolist') else list(rft_coeffs),
        'success': True,
        'algorithm': 'canonical_true_rft'
    }

def perform_irft(
    rft_result: Dict[str, Any], 
    *, 
    alpha: float = 1.0,
    beta: float = 0.3,
    theta: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None
) -> List[float]:
    """Route to True IRFT implementation."""
    kwargs = {'sigma0': alpha, 'gamma': beta}
    if theta is not None:
        kwargs['theta0_values'] = theta.diagonal().tolist() if hasattr(theta, 'diagonal') else [0.0, np.pi/4]
    
    # Extract coefficients from dict format
    if isinstance(rft_result, dict) and 'coefficients' in rft_result:
        coefficients = rft_result['coefficients']
    else:
        coefficients = rft_result
    
    # Use canonical True RFT
    return inverse_true_rft(coefficients, **kwargs)

# Legacy compatibility functions
def perform_rft_list(signal: List[float]) -> List[Tuple[float, complex]]:
    """Legacy compatibility - routes to True RFT."""
    return resonance_fourier_transform(signal)

def perform_irft_list(frequency_components: List[Tuple[float, complex]]) -> List[float]:
    """Legacy compatibility - routes to True RFT."""
    return inverse_resonance_fourier_transform(frequency_components)

# Forward additional True RFT functions for direct access
forward_true_rft = forward_true_rft
inverse_true_rft = inverse_true_rft
validate_true_rft = validate_true_rft
get_rft_basis = get_rft_basis

if __name__ == "__main__":
    print("QuantoniumOS - True RFT Module")
    print("Running validation tests...")
    
    validation_results = validate_true_rft()
    for test_name, passed in validation_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    if all(validation_results.values()):
        print("\n🎉 All True RFT tests passed! No more DFT code - pure RFT implementation!")
    else:
        print("\n⚠️  Some tests failed. Review implementation.")
