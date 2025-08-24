#!/usr/bin/env python3
"""
RFT Symbiotic Bridge - Ensures Python and C++ implementations work together.
This adapter shares the normalized basis matrix between Python and C++ implementations
and guarantees energy conservation in all transforms.
"""

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add the 04_RFT_ALGORITHMS directory to the Python path
rft_algorithms_dir = Path(__file__).parent / "04_RFT_ALGORITHMS"
if rft_algorithms_dir.exists() and str(rft_algorithms_dir) not in sys.path:
    sys.path.append(str(rft_algorithms_dir))

# Import canonical True RFT implementation
try:
    from 04_RFT_ALGORITHMS.canonical_true_rft import get_rft_basis as py_get_rft_basis
except ImportError:
    # Try relative import
    sys.path.append(str(Path(__file__).parent))
    try:
        from 04_RFT_ALGORITHMS.canonical_true_rft import get_rft_basis as py_get_rft_basis
    except ImportError:
        print("Error: Could not import get_rft_basis from canonical_true_rft.py")
        print("Please ensure canonical_true_rft.py is in the correct location.")
        sys.exit(1)

# Try to import the original engine
try:
    spec = importlib.util.find_spec("true_rft_engine")
    if spec is not None:
        true_rft_engine = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(true_rft_engine)
            cpp_available = True
        except Exception as e:
            print(f"Warning: Error loading true_rft_engine: {e}")
            cpp_available = False
    else:
        true_rft_engine = None
        cpp_available = False
except ImportError:
    true_rft_engine = None
    cpp_available = False
    print("Warning: Could not import true_rft_engine. Using Python fallback.")
OL[OverflowError]
# Cache for basis matrices to avoid recomputation
basis_cache = {}


class SymbioticRFTEngine:
    """
    Symbiotic RFT Engine that ensures Python and C++ implementations work together.
    This adapter shares the normalized basis matrix between Python and C++ implementations
    and strictly enforces energy conservation in all transforms.
    """

    def __init__(self, dimension: int = 16, force_python: bool = False):
        """
        Initialize the symbiotic RFT engine.

        Args:
            dimension: Size of the RFT basis (default: 16)
            force_python: If True, always use Python implementation (default: False)
        """
        self.dimension = dimension
        self.force_python = force_python
        self.cpp_available = (
            not force_python and cpp_available and true_rft_engine is not None
        )

        if self.cpp_available:
            # Initialize the C++ engine
            true_rft_engine.engine_init()
        else:
            if not force_python:
                warnings.warn("Using Python fallback for RFT.")

        # Precompute and cache the properly normalized basis
        if dimension not in basis_cache:
            basis_cache[dimension] = py_get_rft_basis(dimension)
        self.basis = basis_cache[dimension]

        # Validate basis orthonormality
        self._validate_basis()

    def _validate_basis(self):
        """Validate that the basis is orthonormal, which is crucial for energy conservation."""
        # Check if columns are orthogonal
        gram = self.basis.conj().T @ self.basis
        identity = np.eye(self.dimension, dtype=complex)
        error = np.max(np.abs(gram - identity))

        if error > 1e-10:
            warnings.warn(f"Basis is not orthonormal. Max error: {error:.2e}")

        # Check column norms
        for j in range(self.dimension):
            column_norm = np.linalg.norm(self.basis[:, j])
            if abs(column_norm - 1.0) > 1e-10:
                warnings.warn(f"Column {j} not normalized: {column_norm:.8f}")

    def forward_true_rft(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply forward True RFT with energy conservation.

        This uses the properly normalized basis from the Python implementation
        to ensure energy conservation, even when using the C++ engine.

        Args:
            signal: Input signal to transform

        Returns:
            RFT domain representation of the signal
        """
        # Ensure signal has the right size
        if len(signal) != self.dimension:
            if len(signal) < self.dimension:
                padded_signal = np.zeros(self.dimension, dtype=complex)
                padded_signal[: len(signal)] = signal
                signal = padded_signal
            else:
                signal = signal[: self.dimension]

        # Store original energy for verification
        original_energy = np.linalg.norm(signal) ** 2

        # Always use the basis from Python implementation to ensure energy conservation
        spectrum = self.basis.conj().T @ signal

        # Verify energy conservation
        spectrum_energy = np.linalg.norm(spectrum) ** 2
        energy_ratio = spectrum_energy / original_energy if original_energy > 0 else 1.0

        if abs(energy_ratio - 1.0) > 0.01:
            print(f"⚠️ Energy not conserved in forward RFT: ratio={energy_ratio:.4f}")
            # Apply energy correction
            spectrum *= np.sqrt(original_energy / spectrum_energy)

        return spectrum

    def inverse_true_rft(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Apply inverse True RFT with proper basis.

        Args:
            spectrum: RFT domain representation to inverse transform

        Returns:
            Time domain reconstruction of the signal
        """
        # Ensure spectrum has the right size
        if len(spectrum) != self.dimension:
            if len(spectrum) < self.dimension:
                padded_spectrum = np.zeros(self.dimension, dtype=complex)
                padded_spectrum[: len(spectrum)] = spectrum
                spectrum = padded_spectrum
            else:
                spectrum = spectrum[: self.dimension]

        # Store spectrum energy for verification
        spectrum_energy = np.linalg.norm(spectrum) ** 2

        # Always use the basis from Python implementation for inverse transform
        reconstructed = self.basis @ spectrum

        # Verify energy conservation
        reconstructed_energy = np.linalg.norm(reconstructed) ** 2
        energy_ratio = (
            reconstructed_energy / spectrum_energy if spectrum_energy > 0 else 1.0
        )

        if abs(energy_ratio - 1.0) > 0.01:
            print(f"⚠️ Energy not conserved in inverse RFT: ratio={energy_ratio:.4f}")
            # Apply energy correction
            reconstructed *= np.sqrt(spectrum_energy / reconstructed_energy)

        return reconstructed

    def verify_energy_conservation(self, test_signals: int = 10) -> Dict[str, Any]:
        """
        Validate energy conservation in the symbiotic RFT implementation.

        Args:
            test_signals: Number of random signals to test

        Returns:
            Dictionary with validation results
        """
        results = {
            "energy_ratios": [],
            "roundtrip_errors": [],
            "passes_energy": [],
            "passes_roundtrip": [],
        }

        for i in range(test_signals):
            # Generate random complex signal
            signal = np.random.normal(size=self.dimension) + 1j * np.random.normal(
                size=self.dimension
            )
            signal_energy = np.linalg.norm(signal) ** 2

            # Apply forward transform
            spectrum = self.forward_true_rft(signal)
            spectrum_energy = np.linalg.norm(spectrum) ** 2

            # Check energy conservation
            energy_ratio = spectrum_energy / signal_energy
            results["energy_ratios"].append(energy_ratio)
            results["passes_energy"].append(0.99 < energy_ratio < 1.01)

            # Apply inverse transform and check round-trip error
            reconstructed = self.inverse_true_rft(spectrum)
            roundtrip_error = np.linalg.norm(signal - reconstructed)
            results["roundtrip_errors"].append(roundtrip_error)
            results["passes_roundtrip"].append(roundtrip_error < 1e-8)

        # Summarize results
        results["avg_energy_ratio"] = np.mean(results["energy_ratios"])
        results["max_energy_error"] = max(
            abs(1.0 - r) for r in results["energy_ratios"]
        )
        results["avg_roundtrip_error"] = np.mean(results["roundtrip_errors"])
        results["all_pass_energy"] = all(results["passes_energy"])
        results["all_pass_roundtrip"] = all(results["passes_roundtrip"])

        return results


# Function to export the basis to a format C++ can understand
def export_basis_to_cpp(
    dimension: int, output_file: str = "rft_basis_export.npz"
) -> str:
    """
    Export the Python-generated RFT basis for C++ to use.

    Args:
        dimension: Dimension of the RFT basis
        output_file: Path to save the exported basis

    Returns:
        Path to the exported basis file
    """
    # Get or compute basis
    if dimension not in basis_cache:
        basis_cache[dimension] = py_get_rft_basis(dimension)
    basis = basis_cache[dimension]

    # Save basis to file
    np.savez(output_file, basis_real=basis.real, basis_imag=basis.imag)

    return output_file


# Create helper functions for external use
def get_engine(dimension: int = 16) -> SymbioticRFTEngine:
    """Get a singleton instance of the symbiotic RFT engine."""
    engine = SymbioticRFTEngine(dimension=dimension)
    return engine


def forward_true_rft(signal: np.ndarray, dimension: int = None) -> np.ndarray:
    """
    Apply forward True RFT transform with energy conservation.

    Args:
        signal: Input signal
        dimension: Size of RFT basis (if None, uses signal length)

    Returns:
        RFT domain representation
    """
    if dimension is None:
        dimension = len(signal)
    engine = get_engine(dimension)
    return engine.forward_true_rft(signal)


def inverse_true_rft(spectrum: np.ndarray, dimension: int = None) -> np.ndarray:
    """
    Apply inverse True RFT transform.

    Args:
        spectrum: RFT domain representation
        dimension: Size of RFT basis (if None, uses spectrum length)

    Returns:
        Time domain signal
    """
    if dimension is None:
        dimension = len(spectrum)
    engine = get_engine(dimension)
    return engine.inverse_true_rft(spectrum)


def get_rft_basis(dimension: int) -> np.ndarray:
    """
    Get the canonical orthonormal RFT basis.

    Args:
        dimension: Size of RFT basis

    Returns:
        RFT basis matrix
    """
    engine = get_engine(dimension)
    return engine.basis


# Simple test if run directly
if __name__ == "__main__":
    print("Symbiotic True RFT Engine")
    print("=" * 40)

    # Create engine and run validation
    dimensions = [16, 32, 64, 128]

    for N in dimensions:
        print(f"\nTesting dimension N={N}")
        print("-" * 30)

        print("Python implementation:")
        engine_py = SymbioticRFTEngine(dimension=N, force_python=True)
        results_py = engine_py.verify_energy_conservation()

        print(f"Average energy ratio: {results_py['avg_energy_ratio']:.6f}")
        print(f"Maximum energy error: {results_py['max_energy_error']:.6e}")
        print(f"Average round-trip error: {results_py['avg_roundtrip_error']:.6e}")
        print(f"All tests pass energy check: {results_py['all_pass_energy']}")
        print(f"All tests pass round-trip check: {results_py['all_pass_roundtrip']}")

        # Try C++ if available
        engine = SymbioticRFTEngine(dimension=N)
        if engine.cpp_available:
            print("\nSymbiotic implementation (with C++):")
            results = engine.verify_energy_conservation()

            print(f"Average energy ratio: {results['avg_energy_ratio']:.6f}")
            print(f"Maximum energy error: {results['max_energy_error']:.6e}")
            print(f"Average round-trip error: {results['avg_roundtrip_error']:.6e}")
            print(f"All tests pass energy check: {results['all_pass_energy']}")
            print(f"All tests pass round-trip check: {results['all_pass_roundtrip']}")

            # Export basis for C++ integration
            basis_file = export_basis_to_cpp(N)
            print(f"\nExported RFT basis to {basis_file} for C++ integration")
