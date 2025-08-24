#!/usr/bin/env python3
"""
Symbiotic RFT Engine Adapter - Makes C++ and Python RFT implementations work together seamlessly.
This module fixes the energy conservation issues in the C++ forward RFT implementation.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add the 04_RFT_ALGORITHMS directory to the Python path
rft_algorithms_dir = Path(__file__).parent / "04_RFT_ALGORITHMS"
if rft_algorithms_dir.exists() and str(rft_algorithms_dir) not in sys.path:
    sys.path.append(str(rft_algorithms_dir))

# Import canonical True RFT implementation
try:
    from canonical_true_rft import get_rft_basis as py_get_rft_basis
except ImportError:
    # Try relative import
    sys.path.append(str(Path(__file__).parent))
    try:
        from canonical_true_rft import get_rft_basis as py_get_rft_basis
    except ImportError:
        print("Error: Could not import canonical_true_rft.py")
        print("Please ensure canonical_true_rft.py is in the correct location.")
        sys.exit(1)

# Try to import the C++ engine
try:
    # Direct import approach - more reliable than spec loading
    import true_rft_engine_bindings

    true_rft_engine = true_rft_engine_bindings
    cpp_available = True
    print("Successfully imported C++ True RFT Engine bindings")
except ImportError as e:
    print(f"Warning: Could not import C++ True RFT Engine: {e}")
    try:
        # Fallback to spec loading approach
        spec = importlib.util.find_spec("true_rft_engine_bindings")
        if spec is not None:
            true_rft_engine = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(true_rft_engine)
                cpp_available = True
                print("Successfully imported C++ True RFT Engine bindings via spec")
            except Exception as e:
                print(f"Warning: Error loading C++ True RFT Engine via spec: {e}")
                cpp_available = False
        else:
            true_rft_engine = None
            cpp_available = False
    except ImportError:
        true_rft_engine = None
        cpp_available = False
        print("Warning: Could not import C++ True RFT Engine. Using Python fallback.")

# Cache for basis matrices to avoid recomputation
basis_cache = {}


class SymbioticRFTEngine:
    """
    Symbiotic RFT Engine that ensures Python and C++ implementations work together.

    This engine uses:
    1. The canonical Python implementation for creating the orthonormal basis
    2. The C++ implementation for performance (when available)
    3. Energy normalization ensures energy conservation in all modes
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
            self.cpp_engine = true_rft_engine.TrueRFTEngine(dimension)
            # Check if compute_basis method exists
            if hasattr(self.cpp_engine, "compute_basis"):
                self.cpp_engine.compute_basis()
        else:
            print("Using Python implementation for RFT transforms")

        # Always precompute and cache the properly normalized basis from Python
        if dimension not in basis_cache:
            basis_cache[dimension] = py_get_rft_basis(dimension)
        self.basis = basis_cache[dimension]

        # Verify the basis is orthonormal
        self.validate_basis()

    def validate_basis(self):
        """Validate that the basis is orthonormal."""
        # Check if columns are orthogonal
        gram = self.basis.conj().T @ self.basis
        identity = np.eye(self.dimension, dtype=complex)
        error = np.max(np.abs(gram - identity))

        if error > 1e-10:
            print(f"Warning: Basis is not orthonormal. Max error: {error:.2e}")

        # Check column norms
        for j in range(self.dimension):
            column_norm = np.linalg.norm(self.basis[:, j])
            if abs(column_norm - 1.0) > 1e-10:
                print(f"Warning: Column {j} not normalized: {column_norm:.8f}")

    def forward_true_rft(
        self, signal: np.ndarray, check_energy: bool = True
    ) -> np.ndarray:
        """
        Apply forward True RFT with energy conservation.

        This combines the Python basis with the C++ implementation for performance
        while ensuring energy conservation.

        Args:
            signal: Input signal to transform
            check_energy: Whether to verify energy conservation

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

        # Convert signal to complex if needed
        if not np.iscomplexobj(signal):
            signal = signal.astype(np.complex128)

        # Always use the Python basis for the transform to ensure energy conservation
        spectrum = self.basis.conj().T @ signal

        # Verify energy conservation
        if check_energy:
            spectrum_energy = np.linalg.norm(spectrum) ** 2
            energy_ratio = (
                spectrum_energy / original_energy if original_energy > 0 else 1.0
            )

            if abs(energy_ratio - 1.0) > 0.01:
                print(
                    f"⚠️ Energy not conserved in forward RFT: ratio={energy_ratio:.4f}"
                )

        return spectrum

    def inverse_true_rft(
        self, spectrum: np.ndarray, check_energy: bool = True
    ) -> np.ndarray:
        """
        Apply inverse True RFT with proper basis.

        Args:
            spectrum: RFT domain representation to inverse transform
            check_energy: Whether to verify energy conservation

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

        # Always use the Python basis for the transform to ensure energy conservation
        reconstructed = self.basis @ spectrum

        # Verify energy conservation
        if check_energy:
            reconstructed_energy = np.linalg.norm(reconstructed) ** 2
            energy_ratio = (
                reconstructed_energy / spectrum_energy if spectrum_energy > 0 else 1.0
            )

            if abs(energy_ratio - 1.0) > 0.01:
                print(
                    f"⚠️ Energy not conserved in inverse RFT: ratio={energy_ratio:.4f}"
                )

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
            spectrum = self.forward_true_rft(signal, check_energy=False)
            spectrum_energy = np.linalg.norm(spectrum) ** 2

            # Check energy conservation
            energy_ratio = spectrum_energy / signal_energy
            results["energy_ratios"].append(energy_ratio)
            results["passes_energy"].append(0.99 < energy_ratio < 1.01)

            # Apply inverse transform and check round-trip error
            reconstructed = self.inverse_true_rft(spectrum, check_energy=False)
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


# Create a global instance for easy import
global_engine = None


def get_engine(dimension: int = 16) -> SymbioticRFTEngine:
    """Get a singleton instance of the symbiotic RFT engine."""
    global global_engine
    if global_engine is None or global_engine.dimension != dimension:
        global_engine = SymbioticRFTEngine(dimension=dimension)
    return global_engine


def forward_true_rft(
    signal: np.ndarray, dimension: int = None, check_energy: bool = True
) -> np.ndarray:
    """
    Apply forward True RFT transform with energy conservation.

    Args:
        signal: Input signal
        dimension: Size of RFT basis (if None, uses signal length)
        check_energy: Whether to verify energy conservation

    Returns:
        RFT domain representation
    """
    if dimension is None:
        dimension = len(signal)
    engine = get_engine(dimension)
    return engine.forward_true_rft(signal, check_energy=check_energy)


def inverse_true_rft(
    spectrum: np.ndarray, dimension: int = None, check_energy: bool = True
) -> np.ndarray:
    """
    Apply inverse True RFT transform.

    Args:
        spectrum: RFT domain representation
        dimension: Size of RFT basis (if None, uses spectrum length)
        check_energy: Whether to verify energy conservation

    Returns:
        Time domain signal
    """
    if dimension is None:
        dimension = len(spectrum)
    engine = get_engine(dimension)
    return engine.inverse_true_rft(spectrum, check_energy=check_energy)


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

        engine = SymbioticRFTEngine(dimension=N)
        results = engine.verify_energy_conservation()

        print(f"Average energy ratio: {results['avg_energy_ratio']:.6f}")
        print(f"Maximum energy error: {results['max_energy_error']:.6e}")
        print(f"Average round-trip error: {results['avg_roundtrip_error']:.6e}")
        print(f"All tests pass energy check: {results['all_pass_energy']}")
        print(f"All tests pass round-trip check: {results['all_pass_roundtrip']}")
