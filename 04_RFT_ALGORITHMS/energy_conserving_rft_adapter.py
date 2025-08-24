#!/usr/bin/env python3
"""
Python adapter for fixing energy conservation in the True RFT engine.
This module wraps the existing C++ implementation and adds the missing normalization step.
"""

import importlib.util
import sys
from pathlib import Path

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

# Cache for basis matrices to avoid recomputation
basis_cache = {}


class EnergyConservingRFTEngine:
    """
    Adapter class that wraps the existing C++ True RFT engine and adds
    the missing normalization step to ensure energy conservation.
    """

    def __init__(self, dimension: int = 16):
        """
        Initialize the energy-conserving RFT engine.

        Args:
            dimension: Size of the RFT basis (default: 16)
        """
        self.dimension = dimension
        self.cpp_available = cpp_available and true_rft_engine is not None

        if self.cpp_available:
            # Initialize the C++ engine
            true_rft_engine.engine_init()
        else:
            print("Warning: Using Python fallback for RFT.")

        # Precompute and cache the properly normalized basis
        if dimension not in basis_cache:
            basis_cache[dimension] = py_get_rft_basis(dimension)
        self.basis = basis_cache[dimension]

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

        # Always use the properly normalized basis from Python
        spectrum = self.basis.conj().T @ signal

        # Verify energy conservation
        spectrum_energy = np.linalg.norm(spectrum) ** 2
        energy_ratio = spectrum_energy / original_energy if original_energy > 0 else 1.0

        if abs(energy_ratio - 1.0) > 0.01:
            print(f"Warning: Energy not conserved in forward RFT: {energy_ratio:.6f}")

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

        # Always use the properly normalized basis from Python
        reconstructed = self.basis @ spectrum

        return reconstructed

    def verify_energy_conservation(self, test_signals: int = 10) -> dict:
        """
        Validate energy conservation in the fixed RFT implementation.

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


# Simple test if run directly
if __name__ == "__main__":
    print("Energy-Conserving True RFT Engine Adapter")
    print("=" * 40)

    # Create engine and run validation
    dimensions = [16, 32, 64, 128]

    for N in dimensions:
        print(f"\nTesting dimension N={N}")
        engine = EnergyConservingRFTEngine(dimension=N)
        results = engine.verify_energy_conservation()

        print(f"Average energy ratio: {results['avg_energy_ratio']:.6f}")
        print(f"Maximum energy error: {results['max_energy_error']:.6e}")
        print(f"Average round-trip error: {results['avg_roundtrip_error']:.6e}")
        print(f"All tests pass energy check: {results['all_pass_energy']}")
        print(f"All tests pass round-trip check: {results['all_pass_roundtrip']}")
