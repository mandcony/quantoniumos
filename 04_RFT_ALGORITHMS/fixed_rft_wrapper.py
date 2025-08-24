#!/usr/bin/env python3
"""
Python wrapper for the fixed True RFT engine with proper energy conservation.
This module imports the fixed C++ RFT implementation and provides a clean Python interface.
"""

import importlib.util
import sys

import numpy as np

# Try to import the fixed engine
try:
    spec = importlib.util.find_spec("fixed_true_rft_engine_bindings")
    if spec is not None:
        fixed_rft_engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixed_rft_engine)
    else:
        fixed_rft_engine = None
        print(
            "Warning: Fixed RFT engine module not found. Energy conservation may not be guaranteed."
        )
except ImportError:
    fixed_rft_engine = None
    print(
        "Warning: Could not import fixed RFT engine. Energy conservation may not be guaranteed."
    )


class FixedTrueRFTEngine:
    """
    Wrapper for the fixed C++ True RFT engine with proper energy conservation.

    This class provides a clean Python interface to the C++ implementation that
    properly normalizes the RFT basis to ensure energy conservation.
    """

    def __init__(self, dimension: int = 16):
        """
        Initialize the fixed True RFT engine.

        Args:
            dimension: Size of the RFT basis (default: 16)
        """
        self.dimension = dimension
        self.cpp_available = fixed_rft_engine is not None

        if self.cpp_available:
            # Initialize the C++ engine
            fixed_rft_engine.engine_init()
        else:
            print(
                "Warning: Using Python fallback for RFT. Energy conservation may not be guaranteed."
            )

    def forward_true_rft(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply forward True RFT to a signal using the fixed C++ implementation.

        Args:
            signal: Input signal to transform

        Returns:
            RFT domain representation of the signal
        """
        if not self.cpp_available:
            raise RuntimeError(
                "C++ engine not available. Please build the fixed RFT engine."
            )

        # Ensure signal has the right size
        if len(signal) != self.dimension:
            if len(signal) < self.dimension:
                padded_signal = np.zeros(self.dimension, dtype=complex)
                padded_signal[: len(signal)] = signal
                signal = padded_signal
            else:
                signal = signal[: self.dimension]

        # Extract real part of signal for C++ function
        signal_real = np.real(signal).astype(np.float64)

        # Dummy parameters (not used in the fixed implementation)
        dummy_w = np.ones(1, dtype=np.float64)
        dummy_th = np.zeros(1, dtype=np.float64)
        dummy_om = np.zeros(1, dtype=np.float64)

        # Call the C++ forward transform
        result = fixed_rft_engine.rft_basis_forward(
            signal_real, dummy_w, dummy_th, dummy_om, 1.0, 1.0, ""
        )

        # Check energy conservation
        input_energy = np.linalg.norm(signal) ** 2
        output_energy = np.linalg.norm(result) ** 2
        energy_ratio = output_energy / input_energy

        if abs(energy_ratio - 1.0) > 0.01:
            print(
                f"Warning: Energy not well-conserved in forward RFT. Ratio: {energy_ratio:.6f}"
            )

        return result

    def inverse_true_rft(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Apply inverse True RFT to a spectrum using the fixed C++ implementation.

        Args:
            spectrum: RFT domain representation to inverse transform

        Returns:
            Time domain reconstruction of the signal
        """
        if not self.cpp_available:
            raise RuntimeError(
                "C++ engine not available. Please build the fixed RFT engine."
            )

        # Ensure spectrum has the right size
        if len(spectrum) != self.dimension:
            if len(spectrum) < self.dimension:
                padded_spectrum = np.zeros(self.dimension, dtype=complex)
                padded_spectrum[: len(spectrum)] = spectrum
                spectrum = padded_spectrum
            else:
                spectrum = spectrum[: self.dimension]

        # Extract real and imaginary parts for C++ function
        np.real(spectrum).astype(np.float64)
        np.imag(spectrum).astype(np.float64)

        # Dummy parameters (not used in the fixed implementation)
        dummy_w = np.ones(1, dtype=np.float64)
        dummy_th = np.zeros(1, dtype=np.float64)
        dummy_om = np.zeros(1, dtype=np.float64)

        # Call the C++ inverse transform
        result = fixed_rft_engine.rft_basis_inverse(
            spectrum, dummy_w, dummy_th, dummy_om, 1.0, 1.0, ""
        )

        return result

    def validate_energy_conservation(self, test_signals: int = 10) -> dict:
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
    # Check if the engine is available
    if fixed_rft_engine is None:
        print(
            "Fixed RFT engine not found. Please build it first using build_energy_conserving_rft.py"
        )
        sys.exit(1)

    print("Testing Fixed True RFT Engine")
    print("=" * 40)

    # Create engine and run validation
    engine = FixedTrueRFTEngine(dimension=16)
    results = engine.validate_energy_conservation()

    print("\nEnergy Conservation Validation:")
    print(f"Average energy ratio: {results['avg_energy_ratio']:.6f}")
    print(f"Maximum energy error: {results['max_energy_error']:.6e}")
    print(f"Average round-trip error: {results['avg_roundtrip_error']:.6e}")
    print(f"All tests pass energy check: {results['all_pass_energy']}")
    print(f"All tests pass round-trip check: {results['all_pass_roundtrip']}")
