"""
Python wrapper for the fixed True RFT Engine with proper normalization.
This enables easy comparison between the Python and C++ implementations.
"""


import numpy as np

# Try to import the fixed C++ engine
try:
    import true_rft_engine_bindings

    cpp_available = True
    print("Successfully imported fixed C++ True RFT Engine bindings")
except ImportError:
    cpp_available = False
    print(
        "Warning: Fixed C++ True RFT Engine bindings not available. Using Python fallback."
    )

# Import the canonical Python implementation
from canonical_true_rft import forward_true_rft as py_forward_true_rft
from canonical_true_rft import get_rft_basis
from canonical_true_rft import inverse_true_rft as py_inverse_true_rft


class FixedTrueRFTEngine:
    """
    Wrapper for the fixed True RFT Engine with proper normalization.
    Falls back to Python implementation if C++ bindings are not available.
    """

    def __init__(self, dimension=64):
        """
        Initialize the RFT engine with the specified dimension.

        Args:
            dimension: Size of the RFT basis matrix
        """
        self.dimension = dimension
        self.cpp_available = cpp_available

        if self.cpp_available:
            # Initialize the C++ engine
            self.cpp_engine = true_rft_engine_bindings.TrueRFTEngine(dimension)
            # Compute the basis with proper normalization
            self.cpp_engine.compute_basis()
            # Verify unitarity
            if not self.cpp_engine.verify_unitarity():
                print("Warning: C++ basis is not unitary!")

        # Also compute the Python basis for comparison
        self.py_basis = get_rft_basis(dimension)

    def forward_true_rft(self, signal, use_cpp=True, check_energy=True):
        """
        Apply forward True RFT transform using the fixed C++ implementation
        or Python fallback.

        Args:
            signal: Input signal (numpy array)
            use_cpp: Whether to use C++ implementation if available
            check_energy: Whether to verify energy conservation

        Returns:
            RFT domain representation
        """
        # Convert signal to complex if it's not already
        if not np.iscomplexobj(signal):
            signal = signal.astype(np.complex128)

        # Store original energy for verification
        original_energy = np.linalg.norm(signal) ** 2

        if self.cpp_available and use_cpp:
            # Use C++ implementation with proper normalization
            signal_list = [complex(x.real, x.imag) for x in signal]
            rft_result = self.cpp_engine.forward_true_rft(signal_list)
            rft_result = np.array(rft_result)
        else:
            # Use Python implementation
            rft_result = py_forward_true_rft(signal, self.dimension, check_energy=False)

        # Verify energy conservation (Parseval's theorem)
        if check_energy:
            rft_energy = np.linalg.norm(rft_result) ** 2
            energy_ratio = rft_energy / original_energy if original_energy > 0 else 1.0
            if not (0.99 < energy_ratio < 1.01):
                print(
                    f"Warning: Energy not conserved in forward RFT: {energy_ratio:.6f}"
                )

        return rft_result

    def inverse_true_rft(
        self, spectrum, use_cpp=True, check_energy=True, original_signal=None
    ):
        """
        Apply inverse True RFT transform using the fixed C++ implementation
        or Python fallback.

        Args:
            spectrum: RFT domain representation
            use_cpp: Whether to use C++ implementation if available
            check_energy: Whether to verify energy conservation
            original_signal: Original signal for round-trip error verification

        Returns:
            Time domain reconstruction
        """
        # Convert spectrum to complex if it's not already
        if not np.iscomplexobj(spectrum):
            spectrum = spectrum.astype(np.complex128)

        # Store original energy for verification
        original_energy = np.linalg.norm(spectrum) ** 2

        if self.cpp_available and use_cpp:
            # Use C++ implementation with proper normalization
            spectrum_list = [complex(x.real, x.imag) for x in spectrum]
            reconstructed = self.cpp_engine.inverse_true_rft(spectrum_list)
            reconstructed = np.array(reconstructed)
        else:
            # Use Python implementation
            reconstructed = py_inverse_true_rft(
                spectrum, self.dimension, original_signal, check_roundtrip=False
            )

        # Verify energy conservation
        if check_energy:
            recon_energy = np.linalg.norm(reconstructed) ** 2
            energy_ratio = (
                recon_energy / original_energy if original_energy > 0 else 1.0
            )
            if not (0.99 < energy_ratio < 1.01):
                print(
                    f"Warning: Energy not conserved in inverse RFT: {energy_ratio:.6f}"
                )

        # Verify round-trip reconstruction accuracy if requested
        if original_signal is not None:
            error = np.linalg.norm(original_signal - reconstructed)
            if error > 1e-8:
                print(f"Warning: Round-trip error too large: {error:.2e}")

        return reconstructed

    def compare_implementations(self, signal=None):
        """
        Compare Python and C++ implementations on a test signal.

        Args:
            signal: Test signal (or generate random if None)

        Returns:
            Dictionary of comparison results
        """
        if signal is None:
            # Generate a random test signal
            signal = np.random.normal(size=self.dimension) + 1j * np.random.normal(
                size=self.dimension
            )

        # Ensure the signal has the right length
        if len(signal) != self.dimension:
            padded_signal = np.zeros(self.dimension, dtype=complex)
            padded_signal[: min(len(signal), self.dimension)] = signal[
                : min(len(signal), self.dimension)
            ]
            signal = padded_signal

        # Compute forward RFT with both implementations
        py_forward = py_forward_true_rft(signal, self.dimension)

        if self.cpp_available:
            cpp_forward = self.forward_true_rft(signal, use_cpp=True)

            # Compute inverse RFT with both implementations
            py_inverse = py_inverse_true_rft(py_forward, self.dimension)
            cpp_inverse = self.inverse_true_rft(cpp_forward, use_cpp=True)

            # Compare results
            py_energy_ratio = (
                np.linalg.norm(py_forward) ** 2 / np.linalg.norm(signal) ** 2
            )
            cpp_energy_ratio = (
                np.linalg.norm(cpp_forward) ** 2 / np.linalg.norm(signal) ** 2
            )

            py_roundtrip_error = np.linalg.norm(signal - py_inverse)
            cpp_roundtrip_error = np.linalg.norm(signal - cpp_inverse)

            implementation_diff = np.linalg.norm(
                py_forward - cpp_forward
            ) / np.linalg.norm(py_forward)

            return {
                "python_energy_ratio": py_energy_ratio,
                "cpp_energy_ratio": cpp_energy_ratio,
                "python_roundtrip_error": py_roundtrip_error,
                "cpp_roundtrip_error": cpp_roundtrip_error,
                "implementation_difference": implementation_diff,
                "energy_ratio_diff": abs(py_energy_ratio - cpp_energy_ratio),
                "cpp_available": True,
            }
        else:
            # C++ implementation not available
            py_inverse = py_inverse_true_rft(py_forward, self.dimension)
            py_energy_ratio = (
                np.linalg.norm(py_forward) ** 2 / np.linalg.norm(signal) ** 2
            )
            py_roundtrip_error = np.linalg.norm(signal - py_inverse)

            return {
                "python_energy_ratio": py_energy_ratio,
                "python_roundtrip_error": py_roundtrip_error,
                "cpp_available": False,
            }


# Test function
def main():
    # Test dimensions
    dimensions = [8, 16, 32, 64, 128]

    for dim in dimensions:
        print(f"\nTesting dimension {dim}:")
        engine = FixedTrueRFTEngine(dim)

        # Generate a test signal
        signal = np.random.normal(size=dim) + 1j * np.random.normal(size=dim)

        # Compare implementations
        results = engine.compare_implementations(signal)

        # Print results
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6e}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
