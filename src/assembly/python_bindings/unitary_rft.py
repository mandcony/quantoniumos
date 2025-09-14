"""
Python interface to the Bare Metal Unitary RFT implementation.

This module provides Python bindings to the C/ASM implementation of the
Resonance Field Theory (RFT) algorithm implemented at the bare metal level.
"""

import os
import ctypes
import numpy as np
from ctypes import c_int, c_size_t, c_double, c_uint32, c_void_p, c_bool, Structure, POINTER, cdll
from typing import Tuple, List, Optional, Union

# Define C structures
class RFTComplex(Structure):
    """Complex number structure matching the C implementation."""
    _fields_ = [("real", c_double),
                ("imag", c_double)]

class RFTEngine(Structure):
    """RFT engine structure matching the C implementation."""
    _fields_ = [("size", c_size_t),
                ("basis", POINTER(RFTComplex)),
                ("eigenvalues", POINTER(c_double)),
                ("initialized", c_bool),
                ("flags", c_uint32),
                ("qubit_count", c_size_t),
                ("quantum_context", c_void_p)]

# RFT flags
RFT_FLAG_DEFAULT = 0x00000000
RFT_FLAG_OPTIMIZE_SIMD = 0x00000001
RFT_FLAG_HIGH_PRECISION = 0x00000002
RFT_FLAG_QUANTUM_SAFE = 0x00000004
RFT_FLAG_UNITARY = 0x00000008
RFT_FLAG_USE_RESONANCE = 0x00000010

class UnitaryRFT:
    """Python interface to the Bare Metal Unitary RFT implementation."""
    
    def __init__(self, size: int, flags: int = RFT_FLAG_UNITARY):
        """Initialize the Unitary RFT engine.
        
        Args:
            size: Size of the transform (power of 2 recommended)
            flags: Configuration flags
        
        Raises:
            RuntimeError: If the RFT library cannot be loaded or initialized
        """
        self.size = size
        try:
            self.lib = self._load_library()
            self.engine = self._init_engine(size, flags)
            self._is_mock = False
        except RuntimeError:
            # Fallback to mock implementation
            self.lib = None
            self.engine = None
            self._is_mock = True
            print(f"Warning: Using mock UnitaryRFT implementation for size {size}")
    
    def _load_library(self):
        """Load the RFT library."""
        # Find the library path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lib_paths = [
            # First check the compiled directory (actual location)
            os.path.join(script_dir, "..", "compiled", "libquantum_symbolic.dll"),
            os.path.join(script_dir, "..", "compiled", "libquantum_symbolic.so"),
            os.path.join(script_dir, "..", "compiled", "rftkernel.dll"),
            os.path.join(script_dir, "..", "compiled", "rftkernel.so"),
            os.path.join(script_dir, "..", "compiled", "librftkernel.dll"),
            os.path.join(script_dir, "..", "compiled", "librftkernel.so"),
            # Then fallback to other possible locations
            os.path.join(script_dir, "librftkernel.dll"),
            os.path.join(script_dir, "librftkernel.so"),
            os.path.join(script_dir, "..", "build", "librftkernel.so"),
            os.path.join(script_dir, "..", "build", "librftkernel.dll"),
            os.path.join(script_dir, "..", "..", "ASSEMBLY", "build", "librftkernel.so"),
            os.path.join(script_dir, "..", "..", "ASSEMBLY", "build", "librftkernel.dll"),
            os.path.join(script_dir, "..", "..", "ASSEMBLY", "kernel", "build", "librftkernel.dll"),
            os.path.join(script_dir, "..", "..", "ASSEMBLY", "kernel", "build", "librftkernel.so")
        ]
        
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                try:
                    lib = cdll.LoadLibrary(lib_path)
                    
                    # Try to set function argument and return types
                    # If any function is missing, we'll catch the AttributeError
                    try:
                        lib.rft_init.argtypes = [POINTER(RFTEngine), c_size_t, c_uint32]
                        lib.rft_init.restype = c_int
                        
                        lib.rft_cleanup.argtypes = [POINTER(RFTEngine)]
                        lib.rft_cleanup.restype = c_int
                        
                        lib.rft_forward.argtypes = [POINTER(RFTEngine), POINTER(RFTComplex), 
                                                   POINTER(RFTComplex), c_size_t]
                        lib.rft_forward.restype = c_int
                        
                        lib.rft_inverse.argtypes = [POINTER(RFTEngine), POINTER(RFTComplex), 
                                                   POINTER(RFTComplex), c_size_t]
                        lib.rft_inverse.restype = c_int
                        
                        lib.rft_quantum_basis.argtypes = [POINTER(RFTEngine), c_size_t]
                        lib.rft_quantum_basis.restype = c_int
                        
                        lib.rft_entanglement_measure.argtypes = [POINTER(RFTEngine), 
                                                                POINTER(RFTComplex),
                                                                POINTER(c_double),
                                                                c_size_t]
                        lib.rft_entanglement_measure.restype = c_int
                        
                        # New mathematical validation functions
                        lib.rft_von_neumann_entropy.argtypes = [POINTER(RFTEngine), 
                                                              POINTER(RFTComplex),
                                                              POINTER(c_double),
                                                              c_size_t]
                        lib.rft_von_neumann_entropy.restype = c_int
                        
                        lib.rft_validate_bell_state.argtypes = [POINTER(RFTEngine),
                                                              POINTER(RFTComplex),
                                                              POINTER(c_double),
                                                              c_double]
                        lib.rft_validate_bell_state.restype = c_int
                        
                        lib.rft_validate_golden_ratio_properties.argtypes = [POINTER(RFTEngine),
                                                                           POINTER(c_double),
                                                                           c_double]
                        lib.rft_validate_golden_ratio_properties.restype = c_int
                    except AttributeError as func_error:
                        print(f"Warning: Some RFT functions not available in {lib_path}: {func_error}")
                        # Still return the library for basic functionality
                    
                    return lib
                except (OSError, AttributeError) as e:
                    print(f"Failed to load library {lib_path}: {e}")
        
        raise RuntimeError("Could not find or load the RFT kernel library")
    
    def _load_library(self):
        """Load the RFT library."""
        # Find the library path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lib_paths = [
            # First check the compiled directory (actual location)
            os.path.join(script_dir, "..", "compiled", "libquantum_symbolic.dll"),
            os.path.join(script_dir, "..", "compiled", "libquantum_symbolic.so"),
            os.path.join(script_dir, "..", "compiled", "rftkernel.dll"),
            os.path.join(script_dir, "..", "compiled", "rftkernel.so"),
            os.path.join(script_dir, "..", "compiled", "librftkernel.dll"),
            os.path.join(script_dir, "..", "compiled", "librftkernel.so"),
            # Then fallback to other possible locations
            os.path.join(script_dir, "librftkernel.dll"),
            os.path.join(script_dir, "librftkernel.so"),
            os.path.join(script_dir, "..", "build", "librftkernel.so"),
            os.path.join(script_dir, "..", "build", "librftkernel.dll"),
            os.path.join(script_dir, "..", "..", "ASSEMBLY", "build", "librftkernel.so"),
            os.path.join(script_dir, "..", "..", "ASSEMBLY", "build", "librftkernel.dll"),
            os.path.join(script_dir, "..", "..", "ASSEMBLY", "kernel", "build", "librftkernel.dll"),
            os.path.join(script_dir, "..", "..", "ASSEMBLY", "kernel", "build", "librftkernel.so")
        ]
        
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                try:
                    lib = cdll.LoadLibrary(lib_path)
                    
                    # Try to set function argument and return types
                    # If any function is missing, we'll catch the AttributeError
                    try:
                        lib.rft_init.argtypes = [POINTER(RFTEngine), c_size_t, c_uint32]
                        lib.rft_init.restype = c_int
                        
                        lib.rft_cleanup.argtypes = [POINTER(RFTEngine)]
                        lib.rft_cleanup.restype = c_int
                        
                        lib.rft_forward.argtypes = [POINTER(RFTEngine), POINTER(RFTComplex), 
                                                   POINTER(RFTComplex), c_size_t]
                        lib.rft_forward.restype = c_int
                        
                        lib.rft_inverse.argtypes = [POINTER(RFTEngine), POINTER(RFTComplex), 
                                                   POINTER(RFTComplex), c_size_t]
                        lib.rft_inverse.restype = c_int
                        
                        lib.rft_quantum_basis.argtypes = [POINTER(RFTEngine), c_size_t]
                        lib.rft_quantum_basis.restype = c_int
                        
                        lib.rft_entanglement_measure.argtypes = [POINTER(RFTEngine), 
                                                                POINTER(RFTComplex),
                                                                POINTER(c_double),
                                                                c_size_t]
                        lib.rft_entanglement_measure.restype = c_int
                        
                        # New mathematical validation functions
                        lib.rft_von_neumann_entropy.argtypes = [POINTER(RFTEngine), 
                                                              POINTER(RFTComplex),
                                                              POINTER(c_double),
                                                              c_size_t]
                        lib.rft_von_neumann_entropy.restype = c_int
                        
                        lib.rft_validate_bell_state.argtypes = [POINTER(RFTEngine),
                                                              POINTER(RFTComplex),
                                                              POINTER(c_double),
                                                              c_double]
                        lib.rft_validate_bell_state.restype = c_int
                        
                        lib.rft_validate_golden_ratio_properties.argtypes = [POINTER(RFTEngine),
                                                                           POINTER(c_double),
                                                                           c_double]
                        lib.rft_validate_golden_ratio_properties.restype = c_int
                    except AttributeError as func_error:
                        print(f"Warning: Some RFT functions not available in {lib_path}: {func_error}")
                        # Still return the library for basic functionality
                    
                    return lib
                except (OSError, AttributeError) as e:
                    print(f"Failed to load library {lib_path}: {e}")
        
        raise RuntimeError("Could not find or load the RFT kernel library")
    
    def _init_engine(self, size: int, flags: int) -> RFTEngine:
        """Initialize the RFT engine."""
        engine = RFTEngine()
        result = self.lib.rft_init(ctypes.byref(engine), size, flags)
        if result != 0:
            raise RuntimeError(f"Failed to initialize RFT engine: error code {result}")
        return engine
    
    def __del__(self):
        """Clean up the RFT engine."""
        if hasattr(self, 'lib') and hasattr(self, 'engine') and self.lib is not None:
            try:
                if hasattr(self.lib, 'rft_cleanup'):
                    self.lib.rft_cleanup(ctypes.byref(self.engine))
            except (RuntimeError, TypeError, AttributeError):
                # Ignore cleanup errors for mock libraries
                pass
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Perform the forward unitary RFT transform.
        
        Args:
            input_data: Complex input array
        
        Returns:
            Complex output array
        """
        if len(input_data) != self.size:
            raise ValueError(f"Input size must be {self.size}")
        
        # Check if we should use mock implementation
        use_mock = self._is_mock or self.lib is None
        if not use_mock:
            try:
                # Try to check if the function exists
                hasattr(self.lib, 'rft_forward')
                use_mock = not hasattr(self.lib, 'rft_forward')
            except (RuntimeError, TypeError):
                # If checking hasattr raises an exception, it's likely a mock library
                use_mock = True
        
        if use_mock:
            # Mock implementation using FFT with phase modulation
            import math
            phi = (1 + math.sqrt(5)) / 2  # Golden ratio
            
            # Apply FFT
            fft_result = np.fft.fft(input_data)
            
            # Apply golden ratio phase modulation for forward transform
            phases = np.exp(1j * phi * np.arange(self.size))
            result = fft_result * phases
            
            # Normalize to maintain unitarity (this is approximate)
            result = result / np.sqrt(self.size)
            
            return result
        
        # Convert numpy array to C array
        input_c = (RFTComplex * self.size)()
        for i in range(self.size):
            input_c[i].real = input_data[i].real
            input_c[i].imag = input_data[i].imag
        
        # Prepare output array
        output_c = (RFTComplex * self.size)()
        
        # Call C function
        result = self.lib.rft_forward(ctypes.byref(self.engine), input_c, output_c, self.size)
        if result != 0:
            raise RuntimeError(f"Forward transform failed: error code {result}")
        
        # Convert C array to numpy array
        output = np.zeros(self.size, dtype=np.complex128)
        for i in range(self.size):
            output[i] = complex(output_c[i].real, output_c[i].imag)
        
        return output
    
    def inverse(self, input_data: np.ndarray) -> np.ndarray:
        """Perform the inverse unitary RFT transform.
        
        Args:
            input_data: Complex input array
        
        Returns:
            Complex output array
        """
        if len(input_data) != self.size:
            raise ValueError(f"Input size must be {self.size}")
        
        # Check if we should use mock implementation
        use_mock = self._is_mock or self.lib is None
        if not use_mock:
            try:
                # Try to check if the function exists
                hasattr(self.lib, 'rft_inverse')
                use_mock = not hasattr(self.lib, 'rft_inverse')
            except (RuntimeError, TypeError):
                # If checking hasattr raises an exception, it's likely a mock library
                use_mock = True
        
        if use_mock:
            # Mock implementation - inverse of forward transform
            import math
            phi = (1 + math.sqrt(5)) / 2  # Golden ratio
            
            # Undo the normalization from forward transform
            scaled_input = input_data * np.sqrt(self.size)
            
            # Remove golden ratio phase modulation
            phases = np.exp(-1j * phi * np.arange(self.size))
            phased_data = scaled_input * phases
            
            # Apply inverse FFT
            result = np.fft.ifft(phased_data)
            
            return result
        
        # Convert numpy array to C array
        input_c = (RFTComplex * self.size)()
        for i in range(self.size):
            input_c[i].real = input_data[i].real
            input_c[i].imag = input_data[i].imag
        
        # Prepare output array
        output_c = (RFTComplex * self.size)()
        
        # Call C function
        result = self.lib.rft_inverse(ctypes.byref(self.engine), input_c, output_c, self.size)
        if result != 0:
            raise RuntimeError(f"Inverse transform failed: error code {result}")
        
        # Convert C array to numpy array
        output = np.zeros(self.size, dtype=np.complex128)
        for i in range(self.size):
            output[i] = complex(output_c[i].real, output_c[i].imag)
        
        return output
    
    def init_quantum_basis(self, qubit_count: int) -> None:
        """Initialize the RFT engine for quantum operations.
        
        Args:
            qubit_count: Number of qubits
        
        Raises:
            RuntimeError: If initialization fails
        """
        if 2**qubit_count != self.size:
            raise ValueError(f"qubit_count {qubit_count} must give size {self.size}")
        
        result = self.lib.rft_quantum_basis(ctypes.byref(self.engine), qubit_count)
        if result != 0:
            raise RuntimeError(f"Quantum basis initialization failed: error code {result}")
    
    def measure_entanglement(self, state: np.ndarray) -> float:
        """Compute entanglement measure for a quantum state.
        
        Args:
            state: Quantum state vector
        
        Returns:
            Entanglement measure (von Neumann entropy)
        
        Raises:
            RuntimeError: If measurement fails
        """
        if len(state) != self.size:
            raise ValueError(f"State size must be {self.size}")
        
        # Convert numpy array to C array
        state_c = (RFTComplex * self.size)()
        for i in range(self.size):
            state_c[i].real = state[i].real
            state_c[i].imag = state[i].imag
        # Prepare output variable
        measure = c_double()
        
        # Call C function
        result = self.lib.rft_entanglement_measure(ctypes.byref(self.engine), state_c, 
                                                  ctypes.byref(measure), self.size)
        if result != 0:
            raise RuntimeError(f"Entanglement measurement failed: error code {result}")
        
        return measure.value


# Convenience top-level function so callers can use `from unitary_rft import rft`
# Cache engines by size to avoid repeated initialization overhead.
_RFT_ENGINE_CACHE = {}

def rft(input_data: np.ndarray) -> np.ndarray:
    """Top-level helper: perform RFT on a complex numpy array.

    This function caches UnitaryRFT engine instances by size for reuse.
    If the UnitaryRFT cannot be initialized (mock fallback), it will still
    attempt to return a usable result via the class's mock implementation.
    """
    if not isinstance(input_data, np.ndarray):
        input_data = np.asarray(input_data)

    N = input_data.shape[0]
    engine = _RFT_ENGINE_CACHE.get(N)
    # Validate cached engine
    if engine is None or not hasattr(engine, 'size') or not hasattr(engine, 'forward'):
        try:
            engine = UnitaryRFT(N)
        except Exception:
            # Last-resort: create a mock instance (UnitaryRFT handles mock fallback)
            engine = UnitaryRFT(N)
        _RFT_ENGINE_CACHE[N] = engine

    # Try calling forward; if it fails, attempt one retry with a fresh engine
    try:
        return engine.forward(input_data)
    except Exception as e:
        # recreate and retry once
        try:
            engine = UnitaryRFT(N)
            _RFT_ENGINE_CACHE[N] = engine
            return engine.forward(input_data)
        except Exception:
            # Re-raise the original exception for visibility
            raise e
    
    def von_neumann_entropy(self, state: np.ndarray) -> float:
        """Compute von Neumann entropy of a quantum state.
        
        Args:
            state: Quantum state vector
        
        Returns:
            Von Neumann entropy S = -Tr(ρ log₂ ρ)
        
        Raises:
            RuntimeError: If calculation fails
        """
        if len(state) != self.size:
            raise ValueError(f"State size must be {self.size}")
        
        # Convert numpy array to C array
        state_c = (RFTComplex * self.size)()
        for i in range(self.size):
            state_c[i].real = state[i].real
            state_c[i].imag = state[i].imag
        
        # Prepare output variable
        entropy = c_double()
        
        # Call C function
        result = self.lib.rft_von_neumann_entropy(ctypes.byref(self.engine), state_c, 
                                                 ctypes.byref(entropy), self.size)
        if result != 0:
            raise RuntimeError(f"Entropy calculation failed: error code {result}")
        
        return entropy.value

    def validate_bell_state(self, bell_state: np.ndarray, tolerance: float = 0.1) -> tuple[float, bool]:
        """Validate Bell state entanglement properties.
        
        Args:
            bell_state: Bell state to validate
            tolerance: Tolerance for validation
        
        Returns:
            Tuple of (measured_entanglement, is_valid)
        
        Raises:
            RuntimeError: If validation fails
        """
        if len(bell_state) != self.size:
            raise ValueError(f"State size must be {self.size}")
        
        # Convert numpy array to C array
        state_c = (RFTComplex * self.size)()
        for i in range(self.size):
            state_c[i].real = bell_state[i].real
            state_c[i].imag = bell_state[i].imag
        
        # Prepare output variable
        measured_entanglement = c_double()
        
        # Call C function
        result = self.lib.rft_validate_bell_state(ctypes.byref(self.engine), state_c, 
                                                 ctypes.byref(measured_entanglement), tolerance)
        
        is_valid = (result == 0)
        return measured_entanglement.value, is_valid

    def validate_golden_ratio_properties(self, tolerance: float = 0.1) -> tuple[float, bool]:
        """Validate golden ratio properties in the transform matrix.
        
        Args:
            tolerance: Tolerance for validation
        
        Returns:
            Tuple of (phi_presence, has_golden_ratio_properties)
        
        Raises:
            RuntimeError: If validation fails
        """
        # Prepare output variable
        phi_presence = c_double()
        
        # Call C function
        result = self.lib.rft_validate_golden_ratio_properties(ctypes.byref(self.engine), 
                                                              ctypes.byref(phi_presence), tolerance)
        
        has_properties = (result == 0)
        return phi_presence.value, has_properties


# Example usage
if __name__ == "__main__":
    # Test with a simple signal
    size = 16
    rft = UnitaryRFT(size, RFT_FLAG_UNITARY | RFT_FLAG_QUANTUM_SAFE)
    
    # Create a test signal
    signal = np.zeros(size, dtype=np.complex128)
    signal[0] = 1.0  # Impulse at the beginning
    
    # Perform forward transform
    spectrum = rft.forward(signal)
    print("Spectrum:")
    for i in range(min(4, size)):
        print(f"  [{i}]: {spectrum[i]}")
    
    # Verify unitarity (norm preservation)
    signal_norm = np.sqrt(np.sum(np.abs(signal)**2))
    spectrum_norm = np.sqrt(np.sum(np.abs(spectrum)**2))
    print(f"Norm preservation: input={signal_norm}, output={spectrum_norm}, ratio={spectrum_norm/signal_norm}")
    
    # Perform inverse transform
    reconstructed = rft.inverse(spectrum)
    
    # Calculate reconstruction error
    error = np.max(np.abs(signal - reconstructed))
    print(f"Reconstruction error: {error}")
    
    # Test quantum capabilities
    qubit_count = int(np.log2(size))
    rft.init_quantum_basis(qubit_count)
    
    # Create a Bell state (|00⟩ + |11⟩)/√2
    bell_state = np.zeros(size, dtype=np.complex128)
    bell_state[0] = 1.0 / np.sqrt(2)  # |00⟩
    bell_state[3] = 1.0 / np.sqrt(2)  # |11⟩
    
    # Measure entanglement
    entanglement = rft.measure_entanglement(bell_state)
    print(f"Entanglement of Bell state: {entanglement}")


# Simplified interface for applications
class RFTProcessor:
    """Simplified RFT processor interface for applications."""
    
    def __init__(self, size: int = 64):
        """Initialize the RFT processor."""
        try:
            self.rft = UnitaryRFT(size, RFT_FLAG_UNITARY | RFT_FLAG_USE_RESONANCE)
            self.available = True
        except Exception as e:
            print(f"RFT processor initialization failed: {e}")
            self.rft = None
            self.available = False
    
    def process_quantum_field(self, data) -> any:
        """Process quantum field data using RFT."""
        if not self.available:
            return self._fallback_processing(data)
        
        try:
            # Convert input to numpy array if needed
            if isinstance(data, (list, tuple)):
                data = np.array(data, dtype=complex)
            elif isinstance(data, str):
                # Convert string to numeric data for processing
                data = np.array([complex(ord(c), 0) for c in data])
            
            # Ensure data fits our RFT size
            if len(data) > self.rft.size:
                data = data[:self.rft.size]
            elif len(data) < self.rft.size:
                # Pad with zeros
                padded = np.zeros(self.rft.size, dtype=complex)
                padded[:len(data)] = data
                data = padded
            
            # Process using RFT
            result = self.rft.forward(data)
            return result
            
        except Exception as e:
            print(f"RFT processing failed: {e}")
            return self._fallback_processing(data)
    
    def _fallback_processing(self, data):
        """Fallback processing when RFT is not available."""
        # Simple identity transformation for fallback
        if isinstance(data, str):
            return data
        elif isinstance(data, (list, tuple, np.ndarray)):
            return np.array(data)
        else:
            return data
    
    def is_available(self) -> bool:
        """Check if RFT processing is available."""
        return self.available
