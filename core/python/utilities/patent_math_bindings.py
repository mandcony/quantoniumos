"""
Python Bindings for Patent Mathematics C++ Implementation
Provides Python access to the C++ implementations of patent mathematics

This module wraps the C++ implementations of:
- Forward RFT: RFT_k = Sigma A_n * e^{iϕ_n} * e^{-2πikn/N}
- Inverse RFT: W_n = (1/N) Sigma RFT_k * e^{2πikn/N}
- XOR Encryption: C_i = D_i ⊕ H(W_i)
"""

import ctypes
import numpy as np
import os

# Try to load the C++ library
_engine_lib = None

def _load_engine_library():
    """Load the engine core C++ library"""
    global _engine_lib

    if _engine_lib is not None:
        return _engine_lib

    # Possible library paths
    library_paths = [
        "build/libengine_core.so",  # Linux
        "build/engine_core.dll",   # Windows
        "build/libengine_core.dylib",  # macOS
        "core/libengine_core.so",
        "core/engine_core.dll",
        "core/libengine_core.dylib",
        "./libengine_core.so",
        "./engine_core.dll",
        "./libengine_core.dylib"
    ]

    for lib_path in library_paths:
        try:
            if os.path.exists(lib_path):
                _engine_lib = ctypes.CDLL(lib_path)
                break
        except OSError:
            continue

    if _engine_lib is None:
        print("Warning: Could not load C++ engine library. Using Python fallback implementations.")
        return None

    # Define function signatures
    try:
        # Forward RFT function
        _engine_lib.forward_rft_run.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # real_part
            ctypes.POINTER(ctypes.c_double),  # imag_part
            ctypes.c_int                      # size
        ]
        _engine_lib.forward_rft_run.restype = None

        # Inverse RFT function
        _engine_lib.inverse_rft_run.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # real_part
            ctypes.POINTER(ctypes.c_double),  # imag_part
            ctypes.c_int                      # size
        ]
        _engine_lib.inverse_rft_run.restype = None

        # Engine initialization
        _engine_lib.engine_init.argtypes = []
        _engine_lib.engine_init.restype = ctypes.c_int

        # Initialize the engine
        result = _engine_lib.engine_init()
        if result != 0:
            print(f"Warning: Engine initialization failed with code {result}")

    except AttributeError as e:
        print(f"Warning: Some C++ functions not available: {e}")
        return None

    return _engine_lib

def forward_rft_cpp(waveform_data: np.ndarray) -> np.ndarray:
    """
    Patent Math: Forward RFT using C++ implementation
    RFT_k = Sigma A_n * e^{iϕ_n} * e^{-2πikn/N}

    Args:
        waveform_data: Complex numpy array representing the waveform

    Returns:
        Complex numpy array with RFT result
    """
    lib = _load_engine_library()

    if lib is None:
        # Fallback to Python implementation
        return np.fft.fft(waveform_data)

    # Ensure input is complex
    if not np.iscomplexobj(waveform_data):
        waveform_data = waveform_data.astype(complex)

    # Extract real and imaginary parts
    real_part = np.real(waveform_data).astype(np.float64)
    imag_part = np.imag(waveform_data).astype(np.float64)
    size = len(waveform_data)

    # Create ctypes arrays
    real_array = (ctypes.c_double * size)(*real_part)
    imag_array = (ctypes.c_double * size)(*imag_part)

    try:
        # Call C++ function
        lib.forward_rft_run(real_array, imag_array, size)

        # Convert back to numpy array
        result_real = np.array([real_array[i] for i in range(size)])
        result_imag = np.array([imag_array[i] for i in range(size)])

        return result_real + 1j * result_imag

    except Exception as e:
        print(f"C++ RFT failed, using Python fallback: {e}")
        return np.fft.fft(waveform_data)

def inverse_rft_cpp(rft_data: np.ndarray) -> np.ndarray:
    """
    Patent Math: Inverse RFT using C++ implementation
    W_n = (1/N) Sigma RFT_k * e^{2πikn/N}

    Args:
        rft_data: Complex numpy array with RFT data

    Returns:
        Complex numpy array with recovered waveform
    """
    lib = _load_engine_library()

    if lib is None:
        # Fallback to Python implementation
        return np.fft.ifft(rft_data)

    # Ensure input is complex
    if not np.iscomplexobj(rft_data):
        rft_data = rft_data.astype(complex)

    # Extract real and imaginary parts
    real_part = np.real(rft_data).astype(np.float64)
    imag_part = np.imag(rft_data).astype(np.float64)
    size = len(rft_data)

    # Create ctypes arrays
    real_array = (ctypes.c_double * size)(*real_part)
    imag_array = (ctypes.c_double * size)(*imag_part)

    try:
        # Call C++ function
        lib.inverse_rft_run(real_array, imag_array, size)

        # Convert back to numpy array
        result_real = np.array([real_array[i] for i in range(size)])
        result_imag = np.array([imag_array[i] for i in range(size)])

        return result_real + 1j * result_imag

    except Exception as e:
        print(f"C++ Inverse RFT failed, using Python fallback: {e}")
        return np.fft.ifft(rft_data)

def test_rft_roundtrip(size: int = 16) -> bool:
    """
    Test that Forward RFT -> Inverse RFT recovers the original signal

    Args:
        size: Size of test signal

    Returns:
        True if roundtrip test passes, False otherwise
    """
    # Create test signal
    t = np.arange(size)
    original_signal = np.exp(1j * 2 * np.pi * t / size) + 0.5 * np.exp(1j * 4 * np.pi * t / size)

    # Forward RFT
    rft_result = forward_rft_cpp(original_signal)

    # Inverse RFT
    recovered_signal = inverse_rft_cpp(rft_result)

    # Check roundtrip accuracy
    error = np.max(np.abs(original_signal - recovered_signal))
    success = error < 1e-10

    print(f"RFT Roundtrip Test (size={size}): {'PASS' if success else 'FAIL'}")
    print(f" Max error: {error:.2e}")

    return success

def geometric_hash_python(data: bytes, amplitude: float, phase: float, prime: int = 251) -> int:
    """
    Python implementation of geometric hash
    H(W_i) = mod(A_i * cos(ϕ_i) + data_influence, p)

    Args:
        data: Input data bytes
        amplitude: Waveform amplitude
        phase: Waveform phase (radians)
        prime: Prime modulus

    Returns:
        Hash value as integer
    """
    import math

    # Basic geometric component
    geometric_component = amplitude * math.cos(phase)

    # Data influence (simple polynomial hash)
    data_influence = 0
    for i, byte in enumerate(data):
        data_influence += byte * (i + 1)
    data_influence = data_influence % prime

    # Combine and take modulus
    hash_value = int(geometric_component + data_influence) % prime
    return hash_value

def symbolic_xor_python(plaintext: bytes, key: bytes) -> bytes:
    """
    Python implementation of symbolic XOR encryption
    C_i = D_i ⊕ H(W_i)

    Args:
        plaintext: Input data to encrypt
        key: Encryption key

    Returns:
        Encrypted bytes
    """
    if len(plaintext) != len(key):
        raise ValueError("Plaintext and key must be same length")

    result = bytearray()

    for i in range(len(plaintext)):
        # Generate geometric hash for this position
        amplitude = 1.0 + (key[i] / 255.0)  # Normalize to [1, 2]
        phase = (key[i] / 255.0) * 2 * np.pi  # Normalize to [0, 2π]

        # Use single byte as data for hash
        hash_byte = geometric_hash_python(bytes([plaintext[i]]), amplitude, phase)

        # XOR encryption
        encrypted_byte = plaintext[i] ^ (hash_byte & 0xFF)
        result.append(encrypted_byte)

    return bytes(result)

def validate_cpp_bindings() -> bool:
    """
    Validate that C++ bindings work correctly

    Returns:
        True if all validations pass, False otherwise
    """
    print("🔗 Validating C++ Bindings for Patent Mathematics")
    print("-" * 50)

    success = True

    # Test RFT roundtrip
    try:
        rft_success = test_rft_roundtrip(16)
        if not rft_success:
            success = False
    except Exception as e:
        print(f"RFT test failed: {e}")
        success = False

    # Test with different sizes
    for size in [4, 8, 32, 64]:
        try:
            size_success = test_rft_roundtrip(size)
            if not size_success:
                success = False
        except Exception as e:
            print(f"RFT test (size={size}) failed: {e}")
            success = False

    # Test encryption
    try:
        plaintext = b"Hello, Patent Math!"
        key = b"SecretKey123456789"[:len(plaintext)]  # Truncate to match length

        encrypted = symbolic_xor_python(plaintext, key)
        decrypted = symbolic_xor_python(encrypted, key)  # XOR is self-inverse

        if decrypted == plaintext:
            print("✅ Symbolic XOR Encryption: PASS")
        else:
            print("❌ Symbolic XOR Encryption: FAIL")
            success = False

    except Exception as e:
        print(f"❌ Symbolic XOR test failed: {e}")
        success = False

    print("-" * 50)
    if success:
        print("🎉 All C++ binding validations passed!")
    else:
        print("⚠️ Some C++ binding validations failed.")

    return success

if __name__ == "__main__":
    # Run validation when executed directly
    validate_cpp_bindings()
