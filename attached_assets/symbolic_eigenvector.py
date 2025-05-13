import ctypes
import numpy as np
import os
import platform

# Ensure the DLL directory is in the PATH to find dependencies
dll_dir = r'C:\quantonium_os\bin'
if platform.system() == "Windows":
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dll_dir)
    os.environ['PATH'] = dll_dir + os.pathsep + os.environ['PATH']

# Attempt to load the DLL and log specific error if it fails
engine_core_path = os.path.join(dll_dir, 'engine_core.dll')
try:
    lib = ctypes.WinDLL(engine_core_path)
    print(f"✅ Loaded DLL: {engine_core_path}")
except OSError as e:
    print(f"❌ Error loading DLL: {e}")
    raise

# Define argument and return types for exposed DLL functions
try:
    lib.U.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                      ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_double]
    lib.U.restype = None

    lib.T.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                      ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    lib.T.restype = None

    lib.ComputeEigenvectors.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.ComputeEigenvectors.restype = None

    lib.encode_resonance.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.encode_resonance.restype = None

    lib.decode_resonance.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.decode_resonance.restype = None

    lib.compute_similarity.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.compute_similarity.restype = ctypes.c_double
    print("✅ All DLL function prototypes assigned successfully.")
except AttributeError as e:
    print(f"❌ DLL function not found or not exported: {e}")
    raise

def apply_u(state, derivative, dt):
    n = len(state)
    out = np.zeros(n, dtype=np.float64)
    state_arr = (ctypes.c_double * n)(*state)
    derivative_arr = (ctypes.c_double * n)(*derivative)
    out_arr = (ctypes.c_double * n)(*out)
    lib.U(state_arr, derivative_arr, n, out_arr, ctypes.c_double(dt))
    return np.frombuffer(out_arr, dtype=np.float64)

def apply_t(state, transform):
    n = len(state)
    out = np.zeros(n, dtype=np.float64)
    state_arr = (ctypes.c_double * n)(*state)
    transform_arr = (ctypes.c_double * n)(*transform)
    out_arr = (ctypes.c_double * n)(*out)
    lib.T(state_arr, transform_arr, n, out_arr)
    return np.frombuffer(out_arr, dtype=np.float64)

def compute_eigenvectors(state):
    n = len(state)
    eigenvalues = np.zeros(n, dtype=np.float64)
    eigenvectors = np.zeros((n, n), dtype=np.float64)
    state_arr = (ctypes.c_double * n)(*state)
    eigenvalues_arr = (ctypes.c_double * n)(*eigenvalues)
    eigenvectors_arr = (ctypes.c_double * (n * n))(*eigenvectors.flatten())
    lib.ComputeEigenvectors(state_arr, n, eigenvalues_arr, eigenvectors_arr)
    return (np.frombuffer(eigenvalues_arr, dtype=np.float64),
            np.frombuffer(eigenvectors_arr, dtype=np.float64).reshape(n, n))

def encode_resonance(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    data_len = len(data) + 1
    out_len = ctypes.c_int(0)
    out = ctypes.create_string_buffer(data_len * 4)
    lib.encode_resonance(data, out, ctypes.byref(out_len))
    return out.value.decode('utf-8')[:out_len.value - 1]

def decode_resonance(encoded_data):
    if isinstance(encoded_data, str):
        encoded_data = encoded_data.encode('utf-8')
    encoded_len = len(encoded_data) + 1
    out_len = ctypes.c_int(0)
    out = ctypes.create_string_buffer(encoded_len * 2)
    lib.decode_resonance(encoded_data, out, ctypes.byref(out_len))
    return out.value.decode('utf-8')[:out_len.value - 1]

def compute_similarity(url1, url2):
    if not isinstance(url1, str) or not isinstance(url2, str):
        raise ValueError("URLs must be strings")
    return lib.compute_similarity(url1.encode('utf-8'), url2.encode('utf-8'))

if __name__ == "__main__":
    state = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    derivative = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    transform = np.array([2.0, 3.0, 4.0], dtype=np.float64)

    result_u = apply_u(state, derivative, 0.1)
    print("U result:", result_u)

    result_t = apply_t(state, transform)
    print("T result:", result_t)

    eigenvalues, eigenvectors = compute_eigenvectors(state)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

    encoded = encode_resonance("https://example.com")
    print("Encoded:", encoded)
    decoded = decode_resonance(encoded)
    print("Decoded:", decoded)
    similarity = compute_similarity("https://example.com", "https://duckduckgo.com")
    print("Similarity:", similarity)