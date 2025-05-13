import sys
import time
import random
import string
import ctypes
import numpy as np
import os

# Load the DLL (changed to engine_core.dll)
DLL_PATH = r"C:\quantonium_os\bin\engine_core.dll"
try:
    engine_core = ctypes.CDLL(DLL_PATH)
    engine_core.ParallelXOREncrypt.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # input
        ctypes.c_int,                    # input_len
        ctypes.POINTER(ctypes.c_uint8),  # key
        ctypes.c_int,                    # key_len
        ctypes.POINTER(ctypes.c_uint8)   # output
    ]
    engine_core.ParallelXOREncrypt.restype = None
    print(f"✅ Loaded DLL: {DLL_PATH}")
except Exception as e:
    print(f"❌ Failed to load DLL: {DLL_PATH}")
    print(f"Error: {e}")
    print(f"Does the file exist? {os.path.exists(DLL_PATH)}")
    if os.path.exists(DLL_PATH):
        print("DLL exists but failed to load. Check dependencies (e.g., libgomp-1.dll) or architecture mismatch.")
    sys.exit(1)

def vectorized_xor_encrypt(input_bytes: bytes, key_bytes: bytes) -> bytes:
    """Wrapper for the ParallelXOREncrypt function in engine_core.dll."""
    input_len = len(input_bytes)
    key_len = len(key_bytes)
    input_arr = np.frombuffer(input_bytes, dtype=np.uint8)
    key_arr = np.frombuffer(key_bytes, dtype=np.uint8)
    output_arr = np.empty(input_len, dtype=np.uint8)
    
    input_ptr = input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    key_ptr = key_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    
    engine_core.ParallelXOREncrypt(input_ptr, input_len, key_ptr, key_len, output_ptr)
    return output_arr.tobytes()

def run_performance_test():
    # Generate 100 MB payload
    size_mb = 100
    num_bytes = size_mb * 1024 * 1024
    print(f"Generating {size_mb} MB payload...")
    payload = ''.join(random.choices(string.ascii_letters + string.digits, k=num_bytes))
    payload_bytes = payload.encode('utf-8')
    print(f"Payload size: {len(payload_bytes) / (1024 * 1024):.2f} MB")

    # Simple key (no tiling needed in Python now)
    key_bytes = b"secret"
    
    # Test parameters
    iterations = 5
    enc_times = []
    dec_times = []

    print(f"\nRunning {iterations} iterations...")
    for i in range(iterations):
        # Encryption
        start = time.time()
        encrypted = vectorized_xor_encrypt(payload_bytes, key_bytes)
        enc_time = time.time() - start
        enc_times.append(enc_time)

        # Decryption (XOR is symmetric)
        start = time.time()
        decrypted = vectorized_xor_encrypt(encrypted, key_bytes)
        dec_time = time.time() - start
        dec_times.append(dec_time)

        # Verify
        if decrypted != payload_bytes:
            print(f"❌ Iteration {i+1}: Decryption failed!")
            return
        
        print(f"Iteration {i+1}: Enc {enc_time:.4f}s, Dec {dec_time:.4f}s")

    # Calculate metrics
    avg_enc = sum(enc_times) / iterations
    avg_dec = sum(dec_times) / iterations
    throughput_enc = (num_bytes / (1024 * 1024)) / avg_enc  # MB/s
    throughput_dec = (num_bytes / (1024 * 1024)) / avg_dec  # MB/s

    # Print results
    print("\nPerformance Metrics:")
    print(f"Avg Encryption Time: {avg_enc:.4f} seconds")
    print(f"Avg Decryption Time: {avg_dec:.4f} seconds")
    print(f"Encryption Throughput: {throughput_enc:.2f} MB/s")
    print(f"Decryption Throughput: {throughput_dec:.2f} MB/s")
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    run_performance_test()