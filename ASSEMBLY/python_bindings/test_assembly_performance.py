#!/usr/bin/env python3
"""
Assembly Performance Validation Test
Tests if the        # 3. Assembly RFT (if available)
        if assembly_available:
            try:
                # Define C structures properly
                class RFTComplex(ctypes.Structure):
                    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]
                
                class RFTEngine(ctypes.Structure):
                    _fields_ = [
                        ("size", ctypes.c_size_t),
                        ("basis", ctypes.POINTER(RFTComplex)),
                        ("eigenvalues", ctypes.POINTER(ctypes.c_double)),
                        ("initialized", ctypes.c_bool),
                        ("flags", ctypes.c_uint32),
                        ("qubit_count", ctypes.c_size_t),
                        ("quantum_context", ctypes.c_void_p)
                    ]
                
                # Set up function signatures correctly
                rft_lib.rft_init.argtypes = [ctypes.POINTER(RFTEngine), ctypes.c_size_t, ctypes.c_uint32]
                rft_lib.rft_init.restype = ctypes.c_int
                rft_lib.rft_forward.argtypes = [ctypes.POINTER(RFTEngine), ctypes.POINTER(RFTComplex), 
                                               ctypes.POINTER(RFTComplex), ctypes.c_size_t]
                rft_lib.rft_forward.restype = ctypes.c_int
                rft_lib.rft_cleanup.argtypes = [ctypes.POINTER(RFTEngine)]
                rft_lib.rft_cleanup.restype = ctypes.c_int
                
                # Initialize RFT engine
                engine = RFTEngine()
                result = rft_lib.rft_init(ctypes.byref(engine), size, 0)
                if result != 0:
                    raise Exception(f"Init failed with code {result}")
                
                # Convert data to C arrays
                input_c = (RFTComplex * size)()
                output_c = (RFTComplex * size)()
                for i in range(size):
                    input_c[i].real = data[i].real
                    input_c[i].imag = data[i].imag
                
                start = time.perf_counter()
                for _ in range(100):
                    rft_lib.rft_forward(ctypes.byref(engine), input_c, output_c, size)
                assembly_time = (time.perf_counter() - start) / 100
                
                rft_lib.rft_cleanup(ctypes.byref(engine))tions are actually providing speedup
"""

import time
import numpy as np
import ctypes
import os
from scipy.fft import fft

def load_assembly_dll():
    """Load the assembly-optimized RFT kernel"""
    dll_path = "../build/rftkernel.dll"
    if not os.path.exists(dll_path):
        dll_path = "../compiled/rftkernel.dll"
    
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"RFT kernel DLL not found")
    
    lib = ctypes.cdll.LoadLibrary(dll_path)
    
    # Set up function signatures
    lib.rft_init.argtypes = [ctypes.c_int]
    lib.rft_init.restype = ctypes.c_void_p
    
    lib.rft_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.rft_forward.restype = ctypes.c_int
    
    lib.rft_cleanup.argtypes = [ctypes.c_void_p]
    lib.rft_cleanup.restype = None
    
    return lib

def benchmark_assembly_vs_python():
    """Benchmark assembly RFT vs pure Python vs NumPy FFT"""
    
    print("🚀 Assembly Performance Validation Test")
    print("=" * 50)
    
    sizes = [64, 128, 256, 512, 1024]
    results = {}
    
    try:
        rft_lib = load_assembly_dll()
        assembly_available = True
        print("✅ Assembly RFT kernel loaded")
    except Exception as e:
        print(f"❌ Assembly RFT not available: {e}")
        assembly_available = False
    
    for size in sizes:
        print(f"\n📊 Testing size {size}:")
        
        # Generate test data
        np.random.seed(42)
        data = np.random.randn(size) + 1j * np.random.randn(size)
        
        results[size] = {}
        
        # 1. NumPy FFT baseline
        start = time.perf_counter()
        for _ in range(100):
            fft_result = fft(data)
        numpy_time = (time.perf_counter() - start) / 100
        results[size]['numpy_fft'] = numpy_time
        print(f"   NumPy FFT: {numpy_time*1000:.3f} ms")
        
        # 2. Pure Python RFT (O(N²))
        start = time.perf_counter()
        result = np.zeros(size, dtype=complex)
        for k in range(size):
            for n in range(size):
                result[k] += data[n] * np.exp(-2j * np.pi * k * n / size)
        python_time = time.perf_counter() - start
        results[size]['python_rft'] = python_time
        print(f"   Python RFT: {python_time*1000:.3f} ms")
        
        # 3. Assembly RFT (if available)
        if assembly_available:
            try:
                # Initialize RFT context
                rft_ctx = rft_lib.rft_init(size)
                
                # Convert data to C arrays
                real_data = (ctypes.c_double * size)(*data.real)
                imag_data = (ctypes.c_double * size)(*data.imag)
                real_output = (ctypes.c_double * size)()
                imag_output = (ctypes.c_double * size)()
                
                start = time.perf_counter()
                for _ in range(100):
                    rft_lib.rft_forward(rft_ctx, real_data, imag_data)
                assembly_time = (time.perf_counter() - start) / 100
                
                rft_lib.rft_cleanup(rft_ctx)
                
                results[size]['assembly_rft'] = assembly_time
                print(f"   Assembly RFT: {assembly_time*1000:.3f} ms")
                
                # Calculate speedups
                speedup_vs_numpy = numpy_time / assembly_time
                speedup_vs_python = python_time / assembly_time
                
                print(f"   🔥 Speedup vs NumPy: {speedup_vs_numpy:.1f}x")
                print(f"   🔥 Speedup vs Python: {speedup_vs_python:.1f}x")
                
            except Exception as e:
                print(f"   ❌ Assembly test failed: {e}")
    
    # Summary
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("=" * 50)
    
    for size in sizes:
        if 'assembly_rft' in results[size]:
            numpy_speedup = results[size]['numpy_fft'] / results[size]['assembly_rft']
            python_speedup = results[size]['python_rft'] / results[size]['assembly_rft']
            print(f"Size {size:4d}: {numpy_speedup:5.1f}x vs NumPy, {python_speedup:5.1f}x vs Python")
    
    return results

if __name__ == "__main__":
    benchmark_assembly_vs_python()
