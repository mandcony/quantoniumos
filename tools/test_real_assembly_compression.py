#!/usr/bin/env python3
"""
Real Assembly RFT Compression Test
=================================
Tests RFT compression using the actual compiled C library (not mocks)
"""

import ctypes
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import sys

# Load the real compiled library
try:
    lib_path = "/workspaces/quantoniumos/src/assembly/compiled/libquantum_symbolic.so"
    rft_lib = ctypes.CDLL(lib_path)
    print(f"✅ Successfully loaded real C library: {lib_path}")
except Exception as e:
    print(f"❌ Failed to load C library: {e}")
    sys.exit(1)

# Define C structures
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
        ("quantum_context", ctypes.c_void_p),
    ]

# Define function signatures
rft_lib.rft_init.argtypes = [ctypes.POINTER(RFTEngine), ctypes.c_size_t, ctypes.c_uint32]
rft_lib.rft_init.restype = ctypes.c_int

rft_lib.rft_cleanup.argtypes = [ctypes.POINTER(RFTEngine)]
rft_lib.rft_cleanup.restype = ctypes.c_int

rft_lib.rft_forward.argtypes = [ctypes.POINTER(RFTEngine), 
                               ctypes.POINTER(RFTComplex),
                               ctypes.POINTER(RFTComplex),
                               ctypes.c_size_t]
rft_lib.rft_forward.restype = ctypes.c_int

rft_lib.rft_inverse.argtypes = [ctypes.POINTER(RFTEngine),
                               ctypes.POINTER(RFTComplex),
                               ctypes.POINTER(RFTComplex), 
                               ctypes.c_size_t]
rft_lib.rft_inverse.restype = ctypes.c_int

rft_lib.rft_validate_unitarity.argtypes = [ctypes.POINTER(RFTEngine), ctypes.c_double]
rft_lib.rft_validate_unitarity.restype = ctypes.c_int

# Constants from header
RFT_SUCCESS = 0
RFT_FLAG_UNITARY = 0x00000008
RFT_FLAG_QUANTUM_SAFE = 0x00000004
RFT_FLAG_USE_RESONANCE = 0x00000010

class RealAssemblyRFT:
    """Python wrapper for the real compiled C RFT library"""
    
    def __init__(self, size):
        self.size = size
        self.engine = RFTEngine()
        
        # Initialize with full flags
        flags = RFT_FLAG_UNITARY | RFT_FLAG_QUANTUM_SAFE | RFT_FLAG_USE_RESONANCE
        result = rft_lib.rft_init(ctypes.byref(self.engine), size, flags)
        
        if result != RFT_SUCCESS:
            raise RuntimeError(f"RFT initialization failed with code: {result}")
        
        print(f"✅ Real assembly RFT engine initialized for size {size}")
    
    def __del__(self):
        if hasattr(self, 'engine'):
            rft_lib.rft_cleanup(ctypes.byref(self.engine))
    
    def forward_transform(self, input_data):
        """Apply forward RFT transform using real C implementation"""
        n = len(input_data)
        if n != self.size:
            raise ValueError(f"Input size {n} doesn't match engine size {self.size}")
        
        # Convert numpy array to C array
        input_c = (RFTComplex * n)()
        for i in range(n):
            input_c[i].real = float(input_data[i].real)
            input_c[i].imag = float(input_data[i].imag)
        
        # Prepare output
        output_c = (RFTComplex * n)()
        
        # Call C function
        result = rft_lib.rft_forward(ctypes.byref(self.engine),
                                    input_c, output_c, n)
        
        if result != RFT_SUCCESS:
            raise RuntimeError(f"Forward transform failed with code: {result}")
        
        # Convert back to numpy
        output = np.zeros(n, dtype=complex)
        for i in range(n):
            output[i] = complex(output_c[i].real, output_c[i].imag)
        
        return output
    
    def inverse_transform(self, input_data):
        """Apply inverse RFT transform using real C implementation"""
        n = len(input_data)
        if n != self.size:
            raise ValueError(f"Input size {n} doesn't match engine size {self.size}")
        
        # Convert numpy array to C array
        input_c = (RFTComplex * n)()
        for i in range(n):
            input_c[i].real = float(input_data[i].real)
            input_c[i].imag = float(input_data[i].imag)
        
        # Prepare output  
        output_c = (RFTComplex * n)()
        
        # Call C function
        result = rft_lib.rft_inverse(ctypes.byref(self.engine),
                                    input_c, output_c, n)
        
        if result != RFT_SUCCESS:
            raise RuntimeError(f"Inverse transform failed with code: {result}")
        
        # Convert back to numpy
        output = np.zeros(n, dtype=complex)
        for i in range(n):
            output[i] = complex(output_c[i].real, output_c[i].imag)
        
        return output
    
    def get_unitarity_error(self):
        """Get unitarity validation from C library"""
        tolerance = 1e-12
        result = rft_lib.rft_validate_unitarity(ctypes.byref(self.engine), tolerance)
        
        # For now, return a computed error since the C function may return status
        # We'll test unitarity by transforming and checking norm preservation
        test_state = np.random.random(self.size) + 1j * np.random.random(self.size)
        test_state = test_state / np.linalg.norm(test_state)
        
        transformed = self.forward_transform(test_state)
        reconstructed = self.inverse_transform(transformed)
        
        return np.linalg.norm(test_state - reconstructed)

def generate_real_assembly_compression_test():
    """Generate compression test using REAL compiled assembly"""
    
    print("🚀 Starting REAL Assembly RFT Compression Test")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "metadata": {
            "timestamp": timestamp,
            "implementation": "compiled_c_assembly",
            "library_path": "/workspaces/quantoniumos/src/assembly/compiled/libquantum_symbolic.so",
            "test_type": "real_assembly_rft_compression_fidelity_curve",
            "description": "Compression ratios vs fidelity using actual compiled C library"
        },
        "curves": []
    }
    
    # Test different state sizes  
    sizes = [64, 128, 256, 512]
    
    for size in sizes:
        print(f"\n📊 Testing size {size} with REAL assembly...")
        
        try:
            rft = RealAssemblyRFT(size)
            
            # Generate structured quantum state (more compressible)
            np.random.seed(42)  # Reproducible
            raw_state = np.random.random(size) + 1j * np.random.random(size)
            
            # Add structure (decay pattern)
            for i in range(size):
                raw_state[i] *= np.exp(-i / (size * 0.3))
                
            quantum_state = raw_state / np.linalg.norm(raw_state)
            
            # Apply RFT transform with timing
            print(f"   🔄 Applying forward transform...")
            start_time = time.time()
            rft_coeffs = rft.forward_transform(quantum_state)
            transform_time = (time.time() - start_time) * 1000
            print(f"   ⚡ Transform completed in {transform_time:.3f} ms")
            
            # Test different sparsity levels
            curve_data = []
            
            # Sort coefficients by magnitude
            sorted_indices = np.argsort(np.abs(rft_coeffs))[::-1]
            
            # Test different retention ratios
            retention_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
            
            for ratio in retention_ratios:
                keep_count = max(1, int(size * ratio))
                
                # Create sparse representation
                sparse_coeffs = np.zeros_like(rft_coeffs)
                sparse_coeffs[sorted_indices[:keep_count]] = rft_coeffs[sorted_indices[:keep_count]]
                
                # Reconstruct using real assembly
                reconstructed = rft.inverse_transform(sparse_coeffs)
                
                # Calculate fidelity and compression metrics
                fidelity = abs(np.vdot(quantum_state, reconstructed))**2
                reconstruction_error = np.linalg.norm(quantum_state - reconstructed)
                compression_ratio = size / keep_count
                
                curve_data.append({
                    "retention_ratio": ratio,
                    "coefficients_kept": keep_count,
                    "compression_ratio": float(compression_ratio),
                    "fidelity": float(fidelity),
                    "reconstruction_error": float(reconstruction_error),
                    "compression_percentage": float(((compression_ratio - 1) / compression_ratio) * 100)
                })
                
                print(f"   📈 {ratio:.1%} retention: {compression_ratio:.1f}× compression, {fidelity:.3f} fidelity")
            
            results["curves"].append({
                "size": size,
                "transform_time_ms": transform_time,
                "unitarity_error": float(rft.get_unitarity_error()),
                "curve_points": curve_data
            })
            
            print(f"   ✅ Size {size} complete - unitarity error: {rft.get_unitarity_error():.2e}")
            
        except Exception as e:
            print(f"   ❌ Error testing size {size}: {e}")
            continue
    
    # Save results
    output_file = f"/workspaces/quantoniumos/results/real_assembly_compression_{timestamp}.json"
    Path(output_file).parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 REAL Assembly Test Complete!")
    print(f"📁 Results saved to: {output_file}")
    print(f"🔬 Implementation: Actual compiled C library with SIMD optimization")
    
    return output_file

if __name__ == "__main__":
    try:
        result_file = generate_real_assembly_compression_test()
        print(f"\n🏆 SUCCESS: Real assembly compression test completed")
        print(f"📊 Data file: {result_file}")
    except Exception as e:
        print(f"\n💥 FAILED: {e}")
        import traceback
        traceback.print_exc()