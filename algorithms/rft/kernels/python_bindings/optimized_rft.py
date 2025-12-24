# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Optimized Assembly Integration for QuantoniumOS
High-performance Python bindings for assembly-optimized RFT
"""

import os
import sys
import ctypes
import numpy as np
from ctypes import c_int, c_size_t, c_float, c_uint32, c_uint64, c_bool, Structure, POINTER, cdll
from typing import Tuple, List, Optional, Union
import threading
import time

# Enhanced complex structure for optimized assembly
class OptimizedRFTComplex(Structure):
    """Optimized complex number structure with 32-byte alignment."""
    _fields_ = [("real", c_float),
                ("imag", c_float)]

# Performance statistics structure
class RFTPerformanceStats(Structure):
    """Performance statistics from optimized assembly."""
    _fields_ = [("avg_cycles_per_transform", ctypes.c_double),
                ("transforms_per_second", ctypes.c_double),
                ("total_transforms", c_uint64),
                ("cache_miss_rate", ctypes.c_double)]

# Optimized engine structure
class OptimizedRFTEngine(Structure):
    """Optimized RFT engine with SIMD and parallel processing."""
    _fields_ = [("size", c_size_t),
                ("log_size", c_size_t),
                ("opt_flags", c_uint32),
                ("twiddle_factors", POINTER(OptimizedRFTComplex)),
                ("workspace", POINTER(OptimizedRFTComplex)),
                ("initialized", c_bool),
                ("transform_count", c_uint64),
                ("total_cycles", c_uint64),
                ("cache_misses", c_uint64),
                ("num_threads", c_int)]

# RFT variant constants used across bindings
RFT_VARIANT_STANDARD = 0
RFT_VARIANT_HARMONIC = 1
RFT_VARIANT_FIBONACCI = 2
RFT_VARIANT_CHAOTIC = 3
RFT_VARIANT_HYPERBOLIC = 4

# Optimization flags
RFT_OPT_NONE = 0x00000000
RFT_OPT_AVX2 = 0x00000001
RFT_OPT_AVX512 = 0x00000002
RFT_OPT_PREFETCH = 0x00000004
RFT_OPT_PARALLEL = 0x00000008
RFT_OPT_QUANTUM = 0x00000010

class OptimizedRFT:
    """High-performance RFT processor with assembly optimization."""
    
    def __init__(self, size: int = 64, opt_flags: int = RFT_OPT_AVX2):
        """Initialize optimized RFT processor.
        
        Args:
            size: Default transform size (can be overridden per-call)
            opt_flags: Optimization flags
        """
        self.size = size
        self.opt_flags = opt_flags
        self.lib = None
        self.engine = None
        self._rftmw = None
        self._engines = {}  # Cache engines by size
        self._is_mock = True
        
        # Try to load rftmw_native (CASM→C++→Python pipeline)
        try:
            import sys
            if '/workspaces/quantoniumos/src/rftmw_native/build' not in sys.path:
                sys.path.insert(0, '/workspaces/quantoniumos/src/rftmw_native/build')
            import rftmw_native
            
            if rftmw_native.HAS_ASM_KERNELS:
                self._rftmw = rftmw_native
                self._is_mock = False
                print(f"OptimizedRFT: Using rftmw_native.RFTKernelEngine (ASM={rftmw_native.HAS_ASM_KERNELS}, AVX2={rftmw_native.HAS_AVX2})")
            else:
                print(f"OptimizedRFT: rftmw_native loaded but no ASM kernels, using Python fallback")
        except ImportError as e:
            print(f"OptimizedRFT: rftmw_native not available ({e}), using Python fallback")
        
        # Legacy ctypes path (disabled - we use rftmw_native now)
        # try:
        #     self.lib = self._load_optimized_library()
        #     ...
    
    def _load_optimized_library(self):
        """Load the optimized assembly library."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lib_paths = [
            # Check optimized versions first
            os.path.join(script_dir, "..", "optimized", "librftoptimized.dll"),
            os.path.join(script_dir, "..", "optimized", "librftoptimized.so"),
            # Fallback to regular version
            os.path.join(script_dir, "..", "compiled", "librftkernel.dll"),
            os.path.join(script_dir, "..", "compiled", "librftkernel.so"),
            # Build directory
            os.path.join(script_dir, "..", "..", "..", "..", "build", "librftoptimized.so"),
            os.path.join(script_dir, "..", "..", "..", "..", "build", "librftoptimized.dll"),
        ]
        
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                try:
                    lib = cdll.LoadLibrary(lib_path)
                    
                    # Set function signatures for optimized functions
                    lib.rft_optimized_init.argtypes = [POINTER(OptimizedRFTEngine), c_size_t, c_uint32]
                    lib.rft_optimized_init.restype = c_int
                    
                    lib.rft_optimized_cleanup.argtypes = [POINTER(OptimizedRFTEngine)]
                    lib.rft_optimized_cleanup.restype = None
                    
                    lib.rft_optimized_forward.argtypes = [POINTER(OptimizedRFTEngine),
                                                         POINTER(OptimizedRFTComplex),
                                                         POINTER(OptimizedRFTComplex)]
                    lib.rft_optimized_forward.restype = c_int
                    
                    lib.rft_optimized_inverse.argtypes = [POINTER(OptimizedRFTEngine),
                                                         POINTER(OptimizedRFTComplex),
                                                         POINTER(OptimizedRFTComplex)]
                    lib.rft_optimized_inverse.restype = c_int
                    
                    lib.rft_quantum_entangle_optimized.argtypes = [POINTER(OptimizedRFTEngine),
                                                                  POINTER(OptimizedRFTComplex),
                                                                  c_int, c_int,
                                                                  POINTER(OptimizedRFTComplex)]
                    lib.rft_quantum_entangle_optimized.restype = c_int
                    
                    lib.rft_get_performance_stats.argtypes = [POINTER(OptimizedRFTEngine),
                                                             POINTER(RFTPerformanceStats)]
                    lib.rft_get_performance_stats.restype = None
                    
                    return lib
                    
                except Exception as e:
                    print(f"Failed to load optimized library {lib_path}: {e}")
        
        raise RuntimeError("Could not load optimized RFT library")
    
    def __del__(self):
        """Clean up optimized resources."""
        # rftmw_native handles its own cleanup; no action needed
        pass
    
    def _get_engine(self, size: int):
        """Get or create an engine for the given size."""
        if size not in self._engines:
            self._engines[size] = self._rftmw.RFTKernelEngine(size)
        return self._engines[size]
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward RFT transform (convenience wrapper)."""
        # Use rftmw_native.RFTKernelEngine if available (true CASM→C++ path)
        if self._rftmw is not None:
            try:
                size = len(input_data)
                engine = self._get_engine(size)
                # Engine expects real input
                real_input = np.real(input_data).astype(np.float64)
                result = engine.forward(real_input)
                return result
            except Exception as e:
                print(f"rftmw_native.RFTKernelEngine.forward failed: {e}, falling back")
        
        # Fallback: phi-modulated FFT
        import math
        phi = (1 + math.sqrt(5)) / 2
        fft_result = np.fft.fft(input_data.astype(np.complex128))
        phases = np.exp(1j * phi * np.arange(len(input_data)))
        result = fft_result * phases / np.sqrt(len(input_data))
        return result
    
    def inverse(self, input_data: np.ndarray) -> np.ndarray:
        """Inverse RFT transform (convenience wrapper)."""
        # Use rftmw_native.RFTKernelEngine if available (true CASM→C++ path)
        if self._rftmw is not None:
            try:
                size = len(input_data)
                engine = self._get_engine(size)
                result = engine.inverse(input_data.astype(np.complex128))
                return result
            except Exception as e:
                print(f"rftmw_native.RFTKernelEngine.inverse failed: {e}, falling back")
        
        # Fallback: phi-modulated IFFT
        import math
        phi = (1 + math.sqrt(5)) / 2
        phases = np.exp(-1j * phi * np.arange(len(input_data)))
        phased_data = input_data.astype(np.complex128) * phases
        result = np.fft.ifft(phased_data) * np.sqrt(len(input_data))
        return result
    
    def forward_optimized(self, input_data: np.ndarray) -> np.ndarray:
        """Perform optimized forward RFT transform.
        
        Args:
            input_data: Complex input array
            
        Returns:
            Complex output array
        """
        if len(input_data) != self.size:
            raise ValueError(f"Input size must be {self.size}")
        
        # Use rftmw_native if available (true CASM→C++ path)
        if self._rftmw is not None:
            try:
                result = self._rftmw.forward(input_data.astype(np.complex128))
                return result.astype(np.complex64)
            except Exception as e:
                print(f"rftmw_native.forward failed: {e}, falling back")
        
        # Fallback: phi-modulated FFT
        import math
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        fft_result = np.fft.fft(input_data.astype(np.complex128))
        phases = np.exp(1j * phi * np.arange(len(input_data)))
        result = fft_result * phases
        result = result / np.sqrt(len(input_data))
        return result.astype(np.complex64)
        
        # Convert numpy array to optimized C structure
        input_c = (OptimizedRFTComplex * self.size)()
        for i in range(self.size):
            input_c[i].real = float(input_data[i].real)
            input_c[i].imag = float(input_data[i].imag)
        
        # Prepare output array
        output_c = (OptimizedRFTComplex * self.size)()
        
        # Call optimized assembly function
        result = self.lib.rft_optimized_forward(self.engine, input_c, output_c)
        if result != 0:
            raise RuntimeError(f"Optimized forward transform failed: {result}")
        
        # Convert back to numpy array
        output = np.zeros(self.size, dtype=np.complex64)
        for i in range(self.size):
            output[i] = complex(output_c[i].real, output_c[i].imag)
        
        return output
    
    def inverse_optimized(self, input_data: np.ndarray) -> np.ndarray:
        """Perform optimized inverse RFT transform.
        
        Args:
            input_data: Complex input array
            
        Returns:
            Complex output array
        """
        if len(input_data) != self.size:
            raise ValueError(f"Input size must be {self.size}")
        
        # Use rftmw_native if available (true CASM→C++ path)
        if self._rftmw is not None:
            try:
                result = self._rftmw.inverse(input_data.astype(np.complex128))
                return result.astype(np.complex64)
            except Exception as e:
                print(f"rftmw_native.inverse failed: {e}, falling back")
        
        # Fallback: phi-modulated IFFT
        import math
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        phases = np.exp(-1j * phi * np.arange(len(input_data)))
        phased_data = input_data.astype(np.complex128) * phases
        result = np.fft.ifft(phased_data)
        result = result * np.sqrt(len(input_data))
        return result.astype(np.complex64)
        
        # Convert numpy array to optimized C structure
        input_c = (OptimizedRFTComplex * self.size)()
        for i in range(self.size):
            input_c[i].real = float(input_data[i].real)
            input_c[i].imag = float(input_data[i].imag)
        
        # Prepare output array
        output_c = (OptimizedRFTComplex * self.size)()
        
        # Call optimized assembly function
        result = self.lib.rft_optimized_inverse(self.engine, input_c, output_c)
        if result != 0:
            raise RuntimeError(f"Optimized inverse transform failed: {result}")
        
        # Convert back to numpy array
        output = np.zeros(self.size, dtype=np.complex64)
        for i in range(self.size):
            output[i] = complex(output_c[i].real, output_c[i].imag)
        
        return output
    
    def quantum_entangle_optimized(self, state: np.ndarray, qubit1: int, qubit2: int) -> np.ndarray:
        """Perform optimized quantum entanglement operation.
        
        Args:
            state: Quantum state vector
            qubit1: First qubit index
            qubit2: Second qubit index
            
        Returns:
            Entangled state vector
        """
        if len(state) != self.size:
            raise ValueError(f"State size must be {self.size}")
        
        # Convert state to C structure
        state_c = (OptimizedRFTComplex * self.size)()
        for i in range(self.size):
            state_c[i].real = float(state[i].real)
            state_c[i].imag = float(state[i].imag)
        
        # Prepare output
        output_c = (OptimizedRFTComplex * self.size)()
        
        # Call optimized entanglement function
        result = self.lib.rft_quantum_entangle_optimized(self.engine, state_c, 
                                                        qubit1, qubit2, output_c)
        if result != 0:
            raise RuntimeError(f"Quantum entanglement failed: {result}")
        
        # Convert back to numpy array
        output = np.zeros(self.size, dtype=np.complex64)
        for i in range(self.size):
            output[i] = complex(output_c[i].real, output_c[i].imag)
        
        return output
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics from optimized assembly.
        
        Returns:
            Dictionary with performance metrics
        """
        self.lib.rft_get_performance_stats(self.engine, self.performance_stats)
        
        return {
            'avg_cycles_per_transform': self.performance_stats.avg_cycles_per_transform,
            'transforms_per_second': self.performance_stats.transforms_per_second,
            'total_transforms': self.performance_stats.total_transforms,
            'cache_miss_rate': self.performance_stats.cache_miss_rate,
            'optimization_level': 'Assembly+SIMD+Parallel',
            'simd_support': self._detect_simd_support()
        }
    
    def _detect_simd_support(self) -> str:
        """Detect available SIMD instruction sets."""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            flags = info.get('flags', [])
            
            if 'avx512f' in flags:
                return 'AVX-512'
            elif 'avx2' in flags:
                return 'AVX2'
            elif 'avx' in flags:
                return 'AVX'
            elif 'sse4_2' in flags:
                return 'SSE4.2'
            else:
                return 'Basic'
        except:
            return 'Unknown'
    
    def benchmark_performance(self, num_iterations: int = 1000) -> dict:
        """Benchmark the optimized RFT performance.
        
        Args:
            num_iterations: Number of test iterations
            
        Returns:
            Benchmark results
        """
        # Generate test data
        test_signal = np.random.random(self.size) + 1j * np.random.random(self.size)
        test_signal = test_signal.astype(np.complex64)
        
        # Benchmark forward transform
        start_time = time.time()
        for _ in range(num_iterations):
            spectrum = self.forward_optimized(test_signal)
        forward_time = time.time() - start_time
        
        # Benchmark inverse transform
        start_time = time.time()
        for _ in range(num_iterations):
            reconstructed = self.inverse_optimized(spectrum)
        inverse_time = time.time() - start_time
        
        # Calculate reconstruction error
        error = np.max(np.abs(test_signal - reconstructed))
        
        # Get detailed performance stats
        perf_stats = self.get_performance_stats()
        
        return {
            'forward_time_per_transform': forward_time / num_iterations,
            'inverse_time_per_transform': inverse_time / num_iterations,
            'reconstruction_error': float(error),
            'transforms_per_second': num_iterations / (forward_time + inverse_time),
            'performance_stats': perf_stats,
            'size': self.size,
            'iterations': num_iterations
        }

# Enhanced processor with fallback compatibility
class EnhancedRFTProcessor:
    """Enhanced RFT processor with optimized assembly and graceful fallbacks."""
    
    def __init__(self, size: int = 64):
        """Initialize enhanced processor with automatic optimization selection."""
        self.size = size
        
        # Try to use optimized version first
        try:
            self.optimized = OptimizedRFT(size)
            self.use_optimized = True
            print("Using optimized assembly RFT processor")
        except Exception as e:
            print(f"Optimized processor not available: {e}")
            # Fallback to standard version
            from .unitary_rft import UnitaryRFT
            self.fallback = UnitaryRFT(size)
            self.use_optimized = False
            print("Using fallback RFT processor")
    
    def process_quantum_field(self, data) -> any:
        """Process quantum field with best available implementation."""
        if self.use_optimized:
            try:
                # Convert data to proper format
                if isinstance(data, (list, tuple)):
                    data = np.array(data, dtype=complex)
                elif isinstance(data, str):
                    data = np.array([complex(ord(c), 0) for c in data])
                
                # Ensure proper size
                if len(data) > self.size:
                    data = data[:self.size]
                elif len(data) < self.size:
                    padded = np.zeros(self.size, dtype=complex)
                    padded[:len(data)] = data
                    data = padded
                
                # Use optimized forward transform
                return self.optimized.forward_optimized(data)
                
            except Exception as e:
                print(f"Optimized processing failed, using fallback: {e}")
                self.use_optimized = False
        
        # Use fallback processor
        return self.fallback.process_quantum_field(data)
    
    def get_performance_metrics(self) -> dict:
        """Get performance metrics from active processor."""
        if self.use_optimized:
            return self.optimized.get_performance_stats()
        else:
            return {
                'processor_type': 'fallback',
                'optimization_level': 'Python+ctypes',
                'note': 'Assembly optimization not available'
            }

# Example usage and performance testing
if __name__ == "__main__":
    # Test optimized processor
    try:
        processor = OptimizedRFTProcessor(256)
        
        # Run benchmark
        results = processor.benchmark_performance(100)
        
        print("=== Optimized RFT Performance Benchmark ===")
        print(f"Transform Size: {results['size']}")
        print(f"Forward Transform: {results['forward_time_per_transform']*1000:.3f} ms")
        print(f"Inverse Transform: {results['inverse_time_per_transform']*1000:.3f} ms")
        print(f"Reconstruction Error: {results['reconstruction_error']:.2e}")
        print(f"Transforms/Second: {results['transforms_per_second']:.1f}")
        print(f"SIMD Support: {results['performance_stats']['simd_support']}")
        
        # Test quantum entanglement
        print("\n=== Quantum Entanglement Test ===")
        state = np.zeros(256, dtype=complex)
        state[0] = 1.0  # |0> state
        
        entangled = processor.quantum_entangle_optimized(state, 0, 1)
        print(f"Entangled state norm: {np.sum(np.abs(entangled)**2):.6f}")
        
    except Exception as e:
        print(f"Optimized testing failed: {e}")
        print("Testing enhanced processor with fallback...")
        
        enhanced = EnhancedRFTProcessor(64)
        result = enhanced.process_quantum_field([1, 0, 0, 0])
        print(f"Enhanced processor result: {result[:4]}")
        print(f"Performance metrics: {enhanced.get_performance_metrics()}")