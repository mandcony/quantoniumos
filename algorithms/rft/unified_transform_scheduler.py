#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Unified Transform Scheduler
============================

Schedules RFT transforms across all available backends:
  1. Python (variants/registry.py) - Always available, production-ready
  2. C/ASM (kernels/compiled/libquantum_symbolic.so) - Optional acceleration
  3. C++ Native (src/rftmw_native) - Optional SIMD acceleration

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │            UnifiedTransformScheduler                    │
  │                                                         │
  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
  │  │  Python     │  │  C/ASM      │  │  C++ Native │     │
  │  │  Variants   │  │  Kernel     │  │  RFTMW      │     │
  │  │  (14 all)   │  │  (unitary)  │  │  (AVX2/512) │     │
  │  └─────────────┘  └─────────────┘  └─────────────┘     │
  │       ✅              ⚠️ N≤8          ⚠️ optional       │
  └─────────────────────────────────────────────────────────┘

Usage:
    from algorithms.rft.unified_transform_scheduler import UnifiedTransformScheduler
    
    scheduler = UnifiedTransformScheduler()
    result = scheduler.forward(signal, variant='original')
    
    # Or run benchmarks across all backends
    scheduler.benchmark_all()
"""

import os
import sys
import time
import numpy as np
from typing import Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


class Backend(Enum):
    """Available transform backends"""
    PYTHON = "python"      # Pure Python (always works)
    C_ASM = "c_asm"        # C/ASM kernels (libquantum_symbolic.so)
    CPP_NATIVE = "cpp"     # C++ native (rftmw_native module)


@dataclass
class BackendStatus:
    """Status of a backend"""
    available: bool
    version: str
    accuracy: str  # "production", "limited", "untested"
    max_size: int  # Maximum reliable transform size
    performance: float  # Relative performance (1.0 = baseline)


class UnifiedTransformScheduler:
    """
    Unified scheduler for RFT transforms across Python, C/ASM, and C++ backends.
    
    Automatically selects the best available backend based on:
    - Availability (is the library loaded?)
    - Accuracy requirements (production vs experimental)
    - Transform size (some backends fail at large sizes)
    - Performance targets
    """
    
    def __init__(self, prefer_native: bool = False):
        """
        Initialize the unified scheduler.
        
        Args:
            prefer_native: If True, try C/ASM first when available
        """
        self.prefer_native = prefer_native
        self.backends: Dict[Backend, BackendStatus] = {}
        self.variant_generators: Dict[str, Callable] = {}
        
        # Initialize all backends
        self._init_python_backend()
        self._init_c_asm_backend()
        self._init_cpp_native_backend()
        
        # Report status
        self._print_status()
    
    def _init_python_backend(self):
        """Initialize Python variant generators (always available)"""
        try:
            from algorithms.rft.variants.registry import VARIANTS
            
            for key, info in VARIANTS.items():
                self.variant_generators[key] = info.generator
            
            self.backends[Backend.PYTHON] = BackendStatus(
                available=True,
                version="1.0",
                accuracy="production",
                max_size=65536,  # Limited by memory, not algorithm
                performance=1.0
            )
        except ImportError as e:
            self.backends[Backend.PYTHON] = BackendStatus(
                available=False,
                version="N/A",
                accuracy="N/A",
                max_size=0,
                performance=0.0
            )
            print(f"⚠️ Python backend unavailable: {e}")
    
    def _init_c_asm_backend(self):
        """Initialize C/ASM kernel (optional)"""
        try:
            from algorithms.rft.kernels.python_bindings.unitary_rft import (
                UnitaryRFT, RFT_FLAG_UNITARY, RFT_VARIANT_STANDARD
            )
            
            # Test if library actually loads
            test_rft = UnitaryRFT(8, RFT_FLAG_UNITARY, RFT_VARIANT_STANDARD)
            if test_rft.engine is not None:
                self.backends[Backend.C_ASM] = BackendStatus(
                    available=True,
                    version="1.0",
                    accuracy="limited",  # Known issues for N > 8
                    max_size=8,  # Only accurate up to N=8
                    performance=0.01  # Currently slower than Python
                )
                self._c_asm_class = UnitaryRFT
            else:
                raise RuntimeError("Engine is None")
                
        except Exception as e:
            self.backends[Backend.C_ASM] = BackendStatus(
                available=False,
                version="N/A",
                accuracy="N/A",
                max_size=0,
                performance=0.0
            )
    
    def _init_cpp_native_backend(self):
        """Initialize C++ native module (optional)"""
        try:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'rftmw_native', 'build'))
            import rftmw_native
            
            self.backends[Backend.CPP_NATIVE] = BackendStatus(
                available=True,
                version=getattr(rftmw_native, '__version__', '1.0'),
                accuracy="production",
                max_size=65536,
                performance=5.0  # Expected 5x speedup with AVX2
            )
            self._cpp_module = rftmw_native
            
        except ImportError:
            self.backends[Backend.CPP_NATIVE] = BackendStatus(
                available=False,
                version="N/A",
                accuracy="N/A",
                max_size=0,
                performance=0.0
            )
    
    def _print_status(self):
        """Print backend status summary"""
        print("\n" + "="*60)
        print(" Unified Transform Scheduler - Backend Status")
        print("="*60)
        
        for backend, status in self.backends.items():
            icon = "✅" if status.available else "❌"
            print(f"{icon} {backend.value:12} | "
                  f"accuracy={status.accuracy:10} | "
                  f"max_n={status.max_size:5} | "
                  f"perf={status.performance:.2f}x")
        
        print(f"\nVariants loaded: {len(self.variant_generators)}")
        print("="*60 + "\n")
    
    def select_backend(self, n: int, require_accuracy: bool = True) -> Backend:
        """
        Select the best backend for a given transform size.
        
        Args:
            n: Transform size
            require_accuracy: If True, only use production-accurate backends
        
        Returns:
            Best available backend
        """
        # Filter by availability and accuracy
        candidates = []
        for backend, status in self.backends.items():
            if not status.available:
                continue
            if require_accuracy and status.accuracy != "production":
                continue
            if n > status.max_size:
                continue
            candidates.append((backend, status))
        
        if not candidates:
            # Fall back to Python (always works)
            return Backend.PYTHON
        
        # Sort by performance (highest first)
        candidates.sort(key=lambda x: x[1].performance, reverse=True)
        
        # If prefer_native and native is available, use it
        if self.prefer_native:
            for backend, status in candidates:
                if backend in (Backend.C_ASM, Backend.CPP_NATIVE):
                    return backend
        
        return candidates[0][0]
    
    def get_basis(self, n: int, variant: str = 'original') -> np.ndarray:
        """
        Get the unitary basis matrix for a variant.
        
        Args:
            n: Transform size
            variant: Variant key (e.g., 'original', 'h3_cascade')
        
        Returns:
            Unitary basis matrix (n x n)
        """
        if variant not in self.variant_generators:
            raise ValueError(f"Unknown variant: {variant}. "
                           f"Available: {list(self.variant_generators.keys())}")
        
        return self.variant_generators[variant](n)
    
    def forward(self, x: np.ndarray, variant: str = 'original',
                backend: Optional[Backend] = None) -> np.ndarray:
        """
        Perform forward RFT transform.
        
        Args:
            x: Input signal (complex array)
            variant: RFT variant to use
            backend: Force specific backend (auto-select if None)
        
        Returns:
            Transformed coefficients
        """
        n = len(x)
        x = np.asarray(x, dtype=np.complex128)
        
        if backend is None:
            backend = self.select_backend(n)
        
        if backend == Backend.PYTHON:
            basis = self.get_basis(n, variant)
            return basis.conj().T @ x
        
        elif backend == Backend.C_ASM:
            if not self.backends[Backend.C_ASM].available:
                return self.forward(x, variant, Backend.PYTHON)
            rft = self._c_asm_class(n)
            return rft.forward(x)
        
        elif backend == Backend.CPP_NATIVE:
            if not self.backends[Backend.CPP_NATIVE].available:
                return self.forward(x, variant, Backend.PYTHON)
            return self._cpp_module.forward(x, variant)
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def inverse(self, X: np.ndarray, variant: str = 'original',
                backend: Optional[Backend] = None) -> np.ndarray:
        """
        Perform inverse RFT transform.
        
        Args:
            X: Transform coefficients
            variant: RFT variant to use
            backend: Force specific backend (auto-select if None)
        
        Returns:
            Reconstructed signal
        """
        n = len(X)
        X = np.asarray(X, dtype=np.complex128)
        
        if backend is None:
            backend = self.select_backend(n)
        
        if backend == Backend.PYTHON:
            basis = self.get_basis(n, variant)
            return basis @ X
        
        elif backend == Backend.C_ASM:
            if not self.backends[Backend.C_ASM].available:
                return self.inverse(X, variant, Backend.PYTHON)
            rft = self._c_asm_class(n)
            return rft.inverse(X)
        
        elif backend == Backend.CPP_NATIVE:
            if not self.backends[Backend.CPP_NATIVE].available:
                return self.inverse(X, variant, Backend.PYTHON)
            return self._cpp_module.inverse(X, variant)
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def benchmark_all(self, sizes: list = None, variants: list = None,
                      iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark all available backends.
        
        Args:
            sizes: Transform sizes to test
            variants: Variants to test
            iterations: Number of iterations per test
        
        Returns:
            Benchmark results dictionary
        """
        if sizes is None:
            sizes = [64, 128, 256, 512, 1024]
        if variants is None:
            variants = ['original', 'h3_cascade', 'fh5_entropy']
        
        results = {
            'sizes': sizes,
            'variants': variants,
            'backends': {},
            'unitarity': {},
            'reconstruction': {}
        }
        
        print("\n" + "="*70)
        print(" UNIFIED TRANSFORM BENCHMARK")
        print("="*70)
        
        for backend in Backend:
            status = self.backends[backend]
            if not status.available:
                continue
            
            results['backends'][backend.value] = {
                'times_ms': {},
                'speedup': {}
            }
            
            print(f"\n[{backend.value.upper()}]")
            
            for n in sizes:
                if n > status.max_size and status.accuracy == "limited":
                    print(f"  N={n:4}: SKIPPED (max_size={status.max_size})")
                    continue
                
                for variant in variants:
                    if variant not in self.variant_generators:
                        continue
                    
                    # Generate test signal
                    np.random.seed(42)
                    x = np.random.randn(n) + 1j * np.random.randn(n)
                    x /= np.linalg.norm(x)
                    
                    # Time forward + inverse
                    start = time.perf_counter()
                    for _ in range(iterations):
                        X = self.forward(x, variant, backend)
                        x_rec = self.inverse(X, variant, backend)
                    elapsed = (time.perf_counter() - start) / iterations * 1000
                    
                    # Check reconstruction error
                    error = np.linalg.norm(x_rec - x) / np.linalg.norm(x)
                    
                    key = f"n={n}_{variant}"
                    results['backends'][backend.value]['times_ms'][key] = elapsed
                    results['reconstruction'][f"{backend.value}_{key}"] = float(error)
                    
                    status_icon = "✅" if error < 1e-10 else "⚠️" if error < 1e-5 else "❌"
                    print(f"  N={n:4} {variant:15}: {elapsed:7.3f}ms  "
                          f"err={error:.2e} {status_icon}")
        
        # Compute speedups relative to Python
        if Backend.PYTHON.value in results['backends']:
            py_times = results['backends'][Backend.PYTHON.value]['times_ms']
            for backend_name, data in results['backends'].items():
                if backend_name == Backend.PYTHON.value:
                    continue
                for key, time_ms in data['times_ms'].items():
                    if key in py_times:
                        data['speedup'][key] = py_times[key] / time_ms
        
        print("\n" + "="*70)
        print(" SUMMARY")
        print("="*70)
        
        # Check unitarity for each variant
        print("\nUnitarity Check (Python backend):")
        for variant in variants[:3]:  # Test first 3
            if variant not in self.variant_generators:
                continue
            basis = self.get_basis(64, variant)
            identity = basis.conj().T @ basis
            error = np.linalg.norm(identity - np.eye(64))
            results['unitarity'][variant] = float(error)
            icon = "✅" if error < 1e-10 else "❌"
            print(f"  {variant:20}: ||Ψ†Ψ - I|| = {error:.2e} {icon}")
        
        return results


# =============================================================================
# MAIN - Run when executed directly
# =============================================================================

if __name__ == "__main__":
    print("QuantoniumOS Unified Transform Scheduler")
    print("="*50)
    
    scheduler = UnifiedTransformScheduler()
    
    # Run benchmarks
    results = scheduler.benchmark_all(
        sizes=[64, 256, 1024],
        variants=['original', 'h3_cascade'],
        iterations=50
    )
    
    # Quick sanity check
    print("\n" + "="*50)
    print("Quick Sanity Check:")
    print("="*50)
    
    x = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)
    X = scheduler.forward(x, 'original')
    x_rec = scheduler.inverse(X, 'original')
    
    print(f"Input:        {x[:4]}...")
    print(f"Coefficients: {np.abs(X[:4])}...")
    print(f"Reconstructed:{x_rec[:4]}...")
    print(f"Error:        {np.linalg.norm(x_rec - x):.2e}")
    
    print("\n✅ Unified Transform Scheduler ready!")
