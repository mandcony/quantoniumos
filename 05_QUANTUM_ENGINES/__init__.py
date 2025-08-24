"""
QuantoniumOS Quantum Engines Package

This package contains the core quantum computing engines and kernels
for QuantoniumOS, including topological quantum algorithms and 
bulletproof quantum kernel implementations.

Modules:
- bulletproof_quantum_kernel: Core production quantum kernel
- topological_quantum_kernel: Topological quantum computing engine
- working_quantum_kernel: Development quantum kernel
"""

# Import key quantum engines for easier access
try:
    __all__ = [
        "BulletproofQuantumKernel",
        "WorkingQuantumKernel",
        "TopologicalQuantumKernel",
    ]
except ImportError:
    # Graceful fallback if modules aren't available
    __all__ = []

__version__ = "1.0.0"
