# QuantoniumOS - Copilot Instructions

## **What This Actually Is:**
QuantoniumOS is a **research prototype** for symbolic quantum-inspired compression techniques. It runs entirely on classical CPUs - there is NO quantum hardware involved. The project's core is a hybrid system integrating Python for high-level logic and C for performance-critical kernels.

## The Big Picture: System Architecture

The system is layered, with clear separation of concerns. Understanding this structure is key.

```
AI Models (e.g., tiny-gpt2) → Enhanced Pipeline → Desktop Environment → Core Algorithms → C Assembly Kernels
```

- **AI Layer (`ai/`)**: Manages HuggingFace models and the quantum compression framework. The only fully verified model is `sshleifer/tiny-gpt2`.
- **Core Layer (`src/core/`)**: Contains the primary Python implementations of the mathematical algorithms. This is the heart of the system's novel mathematics.
- **Assembly Layer (`src/assembly/`)**: Holds performance-critical C kernels with SIMD optimizations. These are called from Python.
- **Frontend Layer (`src/frontend/`)**: A PyQt5-based desktop environment that provides a GUI for the various applications.
- **Applications (`src/apps/`)**: Standalone tools, like the `quantum_simulator.py`, that are dynamically loaded into the desktop environment.

## Critical Developer Workflows

### Building Assembly Kernels
The C kernels provide significant performance improvements. They are built using a standard `Makefile`.

```bash
# Navigate to the assembly directory
cd src/assembly

# Build the C kernels and create shared libraries
make all

# Install the Python bindings
make install
```
**Note:** The system is designed to gracefully fall back to pure Python implementations if the C kernels are not built or fail to load.

### Running the System
- **Full System Boot**: To launch the entire desktop environment with all applications.
  ```bash
  python quantonium_boot.py
  ```
- **Individual Application**: To run a single application, like the quantum simulator.
  ```bash
  python src/apps/quantum_simulator.py
  ```

### Testing
The project has a comprehensive test suite.
- **Run all validations**:
  ```bash
  python tests/comprehensive_validation_suite.py
  ```
- **Run specific test suites**:
  ```bash
  python tests/crypto/crypto_performance_test.py
  python QUANTONIUM_BENCHMARK_SUITE.py
  ```

## Project Conventions & Patterns

### Verified vs. Unverified Claims
Be skeptical of claims in documentation. The project contains many experimental ideas. **Only the following are verified and working:**
- **Symbolic RFT**: `src/core/canonical_true_rft.py` (Unitarity error < 1e-12).
- **Vertex Codec**: `src/core/rft_vertex_codec.py`.
- **Tested Model**: `sshleifer/tiny-gpt2` is the only model with complete encode/decode tests.

### Python/C Integration & Graceful Fallback
A critical pattern is the dynamic check for the compiled C library (`unitary_rft`) and falling back to the Python version if it's not available.

```python
# Example from src/apps/quantum_simulator.py
try:
    # Attempt to import the fast C kernel
    from unitary_rft import UnitaryRFT, RFT_FLAG_QUANTUM_SAFE
    RFT_AVAILABLE = True
except ImportError:
    # Fallback to the pure Python implementation
    from src.core.canonical_true_rft import CanonicalTrueRFT
    RFT_AVAILABLE = False

# Usage:
if RFT_AVAILABLE:
    engine = UnitaryRFT(size)
else:
    engine = CanonicalTrueRFT(size)
```

### Application Integration
Desktop applications are not launched as separate processes. They are Python classes that extend a base class and are dynamically imported by the main desktop manager using `importlib`.
- **Base Class**: `src/apps/launcher_base.py`
- **Desktop Manager**: `src/frontend/quantonium_desktop.py`

### Mathematical Precision
- All RFT-related operations must maintain a **unitarity error below 1e-12**.
- Use the golden ratio constant: `phi = (1 + 5**0.5) / 2`.
- Always validate the unitarity of transform matrices after construction.

```python
# RFT Validation Pattern
unitarity_error = np.linalg.norm(Psi.conj().T @ Psi - np.identity(N), ord=2)
assert unitarity_error < 1e-12, "RFT matrix must be unitary"
```

### Key File Locations
- **Core RFT Algorithm**: `src/core/canonical_true_rft.py`
- **RFT C Kernel**: `src/assembly/kernel/rft_kernel.c`
- **Quantum Simulator**: `src/apps/quantum_simulator.py`
- **Main System Boot**: `quantonium_boot.py`
- **Validation Suite**: `tests/comprehensive_validation_suite.py`