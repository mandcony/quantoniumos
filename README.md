# QuantoniumOS

QuantoniumOS is a quantum computing operating system designed to provide a unified interface for quantum algorithms, simulations, and applications.

## Core Components

### 1. ASSEMBLY (REAL RFT Assembly)

The low-level implementation of Resonance Field Theory (RFT), providing direct hardware integration for quantum operations.

- **kernel/** - C/ASM implementation of the RFT kernel
- **compiled/** - Compiled libraries and executables
- **python_bindings/** - Python interface to the RFT kernel
- **build_scripts/** - Scripts to build the RFT kernel

### 2. Quantum Kernels

Three specialized quantum kernels for different use cases:

- **bulletproof_quantum_kernel.py** - Production-ready quantum kernel
- **working_quantum_kernel.py** - Tested quantum kernel implementation
- **topological_quantum_kernel.py** - Advanced topological quantum algorithms

### 3. RFT Implementations

Multiple implementations of Resonance Field Theory for different purposes:

- **rft_core.py** - Core RFT implementation
- **canonical_true_rft.py** - Canonical True RFT implementation
- **paper_compliant_rft_fixed.py** - Paper-compliant RFT implementation

### 4. User Interface (Cream OS)

- **16_EXPERIMENTAL/prototypes/quantonium_os_unified_cream.py** - Cream OS with circular app dock

### 5. Applications

- **apps/** - Collection of quantum applications and tools

## Getting Started

1. Build the system:
   ```
   build.bat
   ```

2. Launch QuantoniumOS:
   ```
   python os_boot_transition.py
   ```

3. Run the sanity check:
   ```
   python verify_sanity.py
   ```

## Architecture

QuantoniumOS follows a layered architecture:

1. **ASSEMBLY Layer** - Low-level quantum operations (C/ASM)
2. **C++ Layer** - Bindings and interfaces
3. **Python Layer** - Kernels and algorithms
4. **UI Layer** - Cream OS and application interfaces

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
