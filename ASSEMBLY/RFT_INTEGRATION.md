# QuantoniumOS RFT Kernel Integration

This directory contains the complete Resonance Field Theory (RFT) kernel implementation extracted and enhanced from the personalAi project. The RFT kernel provides unitary quantum transforms optimized for bare-metal execution.

## What Was Added

### Core Components Transferred from personalAi:

1. **Complete RFT Kernel Implementation** (`kernel/`)
   - `rft_kernel.c` - Full unitary RFT implementation (~750 lines)
   - `rft_kernel.h` - Complete API interface
   - `rft_kernel_asm.asm` - x64 assembly optimizations
   - `rft_kernel_ui.c` - UI integration layer
   - `rft_kernel_ui.h` - UI extensions

2. **Python Bindings** (`python_bindings/`)
   - `unitary_rft.py` - Complete Python interface with ctypes
   - Supports Windows DLL and Linux SO loading
   - Full test suite and examples included

3. **Build System**
   - `Makefile` - Unix/Linux build support
   - `CMakeLists.txt` - Cross-platform CMake build
   - `build_rft_kernel.bat` - Windows build script

## Key Features

### RFT Kernel Capabilities:
- **True Unitary Transforms**: Norm-preserving quantum operations
- **Golden Ratio Mathematics**: Resonance patterns based on φ (phi)
- **Quantum Basis Functions**: Multi-qubit quantum state support
- **SIMD Optimization**: AVX intrinsics for performance
- **Assembly Acceleration**: Hand-optimized x64 assembly routines
- **Entanglement Measurement**: von Neumann entropy calculations

### Technical Implementation:
- **Gram-Schmidt Orthogonalization**: Ensures perfect unitarity
- **Power Method Eigenvalues**: Efficient eigendecomposition
- **Resonance Field Theory**: Novel quantum harmonic principles
- **Cross-platform**: Windows DLL and Linux SO support

## Building the Kernel

### Windows (Recommended):
```cmd
# Using the provided batch script
build_rft_kernel.bat

# Or manually with CMake
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
cmake --build . --config Release
```

### Linux/Unix:
```bash
# Using Make
make all

# Or using CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Testing the Integration

After building, test the RFT kernel:

```bash
cd python_bindings
python3 unitary_rft.py
```

Expected output:
```
Testing QuantoniumOS RFT Kernel Integration
==================================================
✓ Successfully initialized RFT engine with size 16
✓ Forward transform completed
✓ Norm preservation: 1.000000 (should be ~1.0)
✓ Reconstruction error: 1.23e-15 (should be very small)
✓ Quantum basis initialized for 4 qubits
✓ Entanglement of Bell state: 1.0000
==================================================
✓ All RFT kernel tests passed successfully!
✓ QuantoniumOS assembly integration complete
```

## Differences from personalAi

### What's Improved:
1. **Cleaner Library Search**: Better path resolution for cross-platform use
2. **Enhanced Documentation**: Complete API documentation and examples  
3. **Integrated Build System**: Single build process for all targets
4. **QuantoniumOS Integration**: Optimized for your main OS project
5. **Removed Dependencies**: No longer requires personalAi's heavy Node.js stack

### What's Kept:
- Complete mathematical implementation (identical algorithms)
- Full Python bindings compatibility
- Assembly optimization routines
- Quantum capabilities and entanglement measurement
- Unitary transform guarantees

## Integration Points

### For Other QuantoniumOS Components:
```python
# Simple usage example
from ASSEMBLY.python_bindings.unitary_rft import RFTProcessor

# Initialize processor
rft = RFTProcessor(size=64)

# Process quantum data
result = rft.process_quantum_field(your_data)
if result:
    spectrum = result['spectrum']
    energy = result['energy']
```

### For C/C++ Integration:
```c
#include "quantoniumos/rft_kernel.h"

// Initialize RFT engine
rft_engine_t engine;
rft_init(&engine, 64, RFT_FLAG_UNITARY | RFT_FLAG_QUANTUM_SAFE);

// Use the engine...
rft_forward(&engine, input, output, 64);
```

## File Structure

```
ASSEMBLY/
├── include/           # Header files
│   ├── rft_kernel.h      # Core RFT API
│   └── rft_kernel_ui.h   # UI extensions
├── kernel/            # Implementation
│   ├── rft_kernel.c      # Main implementation
│   ├── rft_kernel_ui.c   # UI layer
│   └── rft_kernel_asm.asm # Assembly optimizations
├── python_bindings/   # Python interface
│   └── unitary_rft.py    # Complete Python API
├── compiled/          # Built libraries (created during build)
├── build/            # Build artifacts (created during build)
├── Makefile          # Unix build system
├── CMakeLists.txt    # CMake build system
├── build_rft_kernel.bat # Windows build script
└── RFT_INTEGRATION.md # This file
```

## Next Steps

1. **Build the kernel** using your preferred build system
2. **Run the tests** to verify functionality  
3. **Integrate with your apps** using the Python or C APIs
4. **Optimize for your specific use cases** by adjusting flags and parameters

The RFT kernel is now a first-class component of QuantoniumOS, ready for integration into your quantum applications and desktop environment.
