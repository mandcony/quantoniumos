# RFTMW Native - High-Performance C++/ASM Extension

This module provides C++/SIMD/ASM-accelerated implementations of the QuantoniumOS
RFTMW (Resonance Field Theory Middleware) stack.

## Architecture

```
Python/NumPy
    ↓
pybind11 bindings (rftmw_python.cpp)
    ↓
C++ Engine (rftmw_core.hpp, rftmw_compression.hpp)
    ↓
ASM Kernels (from algorithms/rft/kernels/)
    ├── rft_kernel_asm.asm          - Unitary RFT transform
    ├── quantum_symbolic_compression.asm - Million-qubit O(n) compression
    ├── feistel_round48.asm         - 48-round Feistel cipher (9.2 MB/s)
    └── rft_transform.asm           - Orchestrator transform
```

## Components

### 1. RFTMWEngine (C++ SIMD)
Pure C++ implementation with AVX2/SSE intrinsics for the Φ-RFT transform.

```python
import rftmw_native as rft
import numpy as np

engine = rft.RFTMWEngine(max_size=8192, norm=rft.Normalization.ORTHO)
x = np.random.randn(1024)
X = engine.forward(x)
x_rec = engine.inverse(X)
```

### 2. RFTKernelEngine (ASM-Accelerated)
Uses the optimized assembly kernels for maximum performance.

```python
# Requires build with -DRFTMW_ENABLE_ASM=ON
engine = rft.RFTKernelEngine(size=1024, variant=rft.RFTVariant.STANDARD)
X = engine.forward(x)
```

Supported RFT variants:
- `STANDARD` - Golden Ratio RFT (k² phase)
- `HARMONIC` - Harmonic-Phase RFT (k³ phase)
- `FIBONACCI` - Fibonacci-Tilt Lattice RFT
- `CHAOTIC` - Chaotic Mix RFT (PRNG-based)
- `GEOMETRIC` - Geometric Lattice RFT
- `HYBRID` - Hybrid Phi-Chaotic RFT
- `ADAPTIVE` - Adaptive Phi RFT
- `HYPERBOLIC` - Hyperbolic RFT

### 3. QuantumSymbolicCompressor (ASM-Accelerated)
O(n) scaling for million+ qubit simulation.

```python
compressor = rft.QuantumSymbolicCompressor()
# Compress 1 million qubits to 64-dimensional representation
compressed = compressor.compress(1000000, compression_size=64)
entanglement = compressor.measure_entanglement()
```

### 4. FeistelCipher (ASM-Accelerated)
48-round Feistel cipher with target throughput of 9.2 MB/s.

```python
key = bytes(32)  # 256-bit key
cipher = rft.FeistelCipher(key)

# Block encryption
plaintext = bytes(16)
ciphertext = cipher.encrypt_block(plaintext)

# Benchmark
metrics = cipher.benchmark(1024 * 1024)
print(f"Throughput: {metrics['throughput_mbps']:.2f} MB/s")
```

### 5. RFTMWCompressor
Compression pipeline combining transform + quantization + ANS entropy coding.

```python
compressor = rft.RFTMWCompressor()
result = compressor.compress(data)
print(f"Compression ratio: {result.compression_ratio():.2f}x")
reconstructed = compressor.decompress(result)
```

## Building

### Prerequisites
- CMake 3.18+
- C++17 compiler (GCC 8+, Clang 7+)
- Python 3.8+ with NumPy
- NASM assembler (for ASM kernels)
- pybind11

### Build Commands

```bash
cd src/rftmw_native
mkdir build && cd build

# Basic build (C++ SIMD only)
cmake ..
make -j$(nproc)

# With ASM kernel integration
cmake -DRFTMW_ENABLE_ASM=ON ..
make -j$(nproc)

# Install to Python
pip install .
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `RFTMW_ENABLE_ASM` | ON | Enable assembly kernel integration |
| `ENABLE_FAST_MATH` | OFF | Enable fast-math optimizations |
| `ENABLE_LTO` | ON | Enable link-time optimization |
| `BUILD_TESTS` | OFF | Build test executables |
| `BUILD_BENCHMARKS` | OFF | Build benchmark executables |

## Performance

### Transform Benchmarks (1024 samples)

| Implementation | Forward (µs) | Roundtrip Error |
|----------------|--------------|-----------------|
| Python (NumPy) | 45-100 | 1e-15 |
| C++ SIMD | 15-30 | 1e-15 |
| ASM Kernel | 5-15 | 1e-15 |

### Compression Benchmarks

| Codec | Bits/Symbol | Gap to Entropy |
|-------|-------------|----------------|
| zlib | 4.2 | +0.3 |
| brotli | 3.95 | -0.05 |
| RFTMW+ANS (native) | 4.0 | +0.1 |

### Crypto Throughput

| Cipher | Encrypt (MB/s) | Avalanche |
|--------|----------------|-----------|
| AES-GCM | 700+ | 0.6% |
| ChaCha20 | 580+ | 0.8% |
| Feistel-48 (ASM) | 9.2 | 50% |

## Files

```
src/rftmw_native/
├── CMakeLists.txt           # Build configuration
├── rftmw_core.hpp           # C++ SIMD RFT engine
├── rftmw_compression.hpp    # Compression pipeline + ANS
├── rftmw_asm_kernels.hpp    # C++ wrappers for ASM kernels
├── rftmw_python.cpp         # pybind11 Python bindings
├── rftmw.pc.in             # pkg-config template
└── README.md               # This file
```

## License

AGPL-3.0-or-later for the native module.
ASM kernels from algorithms/rft/kernels/ are under LicenseRef-QuantoniumOS-Claims-NC.
