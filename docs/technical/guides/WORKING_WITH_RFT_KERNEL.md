# Working with the RFT Kernel

This guide covers everything you need to know about working with the Resonance Fourier Transform (RFT) kernel in QuantoniumOS.

## Overview

The RFT kernel is the mathematical heart of QuantoniumOS. It provides high-performance quantum-inspired transforms using golden ratio parameterization.

**Key Features:**
- Unitary transforms with <1e-12 precision
- SIMD-optimized C implementation
- Python fallback for portability
- Golden ratio (φ) parameterized operations

## Architecture

```
┌─────────────────────────────────────┐
│     Your Application Code           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Python Bindings (ctypes)           │
│  ASSEMBLY/python_bindings/          │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  C Kernel (if compiled)             │
│  src/assembly/kernel/rft_kernel.c   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Python Fallback (always available) │
│  algorithms/rft/core/               │
└─────────────────────────────────────┘
```

## Installation & Building

### Check if RFT Kernel is Available

```python
try:
    from ASSEMBLY.python_bindings.unitary_rft import UnitaryRFT
    print("✅ RFT kernel available")
except ImportError:
    print("⚠️  RFT kernel not available - will use Python fallback")
```

### Build the C Kernel

```bash
# Navigate to assembly directory
cd src/assembly

# Clean previous builds
make clean

# Build release version (optimized)
make all

# Or build with AddressSanitizer for debugging
make asan

# Install Python bindings
make install
```

### Verify Installation

```bash
python -c "from ASSEMBLY.python_bindings.unitary_rft import UnitaryRFT; print('Success!')"
```

## Basic Usage

### Initialize the RFT Engine

```python
from ASSEMBLY.python_bindings.unitary_rft import (
    UnitaryRFT, 
    RFT_FLAG_QUANTUM_SAFE,
    RFT_FLAG_USE_RESONANCE,
    RFT_FLAG_OPTIMIZE_MEMORY
)

# Create RFT engine with default settings
engine = UnitaryRFT(size=256)

# Or with custom flags
engine = UnitaryRFT(
    size=256,
    flags=RFT_FLAG_QUANTUM_SAFE | RFT_FLAG_USE_RESONANCE
)
```

### Transform Data

```python
import numpy as np

# Create input data
input_data = np.random.rand(256) + 1j * np.random.rand(256)

# Apply RFT transform
transformed = engine.transform(input_data)

# Apply inverse transform
reconstructed = engine.inverse_transform(transformed)

# Verify round-trip accuracy
error = np.linalg.norm(input_data - reconstructed)
print(f"Round-trip error: {error:.2e}")  # Should be < 1e-12
```

## Advanced Usage

### Batch Processing

Process multiple arrays efficiently:

```python
# Prepare batch of data
batch_data = [
    np.random.rand(256) + 1j * np.random.rand(256)
    for _ in range(100)
]

# Process batch
results = []
for data in batch_data:
    result = engine.transform(data)
    results.append(result)

# Or use batch method if available
if hasattr(engine, 'batch_transform'):
    results = engine.batch_transform(batch_data)
```

### Quantum State Evolution

Evolve quantum states using RFT:

```python
def evolve_quantum_state(initial_state, time_steps, engine):
    """
    Evolve a quantum state using RFT.
    
    Args:
        initial_state: Initial quantum state vector
        time_steps: Number of evolution steps
        engine: RFT engine instance
        
    Returns:
        Final evolved state
    """
    state = initial_state.copy()
    
    for t in range(time_steps):
        # Apply RFT transform (represents time evolution)
        state = engine.transform(state)
        
        # Renormalize to preserve probability
        norm = np.linalg.norm(state)
        state = state / norm
        
    return state

# Example usage
initial_state = np.random.rand(256) + 1j * np.random.rand(256)
initial_state = initial_state / np.linalg.norm(initial_state)

evolved_state = evolve_quantum_state(initial_state, 100, engine)
```

### Golden Ratio Operations

Leverage the golden ratio parameterization:

```python
def phi_weighted_transform(data, engine, power=1):
    """
    Apply φ-weighted RFT transform.
    
    The golden ratio φ = (1 + √5) / 2 ≈ 1.618
    provides special properties for numerical stability.
    
    Args:
        data: Input array
        engine: RFT engine
        power: Power of φ to use for weighting
        
    Returns:
        Transformed and weighted data
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # Apply RFT transform
    transformed = engine.transform(data)
    
    # Apply φ-weighting
    weighted = transformed * (phi ** power)
    
    return weighted
```

## Performance Optimization

### Choosing the Right Size

RFT performance depends on the transform size:

```python
# Powers of 2 are most efficient
sizes = [128, 256, 512, 1024, 2048]

# Measure performance for each size
import time

for size in sizes:
    engine = UnitaryRFT(size=size)
    data = np.random.rand(size) + 1j * np.random.rand(size)
    
    start = time.time()
    for _ in range(100):
        result = engine.transform(data)
    elapsed = time.time() - start
    
    print(f"Size {size:4d}: {elapsed*10:.2f} ms per transform")
```

### Memory Management

```python
# Use memory-optimized mode for large transforms
engine = UnitaryRFT(
    size=4096,
    flags=RFT_FLAG_OPTIMIZE_MEMORY
)

# Process large dataset in chunks
def process_large_data(large_array, chunk_size=1024):
    results = []
    
    for i in range(0, len(large_array), chunk_size):
        chunk = large_array[i:i+chunk_size]
        
        # Process chunk
        result = engine.transform(chunk)
        results.append(result)
        
    return np.concatenate(results)
```

### Parallel Processing

```python
from multiprocessing import Pool

def transform_chunk(args):
    """Worker function for parallel processing"""
    chunk, size = args
    engine = UnitaryRFT(size=size)
    return engine.transform(chunk)

def parallel_transform(data, n_workers=4):
    """
    Apply RFT transform in parallel.
    
    Args:
        data: Large dataset to transform
        n_workers: Number of parallel workers
        
    Returns:
        Transformed data
    """
    chunk_size = len(data) // n_workers
    chunks = [
        (data[i:i+chunk_size], chunk_size)
        for i in range(0, len(data), chunk_size)
    ]
    
    with Pool(n_workers) as pool:
        results = pool.map(transform_chunk, chunks)
        
    return np.concatenate(results)
```

## Debugging & Validation

### Verify Unitarity

The RFT transform should be unitary (preserve norms):

```python
def validate_unitarity(engine, n_tests=100):
    """
    Verify that RFT preserves unitarity.
    
    Args:
        engine: RFT engine to test
        n_tests: Number of random tests to run
        
    Returns:
        Maximum observed error
    """
    max_error = 0
    
    for _ in range(n_tests):
        # Generate random state
        state = np.random.rand(engine.size) + 1j * np.random.rand(engine.size)
        norm_before = np.linalg.norm(state)
        
        # Transform
        transformed = engine.transform(state)
        norm_after = np.linalg.norm(transformed)
        
        # Check error
        error = abs(norm_before - norm_after)
        max_error = max(max_error, error)
        
    print(f"Maximum unitarity error: {max_error:.2e}")
    assert max_error < 1e-10, "Unitarity violation detected!"
    
    return max_error

# Run validation
validate_unitarity(engine)
```

### Compare with Reference Implementation

```python
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT

def compare_implementations(data):
    """
    Compare C kernel with Python reference.
    
    Args:
        data: Test data
        
    Returns:
        Difference between implementations
    """
    size = len(data)
    
    # C kernel
    c_engine = UnitaryRFT(size=size)
    c_result = c_engine.transform(data)
    
    # Python reference
    py_engine = CanonicalTrueRFT(size=size)
    py_result = py_engine.transform(data)
    
    # Compare
    diff = np.linalg.norm(c_result - py_result)
    print(f"Implementation difference: {diff:.2e}")
    
    return diff

# Test with random data
test_data = np.random.rand(256) + 1j * np.random.rand(256)
compare_implementations(test_data)
```

### Debugging with AddressSanitizer

If you encounter crashes or memory issues:

```bash
# Build with AddressSanitizer
cd src/assembly
make clean
make asan

# Run your script - ASan will detect memory errors
python your_script.py
```

## Common Issues & Solutions

### Issue 1: Import Error

**Problem:**
```python
ImportError: cannot import name 'UnitaryRFT' from 'ASSEMBLY.python_bindings.unitary_rft'
```

**Solution:**
```bash
# Rebuild and reinstall
cd src/assembly
make clean && make all
make install

# Verify installation
python -c "import ASSEMBLY.python_bindings.unitary_rft"
```

### Issue 2: Segmentation Fault

**Problem:** Python crashes with "Segmentation fault"

**Solution:**
1. Rebuild with debug symbols: `make asan`
2. Check array sizes match engine size
3. Ensure data is contiguous: `data = np.ascontiguousarray(data)`
4. Verify data type: `data = data.astype(np.complex128)`

### Issue 3: Poor Performance

**Problem:** RFT is slower than expected

**Solution:**
```python
# Check if you're using the C kernel
import ASSEMBLY.python_bindings.unitary_rft as rft_module
print(f"Using: {rft_module.__file__}")

# Ensure power-of-2 sizes
size = 2 ** int(np.log2(your_size))  # Round to power of 2

# Use appropriate flags
engine = UnitaryRFT(
    size=size,
    flags=RFT_FLAG_OPTIMIZE_MEMORY  # For large sizes
)
```

## Best Practices

### 1. Always Use Context Managers

```python
class RFTContext:
    """Context manager for RFT engine"""
    def __init__(self, size, flags=0):
        self.size = size
        self.flags = flags
        self.engine = None
        
    def __enter__(self):
        self.engine = UnitaryRFT(self.size, self.flags)
        return self.engine
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        if hasattr(self.engine, 'cleanup'):
            self.engine.cleanup()
        return False

# Usage
with RFTContext(256) as engine:
    result = engine.transform(data)
```

### 2. Validate Inputs

```python
def safe_transform(engine, data):
    """Safely transform data with validation"""
    # Check size
    if len(data) != engine.size:
        raise ValueError(f"Data size {len(data)} != engine size {engine.size}")
        
    # Check type
    if not np.iscomplexobj(data):
        data = data.astype(np.complex128)
        
    # Ensure contiguous
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
        
    return engine.transform(data)
```

### 3. Handle Fallback Gracefully

```python
class RFTProcessor:
    """Processor with automatic fallback"""
    def __init__(self, size):
        self.size = size
        try:
            from ASSEMBLY.python_bindings.unitary_rft import UnitaryRFT
            self.engine = UnitaryRFT(size)
            self.use_c_kernel = True
        except ImportError:
            from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
            self.engine = CanonicalTrueRFT(size)
            self.use_c_kernel = False
            
    def transform(self, data):
        return self.engine.transform(data)
```

## Performance Benchmarks

Expected performance on modern hardware (Intel i7, 16GB RAM):

| Size | C Kernel | Python Fallback | Speedup |
|------|----------|-----------------|---------|
| 128  | 2.1 ms   | 45 ms          | 21x     |
| 256  | 4.8 ms   | 180 ms         | 38x     |
| 512  | 11.2 ms  | 720 ms         | 64x     |
| 1024 | 23.8 ms  | 2.9 s          | 122x    |
| 2048 | 52.1 ms  | 11.6 s         | 223x    |

## Further Reading

- [RFT Mathematical Foundations](../../research/MATHEMATICAL_FOUNDATIONS.md)
- [Component Deep Dive](../COMPONENT_DEEP_DIVE.md)
- [Validation Workflow](./VALIDATION_WORKFLOW.md)
