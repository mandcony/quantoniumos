# QuantoniumOS - Copilot Instructions

## System Architecture

QuantoniumOS is a **symbolic quantum-inspired computing platform** with a layered architecture:

```
Applications (PyQt5) → Desktop Environment → Core Algorithms → C Assembly Kernels
```

- **Assembly Layer**: C kernels with SIMD optimization (`src/assembly/kernel/`)
- **Core Layer**: Python mathematical algorithms (`src/core/`)
- **Frontend Layer**: PyQt5 desktop environment (`src/frontend/`)
- **Applications**: Integrated apps running in-process (`src/apps/`)

## Key Components

### RFT (Resonance Fourier Transform) Engine
- **Core Implementation**: `src/core/canonical_true_rft.py`
- **C Kernel**: `src/assembly/kernel/rft_kernel.c` with Python bindings
- **Pattern**: Golden ratio (φ = 1.618...) parameterization for unitary operations
- **Scaling**: O(n) complexity vs O(2^n) traditional quantum simulation

### Quantum Simulator Architecture
- **File**: `src/apps/quantum_simulator.py`
- **Pattern**: Vertex-based encoding instead of binary qubits
- **Scale**: Supports 1000+ symbolic qubits via RFT compression
- **Integration**: Uses `UnitaryRFT` from assembly bindings when available

### Desktop Environment Pattern
- **File**: `src/frontend/quantonium_desktop.py`
- **Pattern**: Single-process PyQt5 desktop with dynamic app importing
- **Golden Ratio UI**: All proportions based on φ mathematical constants
- **App Loading**: Uses `importlib` for dynamic app class detection, NOT subprocess launching

## Development Workflows

### Building Assembly Kernels
```bash
cd src/assembly
make all                    # Build C kernels
make install               # Install Python bindings
```

### Running the System
```bash
python quantonium_boot.py  # Full system boot
python src/apps/quantum_simulator.py  # Individual app
```

### Testing Patterns
- **System Check**: `python comprehensive_system_check.py`
- **Validation Suite**: `tests/comprehensive_validation_suite.py`
- **Crypto Tests**: `tests/crypto/crypto_performance_test.py`
- **Benchmarks**: `QUANTONIUM_BENCHMARK_SUITE.py`

## Critical Conventions

### Application Integration
- Apps extend base classes from `src/apps/launcher_base.py`
- Use in-process importing, never subprocess.Popen()
- Desktop manager uses `importlib.util.spec_from_file_location()`

### Mathematical Precision
- RFT operations must maintain unitarity < 1e-12
- Use golden ratio constants: `phi = (1 + sqrt(5))/2`
- Quantum operations require unitary validation after transforms

### File Naming Patterns
- Core algorithms: `src/core/canonical_true_rft.py`
- Assembly kernels: `src/assembly/kernel/rft_kernel.c`
- App launchers: `src/apps/launch_*.py`
- Test files: `test_*.py` or `*_test.py`

### Error Handling
- Assembly binding failures should gracefully fallback to Python implementations
- RFT_AVAILABLE flag controls kernel vs software paths
- Desktop apps must handle PyQt5 import failures

## Integration Points

### Assembly-Python Interface
```python
from unitary_rft import UnitaryRFT, RFT_FLAG_QUANTUM_SAFE
engine = UnitaryRFT(size, RFT_FLAG_QUANTUM_SAFE | RFT_FLAG_USE_RESONANCE)
```

### Desktop App Pattern
```python
class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My App — QuantoniumOS")
        # Golden ratio proportions
        self.phi = 1.618033988749895
```

### RFT Validation Pattern
```python
# Always validate unitarity
unitarity_error = norm(Psi.conj().T @ Psi - identity, ord=2)
assert unitarity_error < 1e-12, "RFT must be unitary"
```

## Dependencies & Environment

- **Python**: 3.8+ with NumPy, SciPy, PyQt5, matplotlib
- **C Compiler**: GCC (Linux) or MSVC (Windows) for assembly kernels
- **Build Tools**: CMake for complex builds, standard Makefile for kernels
- **ML Stack**: PyTorch, Transformers for AI integration (optional)

## Performance Considerations

- RFT compression enables 1000+ qubit simulation on standard hardware
- C kernels use SIMD optimization (`-march=native -O3`)
- Large datasets stored in `data/`, results in `results/`
- Use vertex encoding for quantum states instead of exponential representations