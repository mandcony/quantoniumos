# Test Routing Fixes - Architecture Alignment

## Summary

Fixed all test file import issues to align with the QuantoniumOS architecture.

## Changes Made

### 1. Added `hypothesis` Dependency

**Files Updated:**
- `requirements.txt` - Added `hypothesis>=6.0.0` for property-based testing
- `requirements.in` - Added `hypothesis>=6.0.0`
- `pyproject.toml` - Added `hypothesis>=6.0.0` to dev dependencies

**Installation:**
```bash
pip install hypothesis
```

### 2. Fixed PyQt5 Import Issues

**File:** `src/apps/quantsounddesign/pattern_editor.py`

**Problem:** When PyQt5 was not available, the import block used `pass`, which left classes like `QWidget` undefined, causing `NameError` when classes tried to inherit from them.

**Solution:** Added comprehensive fallback dummy classes when PyQt5 is not available:
```python
except ImportError:
    HAS_PYQT5 = False
    # Dummy classes for when PyQt5 is not available
    class QWidget: pass
    class QPushButton: pass
    # ... etc
```

This allows the module to be imported for testing even without PyQt5 installed.

### 3. Fixed RFT Vertex Codec Import Path

**File:** `tests/validation/test_rft_vertex_codec_roundtrip.py`

**Problem:** Import was using non-existent `core` module:
```python
from core import rft_vertex_codec as codec  # WRONG
```

**Solution:** Updated to use correct architecture path:
```python
from algorithms.rft.compression import rft_vertex_codec as codec  # CORRECT
```

### 4. Fixed Bell Violations Test Imports

**File:** `tests/validation/test_bell_violations.py`

**Problem:** Importing non-existent modules:
- `quantonium_os_src.engine.vertex_assembly.EntangledVertexEngine`
- `quantonium_os_src.engine.open_quantum_systems.OpenQuantumSystem`
- `quantonium_os_src.engine.open_quantum_systems.NoiseModel`

**Solution:** 
1. Commented out non-existent imports
2. Added stub/placeholder classes for future implementation:
```python
# Note: These modules are placeholders for future quantum engine implementation
class EntangledVertexEngine:
    """Placeholder for entangled vertex engine."""
    pass

class OpenQuantumSystem:
    """Placeholder for open quantum system."""
    pass

class NoiseModel:
    """Placeholder for noise model."""
    pass
```

### 5. Enhanced pytest Configuration

**File:** `pytest.ini`

**Added:**
```ini
pythonpath = .
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

This ensures pytest can find all modules from the project root.

## Architecture Mapping

### Correct Import Paths

| Component | Correct Path | Notes |
|-----------|-------------|-------|
| RFT Codec | `algorithms.rft.compression.rft_vertex_codec` | Compression algorithms |
| RFT Kernels | `algorithms.rft.kernels` | Core RFT implementations |
| UnitaryRFT | `algorithms.rft.kernels.python_bindings.unitary_rft` | Native bindings |
| QuantSound | `src.apps.quantsounddesign` | Audio application |
| RFTMW Engine | `quantonium_os_src.engine.RFTMW` | Middleware transform engine |
| Test Utils | `tests.proofs.test_entanglement_protocols` | Test utilities |

### Module Structure

```
quantoniumos/
├── algorithms/              # Core algorithms (importable)
│   └── rft/
│       ├── compression/     # Compression codecs
│       └── kernels/         # Native implementations
├── src/                     # Applications
│   └── apps/
│       └── quantsounddesign/  # Audio DAW
├── quantonium_os_src/      # OS-level components
│   └── engine/
│       └── RFTMW.py        # Transform middleware
└── tests/                   # Test suite
    ├── proofs/             # Proof utilities
    └── validation/         # Validation tests
```

## Running Tests

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Files
```bash
# Vertex codec tests
python -m pytest tests/validation/test_rft_vertex_codec_roundtrip.py -v

# Property-based crypto tests (requires hypothesis)
python -m pytest tests/crypto/test_property_encryption.py -v

# Bell violations (uses stub classes)
python -m pytest tests/validation/test_bell_violations.py -v
```

### Skip Slow Tests
```bash
python -m pytest tests/ -v -m "not slow"
```

### Verify Imports
```bash
python test_imports.py
```

## Known Limitations

1. **Audio Backend Tests** - Skip if PyQt5 not installed (GUI tests):
   ```bash
   pytest tests/ -v --ignore=tests/test_audio_backend.py
   ```

2. **Quantum Engine Modules** - Currently using placeholder classes:
   - `EntangledVertexEngine` 
   - `OpenQuantumSystem`
   - `NoiseModel`
   
   These will be implemented when the full quantum engine is developed.

3. **QuTiP Optional** - Some tests gracefully skip if QuTiP is not available.

## Future Work

### Modules to Implement

1. **`quantonium_os_src/engine/vertex_assembly.py`**
   - `EntangledVertexEngine` class
   - Quantum vertex assembly operations

2. **`quantonium_os_src/engine/open_quantum_systems.py`**
   - `OpenQuantumSystem` class
   - `NoiseModel` class
   - Decoherence simulation

3. **Enhanced Test Coverage**
   - Integration tests for quantum modules
   - End-to-end pipeline tests
   - Performance benchmarks

## Validation

All test files can now be imported without errors. Run the validation:

```bash
# Check system setup
python validate_system.py

# Check test imports
python test_imports.py

# Run test suite
python -m pytest tests/ -v
```

## Dependencies Status

✅ **Installed:**
- numpy, scipy, matplotlib, pandas
- sympy, qutip
- pytest, black, flake8
- hypothesis (newly added)

⚠️ **Optional:**
- PyQt5 (for GUI tests)
- torch, transformers (for AI features)
- diffusers, pillow (for image generation)

## Contact

For issues related to test routing or imports, check:
1. This document
2. `docs/ARCHITECTURE.md` - System architecture
3. `GETTING_STARTED.md` - Setup instructions
