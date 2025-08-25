# QuantoniumOS

**Quantum Operating System • True RFT Algorithms • Quantum Cryptography • Complete Research Framework**

QuantoniumOS is a comprehensive quantum operating system featuring True Resonance Fourier Transform (RFT) algorithms, quantum-enhanced cryptography, and a complete development framework for quantum computing research.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (3.12 recommended)
- Virtual environment recommended

### Installation & Launch
```bash
# 1) Clone and setup environment
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 2) Install dependencies
pip install numpy scipy networkx flask cryptography PyQt5

# 3) Launch QuantoniumOS (choose one):

# Option A: Unified GUI Launcher
python 11_QUANTONIUMOS/launch.py

# Option B: Web Interface  
python 03_RUNNING_SYSTEMS/app.py

# Option C: Individual Apps
python 11_QUANTONIUMOS/apps/launch_quantum_simulator.py
```

### Quick Validation
```bash
# Verify core functionality
python 02_CORE_VALIDATORS/run_all_validators.py

# Test True RFT algorithms
python 04_RFT_ALGORITHMS/canonical_true_rft.py
```

---

## 📁 Project Structure

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| **01_START_HERE** | Entry point & navigation | README.md, NAVIGATION_GUIDE.md |
| **02_CORE_VALIDATORS** | Scientific validation | run_all_validators.py, validation_results/ |
| **03_RUNNING_SYSTEMS** | Production systems | app.py (Flask web server), main.py |
| **04_RFT_ALGORITHMS** | True RFT implementations | canonical_true_rft.py, true_rft_engine_bindings.py |
| **05_QUANTUM_ENGINES** | Quantum kernels | bulletproof_quantum_kernel.py, topological_quantum_kernel.py |
| **06_CRYPTOGRAPHY** | Quantum cryptography | true_rft_feistel_bindings.py, quantum_cipher.py |
| **07_TESTS_BENCHMARKS** | Test infrastructure | Comprehensive test suite (1000+ tests) |
| **08_RESEARCH_ANALYSIS** | Research tools | Analysis and visualization tools |
| **10_UTILITIES** | Utility functions | third_party/, logging_config.py |
| **11_QUANTONIUMOS** | Main OS & GUI | quantonium_os_unified.py, apps/, core/, frontend/ |
| **13_DOCUMENTATION** | Documentation | This file, guides/, reports/ |
| **14_CONFIGURATION** | Configuration files | Environment configs, build settings |
| **15_DEPLOYMENT** | Deployment scripts | Production deployment tools |
| **16_EXPERIMENTAL** | Experimental features | Research prototypes |
| **17_BUILD_ARTIFACTS** | Build outputs | Compiled artifacts |
| **18_DEBUG_TOOLS** | Debug utilities | Development and debugging tools |
| **src/quantoniumos/** | **Installable package** | **Official Python package for distribution** |

---

## 🎯 Main Entry Points

### For End Users:
- **🖥️ Desktop GUI**: `python 11_QUANTONIUMOS/quantonium_os_unified.py`
- **🌐 Web Interface**: `python 03_RUNNING_SYSTEMS/app.py`
- **🚀 Unified Launcher**: `python 11_QUANTONIUMOS/launch.py`

### For Developers:
- **✅ Run Validators**: `python 02_CORE_VALIDATORS/run_all_validators.py`  
- **🧪 Test RFT**: `python 04_RFT_ALGORITHMS/canonical_true_rft.py`
- **📊 Benchmarks**: Files in `07_TESTS_BENCHMARKS/`

### For Package Users:
```python
# Install as package
pip install -e ./src

# Use in Python
import quantoniumos
kernel = quantoniumos.QuantumKernel()
result = kernel.run_quantum_algorithm(data)
```

---

## 🔬 Core Technologies

- **True RFT Algorithms**: Novel Fourier transform implementations
- **Quantum Kernels**: Bulletproof and topological quantum processing  
- **Quantum Cryptography**: RFT-based encryption and security
- **Web Interface**: Flask-based control panel
- **Desktop GUI**: PyQt5-based user interface
- **Package Distribution**: Standard Python package in `src/`

---

## 📊 Validation & Testing

All core algorithms are scientifically validated:
- **Perfect Reconstruction**: Error < 1e-15
- **Energy Conservation**: Verified to machine precision  
- **Non-DFT Equivalence**: Mathematically proven distinct
- **Quantum Properties**: Full unitarity validation
- **Cryptographic Security**: Statistical randomness verified

Run `python 02_CORE_VALIDATORS/run_all_validators.py` for full verification.

---

## 🏗️ Development

### Configuration
- **No .env files needed** - Configuration via Python classes
- **Settings**: Located in `14_CONFIGURATION/`
- **Runtime options**: Command-line arguments supported

### Security Note
This is research-grade cryptography. For production use, combine with established cryptographic libraries and follow security best practices.

---

## 📄 Citation

```bibtex
@software{QuantoniumOS_2025,
  title   = {QuantoniumOS: Quantum Operating System with True RFT Algorithms},
  author  = {Minier, Luis},
  year    = {2025},
  url     = {https://github.com/mandcony/quantoniumos}
}
```
