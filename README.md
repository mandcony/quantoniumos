# QuantoniumOS

**Hybrid Computational Framework for Quantum & Resonance Simulation**

---

## 🚀 Overview
QuantoniumOS is a post-binary operating system kernel and framework built on
the **Resonance Fourier Transform (RFT)** — a newly proven *unitary transform*
that preserves energy, is perfectly invertible, and abstracts quantum-style
unitarity on classical hardware.

Like Linux powered the Internet era, QuantoniumOS aims to power the **post-quantum era**:
- Unitary transforms at kernel speed (ASM + C backend)
- Symbolic qubit simulation
- Resonance-based cryptography and hashing
- Secure containers and entropy services

---

## 📦 Structure
- `ASSEMBLY/` → bare-metal C/ASM RFT kernel + Python bindings
- `engines/` → RFT core implementations (canonical_true_rft.py, rft_core.py)
- `core/` → Quantum kernel implementations
- `apps/` → Application suite and tools:
  - `quantum_simulator.py` → Full-scale quantum simulation (1000+ qubits)
  - `rft_validation_suite.py` → Mathematical validation of RFT uniqueness
  - `q_notes.py`, `q_vault.py` → Quantum-enhanced applications
- `frontend/` → Desktop management and UI shell
- `ui/` → Stylesheets and visual assets

---

## 🔑 Features
- **True Unitary Resonance Fourier Transform** proven to machine precision
- Forward + Inverse transforms with perfect reconstruction
- SIMD assembly acceleration
- Python API + OS kernel entrypoints
- Quantum-style operations (basis init, entanglement measure)
- Comprehensive validation suite demonstrating RFT ≠ DFT/FFT/DCT/DST

---

## ⚡ Quick Start
```bash
# Clone repo
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos

# Install dependencies
pip install PyQt5 qtawesome pytz psutil numpy

# Run the full OS
python launch_quantonium_os.py

# Run individual components
python apps/quantum_simulator.py      # Quantum simulator with 1000+ qubit support
python apps/rft_validation_suite.py   # Validation suite proving RFT uniqueness
```

## 🧪 RFT Validation
The Resonance Fourier Transform is mathematically distinct from traditional Fourier transforms:
- `apps/rft_validation_suite.py` provides rigorous mathematical proof
- Demonstrates unique spectral decomposition properties
- Validates unitarity and quantum-safe characteristics
- Compares against DFT, FFT, DCT, DST, WHT, and CZT
- Confirms phase preservation via resonance coupling

## 🖥️ Applications
- **Quantum Simulator**: Full-scale quantum simulation with Resonance-based compression
- **RFT Visualizer**: Visual demonstration of Resonance Fourier Transform properties
- **Q-Notes**: Quantum-enhanced text editor
- **Q-Vault**: Secure storage using RFT-based encryption
- **System Monitor**: Real-time monitoring of system and RFT kernel status

## 📚 Documentation
- `PROJECT_STRUCTURE.md` - Complete project architecture
- `DEVELOPMENT_MANUAL.md` - Detailed development guidelines
- `LICENSE.md` - Licensing information
- `PATENT-NOTICE.md` - Patent details
