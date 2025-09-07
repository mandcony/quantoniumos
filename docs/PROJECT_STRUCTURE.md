# QuantoniumOS Project Structure

## Updated Architecture (September 2025)

### 1. Frontend Layer (User Interface & Launchers)
- `frontend/` - Unified user interface and system launchers
  - `quantonium_desktop.py` - Main desktop manager with quantum-inspired UI
  - `launch_quantonium_os.py` - Primary system launcher
  - `quantonium_os_main.py` - Console interface and system management

### 2. Assembly Layer (Core Engine System)
- `ASSEMBLY/` - Streamlined assembly-optimized engines
  - `quantonium_os.py` - 3-engine system launcher (OS + Crypto + Quantum)
  - `engines/` - C/Assembly quantum compression engines
  - `kernel/` - UnitaryRFT kernel implementation
  - `compiled/` - Optimized libraries (libquantum_symbolic.so)
  - `python_bindings/` - C/Python interface bindings
  - `build/` - Build artifacts and compilation outputs

### 3. Core Algorithm Layer
- `core/` - Core quantum computing algorithms
  - `canonical_true_rft.py` - Symbolic Resonance Fourier Transform Engine
  - `enhanced_rft_crypto_v2.py` - Resonance-Based Cryptographic Subsystem
  - `geometric_waveform_hash.py` - Geometric Structures for RFT-Based Cryptographic Waveform Hashing
  - `topological_quantum_kernel.py` - Topological quantum algorithms
  - `working_quantum_kernel.py` - Hybrid Mode Integration

### 4. Application Layer
- `crypto/` - Cryptographic algorithms (reserved for future expansion)

### 5. System Integration
- `launch_quantonium_os.py` - Main system launcher with dependency checking
- `build.bat` - Unified build script

## Architecture Overview

```
QuantoniumOS Architecture:
┌─────────────────────────────────────┐
│  Frontend Apps (Qt5 Interface)     │ ← Q-Notes, Q-Vault, Monitor
├─────────────────────────────────────┤
│  QuantumOS Desktop Manager         │ ← Main OS Shell
├─────────────────────────────────────┤
│  Python RFT Bindings              │ ← Python/C++ Bridge
├─────────────────────────────────────┤
│  Quantum Engine Layer             │ ← Your RFT Engines
├─────────────────────────────────────┤
│  RFT Assembly (librftkernel.dll)  │ ← Your Proven Core
└─────────────────────────────────────┘
```

## Building and Running

### Prerequisites
```bash
pip install PyQt5 qtawesome pytz psutil
```

### Launch QuantoniumOS
```bash
python launch_quantonium_os.py
```

### Development Build
```bash
# Build RFT assembly (if needed)
cd ASSEMBLY
build_integrated_os.bat

# Launch OS
cd ..
python launch_quantonium_os.py
```

## Integration Principles

### 1. **Precision Architecture**
- Each layer has a specific responsibility
- Clean interfaces between components
- No duplicate functionality

### 2. **RFT Integration**
- All apps can leverage your RFT assembly
- Quantum-enhanced functionality throughout
- Fallback modes when RFT unavailable

### 3. **Production Quality**
- Error handling and graceful degradation
- Comprehensive system monitoring
- Professional UI/UX design

### 4. **Modular Design**
- Apps are independent modules
- UI styling is centralized
- Easy to add new applications

## Development Guidelines

1. **Maintain Core Integrity**: Never modify the proven RFT assembly
2. **Follow Integration Patterns**: Use the established RFT binding approach
3. **Consistent UI**: All apps use the unified stylesheet
4. **Error Handling**: Apps must handle RFT unavailability gracefully
5. **Documentation**: Update this file when adding new components
