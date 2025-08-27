# QuantoniumOS Project Structure

## Core Components

### 1. RFT Assembly Implementation (Foundation Layer)
- `ASSEMBLY/` - Contains the proven RFT implementation
  - `compiled/` - Compiled binaries including librftkernel.dll
  - `python_bindings/` - Python bindings for the RFT kernel
  - `build_integrated_os.bat` - Assembly build script

### 2. Quantum Engine Layer
- `engines/` - Quantum computing engines
  - `canonical_true_rft.py` - Canonical RFT implementation
  - `rft_core.py` - Core RFT algorithms
  - `paper_compliant_rft_fixed.py` - Paper-compliant RFT
- `core/` - Quantum kernel implementations
  - `bulletproof_quantum_kernel.py` - Production-ready quantum kernel
  - `working_quantum_kernel.py` - Tested quantum kernel implementation
  - `topological_quantum_kernel.py` - Topological quantum algorithms

### 3. Frontend Integration Layer (NEW)
- `frontend/` - Main OS interface
  - `quantonium_desktop.py` - Primary desktop manager with RFT integration
- `apps/` - Quantum-enhanced applications
  - `q_notes.py` - Quantum text editor
  - `q_vault.py` - RFT-encrypted secure storage
  - `qshll_system_monitor.py` - System monitor with RFT status
- `ui/` - User interface components
  - `styles.qss` - Unified QuantoniumOS stylesheet

### 4. Cryptography Layer
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
