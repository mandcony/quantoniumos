# Quantum Simulator Suite

This directory contains the core quantum simulation applications for QuantoniumOS.

## Components

### Core Simulator
- **`quantum_simulator.py`** - Main quantum state simulator with 1000+ symbolic qubit support
- **`quantum_crypto.py`** - Quantum-inspired cryptographic operations
- **`quantum_parameter_3d_visualizer.py`** - 3D visualization of quantum parameters

## Features

- Symbolic quantum state representation
- Bell state generation and CHSH violation testing
- Quantum entanglement simulation
- Real-time parameter visualization
- Integration with RFT compression algorithms

## Usage

```bash
python quantum_simulator.py        # Launch full simulator
python quantum_crypto.py          # Crypto operations
python quantum_parameter_3d_visualizer.py  # 3D visualization
```

## Integration

The quantum simulator integrates with:
- RFT algorithms (`algorithms/rft/`)
- Assembly kernels (`algorithms/rft/kernels/`)
- Validation tests (`tests/validation/`)