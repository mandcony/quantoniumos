# RFT Visualization Suite

Real-time visualization tools for RFT algorithms and quantum parameters.

## Components

### Core Visualizers
- **`rft_visualizer.py`** - Main RFT algorithm visualizer
- **`rft_visualizer_wrapper.py`** - Wrapper for isolated execution
- **`rft_visualizer_direct.pyw`** - Direct Windows launcher
- **`rft_validation_visualizer.py`** - Validation test visualizer

### Launchers
- **`launch_rft_visualizer_isolated.py`** - Isolated environment launcher
- **`launch_rft_visualizer.bat`** - Windows batch launcher

## Features

- Real-time RFT transform visualization
- Golden ratio parameter tracking
- Unitarity preservation monitoring
- CHSH violation plotting
- Bell state visualization
- Algorithm performance metrics
- Interactive parameter adjustment

## Visualization Types

- **Transform Matrices**: RFT vs DFT comparison
- **Quantum States**: Bell states and entanglement
- **Performance**: Algorithm timing and accuracy
- **Validation**: Test results and proofs

## Usage

```bash
python rft_visualizer.py                    # Main visualizer
python rft_validation_visualizer.py         # Validation plots
python launch_rft_visualizer_isolated.py    # Isolated mode
```