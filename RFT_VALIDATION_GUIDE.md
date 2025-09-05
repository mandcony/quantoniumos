# RFT Scientific Validation Suite - QuantoniumOS

This module provides a comprehensive validation suite for the Resonance Fourier Transform (RFT) implementation in QuantoniumOS. The validation suite tests the mathematical properties, performance characteristics, and cryptographic properties of the RFT implementation.

## Overview

The RFT validation suite is designed to rigorously test the RFT implementation to ensure that it meets the specified requirements and performs as expected. The suite includes tests for:

### A) Mathematical validity (core science)
- **Unitarity / Invertibility**: Tests forward→inverse round-trip on random vectors
- **Energy conservation (Plancherel)**: Tests energy preservation ‖x‖₂² vs ‖RFT(x)‖₂²
- **Operator properties distinct from DFT**: Tests to prove RFT ≠ DFT
- **Linearity**: Tests RFT(ax+by) = a·RFT(x)+b·RFT(y)
- **Time/frequency localization**: Tests expected concentration patterns

### B) Algorithmic performance & numerical robustness
- **Asymptotic scaling**: Tests runtime vs N scaling behavior
- **Precision sweeps**: Tests float32 vs float64 precision
- **CPU feature dispatch**: Tests scalar/AVX2/AVX-512 paths for correctness
- **Stress testing**: Tests long-running operations for stability

### C) Cryptography-adjacent properties
- **Avalanche effect**: Tests bit-flip diffusion
- **Randomness properties**: Tests statistical properties of outputs
- **Side-channel resistance**: Tests for timing and other side-channels

## Usage

The validation suite can be run in several ways:

### Command Line Interface

Run the validation suite from the command line:

```bash
python rft_scientific_validation.py [options]
```

Options:
- `--quick`: Run a quick validation with smaller transform sizes
- `--math-only`: Run only mathematical validity tests
- `--perf-only`: Run only performance tests
- `--crypto-only`: Run only cryptography tests
- `--report FILE`: Save validation report to specified file

Or use the convenience batch script:

```bash
validate_rft.bat [options]
```

### Test Driver

For more flexibility, use the test driver script:

```bash
python test_rft_validation.py [options]
```

Options:
- `--gui`: Launch the GUI visualizer
- `--quick`: Run a quick validation with smaller sizes
- `--math-only`: Run only mathematical validity tests
- `--perf-only`: Run only performance tests
- `--crypto-only`: Run only cryptography tests
- `--report FILE`: Save validation report to specified file

### GUI Visualizer

For interactive testing and visualization, launch the GUI visualizer:

```bash
python apps/rft_validation_visualizer.py
```

The GUI provides:
- Interactive test selection and configuration
- Real-time progress tracking
- Visual results presentation
- Export capabilities for reports and test vectors

## Integration with QuantoniumOS

The RFT validation suite is integrated with the QuantoniumOS desktop environment. Launch it from the desktop by clicking on the "RFT Scientific Validator" icon.

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib (for visualization)
- PyQt5 (for GUI)

## Test Vectors

The validation suite can generate reproducible test vectors for independent verification. Use the "Export Test Vectors" feature in the GUI or run:

```bash
python test_rft_validation.py --report test_vectors.json
```

## Expected Results

For a valid RFT implementation, all tests should pass with the following thresholds:

- **Unitarity**: max error ≤ 1e-12 (float64), mean error ≤ 1e-13 (float64)
- **Energy conservation**: relative error ≤ 1e-12 (float64)
- **Operator distinctness**: RFT must be provably different from DFT
- **Linearity**: relative error ≤ 1e-12 (float64)
- **Asymptotic scaling**: runtime within 25% of O(N log N)

## Further Documentation

For detailed information on the RFT algorithm, mathematical properties, and implementation details, please refer to the associated papers and technical documentation in the `docs` directory.
