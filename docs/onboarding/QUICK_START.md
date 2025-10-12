# Quick Start & Setup Guide

This guide provides the fastest path to getting QuantoniumOS running on your local machine.

## 1. Environment Setup

First, set up your Python virtual environment and install the required dependencies.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Optional: For desktop UI testing
pip install PyQt5
```

**Note:** Keep the virtual environment active for all subsequent build and run commands.

## 2. Build Native Kernels

Compile the C-based assembly kernels for maximum performance. The system will gracefully fall back to a pure Python implementation if this step is skipped, but performance will be significantly reduced.

```bash
# Build the release version of the C kernels
make -C src/assembly all

# Install the Python bindings for the compiled kernels
make -C src/assembly install
```

After any changes to the C source files in `src/assembly/kernel/`, re-run the commands above.

## 3. Run the Validation Suite

After a fresh installation or any changes, run the core validation suite to ensure all components are functioning correctly.

```bash
# Run the comprehensive validation script
python tests/comprehensive_validation_suite.py
```

This script consolidates the key tests from the original development manual into a single, easy-to-run command. Look for a "Validation PASSED" message at the end.

## 4. Boot the System

You are now ready to run the full QuantoniumOS desktop environment.

```bash
# Boot the main application
python quantonium_boot.py
```

If PyQt5 is not installed, the system will launch in a console-only mode.
