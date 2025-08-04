# Contributing to QuantoniumOS

Thank you for your interest in contributing to QuantoniumOS! This document provides guidelines and instructions for contributing to this project.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
  - [Python Setup](#python-setup)
  - [C++ Setup](#c-setup)
  - [Rust Setup](#rust-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Core Scientific Components](#core-scientific-components)

## Getting Started

1. Fork the repository
2. Clone your fork locally: `git clone https://github.com/[your-username]/quantoniumos.git`
3. Set up the development environment (see below)
4. Run the validation tests to ensure everything works correctly

## Development Environment Setup

### Python Setup

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Verify installation
python -c "import quantoniumos; print('Setup successful!')"
```

### C++ Setup

Prerequisites:
- CMake 3.15+
- C++17 compatible compiler (GCC 9+, MSVC 2019+, or Clang 10+)
- Eigen 3.3+

```bash
# Generate build files
mkdir build && cd build
cmake ..

# Build
cmake --build . --config Release

# Run tests
ctest -C Release
```

### Rust Setup

Prerequisites:
- Rust 1.65+ and Cargo

```bash
# Navigate to Rust implementation
cd resonance-core-rs

# Build
cargo build --release

# Run tests
cargo test

# Generate documentation
cargo doc --open
```

## Development Workflow

1. Create a new branch for your feature or fix: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Validate your changes:
   - Ensure all C++ tests pass
   - Verify API endpoints function correctly
   - Check for any regression issues
4. Commit your changes with a descriptive message
5. Push your branch to your fork
6. Open a pull request against the main repository

## Coding Standards

### C++ Code

- Follow C++17 standards
- Use Eigen library for linear algebra operations
- Include OpenMP pragmas for parallel operations
- Document all functions with Doxygen-style comments
- Write comprehensive tests for new functionality

### Python Code

- Follow PEP 8 style guidelines
- Document functions with docstrings
- Use type hints where appropriate
- Write unit tests for new functionality

## Pull Request Process

1. Ensure your code passes all tests
2. Update documentation as necessary
3. Include a clear description of the changes and their purpose
4. Reference any related issues
5. Wait for code review and address any feedback

## Core Scientific Components

When modifying core scientific components, ensure:

1. Mathematical correctness is maintained
2. Performance is not degraded
3. Existing tests still pass
4. New tests are added for new functionality
5. Documentation is updated to reflect changes

## Questions?

If you have any questions about contributing, please open an issue in the repository.

Thank you for helping improve QuantoniumOS!
