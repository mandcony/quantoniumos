# Building QuantoniumOS

This document provides instructions for building QuantoniumOS from source.

## Prerequisites

- Python 3.11 or higher
- Rust toolchain (with Cargo)
- C++ compiler (GCC 11+, MSVC, or Clang)
- CMake 3.15+
- Git

## Dependencies

All Python dependencies are managed through `pyproject.toml` and can be installed with:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Building with Maturin

QuantoniumOS uses Maturin to build the Rust components and integrate them with Python:

```bash
maturin develop --release
```

## Building the C++ Components

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Running Tests

```bash
pytest
```

## Cross-platform Considerations

### Windows

On Windows, make sure to use a developer command prompt for MSVC or have the compiler in your PATH.

### macOS

On macOS, ensure XCode command line tools are installed.

### Linux

On Linux, install development packages for your distribution.
