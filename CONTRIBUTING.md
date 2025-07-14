# Contributing to QuantoniumOS

Thank you for your interest in contributing to QuantoniumOS! This document provides guidelines and instructions for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up the development environment using `.\setup_local_env.ps1`
4. Run the validation tests to ensure everything works correctly:
   - `.\run_simple_test.bat` for C++ components
   - `python test_api_simple.py` for API endpoints

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
