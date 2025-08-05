# Project Structure Documentation

This document explains the purpose of key files and directories in the QuantoniumOS project, with special focus on clarifying files with similar names that exist in different directories.

## Root Directory Files

- **models.py**: Defines Pydantic models for API validation and request schemas
- **routes.py**: Main API routes for symbolic stack modules with token-protected endpoints
- **main.py**: Application entry point
- **app.py**: Flask application setup and configuration

## Directory Structure

### api/ Directory
- **routes.py**: FastAPI routes specifically for the patent validation system
- **resonance_metrics.py**: Implementation of resonance-related algorithms and metrics
- **symbolic_interface.py**: Interface for symbolic engine interactions

### auth/ Directory
- **routes.py**: Authentication-related API routes (key management, JWT auth)
- **models.py**: SQLAlchemy database models for authentication entities
- **jwt_auth.py**: JWT authentication implementation
- **secret_manager.py**: Cryptographic key management and rotation

### core/ Directory
Contains the core algorithmic implementations and computational modules:
- **config.py**: Configuration for core modules
- **encryption/**: Encryption-related implementations
- **testing/**: Internal testing modules

### quantoniumos/ Directory
Contains the C++ implementations and bindings:
- **secure_core/**: C++ secure implementation of core algorithms
- **src/**: C++ source files

## Workflow Files

### .github/workflows/
- **green-wall-ci.yml**: Main CI pipeline for Python and C++ validation
- **cross-implementation-validation.yml**: Cross-implementation validation for all branches
- **cross-validation.yml**: Path-specific cross-implementation validation (only runs when specific files change)

## Build System

- **CMakeLists.txt** (root): Main CMake configuration file for the entire project
- **pyproject.toml**: Python project configuration
- **setup_local_env.py**: Environment setup script

## Next Steps for Project Structure Improvement

1. Consider adopting a more standardized Python package structure
2. Move route definitions into appropriate module directories
3. Consolidate similar functionality into shared modules
4. Create a more consistent naming scheme for parallel implementations
