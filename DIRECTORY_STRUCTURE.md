# QuantoniumOS Directory Structure

## Core Implementation
- `src/` - Main source code
  - `core/` - Mathematical algorithms (RFT, cryptography)
  - `assembly/kernel/` - C implementation of RFT kernel with SIMD optimization
  - `apps/` - PyQt5 applications (Q-Notes, Q-Vault, Quantum Simulator, etc.)
  - `frontend/` - Desktop environment (quantonium_desktop.py)
  - `engine/` - System engines and components

## Development & Testing
- `tests/` - Testing framework
  - `crypto/` - Cryptographic validation and analysis
  - `benchmarks/` - Performance testing
  - `results/` - Test results and reports
- `dev/` - Development tools and utilities
  - `scripts/` - Development and analysis scripts
  - `tools/` - Development utilities
  - `examples/` - Example code and usage

## Documentation
- `docs/` - Technical documentation
  - `reports/` - Technical analysis and validation reports
  - `safety/` - AI safety documentation
  - `technical/` - Technical specifications
  - `audits/` - Audit reports

## Configuration & Assets
- `data/` - Configuration and data files
  - `config/` - System configuration files
  - `logs/` - System logs and chat records
  - `weights/` - Model weights and parameters
- `ui/` - User interface assets
  - `icons/` - Application icons
  - `styles_dark.qss`, `styles_light.qss` - UI themes

## Application Data
- `QNotes/` - Q-Notes application data
- `QVault/` - Q-Vault secure storage data

## Archive & Backup
- `archive/` - Historical data and backups

## Root Files
- `quantonium_boot.py` - Main system launcher
- `requirements.txt` - Python dependencies
- `README.md` - Project overview and usage
- `QUICK_START.md` - Quick start guide
