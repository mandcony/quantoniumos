# QuantoniumOS Configuration System

## Overview

QuantoniumOS uses a code-based configuration system rather than environment variables or `.env` files. This design choice was made to:

1. Enhance type safety through Python's type system
2. Provide better IDE integration and auto-completion
3. Create more explicit dependencies and configuration options
4. Simplify deployment in research environments

## Key Configuration Files

Configuration in QuantoniumOS is managed through:

1. `quantonium_os_unified.py` - Core configuration
2. `quantonium_design_system.py` - UI and visual configuration
3. JSON configuration files in `14_CONFIGURATION/` directory

## Configuration Approach

### Runtime Configuration

The system accepts command-line arguments for runtime configuration:

```bash
python launch_quantoniumos.py --quantum-dimension 16 --debug
```

### Default Configuration

Default configuration values are set in code, typically in class constructors:

```python
def __init__(self, 
             quantum_dimension=8,
             rft_size=64,
             enable_crypto=True,
             debug_mode=False):
    self.config = {
        "quantum_dimension": quantum_dimension,
        "rft_size": rft_size,
        "enable_crypto": enable_crypto,
        "debug_mode": debug_mode
    }
```

### Component-Specific Configuration

Each major component has its own configuration parameters:

1. **Quantum Engine**: Dimension, precision, simulation depth
2. **RFT Engine**: Block size, transform parameters, key derivation
3. **Crypto System**: Algorithm selection, key sizes, hash parameters
4. **UI System**: Window size, colors, fonts, component layout

## Why Not .env?

QuantoniumOS is a research system with complex interdependencies between components. Environment variables don't provide:

1. Type checking - `.env` variables are strings
2. Structure and hierarchical configuration
3. Default value inheritance
4. Complex data types (lists, dictionaries, objects)

## Adding Configuration Options

To add new configuration options:

1. Add parameters to the relevant class constructor
2. Update the internal config dictionary
3. Add getters/setters if needed
4. Document the new parameters
5. Update command-line argument parsing if appropriate

## Example Configuration Usage

```python
# Creating a QuantoniumOS instance with custom configuration
from quantonium_os_unified import QuantoniumOSUnified

# Initialize with custom parameters
os = QuantoniumOSUnified(
    quantum_dimension=16,
    enable_crypto=True,
    debug_mode=True,
    ui_theme="dark"
)

# Access configuration
print(f"Quantum dimension: {os.config['quantum_dimension']}")
```
