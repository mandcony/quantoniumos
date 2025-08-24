# QuantoniumOS Configuration Guide

## Configuration Approach

QuantoniumOS does not use `.env` files for configuration. Instead, the system uses:

1. **Direct imports and class initialization** - Configuration is handled through Python class initialization and parameters
2. **JSON configuration files** - For persistent settings that need to be stored between runs
3. **Command-line arguments** - For runtime configuration options

## How QuantoniumOS Handles Configuration

The main configuration is handled through:

1. `quantonium_os_unified.py` - Core system configuration
2. `quantonium_design_system.py` - UI and design configuration
3. JSON files in the `14_CONFIGURATION` directory

## Key Configuration Parameters

Configuration parameters are set during initialization of key classes:

```python
# Example: Initializing the QuantoniumOS class with configuration
os_instance = QuantoniumOS(
    quantum_dimension=8,
    rft_size=64,
    enable_crypto=True,
    debug_mode=False
)
```

## Runtime Configuration

For runtime configuration, command-line arguments are used:

```bash
python launch_quantoniumos.py --quantum-dimension 16 --debug
```

## Flask Server Configuration

The Flask server (when installed) uses in-code configuration rather than environment variables:

```python
app = Flask("QuantoniumOS")
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['DEBUG'] = False
```

## Why No .env Files?

QuantoniumOS was designed as a research system with configuration that doesn't change frequently. Using direct Python configuration provides:

1. Type safety
2. Better IDE integration
3. More explicit dependencies
4. Simpler deployment in research environments

If you need to add `.env` support in the future, you would need to install the `python-dotenv` package and modify the initialization code to load variables from a `.env` file.
