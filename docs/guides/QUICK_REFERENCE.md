# QuantoniumOS Quick Reference

## Installation

```bash
# One-command setup (recommended)
./quantoniumos-bootstrap.sh

# Installation modes
./quantoniumos-bootstrap.sh --minimal     # Core only
./quantoniumos-bootstrap.sh --dev         # With dev tools
./quantoniumos-bootstrap.sh --hardware    # With HW tools
./quantoniumos-bootstrap.sh --force-rebuild  # Rebuild native modules
```

## Daily Usage

```bash
# Activate environment
source .venv/bin/activate

# Run system validation
python validate_system.py

# Run all benchmarks
python benchmarks/run_all_benchmarks.py

# Run specific benchmark classes
python benchmarks/run_all_benchmarks.py A B  # Quantum + Transform
python benchmarks/run_all_benchmarks.py C D  # Compression + Crypto

# Check dependencies
python benchmarks/run_all_benchmarks.py --deps

# Save benchmark results
python benchmarks/run_all_benchmarks.py --json results.json
```

## Testing

```bash
# Quick validation
pytest tests/validation/ -v

# All tests
pytest tests/ -v

# Specific test categories
pytest tests/rft/ -v           # RFT core tests
pytest tests/crypto/ -v        # Cryptography tests
pytest tests/benchmarks/ -v    # Benchmark tests

# With coverage
pytest tests/ --cov=algorithms --cov-report=html
```

## Hardware Verification

```bash
cd hardware

# Run verification suite
bash verify_fixes.sh

# Generate test vectors
python generate_hardware_test_vectors.py

# Create figures
bash generate_all_figures.sh

# Visualize results
python visualize_hardware_results.py
```

## Python API Quick Examples

### Basic RFT Transform

```python
import numpy as np
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT

# Create RFT engine
rft = CanonicalTrueRFT(size=256)

# Transform data
x = np.random.randn(256)
y = rft.transform(x)

# Check unitarity
error = rft.get_unitarity_error()
print(f"Unitarity error: {error:.2e}")
```

### Using Native Module

```python
import sys
import numpy as np
sys.path.insert(0, 'src/rftmw_native/build')

from unitary_rft import UnitaryRFT

# Create native RFT
rft = UnitaryRFT(512)

# Forward/inverse transform
x = np.random.randn(512)
y = rft.forward(x)
x_reconstructed = rft.inverse(y)

# Check reconstruction error
error = np.max(np.abs(x - x_reconstructed))
print(f"Reconstruction error: {error:.2e}")
```

### RFT-Based Compression

```python
from algorithms.rft.compression.rftmw_codec import RFTMWCodec

# Create codec
codec = RFTMWCodec(block_size=1024)

# Compress data
data = np.random.randn(10000)
compressed = codec.compress(data)
decompressed = codec.decompress(compressed)

# Check compression ratio
ratio = len(data) / len(compressed)
print(f"Compression ratio: {ratio:.2f}x")
```

### Cryptographic Hash

```python
from algorithms.rft.crypto.rft_sis_hash import RFTSISHashV31, Point2D

# Create hash function
hasher = RFTSISHashV31()

# Hash a point
point = Point2D(1.234, 5.678)
hash_value = hasher.hash_point(point)
print(f"Hash: {hash_value.hex()}")
```

## Project Structure

```
quantoniumos/
├── algorithms/              # Core algorithms
│   └── rft/
│       ├── core/           # RFT implementations
│       ├── compression/    # Compression codecs
│       └── crypto/         # Cryptographic primitives
├── benchmarks/             # Benchmark suite
│   ├── class_a_*.py       # Quantum simulation
│   ├── class_b_*.py       # Transform/DSP
│   ├── class_c_*.py       # Compression
│   ├── class_d_*.py       # Cryptography
│   └── class_e_*.py       # Audio/DAW
├── docs/                   # Documentation
│   ├── validation/         # Proofs and theorems
│   ├── technical/          # Technical guides
│   └── manuals/           # User manuals
├── hardware/               # Hardware verification
├── src/                    # Source code
│   ├── apps/              # Applications
│   └── rftmw_native/      # Native modules
├── tests/                  # Test suites
├── experiments/            # Research experiments
└── tools/                 # Development tools
```

## Key Files

| File | Purpose |
|------|---------|
| `quantoniumos-bootstrap.sh` | Complete system setup |
| `organize-release.sh` | Create distribution package |
| `validate_system.py` | System validation |
| `pyproject.toml` | Python project config |
| `requirements.txt` | Python dependencies |
| `pytest.ini` | Test configuration |

## Documentation Paths

| Topic | Location |
|-------|----------|
| Main README | `README.md` |
| Installation | `INSTALL.md` (in release packages) |
| API Documentation | `docs/api/` |
| Validation Proofs | `docs/validation/RFT_THEOREMS.md` |
| Benchmark Results | `docs/research/benchmarks/VERIFIED_BENCHMARKS.md` |
| Developer Manual | `docs/manuals/COMPLETE_DEVELOPER_MANUAL.md` |
| Troubleshooting | `docs/technical/guides/TROUBLESHOOTING.md` |
| License Info | `LICENSE.md`, `LICENSE-CLAIMS-NC.md` |
| Patent Notices | `PATENT_NOTICE.md` |

## Environment Variables

```bash
# Add native module to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}/src/rftmw_native/build"

# Enable verbose CMake output
export CMAKE_VERBOSE_MAKEFILE=ON

# Set number of build threads
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

# Enable optimization flags
export CXXFLAGS="-O3 -march=native"
```

## Common Tasks

### Update Dependencies
```bash
pip install --upgrade pip
pip install -e ".[dev,ai,image]" --upgrade
```

### Rebuild Native Module
```bash
cd src/rftmw_native
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_LTO=ON
make -j$(nproc)
```

### Create Release Package
```bash
./organize-release.sh
# Package created in release/ directory
```

### Clean Build Artifacts
```bash
# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Clean native build
rm -rf src/rftmw_native/build

# Clean test artifacts
rm -rf .pytest_cache htmlcov .coverage
```

## Performance Tips

1. **Use Native Module**: 10-100x faster than pure Python
2. **Enable LTO**: Use `-DENABLE_LTO=ON` in CMake
3. **March Native**: Add `-march=native` for CPU-specific optimizations
4. **Parallel Build**: Use `make -j$(nproc)` for faster compilation
5. **Cache Builds**: Install `ccache` for incremental builds

## Getting Help

```bash
# Bootstrap help
./quantoniumos-bootstrap.sh --help

# Benchmark help
python benchmarks/run_all_benchmarks.py --help

# Python API help
python -c "from algorithms.rft.core import CanonicalTrueRFT; help(CanonicalTrueRFT)"

# Test help
pytest --help
```

## Version Information

```bash
# Check Python version
python --version

# Check package versions
pip list | grep -E "(numpy|scipy|torch|transformers)"

# Check native module
python -c "import sys; sys.path.insert(0, 'src/rftmw_native/build'); import rftmw_native; print('Native module OK')"

# Check CMake version
cmake --version

# Check compiler version
gcc --version
```

## Useful Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias qos-activate='source /path/to/quantoniumos/.venv/bin/activate'
alias qos-test='pytest tests/ -v'
alias qos-bench='python benchmarks/run_all_benchmarks.py'
alias qos-validate='python validate_system.py'
alias qos-hw='cd /path/to/quantoniumos/hardware && bash verify_fixes.sh'
```

---

**For more details, see the complete documentation in `docs/DOCS_INDEX.md`**
