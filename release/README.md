# QuantoniumOS Release Packages

This directory contains release packages for QuantoniumOS - complete, ready-to-deploy bundles with all necessary components.

## Available Packages

### Current Release
- `quantoniumos-benchmarks-20251201.zip` - Complete benchmark suite and core components

### Package Contents

Each release package includes:
- ✓ **Core Algorithms**: All RFT variants and implementations
- ✓ **Native Modules**: C++/Assembly kernel sources
- ✓ **Benchmarks**: Full competitive analysis suite (Classes A-E)
- ✓ **Documentation**: Complete technical and user documentation
- ✓ **Hardware**: SystemVerilog verification modules
- ✓ **Tests**: Comprehensive validation suites
- ✓ **Tools**: Development and analysis utilities
- ✓ **Bootstrap Script**: One-command installation

## Creating New Release Packages

Run the organizer script from the project root:

```bash
# Standard release package
./organize-release.sh

# Minimal package (core only)
./organize-release.sh --minimal

# Include large AI models (not recommended for distribution)
./organize-release.sh --with-models
```

## Installing from a Package

### Quick Installation

1. **Extract the package:**
   ```bash
   unzip quantoniumos-complete-*.zip
   # or
   tar -xzf quantoniumos-complete-*.tar.gz
   ```

2. **Enter the directory:**
   ```bash
   cd quantoniumos-complete-*
   ```

3. **Run the bootstrap installer:**
   ```bash
   ./quantoniumos-bootstrap.sh
   ```

### Installation Modes

```bash
# Full installation (recommended)
./quantoniumos-bootstrap.sh

# Minimal (core only, no AI/ML dependencies)
./quantoniumos-bootstrap.sh --minimal

# Development environment (includes test tools)
./quantoniumos-bootstrap.sh --dev

# Hardware synthesis environment
./quantoniumos-bootstrap.sh --hardware

# Skip virtual environment creation
./quantoniumos-bootstrap.sh --skip-venv

# Force rebuild of native modules
./quantoniumos-bootstrap.sh --force-rebuild
```

## System Requirements

### Minimum
- **OS**: Linux, macOS, or Windows with WSL2
- **Python**: 3.9+
- **RAM**: 8GB
- **Disk**: 5GB free space
- **CPU**: x86_64 with SSE4.2

### Recommended
- **RAM**: 16GB+
- **CPU**: Modern multi-core processor with AVX2
- **Disk**: 10GB+ free space

### Required System Packages

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential cmake gcc g++ nasm python3 python3-pip git
```

**macOS:**
```bash
brew install cmake nasm python3 git
```

## Verification

After installation, verify everything works:

```bash
# Activate virtual environment (if created)
source .venv/bin/activate

# Run system validation
python validate_system.py

# Run quick tests
pytest tests/validation/ -v

# Run benchmarks
python benchmarks/run_all_benchmarks.py
```

## Package Structure

```
quantoniumos-complete-*/
├── quantoniumos-bootstrap.sh  # Main installer script
├── INSTALL.md                 # Detailed installation guide
├── MANIFEST.txt              # Package contents manifest
├── README.md                 # Project overview
├── pyproject.toml            # Python project configuration
├── requirements.txt          # Python dependencies
├── algorithms/               # Core RFT algorithms
├── benchmarks/              # Performance benchmarks
├── docs/                    # Documentation
├── hardware/                # Hardware verification
├── src/                     # Source code
│   ├── apps/               # Applications
│   └── rftmw_native/       # Native modules (C++/ASM)
├── tests/                   # Test suites
├── experiments/            # Research experiments
└── tools/                  # Development tools
```

## Checksum Verification

Each package includes SHA256 checksums for integrity verification:

```bash
# Verify TAR.GZ package
sha256sum -c quantoniumos-complete-*.tar.gz.sha256

# Verify ZIP package
sha256sum -c quantoniumos-complete-*.zip.sha256
```

## Distribution Notes

### What's Included
- All source code and algorithms
- Native module sources (buildable on your system)
- Complete documentation
- Test suites and benchmarks
- Hardware verification tools

### What's NOT Included
- Pre-compiled binaries (built during installation)
- Large AI/ML model weights (unless `--with-models` used)
- Build artifacts and caches
- Development logs and temporary files

### License
All packages include complete licensing information:
- `LICENSE.md` - Main project license
- `LICENSE-CLAIMS-NC.md` - Patent claims license (research/education)
- `PATENT_NOTICE.md` - Patent notices
- `CLAIMS_PRACTICING_FILES.txt` - Files covered by patent claims

## Troubleshooting

### Installation Issues
1. Check `bootstrap.log` in the package directory
2. Ensure all system dependencies are installed
3. Try `--force-rebuild` if native modules fail

### Import Errors
```bash
source .venv/bin/activate  # Activate virtual environment
pip install -e .           # Reinstall in development mode
```

### Native Module Build Fails
```bash
cd src/rftmw_native/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON
make VERBOSE=1
```

## Support

- **Documentation**: See `docs/DOCS_INDEX.md`
- **Troubleshooting**: See `docs/technical/guides/TROUBLESHOOTING.md`
- **Examples**: See `experiments/README.md`
- **Validation**: See `docs/validation/RFT_THEOREMS.md`

## Version History

- **20251201**: Initial complete benchmark suite package
  - All benchmark classes (A-E)
  - Complete documentation
  - Hardware verification
  - Native module sources

---

For the latest releases and updates, see the main QuantoniumOS repository.
