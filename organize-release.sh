#!/usr/bin/env bash
################################################################################
# QuantoniumOS Project Organizer
# ===============================
# 
# Creates a release-ready package with all necessary components:
# - Source code
# - Documentation
# - Benchmarks and validation suites
# - Build scripts
# - Hardware verification
# - Example experiments
#
# Usage:
#   ./organize-release.sh                    # Create standard release
#   ./organize-release.sh --minimal          # Core only
#   ./organize-release.sh --with-models      # Include AI models (large)
#
################################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="${PROJECT_ROOT}/release"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="quantoniumos-complete-${TIMESTAMP}"
PACKAGE_DIR="${RELEASE_DIR}/${PACKAGE_NAME}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
INCLUDE_MODELS=false
MINIMAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-models) INCLUDE_MODELS=true; shift ;;
        --minimal) MINIMAL=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          QuantoniumOS Release Package Creator                 ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Create package directory structure
echo -e "${BLUE}[1/8]${NC} Creating package directory structure..."
mkdir -p "$PACKAGE_DIR"/{algorithms,benchmarks,docs,hardware,src,tests,tools,experiments,data}

# Copy core algorithms
echo -e "${BLUE}[2/8]${NC} Copying core algorithms..."
rsync -a --exclude='__pycache__' \
    "$PROJECT_ROOT/algorithms/" \
    "$PACKAGE_DIR/algorithms/"

# Copy benchmarks
echo -e "${BLUE}[3/8]${NC} Copying benchmark suite..."
rsync -a --exclude='__pycache__' \
    "$PROJECT_ROOT/benchmarks/" \
    "$PACKAGE_DIR/benchmarks/"

# Copy documentation
echo -e "${BLUE}[4/8]${NC} Copying documentation..."
rsync -a \
    "$PROJECT_ROOT/docs/" \
    "$PACKAGE_DIR/docs/"

# Copy hardware components
echo -e "${BLUE}[5/8]${NC} Copying hardware verification..."
rsync -a --exclude='sim_*' --exclude='*.vcd' \
    "$PROJECT_ROOT/hardware/" \
    "$PACKAGE_DIR/hardware/"

# Copy source code
echo -e "${BLUE}[6/8]${NC} Copying source code..."
rsync -a --exclude='__pycache__' --exclude='build' \
    "$PROJECT_ROOT/src/" \
    "$PACKAGE_DIR/src/"
rsync -a --exclude='__pycache__' \
    "$PROJECT_ROOT/quantonium_os_src/" \
    "$PACKAGE_DIR/quantonium_os_src/"

# Copy tests (if not minimal)
if [ "$MINIMAL" = false ]; then
    echo -e "${BLUE}[7/8]${NC} Copying test suites..."
    rsync -a --exclude='__pycache__' \
        "$PROJECT_ROOT/tests/" \
        "$PACKAGE_DIR/tests/"
else
    echo -e "${BLUE}[7/8]${NC} Skipping tests (minimal mode)..."
fi

# Copy experiments and tools
echo -e "${BLUE}[8/8]${NC} Copying experiments and tools..."
rsync -a --exclude='__pycache__' \
    "$PROJECT_ROOT/experiments/" \
    "$PACKAGE_DIR/experiments/"
rsync -a --exclude='__pycache__' \
    "$PROJECT_ROOT/tools/" \
    "$PACKAGE_DIR/tools/"

# Copy root files
echo "Copying root configuration files..."
cp "$PROJECT_ROOT/pyproject.toml" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/requirements.txt" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/requirements.in" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/pytest.ini" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/validate_system.py" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/README.md" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/LICENSE.md" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/LICENSE-CLAIMS-NC.md" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/PATENT_NOTICE.md" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/SECURITY.md" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/CITATION.cff" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/CLAIMS_PRACTICING_FILES.txt" "$PACKAGE_DIR/"

# Copy bootstrap script
cp "$PROJECT_ROOT/quantoniumos-bootstrap.sh" "$PACKAGE_DIR/"
chmod +x "$PACKAGE_DIR/quantoniumos-bootstrap.sh"

# Create installation README
cat > "$PACKAGE_DIR/INSTALL.md" << 'EOF'
# QuantoniumOS Installation Guide

## Quick Start

```bash
# Make bootstrap script executable
chmod +x quantoniumos-bootstrap.sh

# Run full installation
./quantoniumos-bootstrap.sh

# Or choose a specific mode:
./quantoniumos-bootstrap.sh --minimal    # Core only
./quantoniumos-bootstrap.sh --dev        # Development environment
./quantoniumos-bootstrap.sh --hardware   # Hardware synthesis tools
```

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows with WSL2
- **Python**: 3.9 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 5GB free space (50GB with AI models)
- **CPU**: x86_64 with SSE4.2 support

### Required System Packages

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    gcc g++ \
    make \
    nasm \
    python3 python3-pip python3-venv \
    git
```

**macOS:**
```bash
brew install cmake nasm python3 git
```

**Fedora/RHEL:**
```bash
sudo dnf install -y \
    gcc gcc-c++ \
    cmake \
    make \
    nasm \
    python3 python3-pip \
    git
```

## Manual Installation

If you prefer manual setup:

### 1. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### 2. Install Python Dependencies
```bash
# Core dependencies
pip install -e ".[dev]"

# Full installation with AI/ML support
pip install -e ".[dev,ai,image]"
```

### 3. Build Native Modules
```bash
cd src/rftmw_native
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_LTO=ON
make -j$(nproc)
cd ../../..
```

### 4. Verify Installation
```bash
python validate_system.py
```

## Running Tests

```bash
# Quick validation
python validate_system.py

# Full test suite
pytest tests/ -v

# Run benchmarks
python benchmarks/run_all_benchmarks.py

# Specific benchmark classes
python benchmarks/run_all_benchmarks.py A B  # Quantum and Transform only
```

## Hardware Verification

```bash
cd hardware
bash verify_fixes.sh
```

## Directory Structure

```
quantoniumos/
├── algorithms/          # Core RFT algorithms
├── benchmarks/          # Performance benchmarks
├── docs/               # Documentation
├── hardware/           # Hardware verification
├── src/                # Source code
│   ├── apps/          # Applications
│   └── rftmw_native/  # Native C++/ASM modules
├── tests/             # Test suites
├── experiments/       # Research experiments
└── tools/             # Development tools
```

## Troubleshooting

### Native Module Build Fails
```bash
# Force rebuild
./quantoniumos-bootstrap.sh --force-rebuild

# Or manually
cd src/rftmw_native
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make VERBOSE=1
```

### Python Import Errors
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall in development mode
pip install -e .
```

### Hardware Tests Fail
```bash
# Install Verilog simulator
sudo apt-get install iverilog gtkwave  # Ubuntu/Debian
brew install icarus-verilog            # macOS
```

## Getting Help

- **Documentation**: See `docs/DOCS_INDEX.md`
- **Issues**: Check `docs/technical/guides/TROUBLESHOOTING.md`
- **Examples**: Explore `experiments/README.md`

## License

See `LICENSE.md` and `LICENSE-CLAIMS-NC.md` for licensing information.
See `PATENT_NOTICE.md` for patent-related notices.
EOF

# Create package manifest
cat > "$PACKAGE_DIR/MANIFEST.txt" << EOF
QuantoniumOS Complete Package
============================

Package:     ${PACKAGE_NAME}
Created:     $(date)
Mode:        $([ "$MINIMAL" = true ] && echo "Minimal" || echo "Complete")
AI Models:   $([ "$INCLUDE_MODELS" = true ] && echo "Included" || echo "Excluded")

Contents:
  ✓ Core algorithms and RFT implementations
  ✓ Native C++/Assembly kernels (source)
  ✓ Benchmark suite (Classes A-E)
  ✓ Complete documentation
  ✓ Hardware verification (SystemVerilog)
  ✓ Test suites and validation
  ✓ Development tools
  ✓ Research experiments
  ✓ Bootstrap installation script

System Requirements:
  • Python 3.9+
  • GCC/G++ compiler
  • CMake 3.15+
  • NASM assembler
  • 8GB+ RAM

Quick Start:
  1. Extract this package
  2. Run: ./quantoniumos-bootstrap.sh
  3. Follow the on-screen instructions

For detailed installation instructions, see INSTALL.md
EOF

# Create archive
echo ""
echo "Creating compressed archive..."
cd "$RELEASE_DIR"
tar -czf "${PACKAGE_NAME}.tar.gz" "$PACKAGE_NAME"
zip -r -q "${PACKAGE_NAME}.zip" "$PACKAGE_NAME"

# Calculate sizes
TAR_SIZE=$(du -h "${PACKAGE_NAME}.tar.gz" | cut -f1)
ZIP_SIZE=$(du -h "${PACKAGE_NAME}.zip" | cut -f1)

# Generate checksums
echo "Generating checksums..."
sha256sum "${PACKAGE_NAME}.tar.gz" > "${PACKAGE_NAME}.tar.gz.sha256"
sha256sum "${PACKAGE_NAME}.zip" > "${PACKAGE_NAME}.zip.sha256"

echo ""
echo -e "${GREEN}✓ Release package created successfully!${NC}"
echo ""
echo "Package Details:"
echo "  Name:        $PACKAGE_NAME"
echo "  Location:    $RELEASE_DIR"
echo "  TAR.GZ:      $TAR_SIZE"
echo "  ZIP:         $ZIP_SIZE"
echo ""
echo "Files created:"
echo "  • ${PACKAGE_NAME}.tar.gz"
echo "  • ${PACKAGE_NAME}.zip"
echo "  • ${PACKAGE_NAME}.tar.gz.sha256"
echo "  • ${PACKAGE_NAME}.zip.sha256"
echo ""
echo "To install on another system:"
echo "  1. Extract: tar -xzf ${PACKAGE_NAME}.tar.gz"
echo "  2. cd ${PACKAGE_NAME}"
echo "  3. Run: ./quantoniumos-bootstrap.sh"
echo ""
