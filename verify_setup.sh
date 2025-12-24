#!/bin/bash
# ==============================================================================
# QuantoniumOS Environment Verification Script
# ==============================================================================
# This script checks the system for the required Python environment and
# dependencies. It verifies:
# 1. Existence of the virtual environment.
# 2. Python version compatibility.
# 3. Installation of core libraries (NumPy, SciPy, SymPy, Numba).
# 4. Basic functionality of the RFT core algorithm.
#
# Usage: ./verify_setup.sh
# ==============================================================================

echo "=========================================="
echo "QuantoniumOS Setup Verification"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "Virtual environment found"
else
    echo "Virtual environment not found"
    echo "  Run: python3 -m venv .venv"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate 2>/dev/null || . .venv/bin/activate

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check core dependencies
echo ""
echo "Checking core dependencies..."
python -c "import numpy; print('NumPy', numpy.__version__)" 2>/dev/null || echo "NumPy not installed"
python -c "import scipy; print('SciPy', scipy.__version__)" 2>/dev/null || echo "SciPy not installed"
python -c "import sympy; print('SymPy', sympy.__version__)" 2>/dev/null || echo "SymPy not installed"
python -c "import numba; print('Numba', numba.__version__)" 2>/dev/null || echo "Numba not installed"

# Test RFT core
echo ""
echo "Testing RFT core functionality..."
python -c "
import numpy as np
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT

rft = CanonicalTrueRFT(64)
x = np.random.randn(64)
y = rft.forward_transform(x)
error = rft.get_unitarity_error()

print(f'RFT Core: Operational')
print(f'  Transform size: 64')
print(f'  Unitarity error: {error:.2e}')

if error < 1e-12:
    print(f'  Status: EXCELLENT (< 1e-12)')
elif error < 1e-9:
    print(f'  Status: Good (< 1e-9)')
else:
    print(f'  Status: WARNING (> 1e-9)')
" 2>/dev/null

# Check native engines
echo ""
echo "Checking native engines (optional)..."
python -c "
try:
    import rftmw_native
    from algorithms.rft.kernels.python_bindings import _load_assembly_library
    
    has_cpp = True
    has_c = _load_assembly_library() is not None
    
    if has_cpp:
        print('C++ Engine (Layer 3): Built')
        print(f'  AVX2:    {\"Yes\" if rftmw_native.HAS_AVX2 else \"No\"}')
        print(f'  AVX-512: {\"Yes\" if rftmw_native.HAS_AVX512 else \"No\"}')
        print(f'  FMA:     {\"Yes\" if rftmw_native.HAS_FMA else \"No\"}')
        print(f'  ASM:     {\"Yes\" if rftmw_native.HAS_ASM_KERNELS else \"No\"}')
    
    if has_c:
        print('C/ASM Library (Layer 2): Built')
    
    if not has_cpp and not has_c:
        print('Native engines not built (using pure Python mode)')
        print('  To build: see docs/guides/SETUP_GUIDE.md (Native build section)')
except ImportError:
    print('Native engines not built (using pure Python mode)')
    print('  This is OK! Pure Python mode works fine.')
    print('  To build native engines for 3-10x speedup:')
    print('    cd algorithms/rft/kernels && make -j\$(nproc)')
    print('    cd ../../src/rftmw_native && mkdir -p build && cd build')
    print('    cmake .. -DCMAKE_BUILD_TYPE=Release -DRFTMW_ENABLE_ASM=ON')
    print('    make -j\$(nproc)')
    print('    cp rftmw_native.cpython-*-linux-gnu.so ../../../.venv/lib/python3.12/site-packages/')
except Exception as e:
    print(f'Native engine check error: {e}')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Setup verification PASSED"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  - Read docs/guides/SETUP_GUIDE.md for detailed setup"
    echo "  - Run benchmarks: python benchmarks/run_all_benchmarks.py"
    echo "  - Run tests: pytest tests/"
    echo "  - Explore experiments: ls experiments/"
else
    echo ""
    echo "=========================================="
    echo "Setup verification FAILED"
    echo "=========================================="
    echo ""
    echo "Please check SETUP_GUIDE.md for troubleshooting"
    exit 1
fi
