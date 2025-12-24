#!/bin/bash
# ==============================================================================
# QuantoniumOS Master Reproducibility Script
# ==============================================================================
# This script serves as the primary entry point for verifying the integrity and
# performance of the QuantoniumOS stack. It performs the following steps:
# 1. Checks the environment for necessary dependencies.
# 2. Verifies or builds the native C++/Assembly optimization engine.
# 3. Runs the core unit test suite to ensure functional correctness.
# 4. Executes the Class A-E benchmark suite to validate performance claims.
#
# Usage: ./reproduce_results.sh
# ==============================================================================

set -e

echo "============================================================"
echo "   QuantoniumOS Reproducibility Suite"
echo "============================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# 1. Environment Check
echo "[1/4] Checking Environment..."
./verify_setup.sh

# 2. Build Native Engine
echo ""
echo "[2/4] Verifying Native Engine..."
if python3 -c "import sys; sys.path.insert(0, 'src/rftmw_native/build'); import rftmw_native; print('Native Engine Loaded')" 2>/dev/null; then
    echo "Native Engine Active"
else
    echo "Native Engine NOT found. Attempting build..."
    cd src/rftmw_native
    mkdir -p build && cd build
    cmake .. && make
    cd ../../..
fi

# 3. Run Unit Tests
echo ""
echo "[3/4] Running Unit Tests..."
pytest tests/validation/test_unified_scheduler.py

# 4. Run Benchmarks
echo ""
echo "[4/4] Running Verified Benchmarks (Classes A-E)..."
python3 benchmarks/run_all_benchmarks.py --variants

echo ""
echo "============================================================"
echo "REPRODUCIBILITY CHECK COMPLETE"
echo "See docs/scientific_domains/VERIFIED_BENCHMARKS.md for interpretation."
echo "============================================================"
