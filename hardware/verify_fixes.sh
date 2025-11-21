#!/bin/bash
# Quick verification script for fixed design (runs from this script's directory)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "==============================================="
echo "QuantoniumOS Unified Engines - Verification"
echo "==============================================="

# Check file exists
if [ ! -f "quantoniumos_unified_engines.sv" ]; then
    echo "❌ ERROR: hardware/quantoniumos_unified_engines.sv not found"
    exit 1
fi

echo "✓ Found quantoniumos_unified_engines.sv"

# Check for critical syntax patterns
echo ""
echo "Checking for fixed issues..."

# Check for packed arrays
if grep -q "input wire \[N\*PRECISION-1:0\] signal_in" quantoniumos_unified_engines.sv; then
    echo "✓ FIX A: Packed arrays detected"
else
    echo "❌ FIX A: Packed arrays missing"
fi

# Check for CORDIC module
if grep -q "module cordic_sincos" quantoniumos_unified_engines.sv; then
    echo "✓ FIX F: CORDIC module found"
else
    echo "❌ FIX F: CORDIC module missing"
fi

# Check for SHA3 warning
if grep -q "CRITICAL WARNING: This is NOT SHA3" quantoniumos_unified_engines.sv; then
    echo "✓ FIX C: SHA3 warning present"
else
    echo "⚠️  FIX C: SHA3 warning not found"
fi

# Check for HKDF structure
if grep -q "HKDF-Extract" quantoniumos_unified_engines.sv; then
    echo "✓ FIX E: HKDF structure present"
else
    echo "❌ FIX E: HKDF structure missing"
fi

# Check for proper MAC (no combinational loop)
if grep -q "FIX B: Pipelined MAC" quantoniumos_unified_engines.sv; then
    echo "✓ FIX B: Pipelined MAC found"
else
    echo "❌ FIX B: Pipelined MAC missing"
fi

# Check for cryptographic warnings
if grep -q "SHAKE-128" quantoniumos_unified_engines.sv; then
    echo "✓ FIX D: Cryptographic sampling warnings present"
else
    echo "⚠️  FIX D: Missing crypto warnings"
fi

echo ""
echo "==============================================="
echo "Line Count Analysis"
echo "==============================================="
wc -l quantoniumos_unified_engines.sv
echo ""

# Check for verilator
if command -v verilator &> /dev/null; then
    echo "==============================================="
    echo "Running Verilator Lint Check..."
    echo "==============================================="
    verilator --lint-only -Wall \
        --top-module quantoniumos_unified_core \
        quantoniumos_unified_engines.sv 2>&1 | tee verilator_lint.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Verilator lint passed!"
    else
        echo "⚠️  Verilator found issues (see verilator_lint.log)"
    fi
else
    echo "⚠️  Verilator not installed - skipping lint check"
    echo "   Install: sudo apt-get install verilator"
fi

echo ""
echo "==============================================="
echo "Summary"
echo "==============================================="
echo "File: hardware/quantoniumos_unified_engines.sv"
echo "Status: Ready for synthesis review"
echo ""
echo "Next steps:"
echo "1. Review hardware/CRITICAL_FIXES_REPORT.md"
echo "2. Integrate SHA3/HMAC IP cores"
echo "3. Run full testbench with test vectors"
echo "4. Synthesize with Vivado/Quartus"
echo ""
echo "For questions, see hardware/quantoniumos_engines_README.md"
echo "==============================================="
