#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# Generate all hardware visualization figures

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  QuantoniumOS Hardware - Generate All Figures             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Create output directory
mkdir -p figures

echo "📊 Step 1: Hardware test results visualization..."
python visualize_hardware_results.py
echo ""

echo "📊 Step 2: Software vs Hardware comparison..."
python visualize_sw_hw_comparison.py
echo ""

echo "✅ All figures generated successfully!"
echo ""
echo "📁 Output directory: $(pwd)/figures"
echo ""
echo "Generated files:"
ls -lh figures/*.png | awk '{print "  - " $9 " (" $5 ")"}'
echo ""
echo "📄 Documentation:"
echo "  - figures/README.md"
echo "  - HW_VISUALIZATION_REPORT.md"
echo "  - HW_TEST_RESULTS.md"
echo ""
