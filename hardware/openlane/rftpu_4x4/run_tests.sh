#!/bin/bash
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
#
# Run RFTPU Chip Capability Tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TB_DIR="$SCRIPT_DIR/tb"
SRC_DIR="$SCRIPT_DIR/src"
BUILD_DIR="$SCRIPT_DIR/build"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  RFTPU 4×4 Chip Test Runner              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo

# Create build directory
mkdir -p "$BUILD_DIR"

echo -e "${YELLOW}Compiling design and testbench...${NC}"

# Compile with Verilator
verilator --binary --timing \
    -Wall \
    --top-module tb_rftpu_chip \
    -I"$SRC_DIR" \
    "$SRC_DIR/rftpu_4x4_top.sv" \
    "$TB_DIR/tb_rftpu_chip.sv" \
    -Wno-WIDTHEXPAND \
    -Wno-WIDTHTRUNC \
    -Wno-SELRANGE \
    -Wno-UNUSEDSIGNAL \
    -Wno-UNUSEDPARAM \
    -Wno-BLKSEQ \
    --exe \
    -o "$BUILD_DIR/tb_rftpu_chip" \
    2>&1 | tee "$BUILD_DIR/compile.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}❌ Compilation failed! Check $BUILD_DIR/compile.log${NC}"
    exit 1
fi

echo
echo -e "${GREEN}✓ Compilation successful${NC}"
echo
echo -e "${YELLOW}Running tests...${NC}"
echo

# Run simulation
"$BUILD_DIR/obj_dir/tb_rftpu_chip" | tee "$BUILD_DIR/simulation.log"

SIM_EXIT=$?

echo
if [ $SIM_EXIT -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ TESTS COMPLETED SUCCESSFULLY          ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
    echo
    echo "Results saved to: $BUILD_DIR/simulation.log"
else
    echo -e "${RED}╔════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ❌ TESTS FAILED                          ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════╝${NC}"
    echo
    echo "Check $BUILD_DIR/simulation.log for details"
fi

echo
exit $SIM_EXIT
