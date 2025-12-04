#!/bin/bash
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
#
# OpenLane Setup & Run Script
# Handles PDK installation and OpenLane execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARDWARE_DIR="$(dirname "$SCRIPT_DIR")"
DESIGN_DIR="$HARDWARE_DIR/openlane/rftpu_4x4"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  RFTPU Physical Design Flow Setup         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo

# Check if RTL is generated
if [ ! -f "$DESIGN_DIR/src/rftpu_4x4_top.sv" ]; then
    echo -e "${YELLOW}Generating 4×4 RTL variant...${NC}"
    cd "$HARDWARE_DIR"
    python3 scripts/generate_4x4_variant.py
    echo
fi

echo -e "${GREEN}Select your approach:${NC}"
echo
echo "  [1] Use OpenLane Docker (simplest, no local install)"
echo "  [2] Use local OpenLane installation"
echo "  [3] Use OpenLane Web Interface (efabless.com)"
echo "  [4] Alternative: Yosys synthesis only (quick preview)"
echo "  [5] Skip - Just show me what files are needed"
echo
read -p "Select option [1-5]: " option

case $option in
    1)
        echo
        echo -e "${BLUE}════════════════════════════════════════${NC}"
        echo -e "${BLUE}   OpenLane Docker Setup${NC}"
        echo -e "${BLUE}════════════════════════════════════════${NC}"
        echo
        
        # Check if Docker is available
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}Error: Docker not found!${NC}"
            echo "Install Docker: https://docs.docker.com/get-docker/"
            exit 1
        fi
        
        echo -e "${YELLOW}Note: This will download ~2GB and run for 2-6 hours${NC}"
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" != "y" ]; then
            echo "Cancelled."
            exit 0
        fi
        
        echo
        echo -e "${GREEN}Starting OpenLane Docker...${NC}"
        echo "This will:"
        echo "  1. Pull OpenLane container (if needed)"
        echo "  2. Download SkyWater PDK (~1GB)"
        echo "  3. Run full synthesis → place → route"
        echo
        
        # Run with proper volume mounts
        docker run -it --rm \
            -v "$HARDWARE_DIR":/workspace/hardware \
            -v "$DESIGN_DIR":/workspace/design \
            efabless/openlane:latest bash -c "
                echo 'Installing PDK...'
                python3 -m volare fetch sky130
                
                echo 'Setting up design...'
                mkdir -p /openlane/designs/rftpu_4x4
                cp -r /workspace/design/* /openlane/designs/rftpu_4x4/
                
                echo 'Running OpenLane flow...'
                cd /openlane
                flow.tcl -design rftpu_4x4 -tag run_1
                
                echo 'Copying results back...'
                cp -r designs/rftpu_4x4/runs /workspace/design/
                
                echo 'Done! Results in: /workspace/design/runs/run_1'
            "
        ;;
        
    2)
        echo
        echo -e "${BLUE}════════════════════════════════════════${NC}"
        echo -e "${BLUE}   Local OpenLane Installation${NC}"
        echo -e "${BLUE}════════════════════════════════════════${NC}"
        echo
        
        read -p "Enter path to OpenLane directory: " openlane_path
        
        if [ ! -d "$openlane_path" ]; then
            echo -e "${RED}Error: Directory not found!${NC}"
            echo "Clone OpenLane:"
            echo "  git clone https://github.com/The-OpenROAD-Project/OpenLane.git"
            echo "  cd OpenLane && make"
            exit 1
        fi
        
        echo "Copying design..."
        cp -r "$DESIGN_DIR" "$openlane_path/designs/"
        
        echo "Running OpenLane..."
        cd "$openlane_path"
        ./flow.tcl -design rftpu_4x4 -tag run_1
        
        echo "Copying results back..."
        cp -r "$openlane_path/designs/rftpu_4x4/runs" "$DESIGN_DIR/"
        ;;
        
    3)
        echo
        echo -e "${BLUE}════════════════════════════════════════${NC}"
        echo -e "${BLUE}   Efabless Web Interface${NC}"
        echo -e "${BLUE}════════════════════════════════════════${NC}"
        echo
        echo "Steps:"
        echo "  1. Go to: https://efabless.com/projects/create"
        echo "  2. Upload files from: $DESIGN_DIR"
        echo "  3. Run 'Harden' flow online"
        echo "  4. Download GDS results"
        echo
        echo "Files to upload:"
        echo "  - config.json"
        echo "  - src/rftpu_4x4_top.sv"
        echo
        ;;
        
    4)
        echo
        echo -e "${BLUE}════════════════════════════════════════${NC}"
        echo -e "${BLUE}   Quick Synthesis Preview (Yosys)${NC}"
        echo -e "${BLUE}════════════════════════════════════════${NC}"
        echo
        
        if ! command -v yosys &> /dev/null; then
            echo -e "${YELLOW}Installing Yosys...${NC}"
            sudo apt-get update && sudo apt-get install -y yosys
        fi
        
        cd "$HARDWARE_DIR"
        mkdir -p build/synthesis
        
        echo "Running synthesis..."
        yosys -p "
            read_verilog $DESIGN_DIR/src/rftpu_4x4_top.sv
            hierarchy -check -top rftpu_accelerator
            proc; opt; fsm; opt; memory; opt
            techmap; opt
            stat
            write_verilog build/synthesis/rftpu_4x4_synth.v
        " 2>&1 | tee build/synthesis/yosys.log
        
        echo
        echo -e "${GREEN}Synthesis complete!${NC}"
        echo "Output: build/synthesis/rftpu_4x4_synth.v"
        echo "Log: build/synthesis/yosys.log"
        echo
        echo "Gate count and area estimates are in the log."
        ;;
        
    5)
        echo
        echo -e "${BLUE}════════════════════════════════════════${NC}"
        echo -e "${BLUE}   Design Files Summary${NC}"
        echo -e "${BLUE}════════════════════════════════════════${NC}"
        echo
        echo "Design directory: $DESIGN_DIR"
        echo
        echo "Required files:"
        tree -L 2 "$DESIGN_DIR" || find "$DESIGN_DIR" -maxdepth 2 -type f
        echo
        echo "To run manually:"
        echo "  1. Install OpenLane (see openlane/README.md)"
        echo "  2. cd /path/to/OpenLane"
        echo "  3. cp -r $DESIGN_DIR designs/"
        echo "  4. ./flow.tcl -design rftpu_4x4"
        ;;
        
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}   Next Steps${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo
echo "If synthesis/P&R completed successfully:"
echo "  • View layout: ./scripts/view_layout.sh"
echo "  • Check reports: cd $DESIGN_DIR/runs/run_1/reports"
echo "  • See logs: cd $DESIGN_DIR/runs/run_1/logs"
echo
