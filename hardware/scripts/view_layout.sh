#!/bin/bash
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
#
# Layout Viewer Helper Script
# Quick access to view RFTPU chip layouts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARDWARE_DIR="$(dirname "$SCRIPT_DIR")"
OPENLANE_DIR="$HARDWARE_DIR/openlane"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     RFTPU Chip Layout Viewer              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo

# Find available GDS files
echo -e "${YELLOW}Searching for GDS files...${NC}"
GDS_FILES=()
while IFS= read -r -d '' file; do
    GDS_FILES+=("$file")
done < <(find "$OPENLANE_DIR" -name "*.gds" -print0 2>/dev/null)

if [ ${#GDS_FILES[@]} -eq 0 ]; then
    echo -e "${YELLOW}No GDS files found. Have you run OpenLane yet?${NC}"
    echo
    echo "To generate a layout:"
    echo "  1. cd $HARDWARE_DIR"
    echo "  2. python3 scripts/generate_4x4_variant.py"
    echo "  3. Run OpenLane (see openlane/README.md)"
    exit 1
fi

echo -e "${GREEN}Found ${#GDS_FILES[@]} layout file(s):${NC}"
for i in "${!GDS_FILES[@]}"; do
    basename="${GDS_FILES[$i]##*/}"
    size=$(du -h "${GDS_FILES[$i]}" | cut -f1)
    echo "  [$i] $basename ($size)"
done
echo

# Select file
if [ ${#GDS_FILES[@]} -eq 1 ]; then
    SELECTED_FILE="${GDS_FILES[0]}"
    echo -e "${GREEN}Auto-selecting: ${SELECTED_FILE##*/}${NC}"
else
    read -p "Select file number [0-$((${#GDS_FILES[@]}-1))]: " selection
    SELECTED_FILE="${GDS_FILES[$selection]}"
fi

echo
echo -e "${BLUE}Selected: ${SELECTED_FILE}${NC}"
echo

# Choose viewer
echo "Available viewers:"
echo "  [1] KLayout (recommended - fast, colorful)"
echo "  [2] Magic (DRC checks, traditional)"
echo "  [3] Both"
echo
read -p "Select viewer [1-3]: " viewer

case $viewer in
    1)
        echo -e "${GREEN}Launching KLayout...${NC}"
        if command -v klayout &> /dev/null; then
            klayout "$SELECTED_FILE" &
            echo "✓ KLayout opened"
            echo
            echo "Tips:"
            echo "  - Click layer icons to show/hide"
            echo "  - Scroll to zoom"
            echo "  - Shift+drag to pan"
        else
            echo -e "${YELLOW}KLayout not found. Install with:${NC}"
            echo "  sudo apt-get install klayout"
        fi
        ;;
    2)
        echo -e "${GREEN}Launching Magic...${NC}"
        if command -v magic &> /dev/null; then
            PDK_ROOT="${PDK_ROOT:-$HOME/.volare}"
            if [ -d "$PDK_ROOT/sky130A" ]; then
                magic -T "$PDK_ROOT/sky130A/libs.tech/magic/sky130A.tech" "$SELECTED_FILE" &
                echo "✓ Magic opened"
            else
                magic "$SELECTED_FILE" &
                echo "✓ Magic opened (without PDK tech file)"
            fi
            echo
            echo "Magic commands:"
            echo "  load <design>   - Load design"
            echo "  see M2          - Show metal 2"
            echo "  drc check       - Run DRC"
            echo "  what            - Info about selection"
        else
            echo -e "${YELLOW}Magic not found. Install with:${NC}"
            echo "  sudo apt-get install magic"
        fi
        ;;
    3)
        echo -e "${GREEN}Launching both viewers...${NC}"
        if command -v klayout &> /dev/null; then
            klayout "$SELECTED_FILE" &
            echo "✓ KLayout opened"
        fi
        if command -v magic &> /dev/null; then
            PDK_ROOT="${PDK_ROOT:-$HOME/.volare}"
            if [ -d "$PDK_ROOT/sky130A" ]; then
                magic -T "$PDK_ROOT/sky130A/libs.tech/magic/sky130A.tech" "$SELECTED_FILE" &
            else
                magic "$SELECTED_FILE" &
            fi
            echo "✓ Magic opened"
        fi
        ;;
    *)
        echo "Invalid selection"
        exit 1
        ;;
esac

echo
echo -e "${GREEN}Layout viewing complete!${NC}"
