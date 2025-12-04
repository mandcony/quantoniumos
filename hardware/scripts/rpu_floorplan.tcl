# ============================================================================
# QuantoniumOS RPU Floorplan Script (Cadence Innovus)
# Derived from PHYSICAL_DESIGN_SPEC.md — December 2025
# Target: TSMC N7FF, 8.5×8.5 mm die
# ============================================================================
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
#
# Usage:
#   innovus -stylus -files hardware/scripts/rpu_floorplan.tcl
# ============================================================================

puts "========================================================"
puts " QuantoniumOS RPU Floorplan Script"
puts " Target: TSMC N7FF, 64-tile Φ-RFT accelerator"
puts "========================================================"

# ============================================================================
# 1. DESIGN PARAMETERS
# ============================================================================

set DESIGN_NAME       "rpu_accelerator"
set DIE_WIDTH         8500.0      ;# µm
set DIE_HEIGHT        8500.0      ;# µm
set CORE_MARGIN       400.0       ;# Guard ring / ESD / TSV keep-out
set TILE_DIM          8           ;# 8×8 grid
set TILE_COUNT        64
set TILE_SIZE         750.0       ;# µm per tile
set TILE_GAP          62.5        ;# Gap between tiles
set GRID_ORIGIN_X     500.0       ;# Tile grid start X
set GRID_ORIGIN_Y     1200.0      ;# Tile grid start Y (above DMA ingress)

# Spine (SIS + Feistel + Controller) placement
set SPINE_X           7200.0
set SPINE_WIDTH       800.0

# Power grid parameters
set PG_PITCH_H        20.0        ;# Horizontal trunk pitch
set PG_PITCH_V        16.0        ;# Vertical trunk pitch
set PG_WIDTH_TRUNK    8.0         ;# Trunk wire width
set PG_WIDTH_STRAP    4.0         ;# Strap wire width

# ============================================================================
# 2. INITIALIZE FLOORPLAN
# ============================================================================

puts "INFO: Initializing floorplan..."

# Create die and core areas
floorPlan -site core \
    -d $DIE_WIDTH $DIE_HEIGHT \
    $CORE_MARGIN $CORE_MARGIN $CORE_MARGIN $CORE_MARGIN

# Set row utilization target
setPlaceMode -place_global_max_density 0.70

puts "INFO: Die size = ${DIE_WIDTH} x ${DIE_HEIGHT} µm"
puts "INFO: Core margin = ${CORE_MARGIN} µm"

# ============================================================================
# 3. CREATE POWER DOMAINS
# ============================================================================

puts "INFO: Creating power domains..."

# Main compute domain (tiles + NoC)
createPowerDomain PD_COMPUTE -default
addPowerNet VDD_CORE -domain PD_COMPUTE
addPowerNet VSS -domain PD_COMPUTE

# SRAM domain (separate voltage for retention)
createPowerDomain PD_SRAM
addPowerNet VDD_SRAM -domain PD_SRAM
addPowerNet VSS -domain PD_SRAM

# Always-on domain (DMA ingress + control)
createPowerDomain PD_AON
addPowerNet VDD_AON -domain PD_AON
addPowerNet VSS -domain PD_AON

# ============================================================================
# 4. PLACE HARD MACROS (SRAM, PLL, PHY)
# ============================================================================

puts "INFO: Placing hard macros..."

# --- PLL Macros (analog islands, corners) ---
set PLL_SIZE 200.0

# PLL_NW (Northwest corner)
createInstGroup grp_pll_nw -region [expr $CORE_MARGIN] [expr $DIE_HEIGHT - $CORE_MARGIN - $PLL_SIZE] \
    [expr $CORE_MARGIN + $PLL_SIZE] [expr $DIE_HEIGHT - $CORE_MARGIN]
addInstToInstGroup grp_pll_nw [get_cells u_pll_nw]
placeInstance u_pll_nw [expr $CORE_MARGIN + 20] [expr $DIE_HEIGHT - $CORE_MARGIN - $PLL_SIZE + 20] R0

# PLL_SE (Southeast corner)
createInstGroup grp_pll_se -region [expr $DIE_WIDTH - $CORE_MARGIN - $PLL_SIZE] [expr $CORE_MARGIN] \
    [expr $DIE_WIDTH - $CORE_MARGIN] [expr $CORE_MARGIN + $PLL_SIZE]
addInstToInstGroup grp_pll_se [get_cells u_pll_se]
placeInstance u_pll_se [expr $DIE_WIDTH - $CORE_MARGIN - $PLL_SIZE + 20] [expr $CORE_MARGIN + 20] R0

# --- Create halos around PLLs (50 µm isolation) ---
createPlaceBlockage -type hard -box \
    [expr $CORE_MARGIN - 50] [expr $DIE_HEIGHT - $CORE_MARGIN - $PLL_SIZE - 50] \
    [expr $CORE_MARGIN + $PLL_SIZE + 50] [expr $DIE_HEIGHT - $CORE_MARGIN + 50] \
    -name halo_pll_nw

createPlaceBlockage -type hard -box \
    [expr $DIE_WIDTH - $CORE_MARGIN - $PLL_SIZE - 50] [expr $CORE_MARGIN - 50] \
    [expr $DIE_WIDTH - $CORE_MARGIN + 50] [expr $CORE_MARGIN + $PLL_SIZE + 50] \
    -name halo_pll_se

# --- DMA Ingress Region (South edge) ---
set DMA_HEIGHT 500.0
createInstGroup grp_dma_ingress -region \
    $CORE_MARGIN $CORE_MARGIN \
    [expr $DIE_WIDTH - $CORE_MARGIN] [expr $CORE_MARGIN + $DMA_HEIGHT]
addInstToInstGroup grp_dma_ingress [get_cells dma_ingress_inst]

puts "INFO: DMA ingress placed at south edge (y = ${CORE_MARGIN} to [expr $CORE_MARGIN + $DMA_HEIGHT])"

# ============================================================================
# 5. PLACE 64 TILE MACROS (8×8 GRID)
# ============================================================================

puts "INFO: Placing 64 tiles in 8x8 grid..."

for {set row 0} {$row < $TILE_DIM} {incr row} {
    for {set col 0} {$col < $TILE_DIM} {incr col} {
        set tile_id [expr $row * $TILE_DIM + $col]
        set tile_name "gen_rows\[$row\].gen_cols\[$col\].tile_inst"
        
        # Calculate position
        set tile_x [expr $GRID_ORIGIN_X + $col * ($TILE_SIZE + $TILE_GAP)]
        set tile_y [expr $GRID_ORIGIN_Y + $row * ($TILE_SIZE + $TILE_GAP)]
        
        # Create fence for this tile
        createFence $tile_name \
            $tile_x $tile_y \
            [expr $tile_x + $TILE_SIZE] [expr $tile_y + $TILE_SIZE]
        
        # Place SRAM macros within tile (2× 2Kb scratch + 1× 4Kb topo)
        # Scratch SRAM 0 (bottom-left of tile)
        set sram_scratch0 "${tile_name}/scratchpad_sram0"
        if {[sizeof_collection [get_cells -quiet $sram_scratch0]] > 0} {
            placeInstance $sram_scratch0 [expr $tile_x + 20] [expr $tile_y + 20] R0
        }
        
        # Scratch SRAM 1 (bottom-right of tile)
        set sram_scratch1 "${tile_name}/scratchpad_sram1"
        if {[sizeof_collection [get_cells -quiet $sram_scratch1]] > 0} {
            placeInstance $sram_scratch1 [expr $tile_x + 200] [expr $tile_y + 20] R0
        }
        
        # Topo SRAM (top of tile)
        set sram_topo "${tile_name}/topo_mem_sram"
        if {[sizeof_collection [get_cells -quiet $sram_topo]] > 0} {
            placeInstance $sram_topo [expr $tile_x + 20] [expr $tile_y + 400] R0
        }
        
        puts "INFO: Tile $tile_id placed at ($tile_x, $tile_y)"
    }
}

# ============================================================================
# 6. PLACE SPINE (SIS + FEISTEL + CONTROLLER)
# ============================================================================

puts "INFO: Placing spine modules..."

# SIS Hash Engine (large block, center-right)
set SIS_Y_START 3500.0
set SIS_Y_END   5500.0
createFence rft_sis_hash_inst \
    $SPINE_X $SIS_Y_START \
    [expr $SPINE_X + $SPINE_WIDTH] $SIS_Y_END

# Large SRAM for SIS A-matrix cache (256Kb × 2)
set sram_sis_a0 "rft_sis_hash_inst/a_matrix_sram0"
if {[sizeof_collection [get_cells -quiet $sram_sis_a0]] > 0} {
    placeInstance $sram_sis_a0 [expr $SPINE_X + 50] [expr $SIS_Y_START + 100] R0
}
set sram_sis_a1 "rft_sis_hash_inst/a_matrix_sram1"
if {[sizeof_collection [get_cells -quiet $sram_sis_a1]] > 0} {
    placeInstance $sram_sis_a1 [expr $SPINE_X + 50] [expr $SIS_Y_START + 600] R0
}

puts "INFO: SIS hash engine placed at x=${SPINE_X}, y=${SIS_Y_START}-${SIS_Y_END}"

# Feistel-48 Cipher (below SIS)
set FEISTEL_Y_START 2000.0
set FEISTEL_Y_END   3500.0
createFence feistel_engine \
    $SPINE_X $FEISTEL_Y_START \
    [expr $SPINE_X + $SPINE_WIDTH] $FEISTEL_Y_END

puts "INFO: Feistel-48 placed at x=${SPINE_X}, y=${FEISTEL_Y_START}-${FEISTEL_Y_END}"

# Unified Controller (top of spine)
set CTRL_Y_START 5500.0
set CTRL_Y_END   6500.0
createFence unified_ctrl \
    $SPINE_X $CTRL_Y_START \
    [expr $SPINE_X + $SPINE_WIDTH] $CTRL_Y_END

puts "INFO: Unified controller placed at x=${SPINE_X}, y=${CTRL_Y_START}-${CTRL_Y_END}"

# ============================================================================
# 7. NOC ROUTER PLACEMENT (AT TILE EDGES)
# ============================================================================

puts "INFO: Placing NoC routers at tile edges..."

# NoC routers are soft cells, placed at tile boundaries
# Create guide regions along tile grid lines

for {set row 0} {$row < $TILE_DIM} {incr row} {
    for {set col 0} {$col < $TILE_DIM} {incr col} {
        set tile_id [expr $row * $TILE_DIM + $col]
        set router_name "noc_inst/router_${tile_id}"
        
        # Router placed at top-right corner of each tile
        set router_x [expr $GRID_ORIGIN_X + ($col + 1) * ($TILE_SIZE + $TILE_GAP) - $TILE_GAP/2 - 30]
        set router_y [expr $GRID_ORIGIN_Y + ($row + 1) * ($TILE_SIZE + $TILE_GAP) - $TILE_GAP/2 - 30]
        
        createGuide $router_name \
            [expr $router_x - 25] [expr $router_y - 25] \
            [expr $router_x + 25] [expr $router_y + 25]
    }
}

# ============================================================================
# 8. CREATE POWER GRID
# ============================================================================

puts "INFO: Creating power grid..."

# Global power connections
globalNetConnect VDD_CORE -type pgpin -pin VDD -inst * -override
globalNetConnect VDD_SRAM -type pgpin -pin VDDM -inst *sram* -override
globalNetConnect VSS -type pgpin -pin VSS -inst * -override

# Create power rings
addRing -nets {VDD_CORE VSS} \
    -type core_rings \
    -layer {top M12 bottom M12 left M11 right M11} \
    -width $PG_WIDTH_TRUNK \
    -spacing 2.0 \
    -offset 10.0

# Create horizontal stripes (M12)
addStripe -nets {VDD_CORE VSS} \
    -layer M12 \
    -direction horizontal \
    -width $PG_WIDTH_TRUNK \
    -spacing 8.0 \
    -set_to_set_distance $PG_PITCH_H \
    -start_from bottom \
    -start_offset 100.0

# Create vertical stripes (M11)
addStripe -nets {VDD_CORE VSS} \
    -layer M11 \
    -direction vertical \
    -width $PG_WIDTH_TRUNK \
    -spacing 6.0 \
    -set_to_set_distance $PG_PITCH_V \
    -start_from left \
    -start_offset 100.0

# SRAM power stripes (M10/M9)
addStripe -nets {VDD_SRAM VSS} \
    -layer M10 \
    -direction horizontal \
    -width $PG_WIDTH_STRAP \
    -spacing 4.0 \
    -set_to_set_distance 40.0 \
    -start_from bottom \
    -area [list $GRID_ORIGIN_X $GRID_ORIGIN_Y \
           [expr $GRID_ORIGIN_X + $TILE_DIM * ($TILE_SIZE + $TILE_GAP)] \
           [expr $GRID_ORIGIN_Y + $TILE_DIM * ($TILE_SIZE + $TILE_GAP)]]

# Route power to standard cells
sroute -connect {blockPin padPin corePin} \
    -layerChangeRange {M1 M12} \
    -blockPinTarget nearestTarget \
    -padPinPortConnect allPort \
    -checkAlignedSecondaryPin 1 \
    -allowJogging 1 \
    -allowLayerChange 1

puts "INFO: Power grid created with H-trunk on M12, V-trunk on M11"

# ============================================================================
# 9. PIN ASSIGNMENT
# ============================================================================

puts "INFO: Assigning IO pins..."

# DMA pins (South edge, near HBM)
set dma_pins [get_ports {dma_valid dma_ready dma_payload*}]
set num_dma_pins [sizeof_collection $dma_pins]
set dma_start_x [expr $CORE_MARGIN + 500]
set dma_pitch 15.0

editPin -pinWidth 2.0 -pinDepth 5.0 \
    -fixOverlap 1 \
    -unit MICRON \
    -spreadDirection clockwise \
    -side South \
    -layer M10 \
    -spreadType start \
    -start [list $dma_start_x $CORE_MARGIN] \
    -pin $dma_pins

# Config pins (West edge)
set cfg_pins [get_ports {cfg_valid cfg_ready cfg_tile_id* cfg_payload*}]
editPin -pinWidth 2.0 -pinDepth 5.0 \
    -fixOverlap 1 \
    -unit MICRON \
    -side West \
    -layer M11 \
    -spreadType center \
    -pin $cfg_pins

# Status/IRQ pins (East edge)
set status_pins [get_ports {tile_done_bitmap* tile_busy_bitmap* global_irq_*}]
editPin -pinWidth 2.0 -pinDepth 5.0 \
    -fixOverlap 1 \
    -unit MICRON \
    -side East \
    -layer M11 \
    -spreadType center \
    -pin $status_pins

# Clock/Reset (North edge)
set clk_pins [get_ports {clk* rst_n}]
editPin -pinWidth 4.0 -pinDepth 8.0 \
    -fixOverlap 1 \
    -unit MICRON \
    -side North \
    -layer M12 \
    -spreadType center \
    -pin $clk_pins

puts "INFO: Pin assignment complete"

# ============================================================================
# 10. VERIFY & REPORT
# ============================================================================

puts "INFO: Verifying floorplan..."

# Check for overlaps
checkFPlan -reportFile floorplan_check.rpt

# Report utilization
reportFPlan -fences
reportFPlan -macros

# Generate DEF
defOut -floorplan rpu_floorplan.def

puts "========================================================"
puts " Floorplan Complete!"
puts " Output: rpu_floorplan.def"
puts " Report: floorplan_check.rpt"
puts "========================================================"

# ============================================================================
# 11. EXPORT ABSTRACT (FOR HIERARCHICAL FLOW)
# ============================================================================

# Generate LEF abstract for each tile (for top-level integration)
foreach tile_id [list 0 9 18 27 36 45 54 63] {
    set row [expr $tile_id / $TILE_DIM]
    set col [expr $tile_id % $TILE_DIM]
    set tile_name "gen_rows\[$row\].gen_cols\[$col\].tile_inst"
    
    # This would be run after P&R of tile block:
    # abstractOut -lef tile_${tile_id}.lef $tile_name
}

puts "INFO: Run 'abstractOut' after tile P&R to generate LEF abstracts"

# ============================================================================
# END OF FLOORPLAN SCRIPT
# ============================================================================
