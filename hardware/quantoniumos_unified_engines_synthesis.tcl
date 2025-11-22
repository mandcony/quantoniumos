# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier
# Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
# (research/education only). Commercial rights require a separate license.
# ===============================================
# QuantoniumOS Unified Engines - Synthesis Script
# For Xilinx Vivado / Yosys
# ===============================================

# Set design parameters
set DESIGN_NAME "quantoniumos_unified_core"
set TOP_MODULE "quantoniumos_unified_core"
set RTL_FILE "quantoniumos_unified_engines.sv"
set TESTBENCH "tb_quantoniumos_unified"

# Target FPGA (adjust for your board)
set PART "xc7a100tcsg324-1"  # Artix-7 100T (Nexys A7)
# set PART "xc7z020clg484-1"  # Zynq-7000 (Pynq-Z2)
# set PART "xcvu9p-flga2104-2L-e"  # Virtex UltraScale+ (High-end)

puts "==============================================="
puts "QuantoniumOS Unified Engines Synthesis"
puts "Design: $DESIGN_NAME"
puts "Target: $PART"
puts "==============================================="

# Create project
create_project -force $DESIGN_NAME ./build/$DESIGN_NAME -part $PART

# Add design files
add_files $RTL_FILE
set_property top $TOP_MODULE [current_fileset]

# Set synthesis options
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AreaOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]

# Synthesis
launch_runs synth_1
wait_on_run synth_1

# Report utilization
open_run synth_1
report_utilization -file ./build/${DESIGN_NAME}_utilization.rpt
report_timing_summary -file ./build/${DESIGN_NAME}_timing.rpt

# Implementation
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
launch_runs impl_1
wait_on_run impl_1

# Generate reports
open_run impl_1
report_utilization -hierarchical -file ./build/${DESIGN_NAME}_utilization_hier.rpt
report_timing_summary -file ./build/${DESIGN_NAME}_timing_final.rpt
report_power -file ./build/${DESIGN_NAME}_power.rpt

# Generate bitstream
write_bitstream -force ./build/${DESIGN_NAME}.bit

puts "==============================================="
puts "Synthesis Complete!"
puts "Bitstream: ./build/${DESIGN_NAME}.bit"
puts "==============================================="
