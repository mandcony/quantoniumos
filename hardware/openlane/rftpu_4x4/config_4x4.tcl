# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# OpenLane Configuration for 4x4 RFTPU Tile Array
# Physical design flow for SkyWater SKY130 PDK

set ::env(DESIGN_NAME) "rftpu_accelerator_4x4"

# Design sources
set ::env(VERILOG_FILES) [glob $::env(DESIGN_DIR)/src/*.sv]
set ::env(CLOCK_PORT) "clk"
set ::env(CLOCK_NET) "clk"
set ::env(CLOCK_PERIOD) "10.0"

# Floorplan - reduced size for 4x4 tiles
set ::env(FP_SIZING) "absolute"
set ::env(DIE_AREA) "0 0 1500 1500"
set ::env(FP_CORE_UTIL) 30
set ::env(FP_ASPECT_RATIO) 1
set ::env(FP_PDN_VPITCH) 153.18
set ::env(FP_PDN_HPITCH) 153.6

# Placement
set ::env(PL_TARGET_DENSITY) 0.40
set ::env(PL_TIME_DRIVEN) 1
set ::env(PL_ROUTABILITY_DRIVEN) 1
set ::env(PL_RESIZER_DESIGN_OPTIMIZATIONS) 1
set ::env(PL_RESIZER_TIMING_OPTIMIZATIONS) 1

# Global routing
set ::env(GRT_ADJUSTMENT) 0.3
set ::env(GRT_REPAIR_ANTENNAS) 1
set ::env(GRT_ALLOW_CONGESTION) 0
set ::env(GRT_OVERFLOW_ITERS) 150

# Detailed routing
set ::env(DRT_OPT_ITERS) 64

# CTS
set ::env(CTS_TARGET_SKEW) 200
set ::env(CTS_TOLERANCE) 100
set ::env(CTS_MAX_CAP) 1.53169

# Antenna mitigation
set ::env(DIODE_INSERTION_STRATEGY) 3
set ::env(RUN_HEURISTIC_DIODE_INSERTION) 1

# Synthesis
set ::env(SYNTH_STRATEGY) "AREA 0"
set ::env(SYNTH_BUFFERING) 1
set ::env(SYNTH_SIZING) 1
set ::env(SYNTH_MAX_FANOUT) 6
set ::env(SYNTH_DRIVING_CELL) "sky130_fd_sc_hd__inv_2"

# Magic DRC
set ::env(MAGIC_DRC_USE_GDS) 1

# LVS
set ::env(RUN_LVS) 1
set ::env(LVS_INSERT_POWER_PINS) 1

# Output
set ::env(GENERATE_FINAL_SUMMARY_REPORT) 1
