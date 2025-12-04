# ============================================================================
# QuantoniumOS RPU Synthesis Constraints (SDC)
# Derived from PHYSICAL_DESIGN_SPEC.md — December 2025
# Target: TSMC N7FF @ 950 MHz tile clock
# ============================================================================
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos

# ============================================================================
# 1. CLOCK DEFINITIONS
# ============================================================================

# Primary PLL output (3.8 GHz reference)
create_clock -name clk_pll -period 0.263 [get_ports clk_pll_out]

# Tile clock domain: 950 MHz (PLL ÷4)
create_generated_clock -name clk_tile \
    -source [get_ports clk_pll_out] \
    -divide_by 4 \
    [get_pins u_clkgen/clk_tile_buf/Z]

# NoC clock domain: 1.2 GHz (PLL ÷3.17, approximated as ÷3 with phase adjust)
create_generated_clock -name clk_noc \
    -source [get_ports clk_pll_out] \
    -divide_by 3 \
    [get_pins u_clkgen/clk_noc_buf/Z]

# SIS hash clock domain: 475 MHz (PLL ÷8)
create_generated_clock -name clk_sis \
    -source [get_ports clk_pll_out] \
    -divide_by 8 \
    [get_pins u_clkgen/clk_sis_buf/Z]

# Feistel clock domain: 1.4 GHz (PLL ÷2.7, approximated)
create_generated_clock -name clk_feistel \
    -source [get_ports clk_pll_out] \
    -divide_by 3 \
    -multiply_by 1 \
    [get_pins u_clkgen/clk_feistel_buf/Z]

# ============================================================================
# 2. CLOCK UNCERTAINTY & LATENCY
# ============================================================================

# Setup uncertainty (jitter + OCV)
set_clock_uncertainty -setup 0.080 [get_clocks clk_tile]
set_clock_uncertainty -setup 0.065 [get_clocks clk_noc]
set_clock_uncertainty -setup 0.100 [get_clocks clk_sis]
set_clock_uncertainty -setup 0.055 [get_clocks clk_feistel]

# Hold uncertainty
set_clock_uncertainty -hold 0.025 [get_clocks clk_tile]
set_clock_uncertainty -hold 0.020 [get_clocks clk_noc]
set_clock_uncertainty -hold 0.030 [get_clocks clk_sis]
set_clock_uncertainty -hold 0.018 [get_clocks clk_feistel]

# Clock latency (estimated before CTS)
set_clock_latency -source 0.200 [get_clocks clk_tile]
set_clock_latency -source 0.180 [get_clocks clk_noc]
set_clock_latency -source 0.250 [get_clocks clk_sis]
set_clock_latency -source 0.150 [get_clocks clk_feistel]

# Network latency (insertion delay target)
set_clock_latency 0.600 [get_clocks clk_tile]
set_clock_latency 0.500 [get_clocks clk_noc]
set_clock_latency 0.700 [get_clocks clk_sis]
set_clock_latency 0.450 [get_clocks clk_feistel]

# ============================================================================
# 3. CLOCK DOMAIN CROSSINGS
# ============================================================================

# Tile ↔ NoC (synchronizers required)
set_clock_groups -asynchronous \
    -group [get_clocks clk_tile] \
    -group [get_clocks clk_noc]

# Tile ↔ SIS (synchronizers required)
set_clock_groups -asynchronous \
    -group [get_clocks clk_tile] \
    -group [get_clocks clk_sis]

# Tile ↔ Feistel (synchronizers required)
set_clock_groups -asynchronous \
    -group [get_clocks clk_tile] \
    -group [get_clocks clk_feistel]

# NoC ↔ SIS
set_clock_groups -asynchronous \
    -group [get_clocks clk_noc] \
    -group [get_clocks clk_sis]

# False paths for reset synchronizers
set_false_path -from [get_ports rst_n]
set_false_path -from [get_cells */rst_sync_reg*]

# ============================================================================
# 4. INPUT/OUTPUT CONSTRAINTS
# ============================================================================

# --- DMA Interface (HBM-facing, high-speed) ---
set_input_delay -clock clk_tile -max 0.400 [get_ports dma_valid]
set_input_delay -clock clk_tile -min 0.050 [get_ports dma_valid]
set_input_delay -clock clk_tile -max 0.400 [get_ports {dma_payload[*]}]
set_input_delay -clock clk_tile -min 0.050 [get_ports {dma_payload[*]}]

set_output_delay -clock clk_tile -max 0.350 [get_ports dma_ready]
set_output_delay -clock clk_tile -min 0.030 [get_ports dma_ready]

# --- Configuration Interface (host-facing) ---
set_input_delay -clock clk_tile -max 0.500 [get_ports cfg_valid]
set_input_delay -clock clk_tile -min 0.080 [get_ports cfg_valid]
set_input_delay -clock clk_tile -max 0.500 [get_ports {cfg_tile_id[*]}]
set_input_delay -clock clk_tile -min 0.080 [get_ports {cfg_tile_id[*]}]
set_input_delay -clock clk_tile -max 0.500 [get_ports {cfg_payload[*]}]
set_input_delay -clock clk_tile -min 0.080 [get_ports {cfg_payload[*]}]

set_output_delay -clock clk_tile -max 0.400 [get_ports cfg_ready]
set_output_delay -clock clk_tile -min 0.050 [get_ports cfg_ready]

# --- Status & IRQ Outputs ---
set_output_delay -clock clk_tile -max 0.300 [get_ports {tile_done_bitmap[*]}]
set_output_delay -clock clk_tile -max 0.300 [get_ports {tile_busy_bitmap[*]}]
set_output_delay -clock clk_tile -max 0.250 [get_ports global_irq_done]
set_output_delay -clock clk_tile -max 0.250 [get_ports global_irq_error]

# --- Cascade/H3 Links (inter-die) ---
set_input_delay -clock clk_noc -max 0.300 [get_ports {cascade_rx_*}]
set_output_delay -clock clk_noc -max 0.280 [get_ports {cascade_tx_*}]

# ============================================================================
# 5. DESIGN RULE CONSTRAINTS
# ============================================================================

# Max transition (slew)
set_max_transition 0.120 [current_design]
set_max_transition 0.080 [get_clocks clk_feistel]  ;# Tighter for high-speed

# Max capacitance
set_max_capacitance 0.150 [current_design]

# Max fanout
set_max_fanout 32 [current_design]
set_max_fanout 16 [get_pins */clk]  ;# Clock nets

# ============================================================================
# 6. PATH GROUPS & OPTIMIZATION
# ============================================================================

# Critical path groups for focused optimization
group_path -name reg2reg_tile -from [get_clocks clk_tile] -to [get_clocks clk_tile]
group_path -name reg2reg_noc -from [get_clocks clk_noc] -to [get_clocks clk_noc]
group_path -name reg2reg_sis -from [get_clocks clk_sis] -to [get_clocks clk_sis]
group_path -name reg2reg_feistel -from [get_clocks clk_feistel] -to [get_clocks clk_feistel]

group_path -name in2reg -from [get_ports *] -to [get_clocks *]
group_path -name reg2out -from [get_clocks *] -to [get_ports *]

# ============================================================================
# 7. MULTICYCLE PATHS
# ============================================================================

# CORDIC iterations (16 cycles per result)
set_multicycle_path 16 -setup -from [get_pins */cordic_inst/iteration_reg*/Q] \
    -to [get_pins */cordic_inst/*_out_reg*/D]
set_multicycle_path 15 -hold -from [get_pins */cordic_inst/iteration_reg*/Q] \
    -to [get_pins */cordic_inst/*_out_reg*/D]

# RFT matrix build (N×N cycles for full matrix)
set_multicycle_path 64 -setup -from [get_pins */rft_matrix_*/Q] \
    -through [get_pins */signal_out_unpacked_*/D]
set_multicycle_path 63 -hold -from [get_pins */rft_matrix_*/Q] \
    -through [get_pins */signal_out_unpacked_*/D]

# SIS lattice multiply (pipelined accumulation)
set_multicycle_path 4 -setup -from [get_pins */sis_vector_*/Q] \
    -to [get_pins */lattice_point_*/D]
set_multicycle_path 3 -hold -from [get_pins */sis_vector_*/Q] \
    -to [get_pins */lattice_point_*/D]

# ============================================================================
# 8. AREA & POWER CONSTRAINTS
# ============================================================================

# Target area (guide, not hard limit)
set_max_area 2800000  ;# µm² (2.8 mm²)

# Power optimization
set_leakage_optimization true
set_dynamic_optimization true

# Operating conditions
set_operating_conditions -max ss_0p72v_m40c -min ff_0p90v_125c

# ============================================================================
# 9. DONT_TOUCH & PRESERVE
# ============================================================================

# Preserve clock tree buffers
set_dont_touch [get_cells u_clkgen/*]

# Preserve reset synchronizers
set_dont_touch [get_cells */rst_sync_reg*]

# Preserve scan chain connections (for DFT)
set_dont_touch [get_nets */scan_*]

# ============================================================================
# 10. CASE ANALYSIS (for mode-specific optimization)
# ============================================================================

# Default mode: RFT only (mode=0)
set_case_analysis 0 [get_ports {mode[0]}]
set_case_analysis 0 [get_ports {mode[1]}]
set_case_analysis 0 [get_ports {mode[2]}]

# ============================================================================
# END OF SDC
# ============================================================================
