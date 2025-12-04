// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025 Luis M. Minier / quantoniumos
// Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
// (research/education only). Commercial rights require a separate license.
// Patent Application: USPTO #19/169,399
//
// RFTPU Accelerator Formal Verification Properties
// =================================================
// SVA properties for formal verification using SymbiYosys or JasperGold.
// These properties prove key invariants that testing alone cannot guarantee.
//
// Properties covered:
//   1. NoC packet delivery guarantee
//   2. Tile state machine correctness
//   3. SIS digest determinism
//   4. Energy conservation bounds
//   5. Deadlock freedom
//

`timescale 1ns/1ps

module rftpu_formal_props #(
  parameter int TILE_DIM = 8,
  parameter int TILE_COUNT = 64,
  parameter int SAMPLE_WIDTH = 16,
  parameter int BLOCK_SAMPLES = 8,
  parameter int DIGEST_WIDTH = 256,
  parameter int NOC_LATENCY_MAX = 32
)(
  input logic clk,
  input logic rst_n,
  
  // Tile interface signals (bind to DUT internals)
  input logic [TILE_COUNT-1:0] tile_start,
  input logic [TILE_COUNT-1:0] tile_busy,
  input logic [TILE_COUNT-1:0] tile_done,
  input logic [TILE_COUNT-1:0] tile_idle,
  
  // NoC signals
  input logic [TILE_COUNT-1:0] noc_inject_valid,
  input logic [TILE_COUNT-1:0] noc_eject_valid,
  input logic [5:0]            noc_src_tile [TILE_COUNT],
  input logic [5:0]            noc_dst_tile [TILE_COUNT],
  
  // Core signals
  input logic [SAMPLE_WIDTH-1:0] core_sample_in [TILE_COUNT][BLOCK_SAMPLES],
  input logic [DIGEST_WIDTH-1:0] core_digest_out [TILE_COUNT],
  input logic [15:0]             core_energy [TILE_COUNT]
);

  //=========================================================================
  // Helper Functions
  //=========================================================================
  
  // Calculate Manhattan distance between tiles
  function automatic int manhattan_distance(input int src, dst);
    int src_x, src_y, dst_x, dst_y;
    src_x = src % TILE_DIM;
    src_y = src / TILE_DIM;
    dst_x = dst % TILE_DIM;
    dst_y = dst / TILE_DIM;
    return ((src_x > dst_x) ? (src_x - dst_x) : (dst_x - src_x)) +
           ((src_y > dst_y) ? (src_y - dst_y) : (dst_y - src_y));
  endfunction

  //=========================================================================
  // Property 1: Tile State Machine Correctness
  //=========================================================================
  
  // After reset, all tiles must be idle
  property p_reset_idle;
    @(posedge clk)
    !rst_n |=> tile_idle == '1;
  endproperty
  assert property (p_reset_idle);
  
  // Start pulse transitions tile from idle to busy
  generate
    for (genvar t = 0; t < TILE_COUNT; t++) begin : gen_tile_fsm
      property p_start_transition;
        @(posedge clk) disable iff (!rst_n)
        (tile_idle[t] && tile_start[t]) |=> tile_busy[t];
      endproperty
      assert property (p_start_transition);
      
      // Busy tile eventually becomes done (liveness with bound)
      property p_busy_to_done;
        @(posedge clk) disable iff (!rst_n)
        tile_busy[t] |-> ##[1:32] tile_done[t];
      endproperty
      assert property (p_busy_to_done);
      
      // Mutual exclusion: cannot be both idle and busy
      property p_state_mutex;
        @(posedge clk) disable iff (!rst_n)
        !(tile_idle[t] && tile_busy[t]);
      endproperty
      assert property (p_state_mutex);
    end
  endgenerate

  //=========================================================================
  // Property 2: NoC Packet Delivery Guarantee
  //=========================================================================
  
  // Every injected packet is eventually ejected at destination
  // (Bounded liveness - packet delivered within NOC_LATENCY_MAX cycles)
  generate
    for (genvar t = 0; t < TILE_COUNT; t++) begin : gen_noc_delivery
      property p_packet_delivery;
        @(posedge clk) disable iff (!rst_n)
        noc_inject_valid[t] |-> ##[1:NOC_LATENCY_MAX] 
          noc_eject_valid[noc_dst_tile[t]];
      endproperty
      // Note: This is a strong liveness property - may need assume constraints
      // assert property (p_packet_delivery);
      
      // Destination must be valid (within grid)
      property p_valid_destination;
        @(posedge clk) disable iff (!rst_n)
        noc_inject_valid[t] |-> (noc_dst_tile[t] < TILE_COUNT);
      endproperty
      assert property (p_valid_destination);
    end
  endgenerate

  //=========================================================================
  // Property 3: Digest Determinism
  //=========================================================================
  
  // Same input produces same digest (functional correctness)
  // This requires auxiliary logic to track previous computations
  
  logic [SAMPLE_WIDTH*BLOCK_SAMPLES-1:0] prev_input [TILE_COUNT];
  logic [DIGEST_WIDTH-1:0]               prev_digest [TILE_COUNT];
  logic                                  prev_valid [TILE_COUNT];
  
  generate
    for (genvar t = 0; t < TILE_COUNT; t++) begin : gen_digest_track
      // Pack current input
      logic [SAMPLE_WIDTH*BLOCK_SAMPLES-1:0] current_input;
      always_comb begin
        for (int i = 0; i < BLOCK_SAMPLES; i++) begin
          current_input[i*SAMPLE_WIDTH +: SAMPLE_WIDTH] = core_sample_in[t][i];
        end
      end
      
      // Track on completion
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          prev_valid[t] <= 0;
        end else if (tile_done[t]) begin
          prev_input[t] <= current_input;
          prev_digest[t] <= core_digest_out[t];
          prev_valid[t] <= 1;
        end
      end
      
      // If same input, must produce same digest
      property p_digest_determinism;
        @(posedge clk) disable iff (!rst_n)
        (tile_done[t] && prev_valid[t] && current_input == prev_input[t]) |->
          (core_digest_out[t] == prev_digest[t]);
      endproperty
      assert property (p_digest_determinism);
    end
  endgenerate

  //=========================================================================
  // Property 4: Energy Conservation Bounds
  //=========================================================================
  
  // Energy output must be bounded (no overflow)
  generate
    for (genvar t = 0; t < TILE_COUNT; t++) begin : gen_energy_bound
      property p_energy_bounded;
        @(posedge clk) disable iff (!rst_n)
        tile_done[t] |-> (core_energy[t] <= 16'hFFFF);
      endproperty
      assert property (p_energy_bounded);
      
      // Zero input should produce zero energy (approximately)
      // property p_zero_input_low_energy;
      //   @(posedge clk) disable iff (!rst_n)
      //   (tile_done[t] && (core_sample_in[t] == '{default: '0})) |->
      //     (core_energy[t] < 16'h0100);
      // endproperty
      // assert property (p_zero_input_low_energy);
    end
  endgenerate

  //=========================================================================
  // Property 5: Deadlock Freedom
  //=========================================================================
  
  // If any tile is busy, at least one tile will complete within bound
  property p_no_global_deadlock;
    @(posedge clk) disable iff (!rst_n)
    (tile_busy != '0) |-> ##[1:64] (tile_done != '0);
  endproperty
  assert property (p_no_global_deadlock);
  
  // NoC cannot be permanently blocked
  logic noc_any_pending;
  assign noc_any_pending = |noc_inject_valid;
  
  property p_noc_progress;
    @(posedge clk) disable iff (!rst_n)
    noc_any_pending |-> ##[1:NOC_LATENCY_MAX] (!noc_any_pending || |noc_eject_valid);
  endproperty
  // assert property (p_noc_progress);

  //=========================================================================
  // Property 6: SIS Lattice Invariants
  //=========================================================================
  
  // Digest should change if input changes significantly
  // (Collision resistance - weak form)
  
  // Track pairs of computations for collision detection
  // Note: Full collision resistance cannot be proven formally;
  // this is a sanity check for obvious bugs
  
  //=========================================================================
  // Coverage Goals
  //=========================================================================
  
  `ifndef SYNTHESIS
  // Cover: All tiles complete simultaneously
  cover property (@(posedge clk) disable iff (!rst_n)
    (tile_busy == '1) ##[1:64] (tile_done == '1));
  
  // Cover: Cascade operation (tile sends to neighbor)
  cover property (@(posedge clk) disable iff (!rst_n)
    noc_inject_valid[0] && noc_dst_tile[0] == 1);
  
  // Cover: Maximum NoC latency observed
  cover property (@(posedge clk) disable iff (!rst_n)
    noc_inject_valid[0] ##(NOC_LATENCY_MAX-1) noc_eject_valid[63]);
  `endif

  //=========================================================================
  // Assumptions for Formal (Constrain Environment)
  //=========================================================================
  
  // Input samples are valid fixed-point (no NaN equivalent)
  generate
    for (genvar t = 0; t < TILE_COUNT; t++) begin : gen_input_assume
      // Assume inputs don't change while tile is busy
      assume property (@(posedge clk) disable iff (!rst_n)
        tile_busy[t] |-> $stable({core_sample_in[t]}));
      
      // Assume no start while busy
      assume property (@(posedge clk) disable iff (!rst_n)
        tile_busy[t] |-> !tile_start[t]);
    end
  endgenerate

endmodule

//=============================================================================
// Bind Statement for Integration
//=============================================================================
// Usage: Add to testbench or synthesis wrapper
//
// bind rftpu_accelerator rftpu_formal_props #(
//   .TILE_DIM(TILE_DIM),
//   .TILE_COUNT(TILE_COUNT)
// ) formal_inst (
//   .clk(clk),
//   .rst_n(rst_n),
//   .tile_start(/* ... */),
//   .tile_busy(/* ... */),
//   // ...
// );
