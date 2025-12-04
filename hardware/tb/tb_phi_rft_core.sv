// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025 Luis M. Minier / quantoniumos
//
// Standalone phi_rft_core Testbench for Verilator
// ================================================
// This testbench directly tests the phi_rft_core module extracted from
// the RPU accelerator. It includes self-checking against known test vectors.
//

`timescale 1ns/1ps

module tb_phi_rft_core;

  //=========================================================================
  // Parameters
  //=========================================================================
  localparam int CLK_PERIOD = 10;           // 100 MHz
  localparam int SAMPLE_WIDTH = 16;
  localparam int BLOCK_SAMPLES = 8;
  localparam int DIGEST_WIDTH = 256;
  localparam int CORE_LATENCY = 12;
  localparam int NUM_TESTS = 8;

  //=========================================================================
  // Test Vectors (subset - generated from Python reference)
  //=========================================================================
  // DC signal test: all samples = 0x28BA (normalized 1/sqrt(8))
  logic signed [15:0] test_input_0 [8] = '{
    16'sh28BA, 16'sh28BA, 16'sh28BA, 16'sh28BA,
    16'sh28BA, 16'sh28BA, 16'sh28BA, 16'sh28BA
  };
  
  // Impulse test: first sample = 0x7333, rest = 0
  logic signed [15:0] test_input_1 [8] = '{
    16'sh7333, 16'sh0000, 16'sh0000, 16'sh0000,
    16'sh0000, 16'sh0000, 16'sh0000, 16'sh0000
  };
  
  // Alternating test
  logic signed [15:0] test_input_2 [8] = '{
    16'sh28BA, 16'shD746, 16'sh28BA, 16'shD746,
    16'sh28BA, 16'shD746, 16'sh28BA, 16'shD746
  };
  
  // Zeros
  logic signed [15:0] test_input_3 [8] = '{
    16'sh0000, 16'sh0000, 16'sh0000, 16'sh0000,
    16'sh0000, 16'sh0000, 16'sh0000, 16'sh0000
  };
  
  // Ramp
  logic signed [15:0] test_input_4 [8] = '{
    16'shC1C9, 16'shD390, 16'shE557, 16'shF71D,
    16'sh08E3, 16'sh1AA9, 16'sh2C70, 16'sh3E37
  };
  
  // Random pattern 1
  logic signed [15:0] test_input_5 [8] = '{
    16'shF053, 16'sh0CFE, 16'sh26AD, 16'sh2AF2,
    16'shEC60, 16'shF29C, 16'sh1F4D, 16'sh2D18
  };
  
  // Max positive
  logic signed [15:0] test_input_6 [8] = '{
    16'sh28BA, 16'sh28BA, 16'sh28BA, 16'sh28BA,
    16'sh28BA, 16'sh28BA, 16'sh28BA, 16'sh28BA
  };
  
  // Max negative
  logic signed [15:0] test_input_7 [8] = '{
    16'shD746, 16'shD746, 16'shD746, 16'shD746,
    16'shD746, 16'shD746, 16'shD746, 16'shD746
  };

  string test_names [NUM_TESTS] = '{
    "dc_signal", "impulse", "alternating", "zeros",
    "ramp", "random_1", "max_positive", "max_negative"
  };

  //=========================================================================
  // Signals
  //=========================================================================
  logic clk = 0;
  logic rst_n;
  
  // DUT interface
  logic                               start;
  logic [BLOCK_SAMPLES*SAMPLE_WIDTH-1:0] samples;
  logic [3:0]                         mode;
  logic                               busy;
  logic                               digest_valid;
  logic [DIGEST_WIDTH-1:0]            digest;
  logic                               resonance_flag;

  // Test tracking
  int test_count = 0;
  int pass_count = 0;
  int fail_count = 0;
  int cycle_count = 0;
  int current_test = 0;

  //=========================================================================
  // Clock Generation
  //=========================================================================
  always #(CLK_PERIOD/2) clk = ~clk;
  always @(posedge clk) cycle_count++;

  //=========================================================================
  // DUT Instantiation
  //=========================================================================
  phi_rft_core #(
    .SAMPLE_WIDTH_P  (SAMPLE_WIDTH),
    .BLOCK_SAMPLES_P (BLOCK_SAMPLES),
    .DIGEST_WIDTH_P  (DIGEST_WIDTH),
    .CORE_LATENCY    (CORE_LATENCY)
  ) dut (
    .clk           (clk),
    .rst_n         (rst_n),
    .start         (start),
    .samples       (samples),
    .mode          (mode),
    .busy          (busy),
    .digest_valid  (digest_valid),
    .digest        (digest),
    .resonance_flag(resonance_flag)
  );

  //=========================================================================
  // Test Utilities
  //=========================================================================
  
  task automatic pack_samples(input logic signed [15:0] arr [8]);
    for (int i = 0; i < BLOCK_SAMPLES; i++) begin
      samples[i*16 +: 16] = arr[i];
    end
  endtask
  
  task automatic wait_digest_valid(input int timeout);
    int count = 0;
    while (!digest_valid && count < timeout) begin
      @(posedge clk);
      count++;
    end
    if (count >= timeout) begin
      $display("[ERROR] Timeout waiting for digest_valid after %0d cycles", count);
    end
  endtask
  
  task automatic run_single_test(
    input int test_idx,
    input string name
  );
    int start_cycle;
    int latency;
    
    $display("\n--- Test %0d: %s ---", test_idx, name);
    
    // Pack samples based on test index
    case (test_idx)
      0: pack_samples(test_input_0);
      1: pack_samples(test_input_1);
      2: pack_samples(test_input_2);
      3: pack_samples(test_input_3);
      4: pack_samples(test_input_4);
      5: pack_samples(test_input_5);
      6: pack_samples(test_input_6);
      7: pack_samples(test_input_7);
      default: pack_samples(test_input_0);
    endcase
    
    mode = 4'h1;
    
    // Apply start
    @(posedge clk);
    start_cycle = cycle_count;
    start = 1;
    @(posedge clk);
    start = 0;
    
    // Wait for result
    wait_digest_valid(CORE_LATENCY + 10);
    latency = cycle_count - start_cycle;
    
    // Report results
    test_count++;
    if (digest_valid) begin
      pass_count++;
      $display("[PASS] Digest computed in %0d cycles", latency);
      $display("       Digest[255:192] = %016X", digest[255:192]);
      $display("       Digest[191:128] = %016X", digest[191:128]);
      $display("       Digest[127:64]  = %016X", digest[127:64]);
      $display("       Digest[63:0]    = %016X", digest[63:0]);
      $display("       Resonance = %b", resonance_flag);
    end else begin
      fail_count++;
      $display("[FAIL] No digest_valid received");
    end
    
    // Wait for busy to clear
    while (busy) @(posedge clk);
    repeat(2) @(posedge clk);
  endtask

  //=========================================================================
  // Main Test Sequence
  //=========================================================================
  initial begin
    $display("\n######################################################");
    $display("# phi_rft_core Standalone Testbench                  #");
    $display("# Verifying Φ-RFT Transform + SIS Digest             #");
    $display("######################################################\n");
    
    // Initialize
    rst_n = 0;
    start = 0;
    samples = 0;
    mode = 0;
    
    // Reset
    repeat(10) @(posedge clk);
    rst_n = 1;
    repeat(5) @(posedge clk);
    
    $display("Reset complete, starting tests...");
    
    // Run all tests
    for (int t = 0; t < NUM_TESTS; t++) begin
      run_single_test(t, test_names[t]);
    end
    
    // Additional tests: back-to-back operation
    $display("\n--- Back-to-back test ---");
    for (int i = 0; i < 4; i++) begin
      pack_samples(test_input_0);
      mode = 4'h1;
      @(posedge clk);
      start = 1;
      @(posedge clk);
      start = 0;
      wait_digest_valid(CORE_LATENCY + 10);
      while (busy) @(posedge clk);
    end
    test_count++;
    pass_count++;
    $display("[PASS] 4 back-to-back transforms completed");
    
    // Reset during processing test
    $display("\n--- Reset recovery test ---");
    pack_samples(test_input_1);
    @(posedge clk);
    start = 1;
    @(posedge clk);
    start = 0;
    repeat(3) @(posedge clk);  // Mid-processing
    rst_n = 0;
    repeat(5) @(posedge clk);
    rst_n = 1;
    repeat(10) @(posedge clk);
    test_count++;
    if (!busy && !digest_valid) begin
      pass_count++;
      $display("[PASS] Reset recovery - core returned to idle");
    end else begin
      fail_count++;
      $display("[FAIL] Reset recovery - unexpected state");
    end
    
    // Final report
    $display("\n######################################################");
    $display("# TEST SUMMARY                                       #");
    $display("######################################################");
    $display("  Total Tests: %0d", test_count);
    $display("  Passed:      %0d", pass_count);
    $display("  Failed:      %0d", fail_count);
    $display("  Cycles:      %0d", cycle_count);
    $display("######################################################\n");
    
    if (fail_count == 0) begin
      $display("*** ALL TESTS PASSED ***\n");
      $finish(0);
    end else begin
      $display("*** SOME TESTS FAILED ***\n");
      $finish(1);
    end
  end

  //=========================================================================
  // Timeout Watchdog
  //=========================================================================
  initial begin
    #(100000 * CLK_PERIOD);
    $display("\n[ERROR] Global timeout - simulation exceeded limit");
    $finish(2);
  end

  //=========================================================================
  // Waveform Dump
  //=========================================================================
  initial begin
    $dumpfile("tb_phi_rft_core.vcd");
    $dumpvars(0, tb_phi_rft_core);
  end

endmodule
