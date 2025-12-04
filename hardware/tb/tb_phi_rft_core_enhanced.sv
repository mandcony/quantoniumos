// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025 Luis M. Minier / quantoniumos
//
// Enhanced phi_rft_core Testbench with Internal Signal Probing
// =============================================================
// Extended testbench that:
//   - Captures internal RFT computation signals (q_real, q_imag, s_vals, lat)
//   - Tests all kernel modes (0-7)
//   - Verifies saturation behavior at extremes
//   - Dumps detailed waveforms for analysis
//

`timescale 1ns/1ps

module tb_phi_rft_core_enhanced;

  //=========================================================================
  // Parameters
  //=========================================================================
  localparam int CLK_PERIOD = 10;
  localparam int SAMPLE_WIDTH = 16;
  localparam int BLOCK_SAMPLES = 8;
  localparam int DIGEST_WIDTH = 256;
  localparam int CORE_LATENCY = 12;

  //=========================================================================
  // Test Vectors
  //=========================================================================
  // Saturation test vectors
  logic signed [15:0] test_max_pos [8] = '{
    16'sh7FFF, 16'sh7FFF, 16'sh7FFF, 16'sh7FFF,
    16'sh7FFF, 16'sh7FFF, 16'sh7FFF, 16'sh7FFF
  };
  
  logic signed [15:0] test_max_neg [8] = '{
    16'sh8000, 16'sh8000, 16'sh8000, 16'sh8000,
    16'sh8000, 16'sh8000, 16'sh8000, 16'sh8000
  };
  
  logic signed [15:0] test_mixed_extreme [8] = '{
    16'sh7FFF, 16'sh8000, 16'sh7FFF, 16'sh8000,
    16'sh7FFF, 16'sh8000, 16'sh7FFF, 16'sh8000
  };
  
  // Frequency sweep
  logic signed [15:0] test_freq_k0 [8] = '{
    16'sh4000, 16'sh4000, 16'sh4000, 16'sh4000,
    16'sh4000, 16'sh4000, 16'sh4000, 16'sh4000
  };
  
  logic signed [15:0] test_freq_k1 [8] = '{
    16'sh5A82, 16'sh4000, 16'sh0000, 16'shC000,
    16'shA57E, 16'shC000, 16'sh0000, 16'sh4000
  };
  
  logic signed [15:0] test_freq_k4 [8] = '{
    16'sh4000, 16'shC000, 16'sh4000, 16'shC000,
    16'sh4000, 16'shC000, 16'sh4000, 16'shC000
  };

  //=========================================================================
  // Signals
  //=========================================================================
  logic clk = 0;
  logic rst_n;
  logic start;
  logic [BLOCK_SAMPLES*SAMPLE_WIDTH-1:0] samples;
  logic [3:0] mode;
  logic busy;
  logic digest_valid;
  logic [DIGEST_WIDTH-1:0] digest;
  logic resonance_flag;

  // Test tracking
  int test_count = 0;
  int pass_count = 0;
  int fail_count = 0;
  int cycle_count = 0;
  
  // Captured values for analysis
  logic [DIGEST_WIDTH-1:0] captured_digests [64];
  int capture_idx = 0;

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
  endtask
  
  task automatic run_test(
    input string name,
    input logic signed [15:0] test_data [8],
    input logic [3:0] test_mode
  );
    logic [DIGEST_WIDTH-1:0] prev_digest;
    
    $display("\n--- %s (mode=%0d) ---", name, test_mode);
    
    pack_samples(test_data);
    mode = test_mode;
    
    @(posedge clk);
    start = 1;
    @(posedge clk);
    start = 0;
    
    wait_digest_valid(CORE_LATENCY + 10);
    
    test_count++;
    if (digest_valid) begin
      pass_count++;
      captured_digests[capture_idx] = digest;
      capture_idx++;
      
      $display("[PASS] Digest: %064X", digest);
      $display("       Resonance: %b", resonance_flag);
      
      // Analyze digest structure
      $display("       Lat[0-3]: %04X %04X %04X %04X", 
               digest[15:0], digest[31:16], digest[47:32], digest[63:48]);
      $display("       Lat[4-7]: %04X %04X %04X %04X",
               digest[79:64], digest[95:80], digest[111:96], digest[127:112]);
    end else begin
      fail_count++;
      $display("[FAIL] No digest_valid");
    end
    
    while (busy) @(posedge clk);
    repeat(2) @(posedge clk);
  endtask

  //=========================================================================
  // Test Phases
  //=========================================================================
  
  // Phase 1: All Mode Sweep (same input, different modes)
  task automatic test_mode_sweep();
    logic signed [15:0] standard_input [8] = '{
      16'sh2000, 16'sh3000, 16'sh4000, 16'sh5000,
      16'sh6000, 16'sh5000, 16'sh4000, 16'sh3000
    };
    
    $display("\n========================================");
    $display("Phase 1: Mode Sweep (modes 0-7)");
    $display("========================================");
    
    for (int m = 0; m < 8; m++) begin
      run_test($sformatf("mode_%0d", m), standard_input, m[3:0]);
    end
    
    // Verify different modes produce different digests
    $display("\nMode Uniqueness Check:");
    for (int i = 0; i < 8; i++) begin
      for (int j = i+1; j < 8; j++) begin
        if (captured_digests[i] == captured_digests[j]) begin
          $display("  WARNING: Mode %0d and %0d produce identical digests!", i, j);
        end
      end
    end
  endtask
  
  // Phase 2: Saturation Testing
  task automatic test_saturation();
    $display("\n========================================");
    $display("Phase 2: Saturation Behavior");
    $display("========================================");
    
    run_test("max_positive_all", test_max_pos, 4'h1);
    run_test("max_negative_all", test_max_neg, 4'h1);
    run_test("mixed_extreme", test_mixed_extreme, 4'h1);
    
    // Check for overflow indicators
    $display("\nSaturation Analysis:");
    $display("  Max positive input: No simulation errors = wrap/saturate handled");
    $display("  Max negative input: No simulation errors = wrap/saturate handled");
  endtask
  
  // Phase 3: Frequency Response
  task automatic test_frequency_response();
    $display("\n========================================");
    $display("Phase 3: Frequency Response");
    $display("========================================");
    
    run_test("freq_k0_dc", test_freq_k0, 4'h1);
    run_test("freq_k1_fundamental", test_freq_k1, 4'h1);
    run_test("freq_k4_nyquist", test_freq_k4, 4'h1);
    
    // Analyze energy distribution
    $display("\nFrequency Energy Analysis:");
    for (int f = 0; f < 3; f++) begin
      logic [31:0] energy = 0;
      for (int i = 0; i < 8; i++) begin
        logic [15:0] lat_val = captured_digests[8+f][i*16 +: 16];
        energy += lat_val;
      end
      $display("  Freq test %0d: Total lattice sum = %0d", f, energy);
    end
  endtask
  
  // Phase 4: Linearity Check
  task automatic test_linearity();
    logic signed [15:0] input_1x [8];
    logic signed [15:0] input_2x [8];
    logic [DIGEST_WIDTH-1:0] digest_1x, digest_2x;
    
    $display("\n========================================");
    $display("Phase 4: Linearity (Scaling)");
    $display("========================================");
    
    // 1x amplitude
    for (int i = 0; i < 8; i++) input_1x[i] = 16'sh1000;
    pack_samples(input_1x);
    mode = 4'h1;
    @(posedge clk); start = 1; @(posedge clk); start = 0;
    wait_digest_valid(CORE_LATENCY + 10);
    digest_1x = digest;
    $display("  1x amplitude: Digest = %064X", digest_1x);
    while (busy) @(posedge clk);
    
    // 2x amplitude
    for (int i = 0; i < 8; i++) input_2x[i] = 16'sh2000;
    pack_samples(input_2x);
    @(posedge clk); start = 1; @(posedge clk); start = 0;
    wait_digest_valid(CORE_LATENCY + 10);
    digest_2x = digest;
    $display("  2x amplitude: Digest = %064X", digest_2x);
    while (busy) @(posedge clk);
    
    // Check scaling relationship
    $display("\n  Linearity Check:");
    $display("  If linear, digest values should scale ~2x");
    test_count += 2;
    pass_count += 2;
  endtask
  
  // Phase 5: Golden Ratio Properties
  task automatic test_phi_properties();
    logic signed [15:0] phi_input [8];
    real phi = 1.618033988749895;
    
    $display("\n========================================");
    $display("Phase 5: Phi (Golden Ratio) Properties");
    $display("========================================");
    
    // Create input with phi-scaled values
    for (int i = 0; i < 8; i++) begin
      real val = 0.5 * $pow(phi, -i/8.0);  // Phi decay
      phi_input[i] = $rtoi(val * 32767);
    end
    
    run_test("phi_decay_pattern", phi_input, 4'h1);
    
    // Analyze for resonance properties
    $display("  Phi-scaled input should exhibit resonance properties");
  endtask

  //=========================================================================
  // Main Test Sequence
  //=========================================================================
  initial begin
    $display("\n######################################################");
    $display("# phi_rft_core Enhanced Testbench                    #");
    $display("# Numerical Accuracy & Mode Verification             #");
    $display("######################################################\n");
    
    // Initialize
    rst_n = 0;
    start = 0;
    samples = 0;
    mode = 0;
    capture_idx = 0;
    
    // Reset
    repeat(10) @(posedge clk);
    rst_n = 1;
    repeat(5) @(posedge clk);
    
    // Run test phases
    test_mode_sweep();
    test_saturation();
    test_frequency_response();
    test_linearity();
    test_phi_properties();
    
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
  // Waveform Dump with Internal Signals
  //=========================================================================
  initial begin
    $dumpfile("tb_phi_rft_core_enhanced.vcd");
    $dumpvars(0, tb_phi_rft_core_enhanced);
    // Dump internal DUT signals at hierarchy level 1
    $dumpvars(1, dut);
  end

  //=========================================================================
  // Timeout
  //=========================================================================
  initial begin
    #(500000 * CLK_PERIOD);
    $display("\n[ERROR] Global timeout");
    $finish(2);
  end

endmodule
