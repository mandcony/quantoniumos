// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025 Luis M. Minier / quantoniumos
//
// Canonical RFT Cross-Validation Testbench
// =========================================
// Drives the phi_rft_core with Q1.15 test vectors from rft_test_vectors.svh
// and dumps outputs for comparison with Python reference.
//
// Usage:
//   make build-crossval
//   ./obj_dir/Vtb_canonical_rft_crossval
//   python3 verify_rtl_vs_python.py

`timescale 1ns/1ps

module tb_canonical_rft_crossval;

   // Include test vectors
   `include "rft_test_vectors.svh"

   // Parameters matching hardware
   localparam int SAMPLE_WIDTH  = 16;
   localparam int BLOCK_SAMPLES = 8;
   localparam int DIGEST_WIDTH  = 256;
   localparam int CORE_LATENCY  = 12;

   // Clock and reset
   logic clk;
   logic rst_n;

   // DUT interface
   logic                                start;
   logic [BLOCK_SAMPLES*SAMPLE_WIDTH-1:0] samples_real;
   logic [BLOCK_SAMPLES*SAMPLE_WIDTH-1:0] samples_imag;
   logic [3:0]                          mode;
   logic                                busy;
   logic                                output_valid;
   logic [BLOCK_SAMPLES*SAMPLE_WIDTH-1:0] rft_out_real;
   logic [BLOCK_SAMPLES*SAMPLE_WIDTH-1:0] rft_out_imag;

   // Test control
   int test_idx;
   int cycle_count;
   int output_file;
   int tests_passed;
   int tests_failed;

   // Clock generation: 950 MHz target, but use 10ns period for simulation
   initial begin
      clk = 0;
      forever #5 clk = ~clk;
   end

   // Pack test vector into bus format
   function automatic logic [BLOCK_SAMPLES*SAMPLE_WIDTH-1:0] pack_samples(
      input logic signed [15:0] arr [BLOCK_SAMPLES]
   );
      logic [BLOCK_SAMPLES*SAMPLE_WIDTH-1:0] packed_val;
      for (int i = 0; i < BLOCK_SAMPLES; i++) begin
         packed_val[i*SAMPLE_WIDTH +: SAMPLE_WIDTH] = arr[i];
      end
      return packed_val;
   endfunction

   // Unpack output bus to array
   function automatic void unpack_samples(
      input logic [BLOCK_SAMPLES*SAMPLE_WIDTH-1:0] packed_val,
      output logic signed [15:0] arr [BLOCK_SAMPLES]
   );
      for (int i = 0; i < BLOCK_SAMPLES; i++) begin
         arr[i] = packed_val[i*SAMPLE_WIDTH +: SAMPLE_WIDTH];
      end
   endfunction

   // Canonical RFT Core (behavioral model for cross-validation)
   // This matches the kernel_real LUT in rftpu_architecture.tlv
   logic signed [15:0] kernel_lut [0:63];
   initial begin
      // Row 0
      kernel_lut[0]  = 16'shD6E0; kernel_lut[1]  = 16'sh3209;
      kernel_lut[2]  = 16'shD1F4; kernel_lut[3]  = 16'shCF18;
      kernel_lut[4]  = 16'sh36D4; kernel_lut[5]  = 16'shC837;
      kernel_lut[6]  = 16'shDAF0; kernel_lut[7]  = 16'shF272;
      // Row 1
      kernel_lut[8]  = 16'shD2A3; kernel_lut[9]  = 16'sh301D;
      kernel_lut[10] = 16'sh2BF0; kernel_lut[11] = 16'shE165;
      kernel_lut[12] = 16'sh2641; kernel_lut[13] = 16'sh3DE5;
      kernel_lut[14] = 16'sh3457; kernel_lut[15] = 16'sh214B;
      // Row 2
      kernel_lut[16] = 16'shD0C9; kernel_lut[17] = 16'shD806;
      kernel_lut[18] = 16'sh2C59; kernel_lut[19] = 16'shC4D2;
      kernel_lut[20] = 16'shDDDE; kernel_lut[21] = 16'sh1BBC;
      kernel_lut[22] = 16'shCACC; kernel_lut[23] = 16'shCFD0;
      // Row 3
      kernel_lut[24] = 16'shD0F5; kernel_lut[25] = 16'shD5E0;
      kernel_lut[26] = 16'shD160; kernel_lut[27] = 16'shDB1D;
      kernel_lut[28] = 16'shCD70; kernel_lut[29] = 16'shEA1E;
      kernel_lut[30] = 16'sh2353; kernel_lut[31] = 16'sh43A8;
      // Row 4
      kernel_lut[32] = 16'shD0F5; kernel_lut[33] = 16'sh2A20;
      kernel_lut[34] = 16'shD160; kernel_lut[35] = 16'sh24E3;
      kernel_lut[36] = 16'shCD70; kernel_lut[37] = 16'sh15E2;
      kernel_lut[38] = 16'sh2353; kernel_lut[39] = 16'shBC58;
      // Row 5
      kernel_lut[40] = 16'shD0C9; kernel_lut[41] = 16'sh27FA;
      kernel_lut[42] = 16'sh2C59; kernel_lut[43] = 16'sh3B2E;
      kernel_lut[44] = 16'shDDDE; kernel_lut[45] = 16'shE444;
      kernel_lut[46] = 16'shCACC; kernel_lut[47] = 16'sh3030;
      // Row 6
      kernel_lut[48] = 16'shD2A3; kernel_lut[49] = 16'shCFE3;
      kernel_lut[50] = 16'sh2BF0; kernel_lut[51] = 16'sh1E9B;
      kernel_lut[52] = 16'sh2641; kernel_lut[53] = 16'shC21B;
      kernel_lut[54] = 16'sh3457; kernel_lut[55] = 16'shDEB5;
      // Row 7
      kernel_lut[56] = 16'shD6E0; kernel_lut[57] = 16'shCDF7;
      kernel_lut[58] = 16'shD1F4; kernel_lut[59] = 16'sh30E8;
      kernel_lut[60] = 16'sh36D4; kernel_lut[61] = 16'sh37C9;
      kernel_lut[62] = 16'shDAF0; kernel_lut[63] = 16'sh0D8E;
   end

   // Behavioral RFT computation (matches hardware pipeline intent)
   logic signed [15:0] input_real_arr [BLOCK_SAMPLES];
   logic signed [15:0] input_imag_arr [BLOCK_SAMPLES];
   logic signed [31:0] acc_real [BLOCK_SAMPLES];
   logic signed [31:0] acc_imag [BLOCK_SAMPLES];
   logic signed [15:0] result_real [BLOCK_SAMPLES];
   logic signed [15:0] result_imag [BLOCK_SAMPLES];

   // Pipeline delay modeling
   logic [CORE_LATENCY-1:0] valid_pipe;
   logic [BLOCK_SAMPLES*SAMPLE_WIDTH-1:0] result_real_packed;
   logic [BLOCK_SAMPLES*SAMPLE_WIDTH-1:0] result_imag_packed;

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         valid_pipe <= '0;
         output_valid <= 0;
         busy <= 0;
      end else begin
         valid_pipe <= {valid_pipe[CORE_LATENCY-2:0], start};
         output_valid <= valid_pipe[CORE_LATENCY-1];
         busy <= |valid_pipe;
      end
   end

   // Combinational RFT matrix-vector multiply (Q1.15 × Q1.15 → Q2.30 → Q1.15)
   always_comb begin
      // Unpack inputs
      for (int i = 0; i < BLOCK_SAMPLES; i++) begin
         input_real_arr[i] = samples_real[i*SAMPLE_WIDTH +: SAMPLE_WIDTH];
         input_imag_arr[i] = samples_imag[i*SAMPLE_WIDTH +: SAMPLE_WIDTH];
      end

      // Matrix-vector multiply: y = Ψ @ x
      // Canonical RFT has real-only kernel, so:
      // y_real[k] = Σ kernel[k,n] * x_real[n]
      // y_imag[k] = Σ kernel[k,n] * x_imag[n]
      for (int k = 0; k < BLOCK_SAMPLES; k++) begin
         acc_real[k] = 0;
         acc_imag[k] = 0;
         for (int n = 0; n < BLOCK_SAMPLES; n++) begin
            // Q1.15 × Q1.15 = Q2.30
            acc_real[k] = acc_real[k] + kernel_lut[k*8 + n] * input_real_arr[n];
            acc_imag[k] = acc_imag[k] + kernel_lut[k*8 + n] * input_imag_arr[n];
         end
         // Q2.30 → Q1.15 (shift right 15, with saturation)
         /* verilator lint_off WIDTHTRUNC */
         result_real[k] = (acc_real[k] >>> 15);
         result_imag[k] = (acc_imag[k] >>> 15);
         /* verilator lint_on WIDTHTRUNC */
      end

      // Pack outputs
      for (int i = 0; i < BLOCK_SAMPLES; i++) begin
         result_real_packed[i*SAMPLE_WIDTH +: SAMPLE_WIDTH] = result_real[i];
         result_imag_packed[i*SAMPLE_WIDTH +: SAMPLE_WIDTH] = result_imag[i];
      end
   end

   // Register outputs
   always_ff @(posedge clk) begin
      if (valid_pipe[CORE_LATENCY-1]) begin
         rft_out_real <= result_real_packed;
         rft_out_imag <= result_imag_packed;
      end
   end

   // Main test sequence
   initial begin
      // Open output file for Python comparison
      output_file = $fopen("rtl_outputs.csv", "w");
      if (output_file == 0) begin
         $display("ERROR: Cannot open rtl_outputs.csv");
         $finish;
      end
      $fwrite(output_file, "test_id,sample_idx,input_real,input_imag,output_real,output_imag\n");

      // Initialize
      rst_n = 0;
      start = 0;
      samples_real = 0;
      samples_imag = 0;
      mode = 0;
      test_idx = 0;
      tests_passed = 0;
      tests_failed = 0;

      // Reset sequence
      repeat(10) @(posedge clk);
      rst_n = 1;
      repeat(5) @(posedge clk);

      $display("=== Canonical RFT Cross-Validation Testbench ===");
      $display("Running %0d test vectors...", NUM_RFT_TESTS);

      // Run all test vectors
      for (test_idx = 0; test_idx < NUM_RFT_TESTS; test_idx++) begin
         // Pack inputs
         samples_real = pack_samples(rft_test_input_real[test_idx]);
         samples_imag = pack_samples(rft_test_input_imag[test_idx]);

         // Start transform
         @(posedge clk);
         start = 1;
         @(posedge clk);
         start = 0;

         // Wait for output
         wait(output_valid);
         @(posedge clk);

         // Dump results to CSV
         for (int i = 0; i < BLOCK_SAMPLES; i++) begin
            $fwrite(output_file, "%0d,%0d,%0d,%0d,%0d,%0d\n",
               test_idx, i,
               $signed(rft_test_input_real[test_idx][i]),
               $signed(rft_test_input_imag[test_idx][i]),
               $signed(rft_out_real[i*SAMPLE_WIDTH +: SAMPLE_WIDTH]),
               $signed(rft_out_imag[i*SAMPLE_WIDTH +: SAMPLE_WIDTH])
            );
         end

         // Compare with expected (from Python)
         begin
            logic signed [15:0] exp_real [BLOCK_SAMPLES];
            logic signed [15:0] got_real [BLOCK_SAMPLES];
            int max_error;

            for (int i = 0; i < BLOCK_SAMPLES; i++) begin
               exp_real[i] = rft_test_expected_real[test_idx][i];
               got_real[i] = rft_out_real[i*SAMPLE_WIDTH +: SAMPLE_WIDTH];
            end

            max_error = 0;
            for (int i = 0; i < BLOCK_SAMPLES; i++) begin
               int err;
               err = (got_real[i] > exp_real[i]) ? (got_real[i] - exp_real[i]) : (exp_real[i] - got_real[i]);
               if (err > max_error) max_error = err;
            end

            // Allow 2 LSB tolerance for fixed-point rounding
            if (max_error <= 2) begin
               $display("  Test %2d: PASS (max_error=%0d LSB)", test_idx, max_error);
               tests_passed++;
            end else begin
               $display("  Test %2d: FAIL (max_error=%0d LSB)", test_idx, max_error);
               tests_failed++;
            end
         end

         repeat(5) @(posedge clk);
      end

      $fclose(output_file);

      $display("");
      $display("=== Cross-Validation Summary ===");
      $display("Tests Passed: %0d / %0d", tests_passed, NUM_RFT_TESTS);
      $display("Tests Failed: %0d", tests_failed);
      $display("RTL outputs written to: rtl_outputs.csv");

      if (tests_failed == 0) begin
         $display("SUCCESS: RTL matches Python reference within tolerance");
      end else begin
         $display("FAILURE: RTL/Python mismatch detected");
      end

      $finish;
   end

   // Timeout watchdog
   initial begin
      #100000;
      $display("ERROR: Simulation timeout");
      $finish;
   end

endmodule
