`timescale 1ns/1ps
`default_nettype none

module tb_quantoniumos_unified;
  // Clock and reset
  reg clk = 0;
  always #5 clk = ~clk; // 100 MHz

  reg reset = 1;
  reg start = 0;
  reg [2:0] mode = 3'd0;
  reg [255:0] master_key = 256'h0;
  reg [127:0] data_in = 128'hDEADBEEF_CAFEBABE_FEEDFACE_01234567;

  wire [255:0] data_out;
  wire done;

  // Metrics
  wire [31:0] rft_energy;
  wire [15:0] sis_collision_resistance;
  wire [5:0]  feistel_round_count;
  wire [31:0] pipeline_throughput;

  // DUT
  quantoniumos_unified_core #(
    .RFT_SIZE(64),
    .SIS_N(512),
    .FEISTEL_ROUNDS(48)
  ) dut (
    .clk(clk),
    .reset(reset),
    .start(start),
    .mode(mode),
    .master_key(master_key),
    .data_in(data_in),
    .data_out(data_out),
    .done(done),
    .rft_energy(rft_energy),
    .sis_collision_resistance(sis_collision_resistance),
    .feistel_round_count(feistel_round_count),
    .pipeline_throughput(pipeline_throughput)
  );

  // Simple task to run a mode
  task run_mode(input [2:0] m, input [127:0] din, input [255:0] key);
    integer i;
    begin
      mode = m;
      data_in = din;
      master_key = key;
      @(negedge clk);
      start = 1'b1;
      @(negedge clk);
      start = 1'b0;

      // Wait for done with timeout
      i = 0;
      while (!done && i < 200000) begin
        @(negedge clk);
        i = i + 1;
      end
      if (!done) begin
        $display("[TB] TIMEOUT waiting for mode %0d", m);
      end else begin
        $display("[TB] Mode %0d DONE. data_out=%h", m, data_out);
      end
    end
  endtask

  initial begin
    $dumpfile("quantoniumos_unified.vcd");
    $dumpvars(0, tb_quantoniumos_unified);

    // Reset sequence
    repeat (5) @(negedge clk);
    reset = 0;
    repeat (5) @(negedge clk);

    // Test Mode 0: RFT only
    run_mode(3'd0, 128'h00010203_04050607_08090A0B_0C0D0E0F, 256'h0);

    // Test Mode 2: Feistel only (encrypt)
    run_mode(3'd2, 128'h00112233_44556677_8899AABB_CCDDEEFF, 256'h00010203_04050607_08090A0B_0C0D0E0F_10111213_14151617_18191A1B_1C1D1E1F);

    // Test Mode 1: SIS Hash
    run_mode(3'd1, 128'h11223344_55667788_99AABBCC_DDEEFF00, 256'h0);

    // Test Mode 3: Full pipeline
    run_mode(3'd3, 128'hA5A5A5A5_5A5A5A5A_F0F0F0F0_0F0F0F0F, 256'h0BAD_F00D_DEAD_BEEF_CAFE_BABE_FEED_FACE_C001_D00D_0123_4567_89AB_CDEF);

    $display("[TB] Simulation complete.");
    #20;
    $finish;
  end

endmodule

`default_nettype wire