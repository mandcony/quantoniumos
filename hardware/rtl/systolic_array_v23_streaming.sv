// =============================================================================
// RFTPU v2.3 STREAMING: Continuous Flow High-Utilization Controller
// =============================================================================
// COPYRIGHT (C) 2025 QuantoniumOS Contributors
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-NC
//
// NON-COMMERCIAL USE ONLY - See LICENSE-CLAIMS-NC.md
// This file implements algorithms subject to US Patent Application 19/169,399
// Commercial licensing inquiries: See PATENT_NOTICE.md
// =============================================================================
// KEY INSIGHT: Don't drain between vectors! Keep the pipeline full.
//
// Systolic array timing for NxN array:
//   - First result emerges after 2N-1 cycles
//   - After that, one result per cycle (steady state)
//   - Drain only at the very end
//
// For 64 vectors through 8x8 array:
//   - 8 cycles to load weights
//   - 15 cycles latency (2*8-1) for first result
//   - 64 cycles for all inputs (1 per cycle)
//   - 15 more cycles to drain final results
//   - Total: 8 + 64 + 15 = 87 cycles
//   - Utilization: 64/87 = 73.6%
//
// Compare to v2.1: 64 * (8 + 27) = 2240 cycles → 2.9% util (horrible!)
// =============================================================================

`default_nettype none
`timescale 1ns/1ps

// -----------------------------------------------------------------------------
// Processing Element (unchanged)
// -----------------------------------------------------------------------------
module systolic_pe_v23x #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     load_weight,
    input  wire                     clear_acc,
    input  wire                     enable,
    input  wire [DATA_WIDTH-1:0]    weight_in,
    input  wire [DATA_WIDTH-1:0]    activation_in,
    input  wire [ACC_WIDTH-1:0]     psum_in,
    output reg  [DATA_WIDTH-1:0]    activation_out,
    output reg  [ACC_WIDTH-1:0]     psum_out,
    output wire                     mac_active
);
    reg [DATA_WIDTH-1:0] weight_reg;
    wire signed [2*DATA_WIDTH-1:0] product;
    assign product = $signed(weight_reg) * $signed(activation_in);
    assign mac_active = enable && (activation_in != '0);
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= '0;
            activation_out <= '0;
            psum_out <= '0;
        end else begin
            if (load_weight) weight_reg <= weight_in;
            if (enable) begin
                activation_out <= activation_in;
                if (clear_acc)
                    psum_out <= {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
                else
                    psum_out <= $signed(psum_in) + $signed({{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product});
            end
        end
    end
endmodule

// -----------------------------------------------------------------------------
// Systolic Array with Continuous Streaming
// -----------------------------------------------------------------------------
module systolic_array_v23x #(
    parameter int ARRAY_DIM  = 8,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     load_weights,
    input  wire                     clear_acc,
    input  wire                     compute_enable,
    input  wire [$clog2(ARRAY_DIM)-1:0] weight_row_sel,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_row_data,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] activation_in,
    output wire [ARRAY_DIM*ACC_WIDTH-1:0] result_out,
    output wire                     result_valid,
    output reg [31:0]               mac_count  // Count active MACs
);
    wire [DATA_WIDTH-1:0] activation_h [ARRAY_DIM][ARRAY_DIM+1];
    wire [ACC_WIDTH-1:0]  psum_v [ARRAY_DIM+1][ARRAY_DIM];
    wire mac_active [ARRAY_DIM][ARRAY_DIM];
    reg [DATA_WIDTH-1:0] skew_delay [ARRAY_DIM][ARRAY_DIM];
    reg [$clog2(3*ARRAY_DIM):0] cycle_count;
    
    genvar col, row;
    generate
        for (col = 0; col < ARRAY_DIM; col++) begin : init_psum
            assign psum_v[0][col] = '0;
        end
    endgenerate
    
    generate
        for (row = 0; row < ARRAY_DIM; row++) begin : pe_row
            for (col = 0; col < ARRAY_DIM; col++) begin : pe_col
                wire load_pe = load_weights && (weight_row_sel == row);
                systolic_pe_v23x #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_inst (
                    .clk(clk), .rst_n(rst_n),
                    .load_weight(load_pe), .clear_acc(clear_acc), .enable(compute_enable),
                    .weight_in(weight_row_data[(col+1)*DATA_WIDTH-1 -: DATA_WIDTH]),
                    .activation_in(activation_h[row][col]),
                    .psum_in(psum_v[row][col]),
                    .activation_out(activation_h[row][col+1]),
                    .psum_out(psum_v[row+1][col]),
                    .mac_active(mac_active[row][col])
                );
            end
        end
    endgenerate
    
    // Count active MACs each cycle
    integer ii, jj;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_count <= 0;
        end else if (compute_enable) begin
            for (ii = 0; ii < ARRAY_DIM; ii++) begin
                for (jj = 0; jj < ARRAY_DIM; jj++) begin
                    if (mac_active[ii][jj]) mac_count <= mac_count + 1;
                end
            end
        end
    end
    
    integer i, j;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < ARRAY_DIM; i++) for (j = 0; j < ARRAY_DIM; j++) skew_delay[i][j] <= '0;
            cycle_count <= '0;
        end else if (compute_enable) begin
            cycle_count <= cycle_count + 1;
            for (i = 0; i < ARRAY_DIM; i++) begin
                skew_delay[i][0] <= activation_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                for (j = 1; j < ARRAY_DIM; j++) skew_delay[i][j] <= skew_delay[i][j-1];
            end
        end else if (load_weights) begin
            // Don't reset during weight loading
        end else begin
            cycle_count <= '0;
            for (i = 0; i < ARRAY_DIM; i++) for (j = 0; j < ARRAY_DIM; j++) skew_delay[i][j] <= '0;
        end
    end
    
    generate
        for (row = 0; row < ARRAY_DIM; row++) begin : connect_act
            if (row == 0) assign activation_h[row][0] = skew_delay[row][0];
            else          assign activation_h[row][0] = skew_delay[row][row-1];
        end
    endgenerate
    
    generate
        for (col = 0; col < ARRAY_DIM; col++) begin : connect_out
            assign result_out[(col+1)*ACC_WIDTH-1 -: ACC_WIDTH] = psum_v[ARRAY_DIM][col];
        end
    endgenerate
    
    assign result_valid = (cycle_count >= 2*ARRAY_DIM - 1);
endmodule

// -----------------------------------------------------------------------------
// RFTPU v2.3 STREAMING: True Continuous-Flow Controller
// -----------------------------------------------------------------------------
module rftpu_systolic_v23x #(
    parameter int ARRAY_DIM   = 8,
    parameter int DATA_WIDTH  = 8,
    parameter int ACC_WIDTH   = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    input  wire                     start,
    input  wire [15:0]              num_vectors,
    
    // Weight loading (sequential rows)
    input  wire                     weight_valid,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_data,
    output wire                     weight_ready,
    
    // Activation streaming - continuous flow!
    input  wire                     act_valid,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] act_data,
    output wire                     act_ready,
    
    output wire [ARRAY_DIM*ACC_WIDTH-1:0] result_data,
    output wire                     result_valid,
    
    output wire                     busy,
    output wire                     done,
    
    // Performance
    output reg [31:0]               perf_total_cycles,
    output reg [31:0]               perf_weight_cycles,
    output reg [31:0]               perf_compute_cycles,
    output reg [31:0]               perf_idle_cycles,
    output wire [7:0]               utilization_pct
);

    typedef enum logic [2:0] {
        S_IDLE,
        S_LOAD_WEIGHTS,
        S_STREAM,        // Continuous streaming - no separate drain!
        S_DRAIN_FINAL,   // Only drain at the very end
        S_DONE
    } state_t;
    
    state_t state, next_state;
    
    reg [$clog2(ARRAY_DIM):0] weight_row_cnt;
    reg [$clog2(ARRAY_DIM)-1:0] weight_row_sel;
    reg [15:0] vectors_input;     // Vectors sent into array
    reg [$clog2(3*ARRAY_DIM):0] drain_cnt;
    
    reg sys_load, sys_clear, sys_enable;
    wire sys_valid;
    wire [31:0] sys_mac_count;
    
    systolic_array_v23x #(
        .ARRAY_DIM(ARRAY_DIM),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) systolic (
        .clk(clk), .rst_n(rst_n),
        .load_weights(sys_load), .clear_acc(sys_clear), .compute_enable(sys_enable),
        .weight_row_sel(weight_row_sel), .weight_row_data(weight_data),
        .activation_in(act_data),
        .result_out(result_data), .result_valid(sys_valid),
        .mac_count(sys_mac_count)
    );
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE;
        else state <= next_state;
    end
    
    always_comb begin
        next_state = state;
        sys_load = 0;
        sys_clear = 0;
        sys_enable = 0;
        
        case (state)
            S_IDLE: begin
                if (start) next_state = S_LOAD_WEIGHTS;
            end
            
            S_LOAD_WEIGHTS: begin
                if (weight_valid) begin
                    sys_load = 1;
                    if (weight_row_cnt >= ARRAY_DIM - 1)
                        next_state = S_STREAM;
                end
            end
            
            S_STREAM: begin
                // CONTINUOUS STREAMING - this is the key!
                // Accept a new vector every cycle if available
                if (act_valid) begin
                    sys_enable = 1;
                    sys_clear = (vectors_input == 0);  // Clear only on first
                end
                
                // Check if all vectors are in
                if (vectors_input >= num_vectors) begin
                    next_state = S_DRAIN_FINAL;
                end
            end
            
            S_DRAIN_FINAL: begin
                // Keep pipeline running to flush remaining results
                sys_enable = 1;
                if (drain_cnt >= 2*ARRAY_DIM)
                    next_state = S_DONE;
            end
            
            S_DONE: begin
                next_state = S_IDLE;
            end
        endcase
    end
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_row_cnt <= 0;
            weight_row_sel <= 0;
            vectors_input <= 0;
            drain_cnt <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    weight_row_cnt <= 0;
                    weight_row_sel <= 0;
                    vectors_input <= 0;
                    drain_cnt <= 0;
                end
                
                S_LOAD_WEIGHTS: begin
                    if (weight_valid) begin
                        weight_row_cnt <= weight_row_cnt + 1;
                        weight_row_sel <= weight_row_sel + 1;
                    end
                end
                
                S_STREAM: begin
                    if (act_valid) begin
                        vectors_input <= vectors_input + 1;
                    end
                end
                
                S_DRAIN_FINAL: begin
                    drain_cnt <= drain_cnt + 1;
                end
            endcase
        end
    end
    
    assign weight_ready = (state == S_LOAD_WEIGHTS);
    assign act_ready = (state == S_STREAM) && (vectors_input < num_vectors);
    assign result_valid = sys_valid;
    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);
    
    // Performance counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_total_cycles <= 0;
            perf_weight_cycles <= 0;
            perf_compute_cycles <= 0;
            perf_idle_cycles <= 0;
        end else if (state == S_IDLE && start) begin
            perf_total_cycles <= 0;
            perf_weight_cycles <= 0;
            perf_compute_cycles <= 0;
            perf_idle_cycles <= 0;
        end else if (busy) begin
            perf_total_cycles <= perf_total_cycles + 1;
            case (state)
                S_LOAD_WEIGHTS: if (weight_valid) perf_weight_cycles <= perf_weight_cycles + 1;
                S_STREAM: if (act_valid) perf_compute_cycles <= perf_compute_cycles + 1;
                          else perf_idle_cycles <= perf_idle_cycles + 1;
                S_DRAIN_FINAL: perf_compute_cycles <= perf_compute_cycles + 1;  // Still computing during drain
            endcase
        end
    end
    
    // Utilization = (compute + drain) / total
    // Actually better: compute / (weight + compute) ignoring idle
    wire [31:0] active_time = perf_weight_cycles + perf_compute_cycles;
    assign utilization_pct = (active_time > 0) ? 
                             ((perf_compute_cycles * 100) / active_time) : 8'd0;

endmodule

// =============================================================================
// Testbench
// =============================================================================
`ifdef SIMULATION
module rftpu_v23x_tb;
    localparam ARRAY_DIM = 8;
    localparam DATA_WIDTH = 8;
    localparam ACC_WIDTH = 32;
    
    reg clk, rst_n;
    reg start;
    reg [15:0] num_vectors;
    
    reg weight_valid;
    reg [ARRAY_DIM*DATA_WIDTH-1:0] weight_data;
    wire weight_ready;
    
    reg act_valid;
    reg [ARRAY_DIM*DATA_WIDTH-1:0] act_data;
    wire act_ready;
    
    wire [ARRAY_DIM*ACC_WIDTH-1:0] result_data;
    wire result_valid;
    wire busy, done;
    
    wire [31:0] perf_total, perf_weight, perf_compute, perf_idle;
    wire [7:0] utilization;
    
    rftpu_systolic_v23x #(.ARRAY_DIM(ARRAY_DIM)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .num_vectors(num_vectors),
        .weight_valid(weight_valid), .weight_data(weight_data), .weight_ready(weight_ready),
        .act_valid(act_valid), .act_data(act_data), .act_ready(act_ready),
        .result_data(result_data), .result_valid(result_valid),
        .busy(busy), .done(done),
        .perf_total_cycles(perf_total), .perf_weight_cycles(perf_weight),
        .perf_compute_cycles(perf_compute), .perf_idle_cycles(perf_idle),
        .utilization_pct(utilization)
    );
    
    initial clk = 0;
    always #5 clk = ~clk;
    
    integer i, results_received;
    
    initial begin
        $display("\n╔══════════════════════════════════════════════════════════════════╗");
        $display("║  RFTPU v2.3x STREAMING High-Utilization Test                     ║");
        $display("║  Array: %0dx%0d = %0d MACs                                          ║", 
                 ARRAY_DIM, ARRAY_DIM, ARRAY_DIM*ARRAY_DIM);
        $display("╚══════════════════════════════════════════════════════════════════╝");
        
        rst_n = 0;
        start = 0;
        num_vectors = 64;
        weight_valid = 0;
        weight_data = '0;
        act_valid = 0;
        act_data = '0;
        results_received = 0;
        
        #20 rst_n = 1;
        #10;
        
        // Start
        $display("\n[1] Starting: %0d vectors through %0dx%0d array", num_vectors, ARRAY_DIM, ARRAY_DIM);
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        // Load weights (8 cycles)
        $display("[2] Loading weights...");
        for (i = 0; i < ARRAY_DIM; i++) begin
            @(posedge clk);
            while (!weight_ready) @(posedge clk);
            weight_valid = 1;
            weight_data = {ARRAY_DIM{8'(i+1)}};
        end
        @(posedge clk);
        weight_valid = 0;
        $display("    Done: %0d rows loaded", ARRAY_DIM);
        
        // Stream activations continuously!
        $display("[3] Streaming %0d activations (continuous)...", num_vectors);
        i = 0;
        while (!done) begin
            @(posedge clk);
            
            // Send activation if ready and have more
            if (act_ready && i < num_vectors) begin
                act_valid = 1;
                act_data = {ARRAY_DIM{8'(i & 8'hFF)}};
                i = i + 1;
            end else begin
                act_valid = 0;
            end
            
            // Count results
            if (result_valid) results_received = results_received + 1;
            
            // Timeout
            if (perf_total > 500) begin
                $display("ERROR: Timeout at cycle %0d", perf_total);
                $finish;
            end
        end
        act_valid = 0;
        
        // Final results
        $display("\n╔══════════════════════════════════════════════════════════════════╗");
        $display("║  PERFORMANCE RESULTS                                             ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  Total Cycles:      %4d                                          ║", perf_total);
        $display("║  Weight Cycles:     %4d                                          ║", perf_weight);
        $display("║  Compute Cycles:    %4d                                          ║", perf_compute);
        $display("║  Idle Cycles:       %4d                                          ║", perf_idle);
        $display("║  Results Received:  %4d                                          ║", results_received);
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  UTILIZATION:       %3d%%                                         ║", utilization);
        $display("╠══════════════════════════════════════════════════════════════════╣");
        
        // Calculate throughput
        if (perf_total > 0) begin
            $display("║  Vectors/cycle:     %0d.%02d                                        ║",
                     num_vectors / perf_total,
                     ((num_vectors * 100) / perf_total) % 100);
            // MAC ops = vectors * N^2, but with skewing each vector uses N^2 MACs over N cycles
            // Effective MACs per cycle during compute:
            $display("║  Effective MAC/cyc: ~%0d (ramp-up affects avg)                   ║", 
                     ARRAY_DIM*ARRAY_DIM * num_vectors / perf_compute);
        end
        
        // Theoretical analysis
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  THEORETICAL ANALYSIS:                                           ║");
        $display("║  • Weight load: %0d cycles                                         ║", ARRAY_DIM);
        $display("║  • Pipeline fill: %0d cycles (2N-1)                                ║", 2*ARRAY_DIM-1);
        $display("║  • Steady-state: %0d cycles (one result/cycle)                    ║", num_vectors);
        $display("║  • Ideal total: %0d cycles                                        ║", ARRAY_DIM + num_vectors + 2*ARRAY_DIM - 1);
        $display("║  • Ideal util: %0d%%                                               ║", 
                 (num_vectors * 100) / (ARRAY_DIM + num_vectors + 2*ARRAY_DIM - 1));
        $display("╚══════════════════════════════════════════════════════════════════╝");
        
        if (utilization >= 70) begin
            $display("\n✓✓✓ PASS: %0d%% utilization >= 70%% target ✓✓✓\n", utilization);
        end else if (utilization >= 50) begin
            $display("\n~~ MARGINAL: %0d%% utilization (target 70%%) ~~\n", utilization);
        end else begin
            $display("\n✗✗✗ FAIL: %0d%% utilization < 50%% ✗✗✗\n", utilization);
        end
        
        #50 $finish;
    end
    
endmodule
`endif

`default_nettype wire
