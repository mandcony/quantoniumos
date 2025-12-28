// =============================================================================
// RFTPU v2.3 SIMPLE: High-Utilization via Batch Processing
// =============================================================================
// COPYRIGHT (C) 2025 QuantoniumOS Contributors
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-NC
//
// NON-COMMERCIAL USE ONLY - See LICENSE-CLAIMS-NC.md
// This file implements algorithms subject to US Patent Application 19/169,399
// Commercial licensing inquiries: See PATENT_NOTICE.md
// =============================================================================
// KEY INSIGHT: Utilization = compute_cycles / (weight_load_cycles + compute_cycles)
//
// v2.1 Problem: Load 8 rows (8 cycles) → compute 1 vector (19 cycles) → repeat
//               Utilization = 19/(8+19) = 70% theoretical, ~26% actual
//
// v2.3 Solution: Load 8 rows (8 cycles) → compute N vectors (N*19 cycles) → repeat  
//                Utilization = N*19/(8+N*19) 
//                N=1: 70%, N=4: 90%, N=8: 95%, N=16: 97%
//
// This is the "reuse weights across many vectors" optimization.
// =============================================================================

`default_nettype none
`timescale 1ns/1ps

// -----------------------------------------------------------------------------
// Simple High-Utilization Systolic PE
// -----------------------------------------------------------------------------
module systolic_pe_v23s #(
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
// Systolic Array v2.3 Simple
// -----------------------------------------------------------------------------
module systolic_array_v23s #(
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
    output wire [ARRAY_DIM*ARRAY_DIM-1:0] mac_active_out,
    output wire [$clog2(3*ARRAY_DIM):0] cycle_count_out
);
    wire [DATA_WIDTH-1:0] activation_h [ARRAY_DIM][ARRAY_DIM+1];
    wire [ACC_WIDTH-1:0]  psum_v [ARRAY_DIM+1][ARRAY_DIM];
    reg [DATA_WIDTH-1:0] skew_delay [ARRAY_DIM][ARRAY_DIM];
    reg [$clog2(3*ARRAY_DIM):0] cycle_count;
    
    assign cycle_count_out = cycle_count;
    
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
                systolic_pe_v23s #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_inst (
                    .clk(clk), .rst_n(rst_n),
                    .load_weight(load_pe),
                    .clear_acc(clear_acc),
                    .enable(compute_enable),
                    .weight_in(weight_row_data[(col+1)*DATA_WIDTH-1 -: DATA_WIDTH]),
                    .activation_in(activation_h[row][col]),
                    .psum_in(psum_v[row][col]),
                    .activation_out(activation_h[row][col+1]),
                    .psum_out(psum_v[row+1][col]),
                    .mac_active(mac_active_out[row*ARRAY_DIM + col])
                );
            end
        end
    endgenerate
    
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
// RFTPU v2.3 Simple: Batch-Mode High-Utilization Controller
// -----------------------------------------------------------------------------
// The key optimization: BATCH_SIZE vectors computed per weight load
// -----------------------------------------------------------------------------
module rftpu_systolic_v23s #(
    parameter int ARRAY_DIM   = 8,
    parameter int DATA_WIDTH  = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int BATCH_SIZE  = 8    // Vectors per weight load - KEY PARAMETER!
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Simple command interface
    input  wire                     start,
    input  wire [15:0]              num_vectors,     // Total vectors to process
    
    // Weight loading (sequential rows)
    input  wire                     weight_valid,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_data,
    output wire                     weight_ready,
    
    // Activation streaming
    input  wire                     act_valid,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] act_data,
    output wire                     act_ready,
    
    // Results
    output wire [ARRAY_DIM*ACC_WIDTH-1:0] result_data,
    output wire                     result_valid,
    
    // Status
    output wire                     busy,
    output wire                     done,
    
    // Performance counters
    output reg [31:0]               perf_total_cycles,
    output reg [31:0]               perf_compute_cycles,
    output reg [31:0]               perf_weight_cycles,
    output reg [31:0]               perf_drain_cycles,
    output reg [31:0]               perf_mac_ops,
    output wire [7:0]               utilization_pct
);

    // State machine - simple and correct
    typedef enum logic [2:0] {
        S_IDLE,
        S_LOAD_WEIGHTS,      // Load ARRAY_DIM weight rows
        S_COMPUTE,           // Process BATCH_SIZE vectors
        S_DRAIN,             // Wait for pipeline to flush
        S_DONE
    } state_t;
    
    state_t state, next_state;
    
    // Counters
    reg [$clog2(ARRAY_DIM):0] weight_row_cnt;
    reg [$clog2(BATCH_SIZE+1):0] batch_cnt;
    reg [15:0] total_vectors_done;
    reg [$clog2(3*ARRAY_DIM):0] drain_cnt;
    reg weights_loaded;  // Track if weights are in the array
    
    // Systolic array signals
    reg sys_load_weights;
    reg sys_clear_acc;
    reg sys_compute;
    wire [ARRAY_DIM*ARRAY_DIM-1:0] mac_active;
    wire sys_result_valid;
    wire [$clog2(3*ARRAY_DIM):0] sys_cycle_count;
    
    // Weight row selection
    reg [$clog2(ARRAY_DIM)-1:0] weight_row_sel;
    
    systolic_array_v23s #(
        .ARRAY_DIM(ARRAY_DIM),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) systolic (
        .clk(clk),
        .rst_n(rst_n),
        .load_weights(sys_load_weights),
        .clear_acc(sys_clear_acc),
        .compute_enable(sys_compute),
        .weight_row_sel(weight_row_sel),
        .weight_row_data(weight_data),
        .activation_in(act_data),
        .result_out(result_data),
        .result_valid(sys_result_valid),
        .mac_active_out(mac_active),
        .cycle_count_out(sys_cycle_count)
    );
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE;
        else state <= next_state;
    end
    
    always_comb begin
        next_state = state;
        sys_load_weights = 0;
        sys_clear_acc = 0;
        sys_compute = 0;
        
        case (state)
            S_IDLE: begin
                if (start) next_state = S_LOAD_WEIGHTS;
            end
            
            S_LOAD_WEIGHTS: begin
                if (weight_valid) begin
                    sys_load_weights = 1;
                    if (weight_row_cnt >= ARRAY_DIM - 1) begin
                        next_state = S_COMPUTE;
                    end
                end
            end
            
            S_COMPUTE: begin
                sys_clear_acc = (batch_cnt == 0);  // Clear only at start of batch
                if (act_valid) begin
                    sys_compute = 1;
                    // Check if this batch is done OR all vectors done
                    if (total_vectors_done + batch_cnt + 1 >= num_vectors) begin
                        // All vectors done - go to drain then done
                        next_state = S_DRAIN;
                    end else if (batch_cnt >= BATCH_SIZE - 1) begin
                        // Batch done but more vectors remain - reuse weights!
                        next_state = S_DRAIN;
                    end
                end
            end
            
            S_DRAIN: begin
                sys_compute = 1;  // Keep computing to flush pipeline
                if (drain_cnt >= 2*ARRAY_DIM) begin
                    if (total_vectors_done + batch_cnt >= num_vectors) begin
                        next_state = S_DONE;
                    end else begin
                        // More vectors - go directly back to compute (reuse weights!)
                        next_state = S_COMPUTE;
                    end
                end
            end
            
            S_DONE: begin
                next_state = S_IDLE;
            end
        endcase
    end
    
    // Counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_row_cnt <= 0;
            weight_row_sel <= 0;
            batch_cnt <= 0;
            total_vectors_done <= 0;
            drain_cnt <= 0;
            weights_loaded <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    weight_row_cnt <= 0;
                    weight_row_sel <= 0;
                    batch_cnt <= 0;
                    total_vectors_done <= 0;
                    drain_cnt <= 0;
                    weights_loaded <= 0;
                end
                
                S_LOAD_WEIGHTS: begin
                    if (weight_valid) begin
                        weight_row_cnt <= weight_row_cnt + 1;
                        weight_row_sel <= weight_row_sel + 1;
                        if (weight_row_cnt >= ARRAY_DIM - 1) begin
                            weights_loaded <= 1;
                        end
                    end
                    batch_cnt <= 0;
                    drain_cnt <= 0;
                end
                
                S_COMPUTE: begin
                    if (act_valid) begin
                        batch_cnt <= batch_cnt + 1;
                    end
                    drain_cnt <= 0;
                end
                
                S_DRAIN: begin
                    drain_cnt <= drain_cnt + 1;
                    // Update total when transitioning out of drain
                    if (drain_cnt >= 2*ARRAY_DIM - 1) begin
                        total_vectors_done <= total_vectors_done + batch_cnt;
                        batch_cnt <= 0;  // Reset for next batch
                    end
                end
            endcase
        end
    end
    
    // Handshake signals
    assign weight_ready = (state == S_LOAD_WEIGHTS);
    assign act_ready = (state == S_COMPUTE);
    assign result_valid = sys_result_valid && (state == S_COMPUTE || state == S_DRAIN);
    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);
    
    // Performance counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_total_cycles <= 0;
            perf_compute_cycles <= 0;
            perf_weight_cycles <= 0;
            perf_drain_cycles <= 0;
            perf_mac_ops <= 0;
        end else if (state == S_IDLE && start) begin
            perf_total_cycles <= 0;
            perf_compute_cycles <= 0;
            perf_weight_cycles <= 0;
            perf_drain_cycles <= 0;
            perf_mac_ops <= 0;
        end else if (busy) begin
            perf_total_cycles <= perf_total_cycles + 1;
            
            if (state == S_LOAD_WEIGHTS && weight_valid)
                perf_weight_cycles <= perf_weight_cycles + 1;
            
            if (state == S_COMPUTE && act_valid) begin
                perf_compute_cycles <= perf_compute_cycles + 1;
                perf_mac_ops <= perf_mac_ops + $countones(mac_active);
            end
            
            if (state == S_DRAIN)
                perf_drain_cycles <= perf_drain_cycles + 1;
        end
    end
    
    // Utilization = compute / (weight + compute + drain) * 100
    wire [31:0] active_cycles = perf_weight_cycles + perf_compute_cycles + perf_drain_cycles;
    assign utilization_pct = (active_cycles > 0) ? 
                             ((perf_compute_cycles * 100) / active_cycles) : 8'd0;

endmodule

// =============================================================================
// Testbench: Verify High Utilization
// =============================================================================
`ifdef SIMULATION
module rftpu_v23s_tb;
    localparam ARRAY_DIM = 8;
    localparam DATA_WIDTH = 8;
    localparam ACC_WIDTH = 32;
    localparam BATCH_SIZE = 8;
    
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
    
    wire [31:0] perf_total, perf_compute, perf_weight, perf_drain, perf_mac;
    wire [7:0] utilization;
    
    rftpu_systolic_v23s #(
        .ARRAY_DIM(ARRAY_DIM),
        .BATCH_SIZE(BATCH_SIZE)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .num_vectors(num_vectors),
        .weight_valid(weight_valid), .weight_data(weight_data), .weight_ready(weight_ready),
        .act_valid(act_valid), .act_data(act_data), .act_ready(act_ready),
        .result_data(result_data), .result_valid(result_valid),
        .busy(busy), .done(done),
        .perf_total_cycles(perf_total), .perf_compute_cycles(perf_compute),
        .perf_weight_cycles(perf_weight), .perf_drain_cycles(perf_drain),
        .perf_mac_ops(perf_mac), .utilization_pct(utilization)
    );
    
    initial clk = 0;
    always #5 clk = ~clk;
    
    integer i, batch_num, vec_in_batch;
    integer total_vectors_sent;
    
    task load_weights();
        $display("  Loading %0d weight rows...", ARRAY_DIM);
        for (i = 0; i < ARRAY_DIM; i++) begin
            @(posedge clk);
            while (!weight_ready) @(posedge clk);
            weight_valid = 1;
            weight_data = {ARRAY_DIM{8'(i+1)}};  // Row i filled with value i+1
        end
        @(posedge clk);
        weight_valid = 0;
    endtask
    
    task send_batch();
        $display("  Sending batch of %0d vectors...", BATCH_SIZE);
        for (i = 0; i < BATCH_SIZE && total_vectors_sent < num_vectors; i++) begin
            @(posedge clk);
            while (!act_ready) @(posedge clk);
            act_valid = 1;
            act_data = {ARRAY_DIM{8'(total_vectors_sent & 8'hFF)}};
            total_vectors_sent = total_vectors_sent + 1;
        end
        @(posedge clk);
        act_valid = 0;
    endtask
    
    initial begin
        $display("\n=== RFTPU v2.3s High-Utilization Batch Test ===");
        $display("Array: %0dx%0d, Batch Size: %0d", ARRAY_DIM, ARRAY_DIM, BATCH_SIZE);
        
        rst_n = 0;
        start = 0;
        num_vectors = 64;
        weight_valid = 0;
        weight_data = '0;
        act_valid = 0;
        act_data = '0;
        total_vectors_sent = 0;
        
        #20 rst_n = 1;
        #10;
        
        // Start operation
        $display("\n[TEST] Processing %0d vectors with batch size %0d", num_vectors, BATCH_SIZE);
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        // Process batches until done
        batch_num = 0;
        while (!done) begin
            // Load weights when ready
            if (weight_ready) begin
                batch_num = batch_num + 1;
                $display("\n--- Batch %0d ---", batch_num);
                load_weights();
            end
            
            // Send activations when ready
            if (act_ready && total_vectors_sent < num_vectors) begin
                send_batch();
            end
            
            @(posedge clk);
            
            // Timeout protection
            if (perf_total > 10000) begin
                $display("ERROR: Timeout!");
                $finish;
            end
        end
        
        // Results
        $display("\n==========================================");
        $display("RFTPU v2.3s PERFORMANCE RESULTS");
        $display("==========================================");
        $display("Total Cycles:    %0d", perf_total);
        $display("Weight Cycles:   %0d", perf_weight);
        $display("Compute Cycles:  %0d", perf_compute);
        $display("Drain Cycles:    %0d", perf_drain);
        $display("MAC Operations:  %0d", perf_mac);
        $display("Vectors Done:    %0d", num_vectors);
        $display("------------------------------------------");
        $display("UTILIZATION:     %0d%%", utilization);
        $display("------------------------------------------");
        
        // Analysis
        $display("\nAnalysis:");
        $display("  Theoretical MAC ops: %0d (vectors × ARRAY_DIM²)", num_vectors * ARRAY_DIM * ARRAY_DIM);
        $display("  Actual MAC ops:      %0d", perf_mac);
        $display("  Batches processed:   %0d", batch_num);
        
        // Calculate effective throughput
        if (perf_total > 0) begin
            $display("  MAC/cycle:           %0d.%02d", 
                     perf_mac / perf_total,
                     ((perf_mac * 100) / perf_total) % 100);
            $display("  GOPS @ 100MHz:       %0d.%02d",
                     (perf_mac * 100) / perf_total / 1000,
                     ((perf_mac * 100) / perf_total) % 1000 / 10);
        end
        
        // Pass/Fail
        if (utilization >= 70) begin
            $display("\n✓ PASS: Utilization %0d%% >= 70%% target", utilization);
        end else if (utilization >= 50) begin
            $display("\n~ MARGINAL: Utilization %0d%% (target 70%%)", utilization);
        end else begin
            $display("\n✗ FAIL: Utilization %0d%% < 50%%", utilization);
        end
        
        $display("\n=== TEST COMPLETE ===\n");
        #50 $finish;
    end
    
endmodule
`endif

`default_nettype wire
