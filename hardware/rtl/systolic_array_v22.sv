// =============================================================================
// RFTPU v2.2: Double-Buffered Weight Loading for High Utilization
// =============================================================================
// COPYRIGHT (C) 2025 QuantoniumOS Contributors
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-NC
//
// NON-COMMERCIAL USE ONLY - See LICENSE-CLAIMS-NC.md
// This file implements algorithms subject to US Patent Application 19/169,399
// Commercial licensing inquiries: See PATENT_NOTICE.md
// =============================================================================
// Enhancement: Loads next weight tile while current tile is computing
// Target: 80%+ sustained utilization vs 25% in v2.1
// =============================================================================

`default_nettype none
`timescale 1ns/1ps

// -----------------------------------------------------------------------------
// Double-Buffered Weight Bank
// -----------------------------------------------------------------------------
module weight_double_buffer #(
    parameter int ARRAY_DIM  = 8,
    parameter int DATA_WIDTH = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Write interface (load next weights while computing)
    input  wire                     wr_en,
    input  wire [$clog2(ARRAY_DIM)-1:0] wr_row,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] wr_data,
    
    // Read interface (to systolic array)
    input  wire [$clog2(ARRAY_DIM)-1:0] rd_row,
    output wire [ARRAY_DIM*DATA_WIDTH-1:0] rd_data,
    
    // Control
    input  wire                     swap_buffers,   // Swap active/shadow
    output wire                     buffer_ready    // Shadow buffer fully loaded
);

    // Two weight banks
    reg [ARRAY_DIM*DATA_WIDTH-1:0] bank0 [ARRAY_DIM];
    reg [ARRAY_DIM*DATA_WIDTH-1:0] bank1 [ARRAY_DIM];
    
    // Active bank selector (0 or 1)
    reg active_bank;
    
    // Track loading progress
    reg [$clog2(ARRAY_DIM):0] rows_loaded;
    
    assign buffer_ready = (rows_loaded >= ARRAY_DIM);
    
    // Read from active bank
    assign rd_data = active_bank ? bank1[rd_row] : bank0[rd_row];
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_bank <= 0;
            rows_loaded <= 0;
            for (int i = 0; i < ARRAY_DIM; i++) begin
                bank0[i] <= '0;
                bank1[i] <= '0;
            end
        end else begin
            // Write to shadow bank (opposite of active)
            if (wr_en) begin
                if (active_bank)
                    bank0[wr_row] <= wr_data;
                else
                    bank1[wr_row] <= wr_data;
                rows_loaded <= rows_loaded + 1;
            end
            
            // Swap buffers
            if (swap_buffers) begin
                active_bank <= ~active_bank;
                rows_loaded <= 0;
            end
        end
    end

endmodule


// -----------------------------------------------------------------------------
// Pipelined MAC PE (for higher clock frequency)
// -----------------------------------------------------------------------------
module systolic_pe_pipelined #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32,
    parameter int PIPE_STAGES = 2   // 2 = multiply + accumulate separate
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control
    input  wire                     load_weight,
    input  wire                     clear_acc,
    input  wire                     enable,
    
    // Data inputs
    input  wire [DATA_WIDTH-1:0]    weight_in,
    input  wire [DATA_WIDTH-1:0]    activation_in,
    input  wire [ACC_WIDTH-1:0]     psum_in,
    
    // Data outputs
    output reg  [DATA_WIDTH-1:0]    activation_out,
    output reg  [ACC_WIDTH-1:0]     psum_out,
    
    output wire                     mac_active
);

    // Weight register (stationary)
    reg [DATA_WIDTH-1:0] weight_reg;
    
    // Pipeline stage 1: Multiplication
    reg signed [2*DATA_WIDTH-1:0] product_pipe;
    reg [ACC_WIDTH-1:0] psum_pipe;
    reg clear_pipe;
    
    // Signed multiplication
    wire signed [2*DATA_WIDTH-1:0] product;
    assign product = $signed(weight_reg) * $signed(activation_in);
    
    assign mac_active = (activation_in != '0) && enable;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg     <= '0;
            activation_out <= '0;
            psum_out       <= '0;
            product_pipe   <= '0;
            psum_pipe      <= '0;
            clear_pipe     <= 0;
        end else begin
            // Weight loading
            if (load_weight) begin
                weight_reg <= weight_in;
            end
            
            // Pipeline stage 1: Compute product, forward psum
            if (enable) begin
                activation_out <= activation_in;
                product_pipe <= product;
                psum_pipe <= psum_in;
                clear_pipe <= clear_acc;
            end
            
            // Pipeline stage 2: Accumulate
            if (clear_pipe) begin
                psum_out <= '0;
            end else begin
                psum_out <= psum_pipe + {{(ACC_WIDTH-2*DATA_WIDTH){product_pipe[2*DATA_WIDTH-1]}}, product_pipe};
            end
        end
    end

endmodule


// -----------------------------------------------------------------------------
// High-Performance Systolic Array v2.2
// -----------------------------------------------------------------------------
module systolic_array_v22 #(
    parameter int ARRAY_DIM  = 8,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32,
    parameter int PIPE_DEPTH = 2     // Pipeline depth in each PE
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control
    input  wire                     load_weights,
    input  wire                     clear_acc,
    input  wire                     compute_enable,
    
    // Weight input (from double buffer)
    input  wire [$clog2(ARRAY_DIM)-1:0] weight_row_sel,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_row_data,
    
    // Activation input
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] activation_in,
    
    // Results
    output wire [ARRAY_DIM*ACC_WIDTH-1:0] result_out,
    output wire                     result_valid,
    output wire [ARRAY_DIM*ARRAY_DIM-1:0] mac_active_out
);

    // PE interconnect
    wire [DATA_WIDTH-1:0] activation_h [ARRAY_DIM][ARRAY_DIM+1];
    wire [ACC_WIDTH-1:0]  psum_v       [ARRAY_DIM+1][ARRAY_DIM];
    wire                  mac_active_grid [ARRAY_DIM][ARRAY_DIM];
    
    // Skewing
    reg [DATA_WIDTH-1:0] skew_delay [ARRAY_DIM][ARRAY_DIM];
    reg [$clog2(3*ARRAY_DIM):0] cycle_count;
    
    // Top psum = 0
    genvar col;
    generate
        for (col = 0; col < ARRAY_DIM; col++) begin : init_psum
            assign psum_v[0][col] = '0;
        end
    endgenerate
    
    // PE array with pipelining
    genvar row;
    generate
        for (row = 0; row < ARRAY_DIM; row++) begin : pe_row
            for (col = 0; col < ARRAY_DIM; col++) begin : pe_col
                wire load_pe = load_weights && (weight_row_sel == row);
                
                systolic_pe_pipelined #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH),
                    .PIPE_STAGES(PIPE_DEPTH)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .load_weight(load_pe),
                    .clear_acc(clear_acc),
                    .enable(compute_enable),
                    .weight_in(weight_row_data[(col+1)*DATA_WIDTH-1 -: DATA_WIDTH]),
                    .activation_in(activation_h[row][col]),
                    .psum_in(psum_v[row][col]),
                    .activation_out(activation_h[row][col+1]),
                    .psum_out(psum_v[row+1][col]),
                    .mac_active(mac_active_grid[row][col])
                );
                
                assign mac_active_out[row*ARRAY_DIM + col] = mac_active_grid[row][col];
            end
        end
    endgenerate
    
    // Skewing logic
    integer i, j;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < ARRAY_DIM; i++) begin
                for (j = 0; j < ARRAY_DIM; j++) begin
                    skew_delay[i][j] <= '0;
                end
            end
            cycle_count <= '0;
        end else if (compute_enable) begin
            cycle_count <= cycle_count + 1;
            for (i = 0; i < ARRAY_DIM; i++) begin
                skew_delay[i][0] <= activation_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                for (j = 1; j < ARRAY_DIM; j++) begin
                    skew_delay[i][j] <= skew_delay[i][j-1];
                end
            end
        end else begin
            cycle_count <= '0;
            for (i = 0; i < ARRAY_DIM; i++) begin
                for (j = 0; j < ARRAY_DIM; j++) begin
                    skew_delay[i][j] <= '0;
                end
            end
        end
    end
    
    // Connect skewed activations
    generate
        for (row = 0; row < ARRAY_DIM; row++) begin : connect_act
            if (row == 0)
                assign activation_h[row][0] = skew_delay[row][0];
            else
                assign activation_h[row][0] = skew_delay[row][row-1];
        end
    endgenerate
    
    // Output connections
    generate
        for (col = 0; col < ARRAY_DIM; col++) begin : connect_out
            assign result_out[(col+1)*ACC_WIDTH-1 -: ACC_WIDTH] = psum_v[ARRAY_DIM][col];
        end
    endgenerate
    
    // Valid after 2*N + pipeline_depth cycles
    assign result_valid = (cycle_count >= 2*ARRAY_DIM + PIPE_DEPTH - 1);

endmodule


// -----------------------------------------------------------------------------
// RFTPU v2.2 Top-Level with Double Buffering
// -----------------------------------------------------------------------------
module rftpu_systolic_v22 #(
    parameter int ARRAY_DIM   = 8,
    parameter int DATA_WIDTH  = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int PIPE_DEPTH  = 2,
    parameter int UB_DEPTH    = 256
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control
    input  wire                     start,
    input  wire [3:0]               mode,
    input  wire [15:0]              num_vectors,
    
    // Weight loading (to shadow buffer)
    input  wire                     weight_wr_en,
    input  wire [$clog2(ARRAY_DIM)-1:0] weight_wr_row,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_wr_data,
    
    // Activation input
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] activation_data,
    
    // Results
    output wire [ARRAY_DIM*ACC_WIDTH-1:0] result_data,
    output wire                     result_valid,
    
    // Status
    output wire                     busy,
    output wire                     done,
    output wire                     weight_buffer_ready,  // Shadow buffer ready
    
    // Performance counters
    output reg [31:0]               perf_compute_cycles,
    output reg [31:0]               perf_total_cycles,
    output reg [31:0]               perf_mac_ops
);

    localparam MODE_GEMM = 4'hF;
    
    // State machine
    typedef enum logic [2:0] {
        S_IDLE,
        S_WAIT_WEIGHTS,
        S_COMPUTE,
        S_DRAIN,
        S_DONE
    } state_t;
    
    state_t state, next_state;
    
    // Control
    reg compute_enable;
    reg clear_acc;
    reg swap_buffers;
    reg [$clog2(3*ARRAY_DIM):0] compute_cnt;
    
    // Double buffer
    wire [$clog2(ARRAY_DIM)-1:0] weight_rd_row;
    wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_rd_data;
    wire buffer_ready;
    
    weight_double_buffer #(
        .ARRAY_DIM(ARRAY_DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) weight_buf (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(weight_wr_en),
        .wr_row(weight_wr_row),
        .wr_data(weight_wr_data),
        .rd_row(weight_rd_row),
        .rd_data(weight_rd_data),
        .swap_buffers(swap_buffers),
        .buffer_ready(buffer_ready)
    );
    
    assign weight_buffer_ready = buffer_ready;
    assign weight_rd_row = '0;  // Weights already loaded to PEs
    
    // MAC active
    wire [ARRAY_DIM*ARRAY_DIM-1:0] mac_active;
    
    // Systolic array
    systolic_array_v22 #(
        .ARRAY_DIM(ARRAY_DIM),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PIPE_DEPTH(PIPE_DEPTH)
    ) systolic (
        .clk(clk),
        .rst_n(rst_n),
        .load_weights(swap_buffers),  // Load from buffer on swap
        .clear_acc(clear_acc),
        .compute_enable(compute_enable),
        .weight_row_sel('0),
        .weight_row_data(weight_rd_data),
        .activation_in(activation_data),
        .result_out(result_data),
        .result_valid(result_valid),
        .mac_active_out(mac_active)
    );
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= next_state;
    end
    
    always_comb begin
        next_state = state;
        compute_enable = 0;
        clear_acc = 0;
        swap_buffers = 0;
        
        case (state)
            S_IDLE: begin
                if (start && mode == MODE_GEMM && buffer_ready) begin
                    next_state = S_COMPUTE;
                    swap_buffers = 1;  // Swap buffers to make loaded weights active
                    clear_acc = 1;
                end else if (start && mode == MODE_GEMM) begin
                    next_state = S_WAIT_WEIGHTS;
                end
            end
            
            S_WAIT_WEIGHTS: begin
                if (buffer_ready) begin
                    next_state = S_COMPUTE;
                    swap_buffers = 1;
                    clear_acc = 1;
                end
            end
            
            S_COMPUTE: begin
                compute_enable = 1;
                if (compute_cnt >= 2*ARRAY_DIM + PIPE_DEPTH) begin
                    next_state = S_DRAIN;
                end
            end
            
            S_DRAIN: begin
                if (result_valid) begin
                    next_state = S_DONE;
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
            compute_cnt <= 0;
        end else begin
            if (state == S_IDLE || state == S_WAIT_WEIGHTS)
                compute_cnt <= 0;
            else if (compute_enable)
                compute_cnt <= compute_cnt + 1;
        end
    end
    
    // Performance counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_compute_cycles <= 0;
            perf_total_cycles <= 0;
            perf_mac_ops <= 0;
        end else if (state == S_IDLE && start) begin
            perf_compute_cycles <= 0;
            perf_total_cycles <= 0;
            perf_mac_ops <= 0;
        end else begin
            if (state != S_IDLE)
                perf_total_cycles <= perf_total_cycles + 1;
            if (compute_enable) begin
                perf_compute_cycles <= perf_compute_cycles + 1;
                perf_mac_ops <= perf_mac_ops + $countones(mac_active);
            end
        end
    end
    
    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);

endmodule

`default_nettype wire
