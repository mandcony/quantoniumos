// =============================================================================
// RFTPU v2: TPU-Inspired Systolic Matrix Multiply Unit
// =============================================================================
// COPYRIGHT (C) 2025 QuantoniumOS Contributors
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-NC
//
// NON-COMMERCIAL USE ONLY - See LICENSE-CLAIMS-NC.md
// This file implements algorithms subject to US Patent Application 19/169,399
// Commercial licensing inquiries: See PATENT_NOTICE.md
// =============================================================================
// Reference: Google TPU Paper (arXiv:1704.04760)
// "In-Datacenter Performance Analysis of a Tensor Processing Unit"
//
// This module implements a scalable systolic array for matrix-vector multiply,
// following the weight-stationary dataflow pattern used in Google's TPU v1.
//
// Design Parameters:
//   - Configurable array dimensions (default 8x8 for FPGA validation)
//   - INT8 weights and activations (TPU standard)
//   - INT32 accumulators (prevents overflow)
//   - Weight-stationary dataflow (weights loaded once, activations flow through)
//
// Integration with RFTPU:
//   - Complements existing RFT cores
//   - Can accelerate SIS hash matrix operations (MODE_12)
//   - Provides general GEMM capability for ML workloads
// =============================================================================

`default_nettype none
`timescale 1ns/1ps

// -----------------------------------------------------------------------------
// Systolic MAC Processing Element (PE)
// -----------------------------------------------------------------------------
// Single multiply-accumulate unit forming the building block of the array.
// Implements weight-stationary dataflow: weight is loaded and held,
// activations flow from left, partial sums flow from top.
//
// Reference: TPU paper Section 4 "TPU Implementation"
// -----------------------------------------------------------------------------
module systolic_pe #(
    parameter int DATA_WIDTH = 8,      // INT8 standard
    parameter int ACC_WIDTH  = 32      // INT32 accumulator
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control signals
    input  wire                     load_weight,    // Load new weight
    input  wire                     clear_acc,      // Clear accumulator
    
    // Data inputs
    input  wire [DATA_WIDTH-1:0]    weight_in,      // Weight to load
    input  wire [DATA_WIDTH-1:0]    activation_in,  // Activation from left
    input  wire [ACC_WIDTH-1:0]     psum_in,        // Partial sum from above
    
    // Data outputs (systolic forwarding)
    output reg  [DATA_WIDTH-1:0]    activation_out, // Activation to right
    output reg  [ACC_WIDTH-1:0]     psum_out        // Partial sum to below
);

    // Weight register (stationary)
    reg [DATA_WIDTH-1:0] weight_reg;
    
    // Intermediate product (signed multiplication)
    wire signed [2*DATA_WIDTH-1:0] product;
    
    // Signed multiplication of weight and activation
    assign product = $signed(weight_reg) * $signed(activation_in);
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg     <= '0;
            activation_out <= '0;
            psum_out       <= '0;
        end else begin
            // Weight loading (happens during configuration phase)
            if (load_weight) begin
                weight_reg <= weight_in;
            end
            
            // Systolic forwarding: activation moves right with 1 cycle delay
            activation_out <= activation_in;
            
            // MAC operation or clear
            if (clear_acc) begin
                psum_out <= '0;
            end else begin
                // Accumulate: psum_out = psum_in + (weight * activation)
                psum_out <= psum_in + {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
            end
        end
    end

endmodule


// -----------------------------------------------------------------------------
// Systolic Array (NxN Matrix Multiply Unit)
// -----------------------------------------------------------------------------
// NxN array of PEs for matrix-vector multiplication.
// Implements C = A * B where A is weight matrix, B is activation vector.
//
// Dataflow (Weight-Stationary):
//   1. Weights are pre-loaded into each PE (configuration phase)
//   2. Activations flow from left edge, skewed by row
//   3. Partial sums flow from top edge (initialized to 0)
//   4. Results emerge from bottom edge after N+N-1 cycles
//
// Reference: TPU paper Figure 4 "Matrix Multiply Unit data path"
// -----------------------------------------------------------------------------
module systolic_array #(
    parameter int ARRAY_DIM  = 8,      // NxN array (8x8 default, TPU is 256x256)
    parameter int DATA_WIDTH = 8,      // INT8
    parameter int ACC_WIDTH  = 32      // INT32
)(
    input  wire                                     clk,
    input  wire                                     rst_n,
    
    // Control
    input  wire                                     load_weights,   // Enter weight load mode
    input  wire                                     clear_acc,      // Clear all accumulators
    input  wire                                     compute_enable, // Enable computation
    
    // Weight loading interface (row-by-row)
    input  wire [$clog2(ARRAY_DIM)-1:0]            weight_row_sel, // Which row to load
    input  wire [ARRAY_DIM-1:0][DATA_WIDTH-1:0]   weight_row_data, // Row of weights
    
    // Activation input (skewed internally)
    input  wire [ARRAY_DIM-1:0][DATA_WIDTH-1:0]   activation_in,  // Activation vector
    
    // Output results (from bottom of array)
    output wire [ARRAY_DIM-1:0][ACC_WIDTH-1:0]    result_out,     // Output vector
    output wire                                     result_valid    // Results ready
);

    // PE interconnect wires
    wire [DATA_WIDTH-1:0] activation_h [ARRAY_DIM][ARRAY_DIM+1]; // Horizontal (activation flow)
    wire [ACC_WIDTH-1:0]  psum_v       [ARRAY_DIM+1][ARRAY_DIM]; // Vertical (psum flow)
    
    // Load weight signals per PE
    wire load_pe [ARRAY_DIM][ARRAY_DIM];
    
    // Activation skewing delay lines - each row gets progressively more delay
    // Row 0: 0 delay, Row 1: 1 cycle delay, ..., Row N-1: N-1 cycles delay
    reg [DATA_WIDTH-1:0] skew_delay [ARRAY_DIM][ARRAY_DIM]; // [row][delay_stage]
    reg [$clog2(2*ARRAY_DIM)+1:0] cycle_count;
    
    // Initialize top row of psums to zero
    genvar col;
    generate
        for (col = 0; col < ARRAY_DIM; col++) begin : init_psum
            assign psum_v[0][col] = '0;
        end
    endgenerate
    
    // Generate NxN array of PEs
    genvar row;
    generate
        for (row = 0; row < ARRAY_DIM; row++) begin : pe_row
            for (col = 0; col < ARRAY_DIM; col++) begin : pe_col
                
                // Weight load signal: only active for selected row during load phase
                assign load_pe[row][col] = load_weights && (weight_row_sel == row);
                
                systolic_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .load_weight(load_pe[row][col]),
                    .clear_acc(clear_acc),
                    .weight_in(weight_row_data[col]),
                    .activation_in(activation_h[row][col]),
                    .psum_in(psum_v[row][col]),
                    .activation_out(activation_h[row][col+1]),
                    .psum_out(psum_v[row+1][col])
                );
            end
        end
    endgenerate
    
    // Activation skewing logic using shift registers
    // Each row i has i stages of delay before feeding the systolic array
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
            
            // Each row: first stage gets the activation input
            for (i = 0; i < ARRAY_DIM; i++) begin
                skew_delay[i][0] <= activation_in[i];
                // Shift through delay stages
                for (j = 1; j < ARRAY_DIM; j++) begin
                    skew_delay[i][j] <= skew_delay[i][j-1];
                end
            end
        end else begin
            cycle_count <= '0;
            // Clear delay lines when not computing
            for (i = 0; i < ARRAY_DIM; i++) begin
                for (j = 0; j < ARRAY_DIM; j++) begin
                    skew_delay[i][j] <= '0;
                end
            end
        end
    end
    
    // Connect skewed activations to left edge of array
    // Row 0 gets no delay, Row 1 gets 1 cycle delay, ..., Row 7 gets 7 cycles delay
    // skew_delay[row][0] is 1 cycle old, skew_delay[row][1] is 2 cycles old, etc.
    generate
        for (row = 0; row < ARRAY_DIM; row++) begin : connect_act
            if (row == 0) begin
                // Row 0: use current input directly (no extra delay beyond the 1 in skew_delay)
                assign activation_h[row][0] = skew_delay[row][0];
            end else begin
                // Row i (i>0): need i cycles delay total
                // skew_delay[row][row-1] gives us 'row' cycles of delay (0-indexed gives row-1 index for row cycles)
                assign activation_h[row][0] = skew_delay[row][row-1];
            end
        end
    endgenerate
    
    // Output connections from bottom of array
    generate
        for (col = 0; col < ARRAY_DIM; col++) begin : connect_out
            assign result_out[col] = psum_v[ARRAY_DIM][col];
        end
    endgenerate
    
    // Result valid after 2*ARRAY_DIM - 1 cycles
    assign result_valid = (cycle_count >= 2*ARRAY_DIM - 1);

endmodule


// -----------------------------------------------------------------------------
// RFTPU v2 Integration Wrapper
// -----------------------------------------------------------------------------
// Integrates systolic array with existing RFTPU infrastructure.
// Provides command interface compatible with existing tile_ctrl_frame_t.
// -----------------------------------------------------------------------------
module rftpu_systolic_unit #(
    parameter int ARRAY_DIM   = 8,
    parameter int DATA_WIDTH  = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int CTRL_WIDTH  = 128
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control interface (compatible with RFTPU)
    input  wire                     start,
    input  wire [3:0]               mode,           // 0xF = systolic GEMM mode
    input  wire [15:0]              length,         // Number of vectors
    
    // Data interface
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0]  data_in,
    output wire [ARRAY_DIM*ACC_WIDTH-1:0]   data_out,
    
    // Status
    output wire                     busy,
    output wire                     done,
    output wire                     result_valid
);

    // Mode definitions (extend RFTPU modes)
    localparam MODE_SYSTOLIC_GEMM = 4'hF;
    
    // State machine
    typedef enum logic [2:0] {
        S_IDLE,
        S_LOAD_WEIGHTS,
        S_COMPUTE,
        S_OUTPUT,
        S_DONE
    } state_t;
    
    state_t state, next_state;
    
    // Control signals
    reg load_weights_r;
    reg clear_acc_r;
    reg compute_enable_r;
    reg [$clog2(ARRAY_DIM):0] row_counter;
    reg [15:0] vector_counter;
    
    // Systolic array instance
    wire [ARRAY_DIM-1:0][DATA_WIDTH-1:0] activation_vec;
    wire [ARRAY_DIM-1:0][ACC_WIDTH-1:0]  result_vec;
    wire systolic_valid;
    
    // Unpack input data to activation vector
    genvar i;
    generate
        for (i = 0; i < ARRAY_DIM; i++) begin : unpack_act
            assign activation_vec[i] = data_in[(i+1)*DATA_WIDTH-1 : i*DATA_WIDTH];
        end
    endgenerate
    
    // Pack result vector to output data
    generate
        for (i = 0; i < ARRAY_DIM; i++) begin : pack_result
            assign data_out[(i+1)*ACC_WIDTH-1 : i*ACC_WIDTH] = result_vec[i];
        end
    endgenerate
    
    systolic_array #(
        .ARRAY_DIM(ARRAY_DIM),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) systolic_inst (
        .clk(clk),
        .rst_n(rst_n),
        .load_weights(load_weights_r),
        .clear_acc(clear_acc_r),
        .compute_enable(compute_enable_r),
        .weight_row_sel(row_counter[$clog2(ARRAY_DIM)-1:0]),
        .weight_row_data(activation_vec),  // Reuse data path for weight loading
        .activation_in(activation_vec),
        .result_out(result_vec),
        .result_valid(systolic_valid)
    );
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always_comb begin
        next_state = state;
        load_weights_r = 1'b0;
        clear_acc_r = 1'b0;
        compute_enable_r = 1'b0;
        
        case (state)
            S_IDLE: begin
                if (start && mode == MODE_SYSTOLIC_GEMM) begin
                    next_state = S_LOAD_WEIGHTS;
                    clear_acc_r = 1'b1;
                end
            end
            
            S_LOAD_WEIGHTS: begin
                load_weights_r = 1'b1;
                if (row_counter >= ARRAY_DIM) begin
                    next_state = S_COMPUTE;
                end
            end
            
            S_COMPUTE: begin
                compute_enable_r = 1'b1;
                if (systolic_valid && vector_counter >= length) begin
                    next_state = S_OUTPUT;
                end
            end
            
            S_OUTPUT: begin
                if (systolic_valid) begin
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
            row_counter <= '0;
            vector_counter <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    row_counter <= '0;
                    vector_counter <= '0;
                end
                S_LOAD_WEIGHTS: begin
                    row_counter <= row_counter + 1;
                end
                S_COMPUTE: begin
                    if (compute_enable_r) begin
                        vector_counter <= vector_counter + 1;
                    end
                end
            endcase
        end
    end
    
    // Status outputs
    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);
    assign result_valid = systolic_valid && (state == S_OUTPUT || state == S_COMPUTE);

endmodule


// -----------------------------------------------------------------------------
// Testbench
// -----------------------------------------------------------------------------
`ifdef SIMULATION
module systolic_array_tb;
    localparam ARRAY_DIM = 4;  // Smaller for testing
    localparam DATA_WIDTH = 8;
    localparam ACC_WIDTH = 32;
    
    reg clk;
    reg rst_n;
    reg load_weights;
    reg clear_acc;
    reg compute_enable;
    reg [$clog2(ARRAY_DIM)-1:0] weight_row_sel;
    reg [ARRAY_DIM-1:0][DATA_WIDTH-1:0] weight_row_data;
    reg [ARRAY_DIM-1:0][DATA_WIDTH-1:0] activation_in;
    wire [ARRAY_DIM-1:0][ACC_WIDTH-1:0] result_out;
    wire result_valid;
    
    systolic_array #(
        .ARRAY_DIM(ARRAY_DIM),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (.*);
    
    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;
    
    // Test sequence
    initial begin
        rst_n = 0;
        load_weights = 0;
        clear_acc = 0;
        compute_enable = 0;
        weight_row_sel = 0;
        weight_row_data = '0;
        activation_in = '0;
        
        #20 rst_n = 1;
        
        // Load identity matrix weights
        load_weights = 1;
        for (int row = 0; row < ARRAY_DIM; row++) begin
            weight_row_sel = row;
            for (int col = 0; col < ARRAY_DIM; col++) begin
                weight_row_data[col] = (row == col) ? 8'd1 : 8'd0;
            end
            @(posedge clk);
        end
        load_weights = 0;
        
        // Compute with test vector [1, 2, 3, 4]
        @(posedge clk);
        clear_acc = 1;
        @(posedge clk);
        clear_acc = 0;
        
        compute_enable = 1;
        activation_in[0] = 8'd1;
        activation_in[1] = 8'd2;
        activation_in[2] = 8'd3;
        activation_in[3] = 8'd4;
        
        // Wait for result
        @(posedge result_valid);
        $display("Result: %d %d %d %d", result_out[0], result_out[1], result_out[2], result_out[3]);
        $display("Expected: 1 2 3 4 (identity matrix)");
        
        #100 $finish;
    end
    
endmodule
`endif

`default_nettype wire
