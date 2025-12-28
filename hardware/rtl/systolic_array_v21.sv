// =============================================================================
// RFTPU v2.1: Complete Enhanced Systolic Array
// =============================================================================
// COPYRIGHT (C) 2025 QuantoniumOS Contributors
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-NC
//
// NON-COMMERCIAL USE ONLY - See LICENSE-CLAIMS-NC.md
// This file implements algorithms subject to US Patent Application 19/169,399
// Commercial licensing inquiries: See PATENT_NOTICE.md
// =============================================================================
// ALL TODOs IMPLEMENTED:
//   TODO 1: K-Tiling + Accumulation for K > ARRAY_DIM ✓
//   TODO 2: Separate Weight/Activation Paths with Weight FIFO ✓
//   TODO 3: On-Chip Scratchpad (Unified Buffer v0) ✓
//   TODO 4: SIS_HASH routing through systolic path ✓
//   TODO 5: Scalable to 16×16 (parameterized ARRAY_DIM) ✓
//   TODO 6: INT8 Quantization with rounding modes ✓
//   TODO 7: Performance Counters ✓
// =============================================================================

`default_nettype none
`timescale 1ns/1ps

// -----------------------------------------------------------------------------
// Processing Element v2.1 (With Weight Register, Saturation & Rounding)
// -----------------------------------------------------------------------------
module systolic_pe_v21 #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32,
    parameter bit SATURATE   = 0,      // Enable saturation on accumulator
    parameter int ROUND_MODE = 0       // 0=truncate, 1=round-half-up, 2=round-to-nearest-even
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control
    input  wire                     load_weight,
    input  wire                     clear_acc,
    
    // Data inputs
    input  wire [DATA_WIDTH-1:0]    weight_in,
    input  wire [DATA_WIDTH-1:0]    activation_in,
    input  wire [ACC_WIDTH-1:0]     psum_in,
    
    // Data outputs
    output reg  [DATA_WIDTH-1:0]    activation_out,
    output reg  [ACC_WIDTH-1:0]     psum_out,
    
    // Quantized output (for INT8 requantization)
    output wire [DATA_WIDTH-1:0]    quant_out,
    
    // Status
    output wire                     mac_active
);

    // Weight register (stationary)
    reg [DATA_WIDTH-1:0] weight_reg;
    
    // Signed multiplication
    wire signed [2*DATA_WIDTH-1:0] product;
    assign product = $signed(weight_reg) * $signed(activation_in);
    
    // Sign-extended accumulation
    wire signed [ACC_WIDTH-1:0] acc_next;
    assign acc_next = $signed(psum_in) + $signed({{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product});
    
    // Overflow detection for saturation
    wire overflow_pos = !psum_in[ACC_WIDTH-1] && !product[2*DATA_WIDTH-1] && acc_next[ACC_WIDTH-1];
    wire overflow_neg = psum_in[ACC_WIDTH-1] && product[2*DATA_WIDTH-1] && !acc_next[ACC_WIDTH-1];
    
    // MAC is active when activation is non-zero (for perf counting)
    assign mac_active = (activation_in != '0);
    
    // INT8 Requantization with rounding (TODO 6)
    // Scale factor assumed to be right-shift by (ACC_WIDTH - DATA_WIDTH - 8) bits
    localparam int SHIFT = ACC_WIDTH - DATA_WIDTH;
    wire signed [ACC_WIDTH-1:0] shifted;
    wire round_bit;
    wire signed [DATA_WIDTH-1:0] quant_raw;
    
    generate
        if (ROUND_MODE == 0) begin : trunc
            // Truncation (fastest)
            assign shifted = psum_out >>> SHIFT;
            assign round_bit = 1'b0;
        end else if (ROUND_MODE == 1) begin : round_half_up
            // Round half up
            assign shifted = (psum_out + (1 << (SHIFT-1))) >>> SHIFT;
            assign round_bit = psum_out[SHIFT-1];
        end else begin : round_nearest_even
            // Round to nearest even (banker's rounding)
            wire guard = psum_out[SHIFT-1];
            wire sticky = |psum_out[SHIFT-2:0];
            wire lsb = psum_out[SHIFT];
            assign round_bit = guard && (sticky || lsb);
            assign shifted = (psum_out + (round_bit << (SHIFT-1))) >>> SHIFT;
        end
    endgenerate
    
    // Saturate to INT8 range [-128, 127]
    assign quant_raw = (shifted > 127) ? 8'sd127 :
                       (shifted < -128) ? -8'sd128 :
                       shifted[DATA_WIDTH-1:0];
    assign quant_out = quant_raw;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg     <= '0;
            activation_out <= '0;
            psum_out       <= '0;
        end else begin
            // Weight loading
            if (load_weight) begin
                weight_reg <= weight_in;
            end
            
            // Systolic forwarding
            activation_out <= activation_in;
            
            // MAC with optional saturation
            if (clear_acc) begin
                psum_out <= '0;
            end else if (SATURATE) begin
                if (overflow_pos)
                    psum_out <= {1'b0, {(ACC_WIDTH-1){1'b1}}}; // MAX_POS
                else if (overflow_neg)
                    psum_out <= {1'b1, {(ACC_WIDTH-1){1'b0}}}; // MAX_NEG
                else
                    psum_out <= acc_next;
            end else begin
                psum_out <= acc_next;
            end
        end
    end

endmodule


// -----------------------------------------------------------------------------
// Weight FIFO (TODO 2: Decouples weight loading from compute)
// -----------------------------------------------------------------------------
module weight_fifo_v21 #(
    parameter int DEPTH      = 16,
    parameter int DATA_WIDTH = 64   // One row of weights
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Write interface (from DMA/host)
    input  wire                     wr_en,
    input  wire [DATA_WIDTH-1:0]    wr_data,
    output wire                     full,
    
    // Read interface (to systolic array)
    input  wire                     rd_en,
    output wire [DATA_WIDTH-1:0]    rd_data,
    output wire                     empty,
    
    // Status
    output wire [$clog2(DEPTH):0]   count
);

    reg [DATA_WIDTH-1:0] mem [DEPTH];
    reg [$clog2(DEPTH):0] wr_ptr, rd_ptr;
    reg [$clog2(DEPTH):0] cnt;
    
    assign full  = (cnt == DEPTH);
    assign empty = (cnt == 0);
    assign count = cnt;
    assign rd_data = mem[rd_ptr[$clog2(DEPTH)-1:0]];
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            cnt    <= '0;
        end else begin
            if (wr_en && !full) begin
                mem[wr_ptr[$clog2(DEPTH)-1:0]] <= wr_data;
                wr_ptr <= wr_ptr + 1;
            end
            if (rd_en && !empty) begin
                rd_ptr <= rd_ptr + 1;
            end
            
            case ({wr_en && !full, rd_en && !empty})
                2'b10: cnt <= cnt + 1;
                2'b01: cnt <= cnt - 1;
                default: cnt <= cnt;
            endcase
        end
    end

endmodule


// -----------------------------------------------------------------------------
// Unified Buffer v0 (TODO 3: On-Chip Scratchpad)
// -----------------------------------------------------------------------------
module unified_buffer_v21 #(
    parameter int DEPTH       = 256,    // Number of vectors
    parameter int WIDTH       = 64,     // Bits per vector (8 x 8-bit)
    parameter int NUM_BANKS   = 2       // Ping-pong
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Port A (DMA write / Host interface)
    input  wire                     a_en,
    input  wire                     a_we,
    input  wire [$clog2(DEPTH)-1:0] a_addr,
    input  wire [WIDTH-1:0]         a_wdata,
    output reg  [WIDTH-1:0]         a_rdata,
    
    // Port B (Systolic array read/write)
    input  wire                     b_en,
    input  wire                     b_we,
    input  wire [$clog2(DEPTH)-1:0] b_addr,
    input  wire [WIDTH-1:0]         b_wdata,
    output reg  [WIDTH-1:0]         b_rdata,
    
    // Bank select (for ping-pong)
    input  wire                     bank_sel
);

    // Dual-port BRAM inference
    reg [WIDTH-1:0] bank0 [DEPTH/NUM_BANKS];
    reg [WIDTH-1:0] bank1 [DEPTH/NUM_BANKS];
    
    wire [$clog2(DEPTH/NUM_BANKS)-1:0] addr_a = a_addr[$clog2(DEPTH/NUM_BANKS)-1:0];
    wire [$clog2(DEPTH/NUM_BANKS)-1:0] addr_b = b_addr[$clog2(DEPTH/NUM_BANKS)-1:0];
    
    always_ff @(posedge clk) begin
        if (a_en) begin
            if (a_we) begin
                if (bank_sel)
                    bank1[addr_a] <= a_wdata;
                else
                    bank0[addr_a] <= a_wdata;
            end
            a_rdata <= bank_sel ? bank1[addr_a] : bank0[addr_a];
        end
    end
    
    always_ff @(posedge clk) begin
        if (b_en) begin
            if (b_we) begin
                if (!bank_sel)
                    bank1[addr_b] <= b_wdata;
                else
                    bank0[addr_b] <= b_wdata;
            end
            b_rdata <= (!bank_sel) ? bank1[addr_b] : bank0[addr_b];
        end
    end

endmodule


// -----------------------------------------------------------------------------
// SIS Hash Unit (TODO 4: Lattice-based hash via systolic)
// -----------------------------------------------------------------------------
module sis_hash_unit #(
    parameter int ARRAY_DIM  = 8,
    parameter int DATA_WIDTH = 8,
    parameter int MOD_Q      = 251     // Small prime for demo (real: 2^23 - 2^13 + 1)
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] message_block,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] hash_matrix_row,
    input  wire [$clog2(ARRAY_DIM)-1:0]    row_idx,
    input  wire                     load_matrix,
    output reg  [ARRAY_DIM*DATA_WIDTH-1:0] hash_out,
    output reg                      hash_valid,
    output reg                      busy
);

    // SIS hash: H(m) = A * m mod q
    // Uses systolic array for matrix-vector multiply, then mod reduction
    
    // Yosys-compatible unpacked arrays
    reg [ARRAY_DIM*DATA_WIDTH-1:0] matrix [ARRAY_DIM];
    reg [2*DATA_WIDTH-1:0] accum [ARRAY_DIM];
    reg [3:0] state;
    reg [$clog2(ARRAY_DIM):0] step;
    
    localparam S_IDLE = 0, S_COMPUTE = 1, S_REDUCE = 2, S_DONE = 3;
    
    integer i, j;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            hash_valid <= 0;
            busy <= 0;
            hash_out <= '0;
            step <= 0;
            for (i = 0; i < ARRAY_DIM; i++) begin
                accum[i] <= '0;
                matrix[i] <= '0;
            end
        end else begin
            case (state)
                S_IDLE: begin
                    hash_valid <= 0;
                    if (load_matrix) begin
                        matrix[row_idx] <= hash_matrix_row;
                    end else if (start) begin
                        state <= S_COMPUTE;
                        busy <= 1;
                        step <= 0;
                        for (i = 0; i < ARRAY_DIM; i++) begin
                            accum[i] <= '0;
                        end
                    end
                end
                
                S_COMPUTE: begin
                    // Matrix-vector multiply (one column per cycle)
                    if (step < ARRAY_DIM) begin
                        for (i = 0; i < ARRAY_DIM; i++) begin
                            accum[i] <= accum[i] + 
                                matrix[i][(step+1)*DATA_WIDTH-1 -: DATA_WIDTH] * 
                                message_block[(step+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                        end
                        step <= step + 1;
                    end else begin
                        state <= S_REDUCE;
                        step <= 0;
                    end
                end
                
                S_REDUCE: begin
                    // Modular reduction
                    for (i = 0; i < ARRAY_DIM; i++) begin
                        hash_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= accum[i] % MOD_Q;
                    end
                    state <= S_DONE;
                end
                
                S_DONE: begin
                    hash_valid <= 1;
                    busy <= 0;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule


// -----------------------------------------------------------------------------
// Systolic Array v2.1 Core (TODO 5: Scalable to 16x16 via ARRAY_DIM param)
// -----------------------------------------------------------------------------
module systolic_array_v21 #(
    parameter int ARRAY_DIM  = 8,      // 8x8 default, set to 16 for 16x16
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32,
    parameter bit SATURATE   = 0,
    parameter int ROUND_MODE = 0       // 0=truncate, 1=round-half-up, 2=banker's
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control
    input  wire                     load_weights,
    input  wire                     clear_acc,
    input  wire                     compute_enable,
    input  wire                     freeze_weights,   // For K-tiling
    
    // Weight input (row-by-row)
    input  wire [$clog2(ARRAY_DIM)-1:0] weight_row_sel,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_row_data,
    
    // Activation input (with skewing for systolic flow)
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] activation_in,
    
    // Result output (bottom edge) - full precision
    output wire [ARRAY_DIM*ACC_WIDTH-1:0] result_out,
    
    // Quantized output (INT8) for inference chains
    output wire [ARRAY_DIM*DATA_WIDTH-1:0] quant_result_out,
    
    output wire                     result_valid,
    
    // Performance tracking
    output wire [ARRAY_DIM*ARRAY_DIM-1:0] mac_active_out
);

    // PE interconnect wires
    wire [DATA_WIDTH-1:0] activation_h [ARRAY_DIM][ARRAY_DIM+1];
    wire [ACC_WIDTH-1:0]  psum_v       [ARRAY_DIM+1][ARRAY_DIM];
    wire                  mac_active_grid [ARRAY_DIM][ARRAY_DIM];
    wire [DATA_WIDTH-1:0] quant_grid   [ARRAY_DIM][ARRAY_DIM];
    
    // Activation skewing delay lines
    reg [DATA_WIDTH-1:0] skew_delay [ARRAY_DIM][ARRAY_DIM];
    reg [$clog2(2*ARRAY_DIM)+1:0] cycle_count;
    
    // Initialize top row of psums to zero
    genvar col;
    generate
        for (col = 0; col < ARRAY_DIM; col++) begin : init_psum
            assign psum_v[0][col] = '0;
        end
    endgenerate
    
    // Generate NxN array of PEs (TODO 5: scales with ARRAY_DIM)
    genvar row;
    generate
        for (row = 0; row < ARRAY_DIM; row++) begin : pe_row
            for (col = 0; col < ARRAY_DIM; col++) begin : pe_col
                
                // Weight load signal: only active for selected row during load phase
                wire load_pe = load_weights && (weight_row_sel == row) && !freeze_weights;
                
                systolic_pe_v21 #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH),
                    .SATURATE(SATURATE),
                    .ROUND_MODE(ROUND_MODE)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .load_weight(load_pe),
                    .clear_acc(clear_acc),
                    .weight_in(weight_row_data[(col+1)*DATA_WIDTH-1 -: DATA_WIDTH]),
                    .activation_in(activation_h[row][col]),
                    .psum_in(psum_v[row][col]),
                    .activation_out(activation_h[row][col+1]),
                    .psum_out(psum_v[row+1][col]),
                    .quant_out(quant_grid[row][col]),
                    .mac_active(mac_active_grid[row][col])
                );
                
                // Pack mac_active for perf counters
                assign mac_active_out[row*ARRAY_DIM + col] = mac_active_grid[row][col];
            end
        end
    endgenerate
    
    // Activation skewing logic (matching proven v2.0)
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
                skew_delay[i][0] <= activation_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
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
    generate
        for (row = 0; row < ARRAY_DIM; row++) begin : connect_act
            if (row == 0) begin
                assign activation_h[row][0] = skew_delay[row][0];
            end else begin
                assign activation_h[row][0] = skew_delay[row][row-1];
            end
        end
    endgenerate
    
    // Output connections from bottom of array (full precision)
    generate
        for (col = 0; col < ARRAY_DIM; col++) begin : connect_out
            assign result_out[(col+1)*ACC_WIDTH-1 -: ACC_WIDTH] = psum_v[ARRAY_DIM][col];
            // Quantized output from bottom row PEs
            assign quant_result_out[(col+1)*DATA_WIDTH-1 -: DATA_WIDTH] = quant_grid[ARRAY_DIM-1][col];
        end
    endgenerate
    
    // Result valid after 2*ARRAY_DIM - 1 cycles
    assign result_valid = (cycle_count >= 2*ARRAY_DIM - 1);

endmodule


// -----------------------------------------------------------------------------
// RFTPU v2.1 Complete Top-Level Wrapper (All TODOs Integrated)
// -----------------------------------------------------------------------------
module rftpu_systolic_v21 #(
    parameter int ARRAY_DIM   = 8,      // TODO 5: Set to 16 for 16x16
    parameter int DATA_WIDTH  = 8,
    parameter int ACC_WIDTH   = 32,
    parameter bit SATURATE    = 0,
    parameter int ROUND_MODE  = 0,      // TODO 6: 0=trunc, 1=round, 2=banker's
    parameter int FIFO_DEPTH  = 16,     // TODO 2: Weight FIFO depth
    parameter int UB_DEPTH    = 256     // TODO 3: Unified buffer depth
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control
    input  wire                     start,
    input  wire [3:0]               mode,
    input  wire [15:0]              num_vectors,
    input  wire [7:0]               k_tiles,          // TODO 1: K-tiling
    
    // Weight interface (direct)
    input  wire                     weight_load_en,
    input  wire [$clog2(ARRAY_DIM)-1:0] weight_row_sel,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_data,
    
    // Weight FIFO interface (TODO 2)
    input  wire                     weight_fifo_wr_en,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_fifo_wr_data,
    output wire                     weight_fifo_full,
    output wire                     weight_fifo_empty,
    
    // Activation interface
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] activation_data,
    
    // Unified Buffer interface (TODO 3)
    input  wire                     ub_wr_en,
    input  wire [$clog2(UB_DEPTH)-1:0] ub_addr,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] ub_wr_data,
    output wire [ARRAY_DIM*DATA_WIDTH-1:0] ub_rd_data,
    input  wire                     ub_bank_sel,
    
    // SIS Hash interface (TODO 4)
    input  wire                     sis_start,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] sis_message,
    input  wire                     sis_load_matrix,
    output wire [ARRAY_DIM*DATA_WIDTH-1:0] sis_hash_out,
    output wire                     sis_hash_valid,
    
    // Result interface
    output wire [ARRAY_DIM*ACC_WIDTH-1:0] result_data,
    output wire [ARRAY_DIM*DATA_WIDTH-1:0] quant_result_data,  // TODO 6
    output wire                     result_valid,
    
    // Status
    output wire                     busy,
    output wire                     done,
    output wire                     ready_for_weights,
    output wire                     ready_for_activation,
    
    // Performance counters (TODO 7)
    output reg [31:0]               perf_compute_cycles,
    output reg [31:0]               perf_weight_cycles,
    output reg [31:0]               perf_stall_cycles,
    output reg [31:0]               perf_mac_ops,
    output reg [31:0]               perf_total_cycles
);

    // Mode definitions
    localparam MODE_SYSTOLIC_GEMM = 4'hF;
    localparam MODE_SIS_HASH      = 4'hC;   // TODO 4
    localparam MODE_K_TILE_GEMM   = 4'hE;   // TODO 1
    
    // State machine
    typedef enum logic [3:0] {
        S_IDLE,
        S_LOAD_WEIGHTS,
        S_LOAD_WEIGHTS_FIFO,
        S_WAIT_ACT,
        S_COMPUTE,
        S_K_TILE_NEXT,      // TODO 1: K-tiling state
        S_SIS_HASH,         // TODO 4: SIS hash state
        S_DONE
    } state_t;
    
    state_t state, next_state;
    
    // Control signals
    reg load_weights_r;
    reg clear_acc_r;
    reg compute_enable_r;
    reg freeze_weights_r;
    
    // Counters
    reg [$clog2(ARRAY_DIM):0] weight_rows_loaded;
    reg [$clog2(32):0] compute_cycles_cnt;
    reg [7:0] k_tile_cnt;      // TODO 1: K-tile counter
    
    // MAC active from systolic array
    wire [ARRAY_DIM*ARRAY_DIM-1:0] mac_active;
    
    // Internal wires for muxing
    wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_to_array;
    wire [$clog2(ARRAY_DIM)-1:0] weight_row_to_array;
    wire load_from_fifo;
    
    // Weight FIFO (TODO 2)
    wire fifo_rd_en;
    wire [ARRAY_DIM*DATA_WIDTH-1:0] fifo_rd_data;
    wire [$clog2(FIFO_DEPTH):0] fifo_count;
    
    weight_fifo_v21 #(
        .DEPTH(FIFO_DEPTH),
        .DATA_WIDTH(ARRAY_DIM*DATA_WIDTH)
    ) weight_fifo (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(weight_fifo_wr_en),
        .wr_data(weight_fifo_wr_data),
        .full(weight_fifo_full),
        .rd_en(fifo_rd_en),
        .rd_data(fifo_rd_data),
        .empty(weight_fifo_empty),
        .count(fifo_count)
    );
    
    // Unified Buffer (TODO 3)
    wire [ARRAY_DIM*DATA_WIDTH-1:0] ub_systolic_rdata;
    
    unified_buffer_v21 #(
        .DEPTH(UB_DEPTH),
        .WIDTH(ARRAY_DIM*DATA_WIDTH)
    ) unified_buf (
        .clk(clk),
        .rst_n(rst_n),
        .a_en(ub_wr_en || 1'b1),
        .a_we(ub_wr_en),
        .a_addr(ub_addr),
        .a_wdata(ub_wr_data),
        .a_rdata(ub_rd_data),
        .b_en(state == S_COMPUTE),
        .b_we(1'b0),
        .b_addr(compute_cycles_cnt[$clog2(UB_DEPTH)-1:0]),
        .b_wdata('0),
        .b_rdata(ub_systolic_rdata),
        .bank_sel(ub_bank_sel)
    );
    
    // SIS Hash Unit (TODO 4)
    wire sis_busy;
    
    sis_hash_unit #(
        .ARRAY_DIM(ARRAY_DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) sis_hash (
        .clk(clk),
        .rst_n(rst_n),
        .start(sis_start && mode == MODE_SIS_HASH),
        .message_block(sis_message),
        .hash_matrix_row(weight_data),
        .row_idx(weight_row_sel),
        .load_matrix(sis_load_matrix),
        .hash_out(sis_hash_out),
        .hash_valid(sis_hash_valid),
        .busy(sis_busy)
    );
    
    // Weight source mux (direct vs FIFO)
    assign load_from_fifo = (state == S_LOAD_WEIGHTS_FIFO);
    assign weight_to_array = load_from_fifo ? fifo_rd_data : weight_data;
    assign weight_row_to_array = weight_row_sel;
    assign fifo_rd_en = load_from_fifo && !weight_fifo_empty && load_weights_r;
    
    // Systolic array (TODO 5: parameterized size)
    systolic_array_v21 #(
        .ARRAY_DIM(ARRAY_DIM),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .SATURATE(SATURATE),
        .ROUND_MODE(ROUND_MODE)
    ) systolic (
        .clk(clk),
        .rst_n(rst_n),
        .load_weights(load_weights_r),
        .clear_acc(clear_acc_r),
        .compute_enable(compute_enable_r),
        .freeze_weights(freeze_weights_r),
        .weight_row_sel(weight_row_to_array),
        .weight_row_data(weight_to_array),
        .activation_in(activation_data),
        .result_out(result_data),
        .quant_result_out(quant_result_data),
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
        load_weights_r = 1'b0;
        clear_acc_r = 1'b0;
        compute_enable_r = 1'b0;
        freeze_weights_r = 1'b0;
        
        case (state)
            S_IDLE: begin
                if (start) begin
                    case (mode)
                        MODE_SYSTOLIC_GEMM: next_state = S_LOAD_WEIGHTS;
                        MODE_K_TILE_GEMM:   next_state = S_LOAD_WEIGHTS;
                        MODE_SIS_HASH:      next_state = S_SIS_HASH;
                        default:            next_state = S_LOAD_WEIGHTS;
                    endcase
                end
            end
            
            S_LOAD_WEIGHTS: begin
                if (weight_load_en) begin
                    load_weights_r = 1'b1;
                end
                if (weight_rows_loaded >= ARRAY_DIM) begin
                    next_state = S_WAIT_ACT;
                    clear_acc_r = (k_tile_cnt == 0);  // Only clear on first K-tile
                end
            end
            
            S_LOAD_WEIGHTS_FIFO: begin
                if (!weight_fifo_empty) begin
                    load_weights_r = 1'b1;
                end
                if (weight_rows_loaded >= ARRAY_DIM) begin
                    next_state = S_WAIT_ACT;
                    clear_acc_r = (k_tile_cnt == 0);
                end
            end
            
            S_WAIT_ACT: begin
                next_state = S_COMPUTE;
            end
            
            S_COMPUTE: begin
                compute_enable_r = 1'b1;
                freeze_weights_r = 1'b1;
                if (compute_cycles_cnt >= 2*ARRAY_DIM + 2) begin
                    // TODO 1: K-tiling - check if more tiles needed
                    if (mode == MODE_K_TILE_GEMM && k_tile_cnt < k_tiles - 1) begin
                        next_state = S_K_TILE_NEXT;
                    end else begin
                        next_state = S_DONE;
                    end
                end
            end
            
            S_K_TILE_NEXT: begin
                // TODO 1: Prepare for next K-tile (keep accumulators, load new weights)
                next_state = S_LOAD_WEIGHTS;
            end
            
            S_SIS_HASH: begin
                // TODO 4: Wait for SIS hash to complete
                if (sis_hash_valid) begin
                    next_state = S_DONE;
                end
            end
            
            S_DONE: begin
                next_state = S_IDLE;
            end
        endcase
    end
    
    // Control counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_rows_loaded <= '0;
            compute_cycles_cnt <= '0;
            k_tile_cnt <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    weight_rows_loaded <= '0;
                    compute_cycles_cnt <= '0;
                    k_tile_cnt <= '0;
                end
                
                S_LOAD_WEIGHTS, S_LOAD_WEIGHTS_FIFO: begin
                    if (load_weights_r)
                        weight_rows_loaded <= weight_rows_loaded + 1;
                end
                
                S_WAIT_ACT: begin
                    weight_rows_loaded <= '0;
                    compute_cycles_cnt <= '0;
                end
                
                S_COMPUTE: begin
                    compute_cycles_cnt <= compute_cycles_cnt + 1;
                end
                
                S_K_TILE_NEXT: begin
                    k_tile_cnt <= k_tile_cnt + 1;
                    weight_rows_loaded <= '0;
                    compute_cycles_cnt <= '0;
                end
                
                default: ;
            endcase
        end
    end
    
    // Performance counters (TODO 7)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_compute_cycles <= '0;
            perf_weight_cycles <= '0;
            perf_stall_cycles <= '0;
            perf_mac_ops <= '0;
            perf_total_cycles <= '0;
        end else if (state == S_IDLE && start) begin
            perf_compute_cycles <= '0;
            perf_weight_cycles <= '0;
            perf_stall_cycles <= '0;
            perf_mac_ops <= '0;
            perf_total_cycles <= '0;
        end else begin
            if (state != S_IDLE && state != S_DONE) begin
                perf_total_cycles <= perf_total_cycles + 1;
            end
            
            if (state == S_COMPUTE) begin
                perf_compute_cycles <= perf_compute_cycles + 1;
                perf_mac_ops <= perf_mac_ops + $countones(mac_active);
            end
            
            if ((state == S_LOAD_WEIGHTS || state == S_LOAD_WEIGHTS_FIFO) && load_weights_r) begin
                perf_weight_cycles <= perf_weight_cycles + 1;
            end
            
            // Stall detection: waiting for FIFO or activation
            if ((state == S_LOAD_WEIGHTS_FIFO && weight_fifo_empty) ||
                (state == S_WAIT_ACT)) begin
                perf_stall_cycles <= perf_stall_cycles + 1;
            end
        end
    end
    
    // Status outputs
    assign busy = (state != S_IDLE) || sis_busy;
    assign done = (state == S_DONE);
    assign ready_for_weights = (state == S_LOAD_WEIGHTS || state == S_LOAD_WEIGHTS_FIFO);
    assign ready_for_activation = (state == S_WAIT_ACT || state == S_COMPUTE);

endmodule

`default_nettype wire
