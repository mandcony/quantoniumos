// =============================================================================
// RFTPU v2.3: High-Utilization Control Path (Target: 70-90% Utilization)
// =============================================================================
// COPYRIGHT (C) 2025 QuantoniumOS Contributors
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-NC
//
// NON-COMMERCIAL USE ONLY - See LICENSE-CLAIMS-NC.md
// This file implements algorithms subject to US Patent Application 19/169,399
// Commercial licensing inquiries: See PATENT_NOTICE.md
// =============================================================================
// KEY CHANGES FROM v2.2:
//   1. True double-buffer with prefetch (loads next while computing)
//   2. Batch mode: reuses weights across N vectors before reload
//   3. Weight FIFO with prefetch controller (never stall)
//   4. K-tiling accumulation (persistent accumulators for K > ARRAY_DIM)
//   5. Overlapped weight load / compute / drain pipeline
//
// Target: 70-90% sustained utilization vs 25-30% in v2.1/v2.2
//
// Theoretical Analysis:
//   8x8 array: 64 MACs, 19 cycle compute, 8 cycle weight load
//   v2.1: 19/(8+19) = 70% max (actual ~26% due to non-overlap)
//   v2.3: With overlap + batching: (N_batch * 19) / (8 + N_batch * 19)
//         N_batch=8: 152/160 = 95% theoretical
//         N_batch=4: 76/84 = 90% theoretical
// =============================================================================

`default_nettype none
`timescale 1ns/1ps

// -----------------------------------------------------------------------------
// Triple-Buffered Weight Bank (for true prefetch)
// -----------------------------------------------------------------------------
// Three banks: computing | ready | loading
// Allows load to start while previous tile still draining
// -----------------------------------------------------------------------------
module weight_triple_buffer #(
    parameter int ARRAY_DIM  = 8,
    parameter int DATA_WIDTH = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Write interface (load weights into loading bank)
    input  wire                     wr_en,
    input  wire [$clog2(ARRAY_DIM)-1:0] wr_row,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] wr_data,
    
    // Read interface (from computing bank to systolic array)
    input  wire [$clog2(ARRAY_DIM)-1:0] rd_row,
    output wire [ARRAY_DIM*DATA_WIDTH-1:0] rd_data,
    
    // Control
    input  wire                     advance_pipeline,  // Move banks: loading->ready->computing
    output wire                     loading_complete,  // Loading bank is full
    output wire                     ready_available,   // Ready bank has weights
    
    // Status for debug
    output wire [1:0]               active_bank,
    output wire [1:0]               loading_bank
);

    // Three weight banks
    reg [ARRAY_DIM*DATA_WIDTH-1:0] bank [3][ARRAY_DIM];
    
    // Bank rotation: 0->1->2->0
    reg [1:0] compute_bank_idx;   // Currently feeding systolic array
    reg [1:0] ready_bank_idx;     // Loaded, waiting to become active
    reg [1:0] load_bank_idx;      // Currently being loaded
    
    // Loading progress
    reg [$clog2(ARRAY_DIM):0] rows_loaded;
    reg ready_bank_valid;
    
    assign loading_complete = (rows_loaded >= ARRAY_DIM);
    assign ready_available = ready_bank_valid;
    assign active_bank = compute_bank_idx;
    assign loading_bank = load_bank_idx;
    
    // Read from compute bank
    assign rd_data = bank[compute_bank_idx][rd_row];
    
    // Next bank calculation (circular: 0->1->2->0)
    function automatic [1:0] next_bank(input [1:0] b);
        next_bank = (b == 2'd2) ? 2'd0 : b + 1;
    endfunction
    
    integer i;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_bank_idx <= 2'd0;
            ready_bank_idx   <= 2'd1;
            load_bank_idx    <= 2'd2;
            rows_loaded      <= 0;
            ready_bank_valid <= 0;
            for (i = 0; i < ARRAY_DIM; i++) begin
                bank[0][i] <= '0;
                bank[1][i] <= '0;
                bank[2][i] <= '0;
            end
        end else begin
            // Write to loading bank
            if (wr_en && !loading_complete) begin
                bank[load_bank_idx][wr_row] <= wr_data;
                if (wr_row == ARRAY_DIM - 1 || rows_loaded == ARRAY_DIM - 1)
                    rows_loaded <= ARRAY_DIM;  // Mark complete
                else
                    rows_loaded <= rows_loaded + 1;
            end
            
            // When loading completes, mark ready bank as valid
            if (loading_complete && !ready_bank_valid && rows_loaded >= ARRAY_DIM) begin
                // Move loading->ready (swap indices)
                ready_bank_idx <= load_bank_idx;
                load_bank_idx <= ready_bank_idx;  // Old ready becomes new loading
                ready_bank_valid <= 1;
                rows_loaded <= 0;
            end
            
            // Advance pipeline: ready->compute
            if (advance_pipeline && ready_bank_valid) begin
                compute_bank_idx <= ready_bank_idx;
                ready_bank_idx <= compute_bank_idx;  // Old compute becomes ready slot
                ready_bank_valid <= 0;  // Need to load new ready
            end
        end
    end

endmodule


// -----------------------------------------------------------------------------
// High-Throughput Weight FIFO with Prefetch
// -----------------------------------------------------------------------------
// Deep FIFO with prefetch logic to ensure weights always available
// -----------------------------------------------------------------------------
module weight_fifo_prefetch #(
    parameter int DEPTH      = 32,
    parameter int DATA_WIDTH = 64,
    parameter int PREFETCH_THRESHOLD = 8   // Start prefetch when this many remain
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Write interface (from DMA/host)
    input  wire                     wr_en,
    input  wire [DATA_WIDTH-1:0]    wr_data,
    output wire                     full,
    
    // Read interface (to weight buffer)
    input  wire                     rd_en,
    output wire [DATA_WIDTH-1:0]    rd_data,
    output wire                     empty,
    
    // Status
    output wire [$clog2(DEPTH)+1:0] count,
    output wire                     prefetch_request,  // Signal to DMA to send more
    output wire                     critically_low     // About to stall
);

    reg [DATA_WIDTH-1:0] mem [DEPTH];
    reg [$clog2(DEPTH):0] wr_ptr, rd_ptr;
    reg [$clog2(DEPTH)+1:0] cnt;
    
    assign full  = (cnt >= DEPTH);
    assign empty = (cnt == 0);
    assign count = cnt;
    assign rd_data = mem[rd_ptr[$clog2(DEPTH)-1:0]];
    
    // Prefetch request when below threshold
    assign prefetch_request = (cnt < PREFETCH_THRESHOLD) && !full;
    assign critically_low = (cnt < 2);  // Less than 2 entries - about to stall
    
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
// Persistent Accumulator Bank for K-Tiling
// -----------------------------------------------------------------------------
// Holds partial sums across K-tiles without draining
// Supports multiple output vectors for batch processing
// -----------------------------------------------------------------------------
module ktile_accumulator_bank #(
    parameter int ARRAY_DIM  = 8,
    parameter int ACC_WIDTH  = 32,
    parameter int NUM_ACCUM  = 8       // Number of accumulator sets (batch depth)
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Accumulator selection
    input  wire [$clog2(NUM_ACCUM)-1:0] accum_sel,
    
    // Input from systolic array bottom
    input  wire [ARRAY_DIM*ACC_WIDTH-1:0] psum_in,
    input  wire                     psum_valid,
    
    // Control
    input  wire                     clear_accum,      // Clear selected accumulator
    input  wire                     clear_all,        // Clear all accumulators
    input  wire                     accumulate,       // Add to selected accumulator
    
    // Output (selected accumulator)
    output wire [ARRAY_DIM*ACC_WIDTH-1:0] accum_out,
    
    // Status
    output wire [NUM_ACCUM-1:0]     accum_nonzero    // Which accumulators have data
);

    // Accumulator storage
    reg [ARRAY_DIM*ACC_WIDTH-1:0] accum [NUM_ACCUM];
    reg [NUM_ACCUM-1:0] has_data;
    
    assign accum_out = accum[accum_sel];
    assign accum_nonzero = has_data;
    
    integer i;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            has_data <= '0;
            for (i = 0; i < NUM_ACCUM; i++) begin
                accum[i] <= '0;
            end
        end else begin
            if (clear_all) begin
                has_data <= '0;
                for (i = 0; i < NUM_ACCUM; i++) begin
                    accum[i] <= '0;
                end
            end else if (clear_accum) begin
                accum[accum_sel] <= '0;
                has_data[accum_sel] <= 0;
            end else if (accumulate && psum_valid) begin
                // Signed accumulation for K-tiling
                for (i = 0; i < ARRAY_DIM; i++) begin
                    accum[accum_sel][(i+1)*ACC_WIDTH-1 -: ACC_WIDTH] <= 
                        $signed(accum[accum_sel][(i+1)*ACC_WIDTH-1 -: ACC_WIDTH]) +
                        $signed(psum_in[(i+1)*ACC_WIDTH-1 -: ACC_WIDTH]);
                end
                has_data[accum_sel] <= 1;
            end
        end
    end

endmodule


// -----------------------------------------------------------------------------
// Processing Element v2.3 (Same as v2.1 but with batch enable)
// -----------------------------------------------------------------------------
module systolic_pe_v23 #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
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

    reg [DATA_WIDTH-1:0] weight_reg;
    
    wire signed [2*DATA_WIDTH-1:0] product;
    assign product = $signed(weight_reg) * $signed(activation_in);
    
    assign mac_active = enable && (activation_in != '0);
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg     <= '0;
            activation_out <= '0;
            psum_out       <= '0;
        end else begin
            if (load_weight) begin
                weight_reg <= weight_in;
            end
            
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
// Systolic Array v2.3 Core (with batch-aware skewing)
// -----------------------------------------------------------------------------
module systolic_array_v23 #(
    parameter int ARRAY_DIM  = 8,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control
    input  wire                     load_weights,
    input  wire                     clear_acc,
    input  wire                     compute_enable,
    
    // Weight input (row-by-row loading)
    input  wire [$clog2(ARRAY_DIM)-1:0] weight_row_sel,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_row_data,
    
    // Activation input
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] activation_in,
    
    // Results
    output wire [ARRAY_DIM*ACC_WIDTH-1:0] result_out,
    output wire                     result_valid,
    
    // Performance
    output wire [ARRAY_DIM*ARRAY_DIM-1:0] mac_active_out,
    output wire [$clog2(3*ARRAY_DIM):0]   cycle_count_out
);

    // PE interconnect
    wire [DATA_WIDTH-1:0] activation_h [ARRAY_DIM][ARRAY_DIM+1];
    wire [ACC_WIDTH-1:0]  psum_v       [ARRAY_DIM+1][ARRAY_DIM];
    wire                  mac_active_grid [ARRAY_DIM][ARRAY_DIM];
    
    // Skewing
    reg [DATA_WIDTH-1:0] skew_delay [ARRAY_DIM][ARRAY_DIM];
    reg [$clog2(3*ARRAY_DIM):0] cycle_count;
    
    assign cycle_count_out = cycle_count;
    
    // Top psum = 0
    genvar col;
    generate
        for (col = 0; col < ARRAY_DIM; col++) begin : init_psum
            assign psum_v[0][col] = '0;
        end
    endgenerate
    
    // PE array
    genvar row;
    generate
        for (row = 0; row < ARRAY_DIM; row++) begin : pe_row
            for (col = 0; col < ARRAY_DIM; col++) begin : pe_col
                wire load_pe = load_weights && (weight_row_sel == row);
                
                systolic_pe_v23 #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .load_weight(load_pe),
                    .clear_acc(clear_acc && (cycle_count == 0)),
                    .enable(compute_enable),
                    .weight_in(weight_row_data[(col+1)*DATA_WIDTH-1 -: DATA_WIDTH]),
                    .activation_in(activation_h[row][col]),
                    .psum_in(psum_v[row][col]),
                    .activation_out(activation_h[row][col+1]),
                    .psum_out(psum_v[row+1][col])
                );
                
                assign mac_active_grid[row][col] = (activation_h[row][col] != '0) && compute_enable;
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
    
    assign result_valid = (cycle_count >= 2*ARRAY_DIM - 1);

endmodule


// -----------------------------------------------------------------------------
// RFTPU v2.3 Top-Level: High-Utilization Control Path
// -----------------------------------------------------------------------------
// State machine that achieves 70-90% utilization through:
//   1. Overlapped weight loading / compute / drain
//   2. Batch processing (multiple vectors per weight load)
//   3. K-tiling with persistent accumulators
//   4. Prefetch controller for continuous operation
// -----------------------------------------------------------------------------
module rftpu_systolic_v23 #(
    parameter int ARRAY_DIM     = 8,
    parameter int DATA_WIDTH    = 8,
    parameter int ACC_WIDTH     = 32,
    parameter int BATCH_SIZE    = 8,       // Vectors per weight tile (key for utilization!)
    parameter int K_TILES_MAX   = 16,      // Max K-tiles for accumulation
    parameter int WEIGHT_FIFO_DEPTH = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Command interface
    input  wire                     start,
    input  wire [3:0]               mode,
    input  wire [15:0]              M_dim,           // Output rows
    input  wire [15:0]              N_dim,           // Output cols (batch)
    input  wire [15:0]              K_dim,           // Inner dimension
    
    // Weight FIFO write interface (from DMA)
    input  wire                     weight_fifo_wr,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_fifo_data,
    
    // Activation stream interface
    input  wire                     act_valid,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] act_data,
    output wire                     act_ready,       // Backpressure
    
    // Result stream interface  
    output wire [ARRAY_DIM*ACC_WIDTH-1:0] result_data,
    output wire                     result_valid,
    input  wire                     result_ready,    // Downstream ready
    
    // Status
    output wire                     busy,
    output wire                     done,
    output wire                     weight_fifo_full,
    output wire                     prefetch_request,
    
    // Performance counters (TODO 7)
    output reg [31:0]               perf_total_cycles,
    output reg [31:0]               perf_compute_cycles,
    output reg [31:0]               perf_weight_stall_cycles,
    output reg [31:0]               perf_output_stall_cycles,
    output reg [31:0]               perf_mac_ops,
    output reg [31:0]               perf_vectors_processed,
    
    // Utilization output (real-time)
    output wire [7:0]               utilization_pct  // 0-100%
);

    localparam MODE_GEMM = 4'hF;
    localparam COMPUTE_LATENCY = 2*ARRAY_DIM - 1;  // Cycles for result to emerge
    
    // ==========================================================================
    // State Machine: Overlapped Load/Compute/Drain Pipeline
    // ==========================================================================
    typedef enum logic [3:0] {
        S_IDLE,
        S_LOAD_FIRST_WEIGHTS,    // Initial weight load (no overlap yet)
        S_COMPUTE_BATCH,         // Main compute loop (overlap with next load)
        S_DRAIN_PIPELINE,        // Drain systolic array
        S_ACCUMULATE_KTILE,      // K-tile accumulation
        S_OUTPUT_RESULTS,        // Stream out results
        S_DONE
    } state_t;
    
    state_t state, next_state;
    
    // ==========================================================================
    // Weight FIFO with Prefetch
    // ==========================================================================
    wire weight_fifo_empty;
    wire weight_fifo_rd;
    wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_fifo_out;
    wire [$clog2(WEIGHT_FIFO_DEPTH)+1:0] weight_fifo_count;
    wire critically_low;
    
    weight_fifo_prefetch #(
        .DEPTH(WEIGHT_FIFO_DEPTH),
        .DATA_WIDTH(ARRAY_DIM*DATA_WIDTH),
        .PREFETCH_THRESHOLD(ARRAY_DIM*2)
    ) weight_fifo (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(weight_fifo_wr),
        .wr_data(weight_fifo_data),
        .full(weight_fifo_full),
        .rd_en(weight_fifo_rd),
        .rd_data(weight_fifo_out),
        .empty(weight_fifo_empty),
        .count(weight_fifo_count),
        .prefetch_request(prefetch_request),
        .critically_low(critically_low)
    );
    
    // ==========================================================================
    // Triple Buffer for Weights
    // ==========================================================================
    reg weight_buf_wr;
    reg [$clog2(ARRAY_DIM)-1:0] weight_buf_row;
    wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_buf_rd_data;
    reg advance_weight_pipeline;
    wire weight_loading_complete;
    wire weight_ready_available;
    
    weight_triple_buffer #(
        .ARRAY_DIM(ARRAY_DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) weight_buf (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(weight_buf_wr),
        .wr_row(weight_buf_row),
        .wr_data(weight_fifo_out),
        .rd_row(weight_load_row),
        .rd_data(weight_buf_rd_data),
        .advance_pipeline(advance_weight_pipeline),
        .loading_complete(weight_loading_complete),
        .ready_available(weight_ready_available),
        .active_bank(),
        .loading_bank()
    );
    
    // Weight loading state machine (runs in parallel with compute)
    reg [$clog2(ARRAY_DIM):0] weight_load_row;
    reg weight_loading;
    
    assign weight_fifo_rd = weight_loading && !weight_fifo_empty && (weight_load_row < ARRAY_DIM);
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_load_row <= 0;
            weight_loading <= 0;
            weight_buf_wr <= 0;
            weight_buf_row <= 0;
        end else begin
            weight_buf_wr <= 0;
            
            if (state == S_IDLE && start) begin
                weight_loading <= 1;
                weight_load_row <= 0;
            end else if (weight_loading && !weight_fifo_empty) begin
                if (weight_load_row < ARRAY_DIM) begin
                    weight_buf_wr <= 1;
                    weight_buf_row <= weight_load_row[$clog2(ARRAY_DIM)-1:0];
                    weight_load_row <= weight_load_row + 1;
                end else begin
                    // Finished loading one tile, reset for next if needed
                    weight_load_row <= 0;
                end
            end
            
            // Stop loading when done with all tiles
            if (state == S_DONE) begin
                weight_loading <= 0;
            end
        end
    end
    
    // ==========================================================================
    // K-Tile Accumulator Bank
    // ==========================================================================
    wire [ARRAY_DIM*ACC_WIDTH-1:0] ktile_accum_out;
    reg ktile_clear_all;
    reg ktile_accumulate;
    reg [$clog2(BATCH_SIZE)-1:0] ktile_accum_sel;
    
    ktile_accumulator_bank #(
        .ARRAY_DIM(ARRAY_DIM),
        .ACC_WIDTH(ACC_WIDTH),
        .NUM_ACCUM(BATCH_SIZE)
    ) ktile_accum (
        .clk(clk),
        .rst_n(rst_n),
        .accum_sel(ktile_accum_sel),
        .psum_in(systolic_result),
        .psum_valid(systolic_result_valid),
        .clear_accum(1'b0),
        .clear_all(ktile_clear_all),
        .accumulate(ktile_accumulate),
        .accum_out(ktile_accum_out),
        .accum_nonzero()
    );
    
    // ==========================================================================
    // Systolic Array Core
    // ==========================================================================
    reg systolic_load_weights;
    reg systolic_clear_acc;
    reg systolic_compute_enable;
    wire [ARRAY_DIM*ACC_WIDTH-1:0] systolic_result;
    wire systolic_result_valid;
    wire [ARRAY_DIM*ARRAY_DIM-1:0] mac_active;
    wire [$clog2(3*ARRAY_DIM):0] systolic_cycle_count;
    
    systolic_array_v23 #(
        .ARRAY_DIM(ARRAY_DIM),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) systolic (
        .clk(clk),
        .rst_n(rst_n),
        .load_weights(systolic_load_weights),
        .clear_acc(systolic_clear_acc),
        .compute_enable(systolic_compute_enable),
        .weight_row_sel(weight_buf_row),
        .weight_row_data(weight_buf_rd_data),
        .activation_in(act_data),
        .result_out(systolic_result),
        .result_valid(systolic_result_valid),
        .mac_active_out(mac_active),
        .cycle_count_out(systolic_cycle_count)
    );
    
    // ==========================================================================
    // Batch Processing Counters
    // ==========================================================================
    reg [15:0] batch_vector_cnt;     // Vectors in current batch
    reg [15:0] total_vector_cnt;     // Total vectors processed
    reg [15:0] ktile_cnt;            // K-tiles processed
    reg [15:0] k_tiles_needed;       // Total K-tiles = ceil(K_dim / ARRAY_DIM)
    
    // ==========================================================================
    // Main State Machine
    // ==========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= next_state;
    end
    
    reg can_accept_activation;
    assign act_ready = can_accept_activation && (state == S_COMPUTE_BATCH);
    
    // Result output (either direct from systolic or from K-tile accumulator)
    assign result_data = (ktile_cnt > 0) ? ktile_accum_out : systolic_result;
    assign result_valid = (state == S_OUTPUT_RESULTS) && systolic_result_valid;
    
    always_comb begin
        next_state = state;
        systolic_load_weights = 0;
        systolic_clear_acc = 0;
        systolic_compute_enable = 0;
        advance_weight_pipeline = 0;
        ktile_clear_all = 0;
        ktile_accumulate = 0;
        can_accept_activation = 0;
        
        case (state)
            S_IDLE: begin
                if (start && mode == MODE_GEMM) begin
                    next_state = S_LOAD_FIRST_WEIGHTS;
                    ktile_clear_all = 1;
                end
            end
            
            S_LOAD_FIRST_WEIGHTS: begin
                // Wait for first weight tile to be ready
                if (weight_ready_available) begin
                    advance_weight_pipeline = 1;  // Make ready weights active
                    systolic_load_weights = 1;    // Load into PEs
                    next_state = S_COMPUTE_BATCH;
                end
            end
            
            S_COMPUTE_BATCH: begin
                can_accept_activation = 1;
                
                if (act_valid && act_ready) begin
                    systolic_compute_enable = 1;
                    systolic_clear_acc = (batch_vector_cnt == 0);  // Clear on first vector
                end
                
                // Check if batch complete
                if (batch_vector_cnt >= BATCH_SIZE - 1 && act_valid) begin
                    next_state = S_DRAIN_PIPELINE;
                end
                
                // Check if all vectors done
                if (total_vector_cnt >= N_dim - 1 && act_valid) begin
                    next_state = S_DRAIN_PIPELINE;
                end
            end
            
            S_DRAIN_PIPELINE: begin
                // Let pipeline drain, continue computing
                systolic_compute_enable = 1;
                
                if (systolic_result_valid) begin
                    if (ktile_cnt < k_tiles_needed - 1) begin
                        // More K-tiles needed - accumulate and continue
                        ktile_accumulate = 1;
                        next_state = S_ACCUMULATE_KTILE;
                    end else begin
                        // All K-tiles done - output results
                        next_state = S_OUTPUT_RESULTS;
                    end
                end
            end
            
            S_ACCUMULATE_KTILE: begin
                ktile_accumulate = 1;
                
                // Load next weight tile while accumulating
                if (weight_ready_available) begin
                    advance_weight_pipeline = 1;
                    systolic_load_weights = 1;
                    next_state = S_COMPUTE_BATCH;
                end
            end
            
            S_OUTPUT_RESULTS: begin
                // Stream out results
                if (result_ready && systolic_result_valid) begin
                    if (total_vector_cnt >= N_dim) begin
                        next_state = S_DONE;
                    end else begin
                        // More batches to process
                        if (weight_ready_available) begin
                            advance_weight_pipeline = 1;
                            systolic_load_weights = 1;
                            next_state = S_COMPUTE_BATCH;
                        end else begin
                            next_state = S_LOAD_FIRST_WEIGHTS;  // Wait for weights
                        end
                    end
                end
            end
            
            S_DONE: begin
                next_state = S_IDLE;
            end
        endcase
    end
    
    // ==========================================================================
    // Counters
    // ==========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            batch_vector_cnt <= 0;
            total_vector_cnt <= 0;
            ktile_cnt <= 0;
            k_tiles_needed <= 1;
        end else begin
            if (state == S_IDLE && start) begin
                batch_vector_cnt <= 0;
                total_vector_cnt <= 0;
                ktile_cnt <= 0;
                // Calculate K-tiles: ceil(K_dim / ARRAY_DIM)
                k_tiles_needed <= (K_dim + ARRAY_DIM - 1) / ARRAY_DIM;
            end else begin
                if (state == S_COMPUTE_BATCH && act_valid && act_ready) begin
                    batch_vector_cnt <= batch_vector_cnt + 1;
                    total_vector_cnt <= total_vector_cnt + 1;
                end
                
                if (state == S_ACCUMULATE_KTILE && next_state == S_COMPUTE_BATCH) begin
                    ktile_cnt <= ktile_cnt + 1;
                    batch_vector_cnt <= 0;  // Reset batch counter
                end
                
                if (state == S_OUTPUT_RESULTS && next_state == S_COMPUTE_BATCH) begin
                    ktile_cnt <= 0;  // Reset K-tile counter for next output
                    batch_vector_cnt <= 0;
                end
            end
        end
    end
    
    // ==========================================================================
    // Performance Counters (TODO 7)
    // ==========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_total_cycles <= 0;
            perf_compute_cycles <= 0;
            perf_weight_stall_cycles <= 0;
            perf_output_stall_cycles <= 0;
            perf_mac_ops <= 0;
            perf_vectors_processed <= 0;
        end else if (state == S_IDLE && start) begin
            perf_total_cycles <= 0;
            perf_compute_cycles <= 0;
            perf_weight_stall_cycles <= 0;
            perf_output_stall_cycles <= 0;
            perf_mac_ops <= 0;
            perf_vectors_processed <= 0;
        end else if (state != S_IDLE && state != S_DONE) begin
            perf_total_cycles <= perf_total_cycles + 1;
            
            if (systolic_compute_enable) begin
                perf_compute_cycles <= perf_compute_cycles + 1;
                perf_mac_ops <= perf_mac_ops + $countones(mac_active);
            end
            
            // Stall tracking
            if (state == S_LOAD_FIRST_WEIGHTS || 
                (state == S_ACCUMULATE_KTILE && !weight_ready_available)) begin
                perf_weight_stall_cycles <= perf_weight_stall_cycles + 1;
            end
            
            if (state == S_OUTPUT_RESULTS && !result_ready) begin
                perf_output_stall_cycles <= perf_output_stall_cycles + 1;
            end
            
            if (state == S_COMPUTE_BATCH && act_valid && act_ready) begin
                perf_vectors_processed <= perf_vectors_processed + 1;
            end
        end
    end
    
    // ==========================================================================
    // Real-Time Utilization Calculation
    // ==========================================================================
    // Utilization = compute_cycles / total_cycles * 100
    // Using fixed-point: (compute * 100) >> log2(total) approximation
    wire [31:0] util_numer = perf_compute_cycles * 100;
    assign utilization_pct = (perf_total_cycles > 0) ? 
                             (util_numer / perf_total_cycles) : 8'd0;
    
    // Status outputs
    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);

endmodule


// =============================================================================
// Testbench for v2.3 High-Utilization Verification
// =============================================================================
`ifdef SIMULATION
module rftpu_v23_tb;
    localparam ARRAY_DIM = 8;
    localparam DATA_WIDTH = 8;
    localparam ACC_WIDTH = 32;
    localparam BATCH_SIZE = 8;
    
    reg clk, rst_n;
    reg start;
    reg [3:0] mode;
    reg [15:0] M_dim, N_dim, K_dim;
    
    reg weight_fifo_wr;
    reg [ARRAY_DIM*DATA_WIDTH-1:0] weight_fifo_data;
    
    reg act_valid;
    reg [ARRAY_DIM*DATA_WIDTH-1:0] act_data;
    wire act_ready;
    
    wire [ARRAY_DIM*ACC_WIDTH-1:0] result_data;
    wire result_valid;
    reg result_ready;
    
    wire busy, done, weight_fifo_full, prefetch_request;
    wire [31:0] perf_total, perf_compute, perf_weight_stall, perf_output_stall;
    wire [31:0] perf_mac_ops, perf_vectors;
    wire [7:0] utilization;
    
    rftpu_systolic_v23 #(
        .ARRAY_DIM(ARRAY_DIM),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .BATCH_SIZE(BATCH_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .mode(mode),
        .M_dim(M_dim),
        .N_dim(N_dim),
        .K_dim(K_dim),
        .weight_fifo_wr(weight_fifo_wr),
        .weight_fifo_data(weight_fifo_data),
        .act_valid(act_valid),
        .act_data(act_data),
        .act_ready(act_ready),
        .result_data(result_data),
        .result_valid(result_valid),
        .result_ready(result_ready),
        .busy(busy),
        .done(done),
        .weight_fifo_full(weight_fifo_full),
        .prefetch_request(prefetch_request),
        .perf_total_cycles(perf_total),
        .perf_compute_cycles(perf_compute),
        .perf_weight_stall_cycles(perf_weight_stall),
        .perf_output_stall_cycles(perf_output_stall),
        .perf_mac_ops(perf_mac_ops),
        .perf_vectors_processed(perf_vectors),
        .utilization_pct(utilization)
    );
    
    initial clk = 0;
    always #5 clk = ~clk;
    
    integer i, test_pass;
    
    initial begin
        $display("=== RFTPU v2.3 High-Utilization Test ===");
        
        // Initialize
        rst_n = 0;
        start = 0;
        mode = 4'hF;
        M_dim = ARRAY_DIM;
        N_dim = 64;  // 64 vectors = 8 batches
        K_dim = ARRAY_DIM;
        weight_fifo_wr = 0;
        weight_fifo_data = '0;
        act_valid = 0;
        act_data = '0;
        result_ready = 1;
        test_pass = 1;
        
        #20 rst_n = 1;
        #10;
        
        // Test 1: Pre-load weights into FIFO
        $display("\n[TEST 1] Pre-loading weights into FIFO...");
        for (i = 0; i < ARRAY_DIM; i++) begin
            @(posedge clk);
            weight_fifo_wr = 1;
            // Simple weights: row i has value i+1 in all columns
            weight_fifo_data = {ARRAY_DIM{8'(i+1)}};
        end
        @(posedge clk);
        weight_fifo_wr = 0;
        $display("  Weights loaded: %0d rows", ARRAY_DIM);
        
        // Test 2: Start GEMM operation
        $display("\n[TEST 2] Starting GEMM: M=%0d, N=%0d, K=%0d", M_dim, N_dim, K_dim);
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        // Test 3: Stream activations (simulating batch processing)
        $display("\n[TEST 3] Streaming %0d activation vectors...", N_dim);
        for (i = 0; i < N_dim + 20; i++) begin  // Extra cycles for drain
            @(posedge clk);
            if (act_ready && i < N_dim) begin
                act_valid = 1;
                act_data = {ARRAY_DIM{8'(i & 8'hFF)}};  // Test pattern
            end else begin
                act_valid = 0;
            end
            
            // Periodic status
            if (i % 16 == 0 && busy) begin
                $display("  Cycle %0d: util=%0d%%, compute=%0d, total=%0d", 
                         i, utilization, perf_compute, perf_total);
            end
        end
        
        // Wait for completion
        while (!done) @(posedge clk);
        
        // Report results
        $display("\n========================================");
        $display("RFTPU v2.3 Performance Results:");
        $display("========================================");
        $display("Total Cycles:        %0d", perf_total);
        $display("Compute Cycles:      %0d", perf_compute);
        $display("Weight Stall Cycles: %0d", perf_weight_stall);
        $display("Output Stall Cycles: %0d", perf_output_stall);
        $display("MAC Operations:      %0d", perf_mac_ops);
        $display("Vectors Processed:   %0d", perf_vectors);
        $display("Utilization:         %0d%%", utilization);
        $display("----------------------------------------");
        
        // Calculate theoretical
        // 64 vectors * 64 MACs = 4096 MAC ops
        // Ideal: 64 vectors * 19 cycles/vector = 1216 cycles (without batch overlap)
        // With batching (8 vectors/batch): 8 batches * (8 + 8*19 - overlap)
        
        if (utilization >= 50) begin
            $display("✓ PASS: Utilization %0d%% >= 50%% target", utilization);
        end else begin
            $display("✗ FAIL: Utilization %0d%% < 50%% target", utilization);
            test_pass = 0;
        end
        
        if (test_pass)
            $display("\n=== ALL TESTS PASSED ===\n");
        else
            $display("\n=== TESTS FAILED ===\n");
        
        #100 $finish;
    end
    
endmodule
`endif

`default_nettype wire
