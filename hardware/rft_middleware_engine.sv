// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025 Luis M. Minier
// Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
// (research/education only). Commercial rights require a separate license.
// ===============================================
// QUANTONIUMOS COMPLETE - FULL RFT IMPLEMENTATION
// Design modules only - no testbench
// ===============================================

`timescale 1ns/1ps

// ===============================================
// MODULE 1: CORDIC - Full 12-iteration implementation
// ===============================================
module cordic_cartesian_to_polar #(
    parameter WIDTH = 16,
    parameter ITERATIONS = 12
)(
    input wire clk,
    input wire reset,
    input wire start,
    input wire signed [WIDTH-1:0] x_in,
    input wire signed [WIDTH-1:0] y_in,
    output reg [WIDTH-1:0] magnitude,
    output reg signed [WIDTH-1:0] phase,
    output reg valid
);

    reg signed [WIDTH-1:0] atan_table [0:ITERATIONS-1];
    
    initial begin
        atan_table[0]  = 16'h3243;  // atan(2^0)  = 45°
        atan_table[1]  = 16'h1DAC;  // atan(2^-1) = 26.57°
        atan_table[2]  = 16'h0FAD;  // atan(2^-2)
        atan_table[3]  = 16'h07F5;
        atan_table[4]  = 16'h03FE;
        atan_table[5]  = 16'h01FF;
        atan_table[6]  = 16'h00FF;
        atan_table[7]  = 16'h007F;
        atan_table[8]  = 16'h003F;
        atan_table[9]  = 16'h001F;
        atan_table[10] = 16'h000F;
        atan_table[11] = 16'h0007;
    end

    reg signed [WIDTH-1:0] x, y, z;
    reg [3:0] iteration;
    reg [1:0] state;
    
    localparam IDLE = 2'd0;
    localparam ROTATE = 2'd1;
    localparam DONE = 2'd2;
    
    localparam signed [WIDTH-1:0] CORDIC_GAIN = 16'h9B74;  // 0.6073 (unused in this variant)

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            valid <= 0;
            magnitude <= 0;
            phase <= 0;
            iteration <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        x <= (x_in < 0) ? -x_in : x_in;
                        y <= (x_in < 0) ? -y_in : y_in;
                        z <= (x_in < 0) ? 16'h3243 : 16'h0000;
                        iteration <= 0;
                        state <= ROTATE;
                        valid <= 0;
                    end
                end
                
                ROTATE: begin
                    if (iteration < ITERATIONS) begin
                        if (y >= 0) begin
                            x <= x + (y >>> iteration);
                            y <= y - (x >>> iteration);
                            z <= z + atan_table[iteration];
                        end else begin
                            x <= x - (y >>> iteration);
                            y <= y + (x >>> iteration);
                            z <= z - atan_table[iteration];
                        end
                        iteration <= iteration + 1;
                    end else begin
                        state <= DONE;
                    end
                end
                
                DONE: begin
                    // Output raw magnitude without gain compensation
                    magnitude <= x;
                    phase <= z;
                    valid <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule

// ===============================================
// MODULE 2: COMPLEX MULTIPLIER
// ===============================================
module complex_mult #(
    parameter WIDTH = 16
)(
    input wire signed [WIDTH-1:0] a_real,
    input wire signed [WIDTH-1:0] a_imag,
    input wire signed [WIDTH-1:0] b_real,
    input wire signed [WIDTH-1:0] b_imag,
    output wire signed [WIDTH-1:0] c_real,
    output wire signed [WIDTH-1:0] c_imag
);

    wire signed [2*WIDTH-1:0] ac = a_real * b_real;
    wire signed [2*WIDTH-1:0] bd = a_imag * b_imag;
    wire signed [2*WIDTH-1:0] ad = a_real * b_imag;
    wire signed [2*WIDTH-1:0] bc = a_imag * b_real;
    
    assign c_real = (ac - bd) >>> WIDTH;
    assign c_imag = (ad + bc) >>> WIDTH;

endmodule

// ===============================================
// MODULE 3: COMPLETE 8x8 RFT KERNEL ROM
// ===============================================
module rft_kernel_rom #(
    parameter N = 8,
    parameter WIDTH = 16
)(
    input wire [2:0] k,
    input wire [2:0] n,
    output reg signed [WIDTH-1:0] kernel_real,
    output reg signed [WIDTH-1:0] kernel_imag
);

    always @(*) begin
        case ({k, n})
            // k=0: DC component (all equal)
            6'b000_000: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b000_001: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b000_010: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b000_011: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b000_100: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b000_101: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b000_110: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b000_111: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            
            // k=1: e^(-2πi*1*n/8)
            6'b001_000: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b001_001: begin kernel_real = 16'h2000; kernel_imag = 16'hE000; end
            6'b001_010: begin kernel_real = 16'h0000; kernel_imag = 16'hD2BF; end
            6'b001_011: begin kernel_real = 16'hE000; kernel_imag = 16'hE000; end
            6'b001_100: begin kernel_real = 16'hD2BF; kernel_imag = 16'h0000; end
            6'b001_101: begin kernel_real = 16'hE000; kernel_imag = 16'h2000; end
            6'b001_110: begin kernel_real = 16'h0000; kernel_imag = 16'h2D41; end
            6'b001_111: begin kernel_real = 16'h2000; kernel_imag = 16'h2000; end
            
            // k=2
            6'b010_000: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b010_001: begin kernel_real = 16'h0000; kernel_imag = 16'hD2BF; end
            6'b010_010: begin kernel_real = 16'hD2BF; kernel_imag = 16'h0000; end
            6'b010_011: begin kernel_real = 16'h0000; kernel_imag = 16'h2D41; end
            6'b010_100: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b010_101: begin kernel_real = 16'h0000; kernel_imag = 16'hD2BF; end
            6'b010_110: begin kernel_real = 16'hD2BF; kernel_imag = 16'h0000; end
            6'b010_111: begin kernel_real = 16'h0000; kernel_imag = 16'h2D41; end
            
            // k=3
            6'b011_000: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b011_001: begin kernel_real = 16'hE000; kernel_imag = 16'hE000; end
            6'b011_010: begin kernel_real = 16'h0000; kernel_imag = 16'h2D41; end
            6'b011_011: begin kernel_real = 16'h2000; kernel_imag = 16'hE000; end
            6'b011_100: begin kernel_real = 16'hD2BF; kernel_imag = 16'h0000; end
            6'b011_101: begin kernel_real = 16'h2000; kernel_imag = 16'h2000; end
            6'b011_110: begin kernel_real = 16'h0000; kernel_imag = 16'hD2BF; end
            6'b011_111: begin kernel_real = 16'hE000; kernel_imag = 16'h2000; end
            
            // k=4: Nyquist
            6'b100_000: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b100_001: begin kernel_real = 16'hD2BF; kernel_imag = 16'h0000; end
            6'b100_010: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b100_011: begin kernel_real = 16'hD2BF; kernel_imag = 16'h0000; end
            6'b100_100: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b100_101: begin kernel_real = 16'hD2BF; kernel_imag = 16'h0000; end
            6'b100_110: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b100_111: begin kernel_real = 16'hD2BF; kernel_imag = 16'h0000; end
            
            // k=5
            6'b101_000: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b101_001: begin kernel_real = 16'hE000; kernel_imag = 16'h2000; end
            6'b101_010: begin kernel_real = 16'h0000; kernel_imag = 16'hD2BF; end
            6'b101_011: begin kernel_real = 16'h2000; kernel_imag = 16'h2000; end
            6'b101_100: begin kernel_real = 16'hD2BF; kernel_imag = 16'h0000; end
            6'b101_101: begin kernel_real = 16'h2000; kernel_imag = 16'hE000; end
            6'b101_110: begin kernel_real = 16'h0000; kernel_imag = 16'h2D41; end
            6'b101_111: begin kernel_real = 16'hE000; kernel_imag = 16'hE000; end
            
            // k=6
            6'b110_000: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b110_001: begin kernel_real = 16'h0000; kernel_imag = 16'h2D41; end
            6'b110_010: begin kernel_real = 16'hD2BF; kernel_imag = 16'h0000; end
            6'b110_011: begin kernel_real = 16'h0000; kernel_imag = 16'hD2BF; end
            6'b110_100: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b110_101: begin kernel_real = 16'h0000; kernel_imag = 16'h2D41; end
            6'b110_110: begin kernel_real = 16'hD2BF; kernel_imag = 16'h0000; end
            6'b110_111: begin kernel_real = 16'h0000; kernel_imag = 16'hD2BF; end
            
            // k=7
            6'b111_000: begin kernel_real = 16'h2D41; kernel_imag = 16'h0000; end
            6'b111_001: begin kernel_real = 16'h2000; kernel_imag = 16'h2000; end
            6'b111_010: begin kernel_real = 16'h0000; kernel_imag = 16'h2D41; end
            6'b111_011: begin kernel_real = 16'hE000; kernel_imag = 16'h2000; end
            6'b111_100: begin kernel_real = 16'hD2BF; kernel_imag = 16'h0000; end
            6'b111_101: begin kernel_real = 16'hE000; kernel_imag = 16'hE000; end
            6'b111_110: begin kernel_real = 16'h0000; kernel_imag = 16'hD2BF; end
            6'b111_111: begin kernel_real = 16'h2000; kernel_imag = 16'hE000; end
            
            default: begin kernel_real = 16'h0; kernel_imag = 16'h0; end
        endcase
    end

endmodule

// ===============================================
// MODULE 4: MAIN RFT ENGINE - Complete Pipeline
// ===============================================
module rft_middleware_engine #(
    parameter N = 8,
    parameter WIDTH = 16
)(
    input wire clk,
    input wire reset,
    input wire [63:0] raw_data_in,
    input wire start,
    output wire [WIDTH-1:0] vertex_amplitudes [0:N-1],
    output wire signed [WIDTH-1:0] vertex_phases [0:N-1],
    output reg transform_valid,
    output wire [31:0] resonance_energy
);

    reg signed [WIDTH-1:0] input_real [0:N-1];
    reg signed [WIDTH-1:0] input_imag [0:N-1];
    
    reg signed [WIDTH-1:0] rft_real [0:N-1];
    reg signed [WIDTH-1:0] rft_imag [0:N-1];
    
    reg [WIDTH-1:0] vertex_amplitudes_reg [0:N-1];
    reg signed [WIDTH-1:0] vertex_phases_reg [0:N-1];
    
    wire signed [WIDTH-1:0] kernel_real, kernel_imag;
    wire signed [WIDTH-1:0] mult_real, mult_imag;
    
    reg [2:0] k_index;
    reg [2:0] n_index;
    
    reg signed [31:0] acc_real, acc_imag;
    
    reg [2:0] state;
    reg compute_cycle;
    localparam IDLE = 3'd0;
    localparam COMPUTE_RFT = 3'd1;
    localparam EXTRACT_POLAR = 3'd2;
    localparam OUTPUT = 3'd3;
    
    reg cordic_start;
    wire cordic_valid;
    wire [WIDTH-1:0] cordic_magnitude;
    wire signed [WIDTH-1:0] cordic_phase;
    
    reg [31:0] total_energy;
    
    // Instantiate modules
    rft_kernel_rom kernel_rom (
        .k(k_index),
        .n(n_index),
        .kernel_real(kernel_real),
        .kernel_imag(kernel_imag)
    );
    
    complex_mult cmult (
        .a_real(input_real[n_index]),
        .a_imag(input_imag[n_index]),
        .b_real(kernel_real),
        .b_imag(kernel_imag),
        .c_real(mult_real),
        .c_imag(mult_imag)
    );
    
    cordic_cartesian_to_polar cordic (
        .clk(clk),
        .reset(reset),
        .start(cordic_start),
        .x_in(rft_real[k_index]),
        .y_in(rft_imag[k_index]),
        .magnitude(cordic_magnitude),
        .phase(cordic_phase),
        .valid(cordic_valid)
    );
    
    // Connect outputs
    genvar gi;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : output_assignment
            assign vertex_amplitudes[gi] = vertex_amplitudes_reg[gi];
            assign vertex_phases[gi] = vertex_phases_reg[gi];
        end
    endgenerate
    
    reg cordic_processing;
    integer li;
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            transform_valid <= 0;
            k_index <= 0;
            n_index <= 0;
            total_energy <= 0;
            cordic_start <= 0;
            compute_cycle <= 0;
            cordic_processing <= 0;
            for (li = 0; li < N; li = li + 1) begin
                input_real[li] <= '0;
                input_imag[li] <= '0;
                rft_real[li] <= '0;
                rft_imag[li] <= '0;
                vertex_amplitudes_reg[li] <= '0;
                vertex_phases_reg[li] <= '0;
            end
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        // Load input as complex (real only, imag=0)
                        // Scale by 128 (shift left 7) to use upper portion of 16-bit range
                        input_real[0] <= {raw_data_in[7:0], 7'b0};
                        input_real[1] <= {raw_data_in[15:8], 7'b0};
                        input_real[2] <= {raw_data_in[23:16], 7'b0};
                        input_real[3] <= {raw_data_in[31:24], 7'b0};
                        input_real[4] <= {raw_data_in[39:32], 7'b0};
                        input_real[5] <= {raw_data_in[47:40], 7'b0};
                        input_real[6] <= {raw_data_in[55:48], 7'b0};
                        input_real[7] <= {raw_data_in[63:56], 7'b0};
                        
                        for (li = 0; li < N; li = li + 1) begin
                            input_imag[li] <= 16'h0;
                        end
                        
                        k_index <= 0;
                        n_index <= 0;
                        compute_cycle <= 0;
                        state <= COMPUTE_RFT;
                        transform_valid <= 0;
                        total_energy <= 0;
                    end
                end
                
                COMPUTE_RFT: begin
                    // RFT[k] = Σ(input[n] * kernel[k][n])
                    // Need 2 cycles per n: setup and accumulate
                    if (!compute_cycle) begin
                        // Setup cycle - wait for multiplier
                        compute_cycle <= 1;
                    end else begin
                        // Accumulate cycle
                        if (n_index == 0) begin
                            acc_real <= {{16{mult_real[15]}}, mult_real};
                            acc_imag <= {{16{mult_imag[15]}}, mult_imag};
                        end else begin
                            acc_real <= acc_real + {{16{mult_real[15]}}, mult_real};
                            acc_imag <= acc_imag + {{16{mult_imag[15]}}, mult_imag};
                        end
                        
                        if (n_index == (N-1)) begin
                            // Done with this k, save result including last multiply
                            rft_real[k_index] <= acc_real[15:0] + mult_real;
                            rft_imag[k_index] <= acc_imag[15:0] + mult_imag;
                            
                            n_index <= 0;
                            compute_cycle <= 0;
                            
                            if (k_index == (N-1)) begin
                                k_index <= 0;
                                state <= EXTRACT_POLAR;
                            end else begin
                                k_index <= k_index + 1;
                            end
                        end else begin
                            n_index <= n_index + 1;
                            compute_cycle <= 0;
                        end
                    end
                end
                
                EXTRACT_POLAR: begin
                    if (!cordic_processing) begin
                        // Start new CORDIC conversion for current k
                        cordic_start <= 1;
                        cordic_processing <= 1;
                    end else if (cordic_start) begin
                        // Clear start signal after one cycle
                        cordic_start <= 0;
                    end else if (cordic_valid) begin
                        // CORDIC completed, store result
                        vertex_amplitudes_reg[k_index] <= cordic_magnitude;
                        vertex_phases_reg[k_index] <= cordic_phase;
                        
                        total_energy <= total_energy + (cordic_magnitude * cordic_magnitude);
                        
                        cordic_processing <= 0;  // Ready for next k
                        
                        if (k_index == (N-1)) begin
                            state <= OUTPUT;
                        end else begin
                            k_index <= k_index + 1;
                        end
                    end
                end
                
                OUTPUT: begin
                    transform_valid <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end
    
    assign resonance_energy = total_energy;

endmodule
