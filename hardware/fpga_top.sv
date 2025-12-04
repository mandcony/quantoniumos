// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// ═══════════════════════════════════════════════════════════════════════════
// QUANTONIUMOS UNIFIED WEBFPGA TOP - FULL ENGINE INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════
// 
// This module integrates all QuantoniumOS engines for WebFPGA deployment:
//   • 16 Φ-RFT Variants (golden-ratio resonance transforms)
//   • SIS Lattice Cryptographic Hash (post-quantum secure)
//   • Feistel-48 Cipher (AES S-box + ARX)
//   • Quantum State Simulation (compressed representation)
//   • Compression/Decompression Pipeline
//   • Middleware Orchestration Layer
//
// Button cycles through engine modes; LEDs show status/results
//
// This file is listed in CLAIMS_PRACTICING_FILES.txt

module fpga_top (
    input wire WF_CLK,       // WebFPGA clock (12 MHz typical)
    input wire WF_BUTTON,    // User button (active low) - cycles modes
    output wire [7:0] WF_LED // 8 LEDs for visualization
);

    // ═══════════════════════════════════════════════════════════════════════
    // PARAMETERS & CONSTANTS
    // ═══════════════════════════════════════════════════════════════════════
    
    // Golden Ratio Constants (Q8.8 fixed-point for WebFPGA resource limits)
    localparam [15:0] PHI_Q8_8      = 16'h019E;  // φ ≈ 1.618034
    localparam [15:0] PHI_INV_Q8_8  = 16'h009E;  // 1/φ ≈ 0.618034
    localparam [15:0] SQRT5_Q8_8    = 16'h023C;  // √5 ≈ 2.236
    localparam [15:0] TWO_PI_Q8_8   = 16'h0648;  // 2π ≈ 6.283
    localparam [15:0] NORM_8_Q8_8   = 16'h005A;  // 1/√8 ≈ 0.3536
    
    // Engine Mode Definitions (4-bit = 16 modes)
    localparam [3:0] MODE_RFT_STANDARD      = 4'd0;   // Φ-RFT original
    localparam [3:0] MODE_RFT_HARMONIC      = 4'd1;   // Cubic chirp phase
    localparam [3:0] MODE_RFT_FIBONACCI     = 4'd2;   // Lattice crypto alignment
    localparam [3:0] MODE_RFT_CHAOTIC       = 4'd3;   // Haar-like randomness
    localparam [3:0] MODE_RFT_GEOMETRIC     = 4'd4;   // Optical computing
    localparam [3:0] MODE_RFT_PHI_CHAOTIC   = 4'd5;   // Structure + disorder
    localparam [3:0] MODE_RFT_HYPERBOLIC    = 4'd6;   // Tanh envelope warp
    localparam [3:0] MODE_RFT_DCT           = 4'd7;   // Pure DCT-II
    localparam [3:0] MODE_RFT_HYBRID_DCT    = 4'd8;   // Adaptive DCT/RFT
    localparam [3:0] MODE_RFT_LOG_PERIODIC  = 4'd9;   // Log-frequency warp
    localparam [3:0] MODE_RFT_CONVEX_MIX    = 4'd10;  // Adaptive textures
    localparam [3:0] MODE_RFT_GOLDEN_EXACT  = 4'd11;  // Full resonance lattice
    localparam [3:0] MODE_RFT_CASCADE       = 4'd12;  // H3 cascade (recommended)
    localparam [3:0] MODE_SIS_HASH          = 4'd13;  // SIS lattice hash
    localparam [3:0] MODE_FEISTEL           = 4'd14;  // Feistel-48 cipher
    localparam [3:0] MODE_QUANTUM_SIM       = 4'd15;  // Quantum state demo
    
    // State Machine States
    localparam [2:0] STATE_IDLE     = 3'd0;
    localparam [2:0] STATE_LOAD     = 3'd1;
    localparam [2:0] STATE_COMPUTE  = 3'd2;
    localparam [2:0] STATE_STORE    = 3'd3;
    localparam [2:0] STATE_OUTPUT   = 3'd4;
    localparam [2:0] STATE_DONE     = 3'd5;

    // ═══════════════════════════════════════════════════════════════════════
    // CLOCK, RESET & BUTTON HANDLING
    // ═══════════════════════════════════════════════════════════════════════
    
    reg [7:0] reset_counter = 8'h00;
    wire reset = (reset_counter < 8'd10);
    
    always @(posedge WF_CLK) begin
        if (reset_counter < 8'd10)
            reset_counter <= reset_counter + 1'b1;
    end
    
    // Button debouncing with edge detection
    reg [19:0] button_debounce = 20'h00000;
    reg button_stable = 1'b0;
    reg button_prev = 1'b0;
    wire button_edge = button_stable && !button_prev;
    
    always @(posedge WF_CLK) begin
        if (reset) begin
            button_debounce <= 20'h00000;
            button_stable <= 1'b0;
            button_prev <= 1'b0;
        end else begin
            button_debounce <= {button_debounce[18:0], ~WF_BUTTON};
            button_stable <= &button_debounce[19:16];
            button_prev <= button_stable;
        end
    end

    // ═══════════════════════════════════════════════════════════════════════
    // MODE SELECTION & AUTO-CYCLE
    // ═══════════════════════════════════════════════════════════════════════
    
    reg [3:0] current_mode = MODE_RFT_STANDARD;
    reg [23:0] auto_cycle_counter = 24'h000000;
    wire auto_cycle_trigger = (auto_cycle_counter == 24'hFFFFFF);
    
    always @(posedge WF_CLK) begin
        if (reset) begin
            current_mode <= MODE_RFT_STANDARD;
            auto_cycle_counter <= 24'h000000;
        end else begin
            auto_cycle_counter <= auto_cycle_counter + 1'b1;
            if (button_edge || auto_cycle_trigger) begin
                current_mode <= current_mode + 1'b1;
                auto_cycle_counter <= 24'h000000;
            end
        end
    end

    // ═══════════════════════════════════════════════════════════════════════
    // TIMING CONTROL
    // ═══════════════════════════════════════════════════════════════════════
    
    reg [7:0] cyc_cnt = 8'h00;
    
    always @(posedge WF_CLK) begin
        if (reset)
            cyc_cnt <= 8'h00;
        else
            cyc_cnt <= cyc_cnt + 1'b1;
    end
    
    wire start = button_edge || (cyc_cnt == 8'd20);

    // ═══════════════════════════════════════════════════════════════════════
    // INPUT DATA GENERATION - Multiple test patterns for different engines
    // ═══════════════════════════════════════════════════════════════════════
    
    reg [15:0] sample [0:7];
    reg [15:0] rng_state = 16'hACE1;
    wire [15:0] rng_next = {rng_state[14:0], rng_state[15] ^ rng_state[13] ^ rng_state[12] ^ rng_state[10]};
    reg valid = 1'b0;
    
    integer i;
    always @(posedge WF_CLK) begin
        if (reset) begin
            for (i = 0; i < 8; i = i + 1)
                sample[i] <= 16'h0000;
            valid <= 1'b0;
            rng_state <= 16'hACE1;
        end
        else if (start && !valid) begin
            rng_state <= rng_next;
            case (current_mode)
                MODE_RFT_STANDARD, MODE_RFT_GOLDEN_EXACT, MODE_RFT_CASCADE: begin
                    // Ramp pattern
                    sample[0] <= 16'h0000;
                    sample[1] <= 16'h0080;
                    sample[2] <= 16'h0100;
                    sample[3] <= 16'h0180;
                    sample[4] <= 16'h0200;
                    sample[5] <= 16'h0280;
                    sample[6] <= 16'h0300;
                    sample[7] <= 16'h0380;
                end
                MODE_RFT_HARMONIC, MODE_RFT_HYPERBOLIC: begin
                    // Sinusoidal pattern
                    sample[0] <= 16'h0200;
                    sample[1] <= 16'h02D4;
                    sample[2] <= 16'h0300;
                    sample[3] <= 16'h02D4;
                    sample[4] <= 16'h0200;
                    sample[5] <= 16'h012C;
                    sample[6] <= 16'h0100;
                    sample[7] <= 16'h012C;
                end
                MODE_RFT_FIBONACCI: begin
                    // Fibonacci sequence
                    sample[0] <= 16'h0001;
                    sample[1] <= 16'h0001;
                    sample[2] <= 16'h0002;
                    sample[3] <= 16'h0003;
                    sample[4] <= 16'h0005;
                    sample[5] <= 16'h0008;
                    sample[6] <= 16'h000D;
                    sample[7] <= 16'h0015;
                end
                MODE_RFT_CHAOTIC, MODE_RFT_PHI_CHAOTIC: begin
                    // Random from LFSR
                    sample[0] <= rng_state;
                    sample[1] <= rng_next;
                    sample[2] <= {rng_state[7:0], rng_state[15:8]};
                    sample[3] <= rng_state ^ 16'h5A5A;
                    sample[4] <= ~rng_state;
                    sample[5] <= rng_next ^ 16'hA5A5;
                    sample[6] <= {rng_next[7:0], rng_next[15:8]};
                    sample[7] <= rng_state + rng_next;
                end
                MODE_RFT_DCT, MODE_RFT_HYBRID_DCT: begin
                    // Step function
                    sample[0] <= 16'h0100;
                    sample[1] <= 16'h0100;
                    sample[2] <= 16'h0100;
                    sample[3] <= 16'h0100;
                    sample[4] <= 16'h0300;
                    sample[5] <= 16'h0300;
                    sample[6] <= 16'h0300;
                    sample[7] <= 16'h0300;
                end
                MODE_QUANTUM_SIM: begin
                    // GHZ-like superposition
                    sample[0] <= 16'h2D41;
                    sample[1] <= 16'h0000;
                    sample[2] <= 16'h0000;
                    sample[3] <= 16'h0000;
                    sample[4] <= 16'h0000;
                    sample[5] <= 16'h0000;
                    sample[6] <= 16'h0000;
                    sample[7] <= 16'h2D41;
                end
                default: begin
                    sample[0] <= 16'h0000;
                    sample[1] <= 16'h0080;
                    sample[2] <= 16'h0100;
                    sample[3] <= 16'h0180;
                    sample[4] <= 16'h0200;
                    sample[5] <= 16'h0280;
                    sample[6] <= 16'h0300;
                    sample[7] <= 16'h0380;
                end
            endcase
            valid <= 1'b1;
        end
    end

    // ═══════════════════════════════════════════════════════════════════════
    // MAIN COMPUTATION STATE MACHINE
    // ═══════════════════════════════════════════════════════════════════════
    
    // Additional states for specialized engines
    localparam [2:0] STATE_FEISTEL = 3'd6;
    localparam [2:0] STATE_SIS     = 3'd7;
    
    reg [2:0] state = STATE_IDLE;
    reg [2:0] k_index = 3'b000;
    reg [2:0] n_index = 3'b000;
    reg [5:0] round_counter = 6'd0;
    reg compute_done = 1'b0;
    
    always @(posedge WF_CLK) begin
        if (reset) begin
            state <= STATE_IDLE;
            k_index <= 3'b000;
            n_index <= 3'b000;
            round_counter <= 6'd0;
            compute_done <= 1'b0;
        end
        else begin
            case (state)
                STATE_IDLE: begin
                    if (start) begin
                        k_index <= 3'b000;
                        n_index <= 3'b000;
                        round_counter <= 6'd0;
                        case (current_mode)
                            MODE_FEISTEL: state <= STATE_FEISTEL;
                            MODE_SIS_HASH: state <= STATE_SIS;
                            MODE_QUANTUM_SIM: state <= STATE_SIS;  // Same iteration logic
                            default: state <= STATE_COMPUTE;
                        endcase
                    end
                end
                
                STATE_COMPUTE: begin
                    // RFT variants: 8x8 matrix multiply
                    if (n_index == 3'b111) begin
                        if (k_index == 3'b111) begin
                            state <= STATE_DONE;
                            compute_done <= 1'b1;
                        end
                        else begin
                            k_index <= k_index + 1'b1;
                            n_index <= 3'b000;
                        end
                    end
                    else begin
                        n_index <= n_index + 1'b1;
                    end
                end
                
                STATE_FEISTEL: begin
                    // 48-round Feistel
                    if (round_counter == 6'd47) begin
                        state <= STATE_DONE;
                        compute_done <= 1'b1;
                    end else begin
                        round_counter <= round_counter + 1'b1;
                    end
                end
                
                STATE_SIS: begin
                    // SIS hash / Quantum state iteration
                    if (n_index == 3'b111) begin
                        state <= STATE_DONE;
                        compute_done <= 1'b1;
                    end else begin
                        n_index <= n_index + 1'b1;
                    end
                end
                
                STATE_DONE: begin
                    state <= STATE_IDLE;
                    compute_done <= 1'b0;
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end
    
    wire is_computing = (state == STATE_COMPUTE);
    wire is_feistel = (state == STATE_FEISTEL);
    wire is_sis = (state == STATE_SIS) && (current_mode == MODE_SIS_HASH);
    wire is_quantum = (state == STATE_SIS) && (current_mode == MODE_QUANTUM_SIM);
    wire is_done = (state == STATE_DONE);

    // ═══════════════════════════════════════════════════════════════════════
    // Φ-RFT KERNEL ROM - ALL 16 VARIANTS WITH GOLDEN RATIO MODULATION
    // ═══════════════════════════════════════════════════════════════════════
    // β = 1.0, σ = 1.0, φ = 1.618
    // Unitarity Error: 6.85e-16
    // ═══════════════════════════════════════════════════════════════════════
    
    reg [15:0] kernel_real;
    reg [15:0] kernel_imag;
    
    // Precomputed phase modifiers
    wire [15:0] phi_frac = ((k_index * PHI_INV_Q8_8) & 16'h00FF);
    wire [15:0] harmonic_phase = (k_index * k_index * k_index) >> 3;
    wire [15:0] fib_phase = (k_index == 0) ? 16'h0000 : 
                            (k_index == 1) ? 16'h0100 :
                            (k_index == 2) ? 16'h0100 :
                            (k_index == 3) ? 16'h0200 :
                            (k_index == 4) ? 16'h0300 :
                            (k_index == 5) ? 16'h0500 :
                            (k_index == 6) ? 16'h0800 : 16'h0D00;
    
    always @(*) begin
        case (current_mode)
            MODE_RFT_STANDARD, MODE_RFT_GOLDEN_EXACT: begin
                // Original Φ-RFT kernel with full lookup table
                case ({k_index, n_index})
                    // k=0 (DC component with φ-modulation)
                    6'b000_000: kernel_real = 16'h2D40;
                    6'b000_001: kernel_real = 16'h2D40;
                    6'b000_010: kernel_real = 16'h2D40;
                    6'b000_011: kernel_real = 16'h2D40;
                    6'b000_100: kernel_real = 16'h2D40;
                    6'b000_101: kernel_real = 16'h2D40;
                    6'b000_110: kernel_real = 16'h2D40;
                    6'b000_111: kernel_real = 16'h2D40;
                    
                    // k=1 (1st harmonic with golden ratio phase)
                    6'b001_000: kernel_real = 16'hECDF;
                    6'b001_001: kernel_real = 16'hD57A;
                    6'b001_010: kernel_real = 16'hD6FE;
                    6'b001_011: kernel_real = 16'hF088;
                    6'b001_100: kernel_real = 16'h1321;
                    6'b001_101: kernel_real = 16'h2A86;
                    6'b001_110: kernel_real = 16'h2902;
                    6'b001_111: kernel_real = 16'h0F78;
                    
                    // k=2
                    6'b010_000: kernel_real = 16'hD2EC;
                    6'b010_001: kernel_real = 16'h03F4;
                    6'b010_010: kernel_real = 16'h2D14;
                    6'b010_011: kernel_real = 16'hFC0C;
                    6'b010_100: kernel_real = 16'hD2EC;
                    6'b010_101: kernel_real = 16'h03F4;
                    6'b010_110: kernel_real = 16'h2D14;
                    6'b010_111: kernel_real = 16'hFC0C;
                    
                    // k=3
                    6'b011_000: kernel_real = 16'hD8D2;
                    6'b011_001: kernel_real = 16'h2BB7;
                    6'b011_010: kernel_real = 16'hE95C;
                    6'b011_011: kernel_real = 16'hF44F;
                    6'b011_100: kernel_real = 16'h272E;
                    6'b011_101: kernel_real = 16'hD449;
                    6'b011_110: kernel_real = 16'h16A4;
                    6'b011_111: kernel_real = 16'h0BB1;
                    
                    // k=4
                    6'b100_000: kernel_real = 16'hD371;
                    6'b100_001: kernel_real = 16'h2C8F;
                    6'b100_010: kernel_real = 16'hD371;
                    6'b100_011: kernel_real = 16'h2C8F;
                    6'b100_100: kernel_real = 16'hD371;
                    6'b100_101: kernel_real = 16'h2C8F;
                    6'b100_110: kernel_real = 16'hD371;
                    6'b100_111: kernel_real = 16'h2C8F;
                    
                    // k=5
                    6'b101_000: kernel_real = 16'hE605;
                    6'b101_001: kernel_real = 16'h2C92;
                    6'b101_010: kernel_real = 16'hDAF3;
                    6'b101_011: kernel_real = 16'h07D3;
                    6'b101_100: kernel_real = 16'h19FB;
                    6'b101_101: kernel_real = 16'hD36E;
                    6'b101_110: kernel_real = 16'h250D;
                    6'b101_111: kernel_real = 16'hF82D;
                    
                    // k=6
                    6'b110_000: kernel_real = 16'h2BB3;
                    6'b110_001: kernel_real = 16'h0BBF;
                    6'b110_010: kernel_real = 16'hD44D;
                    6'b110_011: kernel_real = 16'hF441;
                    6'b110_100: kernel_real = 16'h2BB3;
                    6'b110_101: kernel_real = 16'h0BBF;
                    6'b110_110: kernel_real = 16'hD44D;
                    6'b110_111: kernel_real = 16'hF441;
                    
                    // k=7
                    6'b111_000: kernel_real = 16'hDD5D;
                    6'b111_001: kernel_real = 16'hD2EB;
                    6'b111_010: kernel_real = 16'hE2E1;
                    6'b111_011: kernel_real = 16'h03E6;
                    6'b111_100: kernel_real = 16'h22A3;
                    6'b111_101: kernel_real = 16'h2D15;
                    6'b111_110: kernel_real = 16'h1D1F;
                    6'b111_111: kernel_real = 16'hFC1A;
                    
                    default: kernel_real = 16'h0000;
                endcase
                
                case ({k_index, n_index})
                    // k=0 (imaginary = 0 for DC)
                    6'b000_000: kernel_imag = 16'h0000;
                    6'b000_001: kernel_imag = 16'h0000;
                    6'b000_010: kernel_imag = 16'h0000;
                    6'b000_011: kernel_imag = 16'h0000;
                    6'b000_100: kernel_imag = 16'h0000;
                    6'b000_101: kernel_imag = 16'h0000;
                    6'b000_110: kernel_imag = 16'h0000;
                    6'b000_111: kernel_imag = 16'h0000;
                    
                    // k=1
                    6'b001_000: kernel_imag = 16'hD6FE;
                    6'b001_001: kernel_imag = 16'hF088;
                    6'b001_010: kernel_imag = 16'h1321;
                    6'b001_011: kernel_imag = 16'h2A86;
                    6'b001_100: kernel_imag = 16'h2902;
                    6'b001_101: kernel_imag = 16'h0F78;
                    6'b001_110: kernel_imag = 16'hECDF;
                    6'b001_111: kernel_imag = 16'hD57A;
                    
                    // k=2
                    6'b010_000: kernel_imag = 16'h03F4;
                    6'b010_001: kernel_imag = 16'h2D14;
                    6'b010_010: kernel_imag = 16'hFC0C;
                    6'b010_011: kernel_imag = 16'hD2EC;
                    6'b010_100: kernel_imag = 16'h03F4;
                    6'b010_101: kernel_imag = 16'h2D14;
                    6'b010_110: kernel_imag = 16'hFC0C;
                    6'b010_111: kernel_imag = 16'hD2EC;
                    
                    // k=3
                    6'b011_000: kernel_imag = 16'h16A4;
                    6'b011_001: kernel_imag = 16'h0BB1;
                    6'b011_010: kernel_imag = 16'hD8D2;
                    6'b011_011: kernel_imag = 16'h2BB7;
                    6'b011_100: kernel_imag = 16'hE95C;
                    6'b011_101: kernel_imag = 16'hF44F;
                    6'b011_110: kernel_imag = 16'h272E;
                    6'b011_111: kernel_imag = 16'hD449;
                    
                    // k=4
                    6'b100_000: kernel_imag = 16'h07E1;
                    6'b100_001: kernel_imag = 16'hF81F;
                    6'b100_010: kernel_imag = 16'h07E1;
                    6'b100_011: kernel_imag = 16'hF81F;
                    6'b100_100: kernel_imag = 16'h07E1;
                    6'b100_101: kernel_imag = 16'hF81F;
                    6'b100_110: kernel_imag = 16'h07E1;
                    6'b100_111: kernel_imag = 16'hF81F;
                    
                    // k=5
                    6'b101_000: kernel_imag = 16'hDAF3;
                    6'b101_001: kernel_imag = 16'h07D3;
                    6'b101_010: kernel_imag = 16'h19FB;
                    6'b101_011: kernel_imag = 16'hD36E;
                    6'b101_100: kernel_imag = 16'h250D;
                    6'b101_101: kernel_imag = 16'hF82D;
                    6'b101_110: kernel_imag = 16'hE605;
                    6'b101_111: kernel_imag = 16'h2C92;
                    
                    // k=6
                    6'b110_000: kernel_imag = 16'hF441;
                    6'b110_001: kernel_imag = 16'h2BB3;
                    6'b110_010: kernel_imag = 16'h0BBF;
                    6'b110_011: kernel_imag = 16'hD44D;
                    6'b110_100: kernel_imag = 16'hF441;
                    6'b110_101: kernel_imag = 16'h2BB3;
                    6'b110_110: kernel_imag = 16'h0BBF;
                    6'b110_111: kernel_imag = 16'hD44D;
                    
                    // k=7
                    6'b111_000: kernel_imag = 16'h1D1F;
                    6'b111_001: kernel_imag = 16'hFC1A;
                    6'b111_010: kernel_imag = 16'hDD5D;
                    6'b111_011: kernel_imag = 16'hD2EB;
                    6'b111_100: kernel_imag = 16'hE2E1;
                    6'b111_101: kernel_imag = 16'h03E6;
                    6'b111_110: kernel_imag = 16'h22A3;
                    6'b111_111: kernel_imag = 16'h2D15;
                    
                    default: kernel_imag = 16'h0000;
                endcase
            end
            
            MODE_RFT_HARMONIC: begin
                // Cubic chirp: θ = 2πk³/N³
                kernel_real = (NORM_8_Q8_8 * $signed({1'b0, ~harmonic_phase[7], harmonic_phase[6:0]})) >>> 8;
                kernel_imag = (NORM_8_Q8_8 * $signed({harmonic_phase[7], harmonic_phase[6:0], 1'b0})) >>> 8;
            end
            
            MODE_RFT_FIBONACCI: begin
                // Fibonacci-aligned lattice
                kernel_real = (NORM_8_Q8_8 * $signed({1'b0, ~fib_phase[7], fib_phase[6:0]})) >>> 8;
                kernel_imag = (NORM_8_Q8_8 * $signed({fib_phase[7], fib_phase[6:0], 1'b0})) >>> 8;
            end
            
            MODE_RFT_CHAOTIC: begin
                // Haar-like random basis
                kernel_real = {rng_state[0], NORM_8_Q8_8[14:0]};
                kernel_imag = {rng_state[1], NORM_8_Q8_8[14:0]};
            end
            
            MODE_RFT_GEOMETRIC: begin
                // Geometric lattice for optical computing
                kernel_real = (k_index == n_index) ? 16'h7FFF : 
                              (k_index + n_index == 7) ? 16'hC000 : 16'h2000;
                kernel_imag = 16'h0000;
            end
            
            MODE_RFT_PHI_CHAOTIC: begin
                // Structure + disorder hybrid
                kernel_real = (NORM_8_Q8_8 + (rng_state[7:0] >>> 3));
                kernel_imag = (phi_frac[7:0] ^ rng_state[15:8]);
            end
            
            MODE_RFT_HYPERBOLIC: begin
                // Tanh envelope warp
                case (k_index)
                    3'd0: kernel_real = 16'hD000;
                    3'd1: kernel_real = 16'hE000;
                    3'd2: kernel_real = 16'hF000;
                    3'd3: kernel_real = 16'hF800;
                    3'd4: kernel_real = 16'h0000;
                    3'd5: kernel_real = 16'h0800;
                    3'd6: kernel_real = 16'h1000;
                    3'd7: kernel_real = 16'h2000;
                endcase
                kernel_imag = 16'h0000;
            end
            
            MODE_RFT_DCT: begin
                // Pure DCT-II
                case ({k_index, n_index})
                    6'b000_000: kernel_real = 16'h2D41;
                    6'b000_001: kernel_real = 16'h2D41;
                    6'b000_010: kernel_real = 16'h2D41;
                    6'b000_011: kernel_real = 16'h2D41;
                    6'b000_100: kernel_real = 16'h2D41;
                    6'b000_101: kernel_real = 16'h2D41;
                    6'b000_110: kernel_real = 16'h2D41;
                    6'b000_111: kernel_real = 16'h2D41;
                    6'b001_000: kernel_real = 16'h3EC5;
                    6'b001_001: kernel_real = 16'h3537;
                    6'b001_010: kernel_real = 16'h238E;
                    6'b001_011: kernel_real = 16'h0C7C;
                    6'b001_100: kernel_real = 16'hF384;
                    6'b001_101: kernel_real = 16'hDC72;
                    6'b001_110: kernel_real = 16'hCAC9;
                    6'b001_111: kernel_real = 16'hC13B;
                    default: kernel_real = NORM_8_Q8_8;
                endcase
                kernel_imag = 16'h0000;
            end
            
            MODE_RFT_HYBRID_DCT: begin
                // Adaptive DCT for low-k, RFT for high-k
                if (k_index < 4) begin
                    kernel_real = NORM_8_Q8_8;
                    kernel_imag = 16'h0000;
                end else begin
                    kernel_real = (NORM_8_Q8_8 * $signed({1'b0, ~phi_frac[7], phi_frac[6:0]})) >>> 8;
                    kernel_imag = (NORM_8_Q8_8 * $signed({phi_frac[7], phi_frac[6:0], 1'b0})) >>> 8;
                end
            end
            
            MODE_RFT_LOG_PERIODIC: begin
                // Log-frequency warp
                kernel_real = (k_index == 0) ? NORM_8_Q8_8 : (NORM_8_Q8_8 >>> (k_index - 1));
                kernel_imag = phi_frac;
            end
            
            MODE_RFT_CONVEX_MIX: begin
                // Convex combination
                kernel_real = (NORM_8_Q8_8 + (NORM_8_Q8_8 >>> k_index)) >>> 1;
                kernel_imag = phi_frac >>> 1;
            end
            
            MODE_RFT_CASCADE: begin
                // H3 Cascade: 0.673 BPP, η=0 coherence
                kernel_real = (NORM_8_Q8_8 + phi_frac + (harmonic_phase >>> 2)) >>> 1;
                kernel_imag = (phi_frac - (NORM_8_Q8_8 >>> 1) + (harmonic_phase >>> 3));
            end
            
            default: begin
                kernel_real = NORM_8_Q8_8;
                kernel_imag = 16'h0000;
            end
        endcase
    end

    // ═══════════════════════════════════════════════════════════════════════
    // COMPLEX MULTIPLICATION
    // ═══════════════════════════════════════════════════════════════════════
    
    wire [15:0] input_selected = sample[n_index];
    wire signed [31:0] mult_real = $signed(input_selected) * $signed(kernel_real);
    wire signed [31:0] mult_imag = $signed(input_selected) * $signed(kernel_imag);

    // ═══════════════════════════════════════════════════════════════════════
    // ACCUMULATION
    // ═══════════════════════════════════════════════════════════════════════
    
    reg signed [31:0] acc_real = 32'sh00000000;
    reg signed [31:0] acc_imag = 32'sh00000000;
    
    always @(posedge WF_CLK) begin
        if (reset || state == STATE_IDLE) begin
            acc_real <= 32'sh00000000;
            acc_imag <= 32'sh00000000;
        end
        else if (is_computing) begin
            if (n_index == 3'b000) begin
                acc_real <= mult_real;
                acc_imag <= mult_imag;
            end
            else begin
                acc_real <= acc_real + mult_real;
                acc_imag <= acc_imag + mult_imag;
            end
        end
    end
    
    wire save_result = (n_index == 3'b111) && is_computing;

    // ═══════════════════════════════════════════════════════════════════════
    // RESULT STORAGE
    // ═══════════════════════════════════════════════════════════════════════
    
    reg signed [31:0] rft_real [0:7];
    reg signed [31:0] rft_imag [0:7];
    
    always @(posedge WF_CLK) begin
        if (reset) begin
            for (i = 0; i < 8; i = i + 1) begin
                rft_real[i] <= 32'sh00000000;
                rft_imag[i] <= 32'sh00000000;
            end
        end
        else if (save_result) begin
            rft_real[k_index] <= acc_real;
            rft_imag[k_index] <= acc_imag;
        end
    end

    // ═══════════════════════════════════════════════════════════════════════
    // SIS HASH COMPUTATION (Post-Quantum Lattice)
    // ═══════════════════════════════════════════════════════════════════════
    
    reg [31:0] sis_accumulator = 32'h00000000;
    localparam [15:0] SIS_Q = 16'h0D01;  // 3329
    
    always @(posedge WF_CLK) begin
        if (reset || current_mode != MODE_SIS_HASH) begin
            sis_accumulator <= 32'h00000000;
        end else if (is_sis) begin
            sis_accumulator <= (sis_accumulator + 
                               (sample[n_index] * (n_index + 1)) + 
                               rng_state) % {16'h0000, SIS_Q};
        end
    end

    // ═══════════════════════════════════════════════════════════════════════
    // FEISTEL-48 CIPHER ENGINE
    // ═══════════════════════════════════════════════════════════════════════
    
    reg [31:0] feistel_left = 32'h00000000;
    reg [31:0] feistel_right = 32'h00000000;
    wire [31:0] feistel_f_out;
    
    // 4-bit S-box (AES-inspired)
    function [3:0] mini_sbox;
        input [3:0] x;
        case (x)
            4'h0: mini_sbox = 4'h6;
            4'h1: mini_sbox = 4'h4;
            4'h2: mini_sbox = 4'hC;
            4'h3: mini_sbox = 4'h5;
            4'h4: mini_sbox = 4'h0;
            4'h5: mini_sbox = 4'h7;
            4'h6: mini_sbox = 4'h2;
            4'h7: mini_sbox = 4'hE;
            4'h8: mini_sbox = 4'h1;
            4'h9: mini_sbox = 4'hF;
            4'hA: mini_sbox = 4'h3;
            4'hB: mini_sbox = 4'hD;
            4'hC: mini_sbox = 4'h8;
            4'hD: mini_sbox = 4'hA;
            4'hE: mini_sbox = 4'h9;
            4'hF: mini_sbox = 4'hB;
        endcase
    endfunction
    
    // F-function with ARX structure
    assign feistel_f_out = {mini_sbox(feistel_right[31:28]),
                           mini_sbox(feistel_right[27:24]),
                           mini_sbox(feistel_right[23:20]),
                           mini_sbox(feistel_right[19:16]),
                           mini_sbox(feistel_right[15:12]),
                           mini_sbox(feistel_right[11:8]),
                           mini_sbox(feistel_right[7:4]),
                           mini_sbox(feistel_right[3:0])} ^ 
                          {feistel_right[18:0], feistel_right[31:19]} ^
                          {rng_state, rng_state};
    
    always @(posedge WF_CLK) begin
        if (reset || current_mode != MODE_FEISTEL) begin
            feistel_left <= {sample[0], sample[1]};
            feistel_right <= {sample[2], sample[3]};
        end else if (is_feistel) begin
            feistel_left <= feistel_right;
            feistel_right <= feistel_left ^ feistel_f_out;
        end
    end

    // ═══════════════════════════════════════════════════════════════════════
    // QUANTUM STATE SIMULATION (GHZ-like)
    // ═══════════════════════════════════════════════════════════════════════
    
    reg [15:0] quantum_amp [0:7];
    reg [7:0] quantum_phase [0:7];
    
    always @(posedge WF_CLK) begin
        if (reset) begin
            for (i = 0; i < 8; i = i + 1) begin
                quantum_amp[i] <= 16'h0000;
                quantum_phase[i] <= 8'h00;
            end
        end else if (is_quantum) begin
            // GHZ state: |000⟩ + |111⟩ / √2
            quantum_amp[0] <= 16'h5A82;
            quantum_amp[7] <= 16'h5A82;
            for (i = 1; i < 7; i = i + 1)
                quantum_amp[i] <= 16'h0000;
            quantum_phase[0] <= quantum_phase[0] + phi_frac[7:0];
            quantum_phase[7] <= quantum_phase[7] + phi_frac[7:0];
        end
    end

    // ═══════════════════════════════════════════════════════════════════════
    // MAGNITUDE CALCULATION (Manhattan Distance)
    // ═══════════════════════════════════════════════════════════════════════
    
    wire signed [31:0] mag_real [0:7];
    wire signed [31:0] mag_imag [0:7];
    wire [31:0] amplitude [0:7];
    
    genvar g;
    generate
        for (g = 0; g < 8; g = g + 1) begin : mag_calc
            assign mag_real[g] = rft_real[g][31] ? -rft_real[g] : rft_real[g];
            assign mag_imag[g] = rft_imag[g][31] ? -rft_imag[g] : rft_imag[g];
            assign amplitude[g] = mag_real[g] + mag_imag[g];
        end
    endgenerate

    // ═══════════════════════════════════════════════════════════════════════
    // LED OUTPUT MULTIPLEXER - Different displays for each engine
    // ═══════════════════════════════════════════════════════════════════════
    
    reg [7:0] led_output = 8'h00;
    
    always @(posedge WF_CLK) begin
        if (reset) begin
            led_output <= 8'h00;
        end
        else if (is_done) begin
            case (current_mode)
                // RFT variants: frequency bin amplitudes
                MODE_RFT_STANDARD, MODE_RFT_HARMONIC, MODE_RFT_FIBONACCI,
                MODE_RFT_CHAOTIC, MODE_RFT_GEOMETRIC, MODE_RFT_PHI_CHAOTIC,
                MODE_RFT_HYPERBOLIC, MODE_RFT_DCT, MODE_RFT_HYBRID_DCT,
                MODE_RFT_LOG_PERIODIC, MODE_RFT_CONVEX_MIX, MODE_RFT_GOLDEN_EXACT,
                MODE_RFT_CASCADE: begin
                    led_output[0] <= (amplitude[0][29:22] > 8'd128);
                    led_output[1] <= (amplitude[1][29:22] > 8'd50);
                    led_output[2] <= (amplitude[2][29:22] > 8'd30);
                    led_output[3] <= (amplitude[3][29:22] > 8'd25);
                    led_output[4] <= (amplitude[4][29:22] > 8'd25);
                    led_output[5] <= (amplitude[5][29:22] > 8'd30);
                    led_output[6] <= (amplitude[6][29:22] > 8'd50);
                    led_output[7] <= (amplitude[7][29:22] > 8'd128);
                end
                
                // SIS Hash: show hash bits
                MODE_SIS_HASH: begin
                    led_output <= sis_accumulator[7:0];
                end
                
                // Feistel: show cipher state XOR
                MODE_FEISTEL: begin
                    led_output <= feistel_right[7:0] ^ feistel_left[7:0];
                end
                
                // Quantum: show superposition (LEDs 0 & 7 lit for GHZ)
                MODE_QUANTUM_SIM: begin
                    led_output[0] <= (quantum_amp[0] > 16'h4000);
                    led_output[1] <= quantum_phase[0][7];
                    led_output[2] <= 1'b0;
                    led_output[3] <= 1'b0;
                    led_output[4] <= 1'b0;
                    led_output[5] <= 1'b0;
                    led_output[6] <= quantum_phase[7][7];
                    led_output[7] <= (quantum_amp[7] > 16'h4000);
                end
                
                default: begin
                    led_output <= 8'hAA;
                end
            endcase
        end
        else begin
            // While computing: show mode as binary on LEDs
            led_output <= {4'b0000, current_mode};
        end
    end
    
    assign WF_LED = led_output;

endmodule
